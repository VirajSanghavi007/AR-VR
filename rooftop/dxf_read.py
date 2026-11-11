"""Phase 2, M2: read a rooftop drawing (DXF) that follows the layer rules in
docs/phase2_dxf.md into a RooftopSpec plus the rooftop structures.

Everything comes out in the plan frame of phase2: the origin is the roof outline's
south-west (min x, min y) corner. The spec stays in mm (as RooftopSpec expects);
features, doors, drains and pipes are in metres (as phase2.Feature expects).

Labels (text such as "H=2700") belong to the nearest shape on their own layer.
A real drawing with other layer names is read with a layers.json mapping:
    {"ROOF_OUTLINE": "A-ROOF-OTLN", "TANK": ["P-TANK", "P-TANK-OH"], ...}
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import shapely
import shapely.affinity
from shapely.geometry import LineString, Point, Polygon

from .cad import RooftopSpec
from .phase2 import Feature

LAYER_NAMES = ("ROOF_OUTLINE", "PARAPET", "STAIR", "STORE", "DOOR", "TANK", "DRAIN", "PIPE",
               "NORTH", "DIMENSIONS", "SITE_INFO")
REQUIRED = ("ROOF_OUTLINE", "PARAPET")
UNIT_MM = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}   # $INSUNITS code: mm per unit
UNIT_CODES = {"in": 1, "ft": 2, "mm": 4, "cm": 5, "m": 6}
TOL = 1.0          # mm: how far a door may sit off a room edge, how far a polyline may miss closing
LABEL_RE = re.compile(r"([A-Za-z_]+)\s*=\s*(-?\d+(?:\.\d+)?)")


class DrawingError(ValueError):
    """The drawing breaks a layer rule; the message says what and where."""


@dataclass(frozen=True)
class Door:
    room: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass(frozen=True)
class Pipe:
    points: tuple          # ((x, y), ...) metres
    diameter: float        # mm


@dataclass(frozen=True)
class Dimension:
    p1: tuple              # metres, plan frame
    p2: tuple
    value: float           # mm, as the drawing states it


@dataclass
class Drawing:
    spec: RooftopSpec
    features: list
    doors: list = field(default_factory=list)
    drains: list = field(default_factory=list)
    pipes: list = field(default_factory=list)
    north_deg: float = 90.0            # direction of north in the plan, degrees from +x
    site: dict = field(default_factory=dict)
    dimensions: list = field(default_factory=list)

    @property
    def latitude(self):
        return self.site.get("LAT")

    def facing_deg(self):
        """Plan direction the panels should face: the equator side (south in the north)."""
        lat = self.latitude if self.latitude is not None else 0.0
        return (self.north_deg + (180.0 if lat >= 0 else 0.0)) % 360.0


def load_mapping(path):
    """layers.json -> {canonical name: [drawing layer names]}; unknown keys are an error."""
    data = json.loads(Path(path).read_text())
    bad = sorted(set(data) - set(LAYER_NAMES))
    if bad:
        raise DrawingError(f"layers.json names unknown layers {bad}; known: {list(LAYER_NAMES)}")
    return {k: [v] if isinstance(v, str) else list(v) for k, v in data.items()}


def parse_labels(text):
    """'H=1500 S=600' -> {'H': 1500.0, 'S': 600.0}; a key given twice is an error."""
    out = {}
    for k, v in LABEL_RE.findall(text):
        k = k.upper()
        if k in out and out[k] != float(v):
            raise DrawingError(f"label {text!r} gives {k} twice")
        out[k] = float(v)
    return out


def unit_scale(doc, units=None):
    """mm per drawing unit, from units ('mm', 'm', ...) or else the drawing's $INSUNITS."""
    if units is not None:
        if units not in UNIT_CODES:
            raise DrawingError(f"unknown unit {units!r}; use one of {list(UNIT_CODES)}")
        return UNIT_MM[UNIT_CODES[units]]
    code = doc.header.get("$INSUNITS", 0)
    if code not in UNIT_MM:
        raise DrawingError(f"the drawing has no usable unit ($INSUNITS={code}); set it in AutoCAD "
                           f"(UNITS command) or pass units='mm'")
    return UNIT_MM[code]


def _entities(msp, names):
    wanted = {n.upper() for n in names}
    return [e for e in msp if e.dxf.get("layer", "0").upper() in wanted]


def _poly_points(e, layer):
    """2D vertices (drawing units) and whether the polyline is closed."""
    t = e.dxftype()
    if t == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in e.get_points("xyb")]
        if any(abs(p[2]) > 1e-9 for p in e.get_points("xyb")):
            raise DrawingError(f"{layer}: polyline {e.dxf.handle} has arcs (bulges); draw straight segments")
        closed = e.closed
    elif t == "POLYLINE":
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        closed = e.is_closed
    else:
        raise DrawingError(f"{layer}: expected a polyline, found {t} ({e.dxf.handle})")
    if len(pts) > 2 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts, closed = pts[:-1], True
    return pts, closed


def _closed_polygon(e, layer, scale):
    pts, closed = _poly_points(e, layer)
    if not closed:
        raise DrawingError(f"{layer}: polyline {e.dxf.handle} is open; close it (PEDIT > Close)")
    if len(pts) < 3:
        raise DrawingError(f"{layer}: polyline {e.dxf.handle} has fewer than 3 vertices")
    poly = Polygon([(x * scale, y * scale) for x, y in pts])
    if not poly.is_valid or poly.area <= 0:
        raise DrawingError(f"{layer}: polyline {e.dxf.handle} crosses itself")
    return poly


def _texts(msp, names, scale):
    out = []
    for e in _entities(msp, names):
        if e.dxftype() == "TEXT":
            s, p = e.dxf.text, e.dxf.insert
        elif e.dxftype() == "MTEXT":
            s, p = e.plain_text(), e.dxf.insert
        else:
            continue
        out.append((s, Point(p.x * scale, p.y * scale)))
    return out


def _assign_labels(shapes, texts, layer):
    """Give each text's labels to the nearest shape; returns one dict per shape."""
    labels = [{} for _ in shapes]
    for s, pt in texts:
        lab = parse_labels(s)
        if not lab or not shapes:
            continue
        i = min(range(len(shapes)), key=lambda j: shapes[j].distance(pt))
        for k, v in lab.items():
            if k in labels[i] and labels[i][k] != v:
                raise DrawingError(f"{layer}: two labels give {k} for the same shape")
            labels[i][k] = v
    return labels


def _need(label, key, layer, i):
    if key not in label:
        raise DrawingError(f"{layer}: shape {i + 1} has no {key}= label (e.g. '{key}=2700' next to it)")
    return label[key]


def _rect(poly, layer, i):
    if abs(poly.area - poly.envelope.area) > TOL * poly.length:
        raise DrawingError(f"{layer}: shape {i + 1} is not an axis-aligned rectangle")
    return poly.bounds


def spec_from_outline(outline, inner, parapet_h, slab=150.0):
    """RooftopSpec (mm) from the outline and parapet inner edge, both already at the origin.
    Supported: a rectangle, or an L with the north-east (+x, +y) corner cut away."""
    x0, y0, L, W = outline.bounds
    if abs(x0) > 1e-6 or abs(y0) > 1e-6:
        raise DrawingError("outline is not at the origin")
    notch = shapely.box(0, 0, L, W).difference(outline)
    nl = nw = 0.0
    if notch.area > TOL * TOL:
        nx0, ny0, nx1, ny1 = notch.bounds
        if (abs(notch.area - notch.envelope.area) > TOL * max(L, W)
                or abs(nx1 - L) > TOL or abs(ny1 - W) > TOL):
            raise DrawingError("ROOF_OUTLINE: only a rectangle or an L with the north-east (+x, +y) "
                               "corner cut away is supported for now")
        nl, nw = L - nx0, W - ny0
    t = inner.bounds[0]
    ring = outline.buffer(-t, join_style=2)
    if t <= 0 or ring.is_empty or ring.symmetric_difference(inner).area > TOL * outline.length:
        raise DrawingError("PARAPET: the inner edge must run at one even distance inside ROOF_OUTLINE")
    return RooftopSpec(round(L, 3), round(W, 3), slab, parapet_h, round(t, 3), round(nl, 3), round(nw, 3))


def read_dxf(path, layers=None, units=None, slab_thickness=150.0):
    """Read a rooftop drawing. layers: a {canonical: [names]} mapping or a layers.json path."""
    import ezdxf

    path = Path(path)
    if not path.exists():
        raise DrawingError(f"no such drawing: {path}")
    try:
        doc = ezdxf.readfile(str(path))
    except (IOError, ezdxf.DXFStructureError) as e:
        raise DrawingError(f"{path.name} is not a readable DXF ({e}); save it from AutoCAD as DXF") from e
    if isinstance(layers, (str, Path)):
        layers = load_mapping(layers)
    names = {k: (layers or {}).get(k, [k]) for k in LAYER_NAMES}
    msp, sc = doc.modelspace(), unit_scale(doc, units)
    present = {e.dxf.get("layer", "0").upper() for e in msp}
    for k in REQUIRED:
        if not present & {n.upper() for n in names[k]}:
            raise DrawingError(f"layer {k} (drawing name {names[k]}) has nothing on it")

    def polys(k):
        return [_closed_polygon(e, k, sc) for e in _entities(msp, names[k])
                if e.dxftype() in ("LWPOLYLINE", "POLYLINE")]

    outs = polys("ROOF_OUTLINE")
    if len(outs) != 1:
        raise DrawingError(f"ROOF_OUTLINE needs exactly one closed polyline, found {len(outs)}")
    ox, oy = outs[0].bounds[:2]

    def mv(g):
        return shapely.affinity.translate(g, -ox, -oy)

    outline = mv(outs[0])
    pars = polys("PARAPET")
    if len(pars) != 1:
        raise DrawingError(f"PARAPET needs exactly one closed polyline, found {len(pars)}")
    inner = mv(pars[0])
    plab = _assign_labels([inner.exterior], [(s, mv(p)) for s, p in _texts(msp, names["PARAPET"], sc)], "PARAPET")
    spec = spec_from_outline(outline, inner, _need(plab[0], "H", "PARAPET", 0), slab_thickness)

    features, rooms = [], {}
    for kind in ("stair", "store"):
        k = kind.upper()
        shapes = [mv(p) for p in polys(k)]
        labs = _assign_labels(shapes, [(s, mv(p)) for s, p in _texts(msp, names[k], sc)], k)
        for i, (poly, lab) in enumerate(zip(shapes, labs)):
            x0, y0, x1, y1 = _rect(poly, k, i)
            name = f"{kind.capitalize()}{i + 1}"
            features.append(Feature(name, kind, x0 / 1000, y0 / 1000, (x1 - x0) / 1000, (y1 - y0) / 1000,
                                    _need(lab, "H", k, i) / 1000))
            rooms[name] = poly

    circles = [e for e in _entities(msp, names["TANK"]) if e.dxftype() == "CIRCLE"]
    tshapes = [mv(Point(c.dxf.center.x * sc, c.dxf.center.y * sc).buffer(c.dxf.radius * sc, 64))
               for c in circles]
    tlabs = _assign_labels(tshapes, [(s, mv(p)) for s, p in _texts(msp, names["TANK"], sc)], "TANK")
    for i, (c, lab) in enumerate(zip(circles, tlabs)):
        r = c.dxf.radius * sc
        cx, cy = c.dxf.center.x * sc - ox, c.dxf.center.y * sc - oy
        h, s = _need(lab, "H", "TANK", i), _need(lab, "S", "TANK", i)
        name = f"Tank{i + 1}"
        features.append(Feature(name, "tank", (cx - r) / 1000, (cy - r) / 1000, 2 * r / 1000, 2 * r / 1000,
                                (h + s) / 1000, stand=s / 1000))

    floor = inner.buffer(TOL)
    for f in features:
        if not floor.contains(shapely.box(f.x * 1000, f.y * 1000, (f.x + f.w) * 1000, (f.y + f.d) * 1000)):
            raise DrawingError(f"{f.name} lies outside the parapet's inner edge")

    doors = []
    for e in _entities(msp, names["DOOR"]):
        if e.dxftype() != "LINE":
            continue
        a, b = e.dxf.start, e.dxf.end
        seg = LineString([(a.x * sc - ox, a.y * sc - oy), (b.x * sc - ox, b.y * sc - oy)])
        host = [n for n, p in rooms.items() if p.exterior.buffer(TOL).contains(seg)]
        if not host:
            raise DrawingError(f"DOOR line {e.dxf.handle} does not lie on a STAIR or STORE room edge")
        (x0, y0), (x1, y1) = seg.coords
        doors.append(Door(host[0], x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000))

    drains = [((e.dxf.location.x * sc - ox) / 1000, (e.dxf.location.y * sc - oy) / 1000)
              for e in _entities(msp, names["DRAIN"]) if e.dxftype() == "POINT"]

    pents = [e for e in _entities(msp, names["PIPE"]) if e.dxftype() in ("LINE", "LWPOLYLINE", "POLYLINE")]
    plines = []
    for e in pents:
        pts = ([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)] if e.dxftype() == "LINE"
               else _poly_points(e, "PIPE")[0])
        plines.append(LineString([(x * sc - ox, y * sc - oy) for x, y in pts]))
    plabs = _assign_labels(plines, [(s, mv(p)) for s, p in _texts(msp, names["PIPE"], sc)], "PIPE")
    pipes = [Pipe(tuple((x / 1000, y / 1000) for x, y in ln.coords), _need(lab, "D", "PIPE", i))
             for i, (ln, lab) in enumerate(zip(plines, plabs))]

    north = 90.0
    nl = [e for e in _entities(msp, names["NORTH"]) if e.dxftype() == "LINE"]
    if len(nl) > 1:
        raise DrawingError(f"NORTH needs one line (tail to tip), found {len(nl)}")
    if nl:
        a, b = nl[0].dxf.start, nl[0].dxf.end
        if math.hypot(b.x - a.x, b.y - a.y) == 0:
            raise DrawingError("NORTH line has zero length")
        north = math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 360.0

    site = {}
    for s, _ in _texts(msp, names["SITE_INFO"], 1.0):
        site.update(parse_labels(s))

    dims = []
    for e in _entities(msp, names["DIMENSIONS"]):
        if e.dxftype() != "DIMENSION" or e.dimtype not in (0, 1):      # linear, aligned
            continue
        p1, p2 = e.dxf.defpoint2, e.dxf.defpoint3
        dims.append(Dimension(((p1.x * sc - ox) / 1000, (p1.y * sc - oy) / 1000),
                              ((p2.x * sc - ox) / 1000, (p2.y * sc - oy) / 1000), e.get_measurement() * sc))

    return Drawing(spec, features, doors, drains, pipes, north, site, dims)
