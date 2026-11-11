"""Phase 2, M1: a sample rooftop drawing (DXF, mm) that follows the layer rules in
docs/phase2_dxf.md, plus the exact true values it was drawn from.

The drawing stands in for a real AutoCAD file until one arrives. Every value in
it comes from SAMPLE below, and measurements_json() writes those same values out
independently of the reader, so reading the DXF back can be checked against them.

Plan: 18 x 12 m L-shaped roof, the north-east corner (6 x 4 m) cut away, north = +y.
Rooms and tanks sit at the north edge, so their winter shadows fall off the roof
for panels facing south (the equator side at LAT=19.07).
"""
import json
from pathlib import Path

LAYERS = {   # name: AutoCAD colour index
    "ROOF_OUTLINE": 7, "PARAPET": 8, "STAIR": 1, "STORE": 3, "DOOR": 6, "TANK": 5,
    "DRAIN": 4, "PIPE": 30, "NORTH": 2, "DIMENSIONS": 9, "SITE_INFO": 7,
}

SAMPLE = {
    "length": 18000, "width": 12000, "notch_length": 6000, "notch_width": 4000,
    "parapet_thickness": 230, "parapet_height": 1000,
    "rooms": [
        {"name": "Stair1", "kind": "stair", "x": 800, "y": 8400, "w": 3000, "d": 3200, "h": 2700,
         "door": [1800, 8400, 2800, 8400]},
        {"name": "Store1", "kind": "store", "x": 4400, "y": 9600, "w": 2400, "d": 2000, "h": 2700,
         "door": [5100, 9600, 6000, 9600]},
    ],
    "tanks": [
        {"name": "Tank1", "cx": 14000, "cy": 6800, "r": 600, "h": 1500, "s": 600},
        {"name": "Tank2", "cx": 16000, "cy": 6800, "r": 600, "h": 1500, "s": 600},
    ],
    "drains": [[600, 600], [17400, 600], [600, 11400]],
    "pipes": [{"points": [[14000, 6200], [14000, 3000], [12500, 3000]], "d": 50}],
    "north": [[9000, 4000], [9000, 5500]],
    "north_deg": 90.0,
    "site": {"lat": 19.07, "lon": 72.88, "tz": 5.5, "floors": 3, "floor_h": 3200},
    "dimensions": [
        {"name": "length", "p1": [0, 0], "p2": [18000, 0], "value": 18000, "base": [0, -1000], "angle": 0},
        {"name": "width", "p1": [0, 0], "p2": [0, 12000], "value": 12000, "base": [-1000, 0], "angle": 90},
        {"name": "notch_length", "p1": [12000, 12000], "p2": [18000, 12000], "value": 6000,
         "base": [0, 13000], "angle": 0},
        {"name": "notch_width", "p1": [18000, 8000], "p2": [18000, 12000], "value": 4000,
         "base": [19000, 0], "angle": 90},
        {"name": "stair_width", "p1": [800, 8400], "p2": [3800, 8400], "value": 3000,
         "base": [0, 7800], "angle": 0},
    ],
}


def outline_points(s=SAMPLE):
    L, W, nl, nw = s["length"], s["width"], s["notch_length"], s["notch_width"]
    return [(0, 0), (L, 0), (L, W - nw), (L - nl, W - nw), (L - nl, W), (0, W)]


def parapet_points(s=SAMPLE):
    t, L, W, nl, nw = s["parapet_thickness"], s["length"], s["width"], s["notch_length"], s["notch_width"]
    return [(t, t), (L - t, t), (L - t, W - nw - t), (L - nl - t, W - nw - t), (L - nl - t, W - t), (t, W - t)]


def build_doc(s=SAMPLE, offset=(0.0, 0.0), units=4):
    """The drawing as an ezdxf document. offset shifts everything (tests use it to check
    the reader moves the origin to the outline's south-west corner); units = $INSUNITS."""
    import ezdxf

    ox, oy = offset

    def P(x, y):
        return (x + ox, y + oy)

    doc = ezdxf.new("R2018", units=units, setup=True)
    for name, colour in LAYERS.items():
        doc.layers.add(name, color=colour)
    msp = doc.modelspace()

    def text(layer, txt, x, y, h=250):
        msp.add_text(txt, height=h, dxfattribs={"layer": layer}).set_placement(P(x, y))

    msp.add_lwpolyline([P(*p) for p in outline_points(s)], close=True, dxfattribs={"layer": "ROOF_OUTLINE"})
    msp.add_lwpolyline([P(*p) for p in parapet_points(s)], close=True, dxfattribs={"layer": "PARAPET"})
    t = s["parapet_thickness"]
    text("PARAPET", f"H={s['parapet_height']:g}", 2 * t, 2 * t + 100)
    for r in s["rooms"]:
        layer = r["kind"].upper()
        x, y, w, d = r["x"], r["y"], r["w"], r["d"]
        msp.add_lwpolyline([P(x, y), P(x + w, y), P(x + w, y + d), P(x, y + d)], close=True,
                           dxfattribs={"layer": layer})
        text(layer, f"H={r['h']:g}", x + 200, y + d / 2)
        x0, y0, x1, y1 = r["door"]
        msp.add_line(P(x0, y0), P(x1, y1), dxfattribs={"layer": "DOOR"})
    for k in s["tanks"]:
        msp.add_circle(P(k["cx"], k["cy"]), k["r"], dxfattribs={"layer": "TANK"})
        text("TANK", f"H={k['h']:g} S={k['s']:g}", k["cx"] - 500, k["cy"] - 150, 120)
    for x, y in s["drains"]:
        msp.add_point(P(x, y), dxfattribs={"layer": "DRAIN"})
    for p in s["pipes"]:
        msp.add_lwpolyline([P(*q) for q in p["points"]], dxfattribs={"layer": "PIPE"})
        (ax, ay), (bx, by) = p["points"][:2]
        text("PIPE", f"D={p['d']:g}", (ax + bx) / 2 + 100, (ay + by) / 2, 120)
    (ax, ay), (bx, by) = s["north"]
    msp.add_line(P(ax, ay), P(bx, by), dxfattribs={"layer": "NORTH"})
    text("NORTH", "N", bx - 100, by + 150, 300)
    for dm in s["dimensions"]:
        msp.add_linear_dim(base=P(*dm["base"]), p1=P(*dm["p1"]), p2=P(*dm["p2"]), angle=dm["angle"],
                           dimstyle="EZDXF", dxfattribs={"layer": "DIMENSIONS"}).render()
    st = s["site"]
    text("SITE_INFO", f"LAT={st['lat']:g} LON={st['lon']:g} TZ={st['tz']:g} FLOORS={st['floors']} FLOOR_H={st['floor_h']:g}", 0, -2500, 300)
    return doc


def write_sample(path, s=SAMPLE, offset=(0.0, 0.0), units=4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_doc(s, offset, units).saveas(path)
    return path


def measurements_json(s=SAMPLE):
    """The true values in mm (plan origin = outline's south-west corner), written from
    SAMPLE directly, not from reading the DXF."""
    return {
        "_note": "True values of data/sample_roof/roof.dxf (mm, origin at the roof outline's "
                 "south-west corner). Written by scripts/make_sample_dxf.py from rooftop/sample_dxf.py.",
        "length": s["length"], "width": s["width"],
        "diagonal": round((s["length"] ** 2 + s["width"] ** 2) ** 0.5, 1),
        "notch_length": s["notch_length"], "notch_width": s["notch_width"],
        "parapet_thickness": s["parapet_thickness"], "parapet_height": s["parapet_height"],
        "rooms": [{k: r[k] for k in ("name", "kind", "x", "y", "w", "d", "h", "door")} for r in s["rooms"]],
        "tanks": [dict(k) for k in s["tanks"]],
        "drains": s["drains"], "pipes": s["pipes"], "north_deg": s["north_deg"],
        "site": dict(s["site"]),
        "dimensions": [{k: d[k] for k in ("name", "p1", "p2", "value")} for d in s["dimensions"]],
    }


def write_measurements(path, s=SAMPLE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(measurements_json(s), indent=1))
    return path
