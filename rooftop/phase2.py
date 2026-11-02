"""Phase 2: given the roof CAD (slab + parapets + rooftop structures), place solar panels automatically.

Units: the spec is millimetres; everything here is metres, z up, origin at the
slab's lower corner. The roof frame sits on the slab top. The footprint is a
rectangle, or an L when the spec has a notch (the +x, +y corner is cut away).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import box
from shapely.ops import unary_union

from . import layout as L
from .ar_export import Material, _materials, box_from_quad, glb_bytes, laminate_quad, Z_TO_Y
from .cad import Box
from .geometry import PlaneFrame
from .panels import panel_corners_3d, quad_solid

TANK_SIDES = 32


@dataclass(frozen=True)
class Feature:
    """A rooftop structure, metres in the plan. kind: stair (headroom + door),
    store (storage room + door) or tank (water tank on a plinth, w = diameter)."""
    name: str
    kind: str
    x: float          # min corner
    y: float
    w: float
    d: float
    h: float

    @property
    def footprint(self):
        return box(self.x, self.y, self.x + self.w, self.y + self.d)


def roof_frame(spec):
    z = np.array([0.0, 0.0, 1.0])
    return PlaneFrame(np.array([0.0, 0.0, spec.slab_thickness / 1000]),
                      np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), z)


def outline(spec):
    """Building footprint in metres: the full rectangle minus the notch."""
    Lm, Wm = spec.length / 1000, spec.width / 1000
    shape = box(0, 0, Lm, Wm)
    if spec.notch_length:
        shape = shape.difference(box(Lm - spec.notch_length / 1000, Wm - spec.notch_width / 1000, Lm, Wm))
    return shape


def inner_outline(spec):
    """Floor inside the parapets."""
    return outline(spec).buffer(-spec.parapet_thickness / 1000, join_style=2)


def rect_cells(poly):
    """Split a rectilinear polygon into non-overlapping (x0, y0, x1, y1) rectangles."""
    xy = shapely.get_coordinates(poly)
    xs, ys = np.unique(np.round(xy[:, 0], 6)), np.unique(np.round(xy[:, 1], 6))
    out = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        run = None
        for x0, x1 in zip(xs[:-1], xs[1:]):
            inside = poly.contains(shapely.Point((x0 + x1) / 2, (y0 + y1) / 2))
            if inside and run is None:
                run = x0
            if not inside and run is not None:
                out.append((run, y0, x0, y1))
                run = None
        if run is not None:
            out.append((run, y0, xs[-1], y1))
    return out


def building_boxes(spec):
    """Slab plus parapet walls (mm), following the footprint, as non-overlapping boxes."""
    t, h, p = spec.slab_thickness, spec.parapet_height, spec.parapet_thickness / 1000
    out = [Box(f"RoofSlab_{i}", (x0 * 1000, y0 * 1000, 0), ((x1 - x0) * 1000, (y1 - y0) * 1000, t))
           for i, (x0, y0, x1, y1) in enumerate(rect_cells(outline(spec)))]
    ring = outline(spec).difference(inner_outline(spec))
    out += [Box(f"Parapet_{i}", (x0 * 1000, y0 * 1000, t), ((x1 - x0) * 1000, (y1 - y0) * 1000, h))
            for i, (x0, y0, x1, y1) in enumerate(rect_cells(ring))]
    return out


def default_features(spec):
    """A believable set for the spec: stair headroom and storage room front-left, two water
    tanks front-right (the low-y edge, so their winter shadows fall off the roof for panels
    facing +y). Anything that does not fit is dropped."""
    p, Lm = spec.parapet_thickness / 1000, spec.length / 1000
    m, y = 0.1, spec.parapet_thickness / 1000 + 0.1      # at the low-y edge: shadows fall off the roof
    cand = [
        Feature("StairHeadroom", "stair", p + m, y, 2.6, 2.8, 2.4),
        Feature("StoreRoom", "store", p + m + 2.6 + 0.5, y, 2.2, 1.8, 2.4),
        Feature("WaterTank1", "tank", Lm - p - m - 2 * 1.4 - 0.5, y, 1.4, 1.4, 1.6),
        Feature("WaterTank2", "tank", Lm - p - m - 1.4, y, 1.4, 1.4, 1.6),
    ]
    floor = inner_outline(spec)
    keep, taken = [], []
    for f in cand:
        fp = f.footprint.buffer(0.3)
        if floor.contains(f.footprint) and not any(fp.intersects(o) for o in taken):
            keep.append(f)
            taken.append(f.footprint)
    return keep


def keepout(features, facing_deg=90.0, latitude=19.0, clearance=0.5):
    """Where panels may not go: each structure grown by clearance plus its winter-noon
    shadow, which falls away from the equator side the panels face."""
    alt = L.winter_sun_altitude(latitude)
    away = -np.array([np.cos(np.radians(facing_deg)), np.sin(np.radians(facing_deg))])
    parts = []
    for f in features:
        fp = f.footprint
        top = f.h + (0.5 if f.kind == "tank" else 0.0)
        s = away * top / np.tan(np.radians(alt))
        shade = shapely.affinity.translate(fp, s[0], s[1])
        parts.append(unary_union([fp, shade]).convex_hull.buffer(clearance, join_style=2))
    return unary_union(parts) if parts else shapely.Polygon()


def usable_area(spec, setback=0.5, obstacles=None):
    """Floor inside the parapets, pulled in by setback (m), minus any keep-out."""
    area = inner_outline(spec).buffer(-setback, join_style=2)
    if obstacles is not None and not obstacles.is_empty:
        area = area.difference(obstacles)
    return area


def design(spec, setback=0.5, tilt_deg=15.0, latitude=19.0, facing_deg=90.0, standoff=0.05, features=()):
    """Return (layout, panel corners as (4,3) metre arrays). facing_deg: direction in the
    roof's x/y plan the panels face (90 = +y); set it to the equator side on site."""
    area = usable_area(spec, setback, keepout(features, facing_deg, latitude))
    kw = {}
    if tilt_deg > 0:
        kw = dict(facing_deg=facing_deg, sun_alt_deg=L.winter_sun_altitude(latitude))
    lay = L.layout_panels(area, tilt_deg=tilt_deg, **kw)
    facing = lay.facing if tilt_deg > 0 else None
    corners = panel_corners_3d(roof_frame(spec), lay.panels, standoff, facing, lay.rise)
    return lay, corners


def _box_m(x, y, z, dx, dy, dz):
    c = np.array([[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z]])
    return box_from_quad(c + [0, 0, dz], dz)


def _box_mesh(b):
    (x, y, z), (dx, dy, dz) = (np.asarray(b.origin) / 1000, np.asarray(b.size) / 1000)
    return _box_m(x, y, z, dx, dy, dz)


def _cylinder_mesh(cx, cy, z, r, h, n=TANK_SIDES):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    v = np.vstack([np.c_[ring, np.full(n, z)], np.c_[ring, np.full(n, z + h)],
                   [[cx, cy, z]], [[cx, cy, z + h]]])
    t = []
    for i in range(n):
        j = (i + 1) % n
        t += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return v, np.array(t)


def feature_parts(f, z0):
    """(verts, tris, uv, material) parts for one rooftop structure standing on z0."""
    if f.kind == "tank":
        plinth = _box_m(f.x, f.y, z0, f.w, f.d, 0.4) + (None, "roof")
        r = f.w / 2 * 0.92
        body = _cylinder_mesh(f.x + f.w / 2, f.y + f.d / 2, z0 + 0.4, r, f.h - 0.4)
        lid = _cylinder_mesh(f.x + f.w / 2, f.y + f.d / 2, z0 + f.h, r * 0.35, 0.12)
        return [plinth, body + (None, "tank"), lid + (None, "tank")]
    parts = [_box_m(f.x, f.y, z0, f.w, f.d, f.h) + (None, "roof"),
             _box_m(f.x - 0.1, f.y - 0.1, z0 + f.h, f.w + 0.2, f.d + 0.2, 0.12) + (None, "roof")]
    dw, dh = (1.0, 2.1) if f.kind == "stair" else (0.9, 2.0)
    parts.append(_box_m(f.x + f.w / 2 - dw / 2, f.y + f.d, z0, dw, 0.06, dh) + (None, "door"))   # door faces the roof
    return parts


def scene_parts(spec, panel_corners, thickness=0.035, features=()):
    """Parts for glb_bytes: (verts, tris, uv, material), z-up, metres."""
    parts = [_box_mesh(b) + (None, "roof") for b in building_boxes(spec)]
    z0 = spec.slab_thickness / 1000
    for f in features:
        parts += feature_parts(f, z0)
    for c in panel_corners:
        v, t = box_from_quad(c, thickness)
        parts.append((v, t, None, "frame"))
        parts.append(laminate_quad(c) + ("glass",))
    return parts


def _scene_materials():
    m = _materials()
    m["tank"] = Material((0.07, 0.18, 0.42), 0.0, 0.55)
    m["door"] = Material((0.22, 0.17, 0.12), 0.0, 0.7)
    return m


def write_glb(path, spec, panel_corners, features=()):
    parts = [(v @ Z_TO_Y.T, t, uv, k) for v, t, uv, k in scene_parts(spec, panel_corners, features=features)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(parts, _scene_materials()))
    return path


def write_layout_json(path, spec, features=()):
    """Footprint cells and structures for the viewer (metres, plan x/y, z up)."""
    data = {"slab": spec.slab_thickness / 1000, "parapet": spec.parapet_height / 1000,
            "cells": [list(c) for c in rect_cells(outline(spec))],
            "features": [{"name": f.name, "kind": f.kind, "x": f.x, "y": f.y, "w": f.w, "d": f.d, "h": f.h}
                         for f in features]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path


def export_step(path, spec, panel_corners, thickness=0.035, features=()):
    """Roof boxes, structures and panel solids in one STEP assembly (mm)."""
    import cadquery as cq

    asm = cq.Assembly(name="RooftopSolar")
    for b in building_boxes(spec):
        asm.add(cq.Workplane("XY").box(*b.size, centered=False).translate(b.origin), name=b.name)
    z0 = spec.slab_thickness
    for f in features:
        if f.kind == "tank":
            asm.add(cq.Workplane("XY").box(f.w * 1000, f.d * 1000, 400, centered=False)
                    .translate((f.x * 1000, f.y * 1000, z0)), name=f.name + "_Plinth")
            asm.add(cq.Workplane("XY").workplane(offset=z0 + 400).center((f.x + f.w / 2) * 1000, (f.y + f.d / 2) * 1000)
                    .circle(f.w / 2 * 920).extrude(f.h * 1000 - 400), name=f.name,
                    color=cq.Color(0.07, 0.18, 0.42))
        else:
            asm.add(cq.Workplane("XY").box(f.w * 1000, f.d * 1000, f.h * 1000, centered=False)
                    .translate((f.x * 1000, f.y * 1000, z0)), name=f.name)
    for i, c in enumerate(panel_corners):
        asm.add(cq.Workplane().add(quad_solid(c, thickness, 1000.0)), name=f"Panel_{i:03d}",
                color=cq.Color(0.1, 0.2, 0.8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    asm.export(str(path), exportType="STEP")
    return path
