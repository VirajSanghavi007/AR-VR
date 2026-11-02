"""Phase 2: given the roof CAD (slab + parapets), place solar panels automatically.

Units: the spec is millimetres; everything here is metres, z up, origin at the
slab's lower corner. The roof frame sits on the slab top.
"""
import numpy as np
from shapely.geometry import box

from . import layout as L
from .ar_export import _materials, box_from_quad, glb_bytes, laminate_quad, Z_TO_Y
from .cad import rooftop_boxes
from .geometry import PlaneFrame
from .panels import panel_corners_3d, quad_solid


def roof_frame(spec):
    z = np.array([0.0, 0.0, 1.0])
    return PlaneFrame(np.array([0.0, 0.0, spec.slab_thickness / 1000]),
                      np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), z)


def usable_area(spec, setback=0.5):
    """Slab top inside the parapets, pulled in by setback (m)."""
    p = spec.parapet_thickness / 1000
    return box(p, p, spec.length / 1000 - p, spec.width / 1000 - p).buffer(-setback, join_style=2)


def design(spec, setback=0.5, tilt_deg=15.0, latitude=19.0, facing_deg=90.0, standoff=0.05):
    """Return (layout, panel corners as (4,3) metre arrays). facing_deg: direction in the
    roof's x/y plan the panels face (90 = +y); set it to the equator side on site."""
    area = usable_area(spec, setback)
    kw = {}
    if tilt_deg > 0:
        kw = dict(facing_deg=facing_deg, sun_alt_deg=L.winter_sun_altitude(latitude))
    lay = L.layout_panels(area, tilt_deg=tilt_deg, **kw)
    facing = lay.facing if tilt_deg > 0 else None
    corners = panel_corners_3d(roof_frame(spec), lay.panels, standoff, facing, lay.rise)
    return lay, corners


def _box_mesh(b):
    (x, y, z), (dx, dy, dz) = (np.asarray(b.origin) / 1000, np.asarray(b.size) / 1000)
    c = np.array([[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z]])
    v, t = box_from_quad(c + [0, 0, dz], dz)
    return v, t


def scene_parts(spec, panel_corners, thickness=0.035):
    """Parts for glb_bytes: (verts, tris, uv, material), z-up, metres."""
    parts = [_box_mesh(b) + (None, "roof") for b in rooftop_boxes(spec)]
    for c in panel_corners:
        v, t = box_from_quad(c, thickness)
        parts.append((v, t, None, "frame"))
        parts.append(laminate_quad(c) + ("glass",))
    return parts


def write_glb(path, spec, panel_corners):
    from pathlib import Path
    parts = [(v @ Z_TO_Y.T, t, uv, k) for v, t, uv, k in scene_parts(spec, panel_corners)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(parts, _materials()))
    return path


def export_step(path, spec, panel_corners, thickness=0.035):
    """Roof boxes + panel solids in one STEP assembly (mm)."""
    import cadquery as cq
    from pathlib import Path

    asm = cq.Assembly(name="RooftopSolar")
    for b in rooftop_boxes(spec):
        asm.add(cq.Workplane("XY").box(*b.size, centered=False).translate(b.origin), name=b.name)
    for i, c in enumerate(panel_corners):
        asm.add(cq.Workplane().add(quad_solid(c, thickness, 1000.0)), name=f"Panel_{i:03d}",
                color=cq.Color(0.1, 0.2, 0.8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    asm.export(str(path), exportType="STEP")
    return path
