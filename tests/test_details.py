"""Unit tests for rooftop/details.py (Phase 2 M3 detailed geometry)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from rooftop import details as D
from rooftop import phase2 as P2
from rooftop.ar_export import Z_TO_Y
from rooftop.cad import RooftopSpec
from rooftop.dxf_read import Door, Drawing, Pipe

SPEC = RooftopSpec(18000, 12000, 150, 1000, 230, 6000, 4000)
T, H = 0.15, 1.0
DRAINS = [(0.6, 0.6), (17.4, 0.6), (0.6, 11.4)]


def stair(**kw):
    a = dict(name="Stair1", kind="stair", x=0.8, y=8.4, w=3.0, d=3.2, h=2.7)
    a.update(kw)
    return P2.Feature(**a)


def store(**kw):
    a = dict(name="Store1", kind="store", x=4.4, y=9.6, w=2.4, d=2.0, h=2.4)
    a.update(kw)
    return P2.Feature(**a)


def tank(stand=0.6, h=2.1, **kw):
    a = dict(name="Tank1", kind="tank", x=13.4, y=6.2, w=1.2, d=1.2, h=h, stand=stand)
    a.update(kw)
    return P2.Feature(**a)


def volume(v, t):
    """Signed volume of a closed triangle mesh (positive when it winds outwards)."""
    v = np.asarray(v, float)[np.asarray(t)]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


def bounds(parts):
    pts = np.vstack([D.part_mesh(p)[0] for p in parts])
    return pts.min(0), pts.max(0)


def closed(t):
    """Every directed edge is matched by its reverse exactly once (closed, consistently wound)."""
    e = Counter()
    for a, b, c in np.asarray(t):
        for u, w in ((a, b), (b, c), (c, a)):
            e[(u, w)] += 1
    return all(n == 1 and e[(w, u)] == 1 for (u, w), n in e.items())


# ---------------------------------------------------------------- primitives

def test_box_corners_and_empty_box():
    p = D.box("G", "concrete", (1, 2, 3), (2, 4, 7))
    assert np.allclose(p.centre, (1.5, 3, 5)) and np.allclose(p.half, (0.5, 1, 2))
    lo, hi = bounds([p])
    assert np.allclose(lo, (1, 2, 3)) and np.allclose(hi, (2, 4, 7))
    assert p.step
    assert not D.box("G", "ground", (0, 0, 0), (1, 1, 1), step=False).step
    for hi in ((1, 4, 7), (2, 2, 7), (2, 4, 3), (0, 4, 7)):
        with pytest.raises(ValueError, match="empty box"):
            D.box("G", "concrete", (1, 2, 3), hi)


def test_beam_runs_p_to_q_with_its_section_towards_up():
    p = D.beam("G", "galv", (0, 0, 0), (3, 4, 0), 0.1, 0.2, up=(0, 0, 1))
    assert np.allclose(p.centre, (1.5, 2, 0))
    assert np.allclose(p.half, (2.5, 0.05, 0.1))
    assert np.allclose(p.axes[0], (0.6, 0.8, 0))
    assert np.allclose(p.axes[2], (0, 0, 1))
    assert np.isclose(np.linalg.det(p.axes), 1)
    v = D.part_mesh(p)[0]
    assert np.isclose(v[:, 2].max(), 0.1) and np.isclose(v[:, 2].min(), -0.1)
    tilted = D.beam("G", "galv", (0, 0, 0), (1, 0, 0), 0.1, 0.2, up=(0, -np.sin(0.3), np.cos(0.3)))
    assert np.allclose(tilted.axes[2], (0, -np.sin(0.3), np.cos(0.3)))
    vert = D.beam("G", "galv", (0, 0, 0), (0, 0, 2), 0.1, 0.1)      # along up: still a valid frame
    assert np.isclose(abs(np.linalg.det(vert.axes)), 1)
    assert np.isclose(volume(*D.part_mesh(vert)[:2]), 0.02)
    with pytest.raises(ValueError, match="zero-length"):
        D.beam("G", "galv", (1, 1, 1), (1, 1, 1), 0.1, 0.1)


def test_cylinder_and_pipe():
    c = D.cyl("G", "tank", (1, 2, 3), (0, 0, 2), 0.5, 1.5)
    assert np.allclose(c.axis, (0, 0, 1)) and c.r == 0.5 and c.length == 1.5
    p = D.pipe("G", "pvc", (0, 0, 1), (0, 3, 1), 0.05)
    assert np.allclose(p.axis, (0, 1, 0)) and np.isclose(p.r, 0.025) and np.isclose(p.length, 3)
    for r, L in ((0, 1), (1, 0), (-1, 1)):
        with pytest.raises(ValueError, match="positive"):
            D.cyl("G", "tank", (0, 0, 0), (0, 0, 1), r, L)


def test_box_mesh_is_closed_outward_with_flat_normals():
    p = D.beam("G", "galv", (0, 0, 0), (2, 1, 0.5), 0.3, 0.2)
    v, t, n, uv = D.part_mesh(p)
    assert v.shape == (24, 3) and t.shape == (12, 3) and uv.shape == (24, 2)
    assert np.isclose(volume(v, t), 0.3 * 0.2 * np.linalg.norm([2, 1, 0.5]))
    assert all(np.dot(v[i] - p.centre, n[i]) > 0 for i in range(24))
    f = v[t]
    fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    assert np.all(np.einsum("ij,ij->i", fn, n[t[:, 0]]) > 0)        # winding agrees with the normal


def test_cylinder_mesh_volume_normals_and_ends():
    c = D.cyl("G", "tank", (1, 1, 0), (0, 0, 1), 0.5, 2.0)
    v, t, n, _ = D.part_mesh(c, sides=48)
    poly = 0.5 * 48 * 0.25 * np.sin(2 * np.pi / 48)
    assert np.isclose(volume(v, t), poly * 2.0)
    assert np.isclose(v[:, 2].min(), 0) and np.isclose(v[:, 2].max(), 2)
    assert np.isclose(np.hypot(v[:, 0] - 1, v[:, 1] - 1).max(), 0.5)
    side = n[:96]
    assert np.allclose(side[:, 2], 0) and np.allclose(np.linalg.norm(side, axis=1), 1)
    assert np.allclose(n[96:144], (0, 0, -1)) and np.allclose(n[144:192], (0, 0, 1))
    f = v[t]
    fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    assert np.all(np.einsum("ij,ij->i", fn, n[t].mean(1)) > 0)          # winding agrees with the normals


def test_mesh_part_unwelds_with_face_normals():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    t = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    p = D.mesh("G", "membrane", v, t)
    mv, mt, mn, _ = D.part_mesh(p)
    assert mv.shape == (12, 3) and np.isclose(volume(mv, mt), 1 / 6)
    assert np.allclose(mn[:3], (0, 0, -1)) and np.allclose(mn[9:], np.ones(3) / np.sqrt(3))


def test_uv_projects_along_the_main_normal_axis():
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
    n = np.array([[1, 0, 0], [0, -1, 0], [0, 0.2, 1]], float)
    assert np.allclose(D._uv(v, n), [[2, 3], [4, 6], [7, 8]])


def test_perp_gives_an_orthonormal_frame():
    for a in ((1, 0, 0), (0, 0, 1), (0.6, 0.8, 0)):
        a = np.asarray(a, float)
        e1, e2 = D._perp(a)
        assert np.isclose(np.dot(e1, a), 0) and np.isclose(np.dot(e2, a), 0) and np.isclose(np.dot(e1, e2), 0)
        assert np.isclose(np.linalg.norm(e1), 1) and np.isclose(np.linalg.norm(e2), 1)


# ---------------------------------------------------------------- solids (CadQuery)

def test_part_solid_box_beam_cylinder_mesh_in_mm():
    b = D.part_solid(D.box("G", "concrete", (1, 2, 3), (2, 4, 7)))
    assert np.isclose(b.Volume(), 1000 * 2000 * 4000)
    bb = b.BoundingBox()
    assert np.allclose((bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax), (1000, 2000, 3000, 2000, 4000, 7000))
    beam = D.part_solid(D.beam("G", "galv", (0, 0, 0), (3, 4, 0), 0.1, 0.2))
    assert np.isclose(beam.Volume(), 5000 * 100 * 200, rtol=1e-6)
    bb = beam.BoundingBox()
    assert np.isclose(bb.zmax, 100) and np.isclose(bb.zmin, -100)
    mv = D.part_mesh(D.beam("G", "galv", (0, 0, 0), (3, 4, 0), 0.1, 0.2))[0] * 1000
    assert np.isclose(bb.xmin, mv[:, 0].min()) and np.isclose(bb.ymax, mv[:, 1].max())
    c = D.part_solid(D.cyl("G", "tank", (1, 1, 0.5), (0, 0, 1), 0.5, 2))
    assert np.isclose(c.Volume(), np.pi * 500 ** 2 * 2000, rtol=1e-6)
    assert np.isclose(c.BoundingBox().zmin, 500) and np.isclose(c.BoundingBox().xmax, 1500)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    m = D.part_solid(D.mesh("G", "m", v, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    assert np.isclose(abs(m.Volume()), 1e9 / 6, rtol=1e-6)


# ---------------------------------------------------------------- roof

def test_screed_height_falls_to_the_nearest_drain():
    z = T + D.SCREED_MIN
    dr = np.array(DRAINS)
    assert np.isclose(D.screed_height(0.6, 0.6, dr, SPEC), z)
    assert np.isclose(D.screed_height(3.6, 4.6, dr, SPEC), z + 0.05)
    assert np.isclose(D.screed_height(17.4, 3.6, dr, SPEC), z + 0.03)
    floor_y0 = 0.23
    assert np.isclose(D.screed_height(5, floor_y0 + 2, [], SPEC), z + 0.02)


def test_grid_lines_keep_extras_and_merge_close_ones():
    g = D._grid_lines(0.0, 3.5, [0.5, 1.02, 3.5, 7, -1])
    assert np.allclose(g, [0, 0.5, 1, 2, 3, 3.5])
    g = D._grid_lines(0.2, 2.21, [])
    assert g[0] == 0.2 and g[-1] == 2.21 and len(g) == 3


def test_screed_is_closed_covers_the_floor_and_slopes():
    p = D.screed_part(SPEC, DRAINS)
    assert p.material == "membrane" and p.kind == "mesh"
    assert closed(p.tris)
    floor = P2.inner_outline(SPEC).buffer(-D.UPTURN_T, join_style=2)
    v = p.verts
    assert np.isclose(v[:, 2].min(), T)
    top = v[p.tris][np.all(v[p.tris][:, :, 2] > T + 1e-9, axis=1)]
    a = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    assert np.all(a[:, 2] > 0)
    assert np.isclose(a[:, 2].sum() / 2, floor.area)
    vol = volume(v, p.tris)
    assert floor.area * D.SCREED_MIN < vol < floor.area * (D.SCREED_MIN + 0.12)
    for x, y in DRAINS:
        k = np.argmin(np.hypot(v[:, 0] - x, v[:, 1] - y) + (v[:, 2] <= T) * 99)
        assert np.isclose(v[k, 2], T + D.SCREED_MIN)
    tops = v[v[:, 2] > T]
    assert np.allclose(tops[:, 2], [D.screed_height(x, y, np.array(DRAINS), SPEC) for x, y, _ in tops])


def test_roof_parts_slab_parapet_coping_upturn():
    parts = D.roof_parts(SPEC, DRAINS)
    g = lambda name: [p for p in parts if p.group == name]
    lo, hi = bounds(g("Slab"))
    assert np.allclose(lo, (0, 0, 0)) and np.allclose(hi, (18, 12, T))
    assert np.isclose(sum(np.prod(2 * p.half) for p in g("Slab")), P2.outline(SPEC).area * T)
    lo, hi = bounds(g("Coping"))
    assert np.isclose(hi[2], T + H) and np.isclose(lo[2], T + H - D.COPING_T)
    assert np.allclose(lo[:2], (-D.COPING_OVER, -D.COPING_OVER)) and np.allclose(hi[:2], (18.03, 12.03))
    ring = P2.outline(SPEC).difference(P2.inner_outline(SPEC))
    assert np.isclose(sum(np.prod(2 * p.half) for p in g("Parapet")), ring.area * (H - D.COPING_T))
    assert np.isclose(bounds(g("Parapet"))[1][2], T + H - D.COPING_T)
    cop = sum(np.prod(2 * p.half[:2]) for p in g("Coping"))
    assert np.isclose(cop, ring.area + (P2.outline(SPEC).length + P2.inner_outline(SPEC).length) * 0.03, rtol=1e-3)
    lo, hi = bounds(g("Upturn"))
    assert np.isclose(hi[2], T + D.UPTURN_H) and np.isclose(lo[2], T)
    up = sum(np.prod(2 * p.half[:2]) for p in g("Upturn"))
    assert np.isclose(up, P2.inner_outline(SPEC).length * D.UPTURN_T, rtol=1e-2)
    assert all(p.material == m for name, m in (("Slab", "concrete"), ("Parapet", "plaster"), ("Coping", "concrete"),
                                                ("Upturn", "membrane")) for p in g(name))
    assert len(g("Screed")) == 1
    assert {p.group for p in parts if p.group.startswith("Drain")} == {"Drain1", "Drain2", "Drain3"}
    assert not [p for p in D.roof_parts(SPEC, []) if p.group.startswith("Drain")]


def test_drain_grate_is_200mm_and_sits_at_z():
    parts = D.drain_parts("Drain1", 2.0, 3.0, 0.2)
    lo, hi = bounds(parts)
    assert np.allclose(lo, (1.9, 2.9, 0.196)) and np.allclose(hi, (2.1, 3.1, 0.212))
    assert sum(p.material == "metal" for p in parts) == 7 and parts[0].material == "drain"


# ---------------------------------------------------------------- rooms

def test_door_on_finds_the_side_and_span():
    f = stair()
    assert D.door_on(f, Door("Stair1", 1.8, 8.4, 2.8, 8.4)) == ("S", pytest.approx(1.0), pytest.approx(2.0))
    assert D.door_on(f, Door("Stair1", 2.8, 11.6, 1.8, 11.6)) == ("N", pytest.approx(1.0), pytest.approx(2.0))
    assert D.door_on(f, Door("Stair1", 0.8, 9.0, 0.8, 10.0)) == ("W", pytest.approx(0.6), pytest.approx(1.6))
    assert D.door_on(f, Door("Stair1", 3.8, 10.0, 3.8, 9.0)) == ("E", pytest.approx(0.6), pytest.approx(1.6))
    assert D.door_on(f, None) == ("N", 1.0, 2.0)
    assert D.door_on(store(), None) == ("N", pytest.approx(0.75), pytest.approx(1.65))
    with pytest.raises(ValueError, match="not on one of its sides"):
        D.door_on(f, Door("Stair1", 1.8, 9.0, 2.8, 9.0))


def test_side_box_runs_along_and_inwards():
    f = stair()
    b = D.side_box("G", "plaster", f, "E", 0.5, 1.5, 0, 0.1, 1, 2)
    lo, hi = bounds([b])
    assert np.allclose(lo, (3.7, 8.9, 1)) and np.allclose(hi, (3.8, 9.9, 2))
    b = D.side_box("G", "plaster", f, "N", 0.5, 1.5, -0.2, 0, 1, 2)
    lo, hi = bounds([b])
    assert np.allclose(lo, (1.3, 11.6, 1)) and np.allclose(hi, (2.3, 11.8, 2))


def covers(parts, pt):
    return any(np.all(np.abs(np.asarray(pt) - p.centre) < p.half - 1e-9) for p in parts if p.kind == "box")


def test_stair_room_top_overhang_door_opening_and_vent():
    f, z0 = stair(), T
    parts = D.room_parts(f, z0, Door("Stair1", 1.8, 8.4, 2.8, 8.4))
    lo, hi = bounds(parts)
    assert np.isclose(hi[2], z0 + f.h) and np.isclose(lo[2], z0)
    assert np.isclose(lo[0], f.x - D.OVERHANG) and np.isclose(hi[1], f.y + f.d + D.OVERHANG)
    walls = [p for p in parts if p.material == "plaster"]
    assert not covers(walls, (2.3, 8.45, z0 + 1.0))                  # the opening
    assert covers(walls, (2.3, 8.45, z0 + 2.3))                       # lintel over it
    assert covers(walls, (1.2, 8.45, z0 + 1.0)) and covers(walls, (3.3, 8.45, z0 + 1.0))
    assert covers(walls, (2.3, 11.55, z0 + 1.0)) and covers(walls, (0.85, 10, z0 + 1.0))
    assert covers(walls, (3.75, 10, z0 + 1.0))
    assert not covers(walls, (2.3, 10, z0 + 1.0))                     # hollow
    wv = sum(np.prod(2 * p.half) for p in walls)
    wh = f.h - D.ROOF_T
    full = (2 * f.w + 2 * (f.d - 2 * D.WALL_T)) * D.WALL_T * wh
    assert np.isclose(wv, full - 1.0 * D.WALL_T * 2.1)
    door = [p for p in parts if p.material == "door"]
    assert len(door) == 1
    dlo, dhi = bounds(door)
    assert np.allclose((dlo[0], dhi[0]), (1.8 + D.FRAME, 2.8 - D.FRAME)) and np.isclose(dhi[2], z0 + 2.1 - D.FRAME)
    assert len([p for p in parts if p.material == "door_frame"]) == 3
    vlo, vhi = bounds([p for p in parts if p.material == "vent"])
    assert np.isclose(vlo[1], 11.6) and np.isclose(vhi[1], 11.63)           # back wall, outside
    assert not [p for p in parts if p.material == "shutter"]
    assert {p.group for p in parts} == {"Stair1"}


def test_door_near_a_corner_keeps_the_short_wall_piece():
    f, z0 = stair(x=2, y=2, w=3, d=2), T
    walls = [p for p in D.room_parts(f, z0, Door("Stair1", 3.5, 2, 4.5, 2)) if p.material == "plaster"]
    assert covers(walls, (4.75, 2.05, z0 + 1.0)) and covers(walls, (2.75, 2.05, z0 + 1.0))
    assert not covers(walls, (4.0, 2.05, z0 + 1.0))


def test_low_room_door_is_shorter_and_store_gets_a_shutter():
    f = store(h=2.0)
    parts = D.room_parts(f, T, None)
    sh = [p for p in parts if p.material == "shutter"]
    assert sh and not [p for p in parts if p.material == "door"]
    dh = min(2.2, 2.0 - D.ROOF_T - 0.2)
    curtain = sh[0]
    lo, hi = bounds([curtain])
    assert np.isclose(hi[2], T + dh) and np.isclose(lo[0], 4.4 + 0.75)
    box_lo, box_hi = bounds([sh[1]])
    assert len([p for p in parts if p.material == "metal"]) == 2
    assert np.isclose(box_lo[2], T + dh) and np.isclose(box_hi[2], T + dh + 0.3)
    assert np.isclose(box_hi[1], 11.6 + 0.25)
    slats = sh[2:]
    assert len(slats) == len(np.arange(T + 0.08, T + dh - 0.05, 0.1))
    walls = [p for p in parts if p.material == "plaster"]
    assert not covers(walls, (5.6, 11.55, T + 1.0)) and covers(walls, (5.6, 11.55, T + dh + 0.05))
    full = D.room_parts(store(), T, None)
    assert np.isclose(bounds([p for p in full if p.material == "shutter"][:1])[1][2], T + min(2.2, 2.4 - D.ROOF_T - 0.2))
    tall = D.room_parts(store(h=3.0), T, None)
    assert np.isclose(bounds([p for p in tall if p.material == "shutter"][:1])[1][2], T + 2.2)
    st = D.room_parts(stair(h=3.0), T, None)
    assert np.isclose(bounds([p for p in st if p.material == "door"])[1][2], T + 2.1 - D.FRAME)


# ---------------------------------------------------------------- tanks

def test_tank_size_top_and_steel_stand():
    f = tank()
    parts = D.tank_parts(f, T)
    body = [p for p in parts if p.material == "tank"]
    lo, hi = bounds(body)
    assert np.isclose(hi[2], T + f.h)
    assert np.isclose(hi[0] - lo[0], f.w, atol=1e-9) and np.isclose(hi[1] - lo[1], f.w, atol=0.02)
    assert np.isclose(lo[2], T + f.stand)
    assert max(p.r for p in body) == pytest.approx(0.6)
    main = body[0]
    assert np.isclose(main.r, 0.6 - D.TANK_GAP) and np.isclose(main.base[2] + main.length, T + f.h - 0.1)
    ribs = [p for p in body if np.isclose(p.r, 0.6)]
    assert len(ribs) == len(np.arange(T + 0.6 + 0.15, T + 2.1 - 0.15 - D.RIB_H, D.RIB_STEP))
    steel = [p for p in parts if p.material == "steel"]
    assert len(steel) == 1 + 4 + 4 and not [p for p in parts if p.material == "brick"]
    slo, shi = bounds(steel)
    assert np.isclose(slo[2], T) and np.isclose(shi[2], T + f.stand)
    pv = [p for p in parts if p.material == "pvc"]
    assert len(pv) == 4 and bounds(pv)[1][2] < T + f.h - 0.2
    assert {p.group for p in parts} == {"Tank1"}


def test_low_stand_is_brick_and_no_stand_is_none():
    parts = D.tank_parts(tank(stand=0.3, h=1.8), T)
    brick = [p for p in parts if p.material == "brick"]
    assert len(brick) == 1 and not [p for p in parts if p.material == "steel"]
    lo, hi = bounds(brick)
    assert np.allclose(lo, (13.4, 6.2, T)) and np.allclose(hi, (14.6, 7.4, T + 0.3))
    edge = D.tank_parts(tank(stand=D.STEEL_STAND, h=1.8), T)
    assert [p for p in edge if p.material == "brick"]
    parts = D.tank_parts(tank(stand=0.0, h=1.5), T)
    assert not [p for p in parts if p.material in ("brick", "steel")]
    assert np.isclose(bounds([p for p in parts if p.material == "tank"])[0][2], T)


def test_short_tank_keeps_one_rib_and_too_short_raises():
    parts = D.tank_parts(tank(stand=0.0, h=0.32), T)
    ribs = [p for p in parts if p.material == "tank" and np.isclose(p.r, 0.6)]
    assert len(ribs) == 1
    assert np.isclose(ribs[0].base[2], (T + T + 0.32 - 0.1 - D.RIB_H) / 2)
    D.tank_parts(tank(stand=0.0, h=0.31), T)
    with pytest.raises(ValueError, match="too short"):
        D.tank_parts(tank(stand=0.0, h=0.29), T)


# ---------------------------------------------------------------- pipes and building

def test_pipes_follow_the_route_above_the_screed_on_supports():
    pp = Pipe(((14.0, 6.2), (14.0, 3.0), (12.5, 3.0)), 50)
    parts = D.pipe_parts([pp], SPEC, DRAINS)
    runs = [p for p in parts if p.kind == "cyl"]
    sup = [p for p in parts if p.kind == "box"]
    assert len(runs) == 2 and all(np.isclose(p.r, 0.025) for p in runs)
    zc = max(D.screed_height(x, y, np.array(DRAINS), SPEC) for x, y in pp.points) + 0.05 + 0.025
    assert all(np.isclose(p.base[2], zc) for p in runs)
    assert np.isclose(runs[0].length, 3.2 + 0.05) and np.isclose(runs[1].length, 1.5 + 0.05)
    assert np.allclose(runs[0].base[:2], (14, 6.225)) and np.allclose(runs[0].axis, (0, -1, 0))
    assert len(sup) == (int(np.ceil(3.2 / 1.5)) + 1) + 2
    assert all(np.isclose(p.centre[2] - p.half[2], T) and np.isclose(p.centre[2] + p.half[2], zc - 0.025) for p in sup)
    assert {p.group for p in parts} == {"Pipe1"}
    assert D.pipe_parts([], SPEC, DRAINS) == []


def test_building_floors_windows_and_ground():
    parts = D.building_parts(SPEC, floors=3, floor_h=3.2)
    bld = [p for p in parts if p.group == "Building"]
    lo, hi = bounds(bld)
    assert np.isclose(lo[2], -9.6) and np.isclose(hi[2], 0)
    blocks = [p for p in bld if p.material == "plaster"]
    assert np.isclose(sum(np.prod(2 * p.half) for p in blocks), P2.outline(SPEC).area * 9.6)
    win = [p for p in parts if p.group == "Windows" and p.material == "window"]
    edges = [18, 8, 6, 4, 12, 12]
    assert len(win) == 3 * sum(int((L - 1.2) // D.WINDOW_STEP) for L in edges)
    out = P2.outline(SPEC)
    for p in win:
        assert np.isclose(np.prod(2 * p.half), D.WINDOW_W * D.WINDOW_H * 0.015)
        assert out.exterior.distance(shapely.Point(p.centre[:2])) < 0.01
        assert not out.buffer(-0.001).contains(shapely.Point(p.centre[:2]))
        z = (p.centre[2] - p.half[2] + 9.6) % 3.2
        assert np.isclose(z, D.SILL)
    ground = [p for p in parts if p.group == "Ground"]
    assert len(ground) == 1 and not ground[0].step
    glo, ghi = bounds(ground)
    assert np.isclose(ghi[2], -9.6) and np.allclose(glo[:2], (-8, -8)) and np.allclose(ghi[:2], (26, 20))
    one = D.building_parts(SPEC, floors=1, floor_h=3.0)
    assert np.isclose(bounds([p for p in one if p.group == "Building"])[0][2], -3.0)


# ---------------------------------------------------------------- mounting

@pytest.fixture(scope="module")
def sample_design():
    feats = [stair(), store(), tank(name="Tank1"), tank(name="Tank2", x=15.4)]
    lay, corners = P2.design(SPEC, latitude=19.07, facing_deg=270.0, features=feats, standoff=0.3, walkway=0.6)
    return feats, lay, corners


def test_panel_edges_and_normal():
    c = np.array([[0, 0, 1], [2, 0, 1], [2, 1, 1.5], [0, 1, 1.5]], float)
    f0, f1, g = D._panel_edges(c)
    assert {tuple(f0), tuple(f1)} == {(0, 0, 1), (2, 0, 1)}
    assert np.allclose(g, (0, 1, 0.5))
    n = D._normal(c)
    assert np.isclose(np.linalg.norm(n), 1) and n[2] > 0 and np.isclose(np.dot(n, g), 0)
    assert np.allclose(D._normal(c[::-1]), n)


def split_mounting(parts):
    plates = [p for p in parts if np.isclose(p.half[0], 0.06)]
    legs = [p for p in parts if np.allclose(p.half[:2], D.RAIL / 2) and not np.isclose(p.half[2], D.RAIL / 2)]
    taken = {id(p) for p in plates + legs}
    rails = [p for p in parts if id(p) not in taken]
    return rails, legs, plates


def test_mounting_rails_under_every_run_and_legs_on_the_screed(sample_design):
    feats, lay, corners = sample_design
    parts = D.mounting_parts(corners, SPEC, DRAINS)
    assert D.mounting_parts([], SPEC, DRAINS) == []
    rails, legs, plates = split_mounting(parts)
    groups = {p.group for p in parts}
    assert all(g.startswith("Mounting_R") for g in groups)
    assert len(rails) == 2 * len(groups)
    assert len(legs) == len(plates) and legs
    dr = np.array(DRAINS)
    for p in plates:
        assert np.isclose(p.centre[2] - p.half[2], D.screed_height(*p.centre[:2], dr, SPEC))
    covered = 0
    for c in corners:
        c = np.asarray(c)
        f0, f1, g = D._panel_edges(c)
        n = D._normal(c)
        for fr in (0.2, 0.8):
            pt = (f0 + f1) / 2 + g * fr - n * (D.PANEL_T + D.RAIL / 2)
            hit = [r for r in rails if np.all(np.abs(r.axes @ (pt - r.centre)) <= r.half + 1e-6)]
            on_axis = [r for r in hit if np.allclose((r.axes @ (pt - r.centre))[1:], 0, atol=1e-6)]
            covered += bool(on_axis)                                  # on the rail's centre line
    assert covered == 2 * len(corners)
    for leg in legs:
        top = leg.centre + leg.half
        assert any(np.all(np.abs(r.axes @ (np.r_[top[:2], top[2] + D.RAIL / 2] - r.centre)) <= r.half + 1e-3)
                   for r in rails if r.group == leg.group)


def test_mounting_splits_runs_at_gaps_and_skips_short_legs():
    def quad(x0, y0, w=1.0, d=1.6, z=0.3, rise=0.4):
        return np.array([[x0, y0, z], [x0 + w, y0, z], [x0 + w, y0 + d, z + rise], [x0, y0 + d, z + rise]])
    corners = [quad(2, 2), quad(3.02, 2), quad(5, 2), quad(2, 5)]
    parts = D.mounting_parts(corners, SPEC, [])
    assert {p.group for p in parts} == {"Mounting_R01_1", "Mounting_R01_2", "Mounting_R02_1"}
    rails = [p for p in split_mounting(parts)[0] if p.group == "Mounting_R01_1"]
    assert len(rails) == 2 and all(np.isclose(2 * r.half[0], 2.02) for r in rails)
    flat = [quad(2, 2, z=T + 0.04, rise=0.0)]
    rails, legs, plates = split_mounting(D.mounting_parts(flat, SPEC, []))
    assert len(rails) == 2 and not legs and not plates
    long = [quad(2 + i * 1.02, 2) for i in range(5)]
    legs = split_mounting(D.mounting_parts(long, SPEC, []))[1]
    span = 5 * 1.02 - 0.02
    per_rail = max(2, int(np.ceil((span - 0.2) / D.LEG_STEP)) + 1)
    assert len(legs) == 2 * per_rail
    xs = sorted({round(p.centre[0], 4) for p in legs})
    assert np.isclose(xs[0], 2.1) and np.isclose(xs[-1], 2 + span - 0.1) and np.diff(xs).max() <= D.LEG_STEP


def test_walkway_and_keepouts_hold(sample_design):
    feats, lay, corners = sample_design
    assert lay.row_gap >= 0.6
    fps = [P2.Feature(**{**f.__dict__}).footprint for f in feats]
    for c in corners:
        q = Polygon(np.asarray(c)[:, :2])
        assert not any(q.intersects(fp) for fp in fps)
    rows = sorted({round(float(np.asarray(c)[:, 1].min()), 3) for c in corners})      # facing 270 = -y
    depth = max(np.ptp(np.asarray(c)[:, 1]) for c in corners)
    gaps = [b - a - depth for a, b in zip(rows[:-1], rows[1:]) if b - a - depth > 0.01]
    assert gaps and min(gaps) >= 0.6 - 1e-6


# ---------------------------------------------------------------- inverter and tray

def test_inverter_on_the_wall_facing_the_array_and_tray_to_a_row(sample_design):
    feats, lay, corners = sample_design
    doors = [Door("Stair1", 1.8, 8.4, 2.8, 8.4)]
    parts = D.inverter_parts(feats, corners, SPEC, doors)
    inv = [p for p in parts if p.material == "inverter"]
    assert len(inv) == 1
    lo, hi = bounds(inv)
    assert np.isclose(lo[2], T + 1.0) and np.isclose(hi[2], T + 1.6)
    centre = np.mean([np.asarray(c)[:, :2].mean(0) for c in corners], 0)
    f = feats[0]
    side = {"S": f.y, "N": f.y + f.d, "W": f.x, "E": f.x + f.w}
    on = [s for s, v in side.items() if (np.isclose(hi[1], v + 0.2) if s == "N" else np.isclose(lo[1], v - 0.2) if s == "S"
                                         else np.isclose(hi[0], v + 0.2) if s == "E" else np.isclose(lo[0], v - 0.2))]
    assert on and on[0] != "S"
    assert on[0] == max((s for s in "NWE"), key=lambda s: np.dot(-np.asarray(D._side(f, s)[2], float),
                                                                 centre - (f.x + f.w / 2, f.y + f.d / 2)))
    tray = [p for p in parts if p.group == "CableTray" and p.material == "galv"]
    assert tray and len(tray) % 3 == 0
    tlo, thi = bounds(tray)
    assert np.isclose(tlo[2], T + D.TRAY_Z) and np.isclose(thi[2], T + D.TRAY_Z + D.TRAY_H)
    feet = [np.asarray(c)[np.argsort(np.asarray(c)[:, 2])[:2]].mean(0)[:2] for c in corners]
    ends = np.vstack([D.part_mesh(p)[0][:, :2] for p in tray])
    assert min(np.linalg.norm(ends - ft, axis=1).min() for ft in feet) < 0.1
    assert D.inverter_parts(feats, [], SPEC, doors) == []
    assert D.inverter_parts([tank()], corners, SPEC, doors) == []
    store_only = D.inverter_parts([store()], corners, SPEC, [])
    assert [p for p in store_only if p.material == "inverter"]


def test_tray_route_avoids_rooms_when_it_can():
    f = stair(x=2, y=2, w=2, d=2)
    blocker = store(x=5.5, y=2.5, w=2, d=1.0)
    corners = [np.array([[8, 6, 0.5], [9, 6, 0.5], [9, 7.6, 0.9], [8, 7.6, 0.9]])]
    parts = D.inverter_parts([f, blocker], corners, SPEC, [Door("Stair1", 2, 2, 3, 2)])
    tray = [p for p in parts if p.group == "CableTray" and p.material == "galv"]
    hulls = [shapely.MultiPoint(D.part_mesh(p)[0][:, :2]).convex_hull for p in tray]
    assert tray and not any(h.intersects(blocker.footprint) for h in hulls)
    free = D.inverter_parts([f], corners, SPEC, [Door("Stair1", 2, 2, 3, 2)])
    xs = [D.part_mesh(p)[0][:, 0] for p in free if p.group == "CableTray" and p.material == "galv"]
    assert max(x.max() for x in xs) > 8                               # with nothing in the way: route 0
    other = store(name="Store2", x=3.6, y=4.5, w=1.2, d=1.0)            # blocks route 1 (up first)
    parts = D.inverter_parts([f, other], corners, SPEC, [Door("Stair1", 2, 2, 3, 2)])
    tray = [p for p in parts if p.group == "CableTray" and p.material == "galv"]
    hulls = [shapely.MultiPoint(D.part_mesh(p)[0][:, :2]).convex_hull for p in tray]
    assert tray and not any(h.intersects(other.footprint) for h in hulls)


def test_tray_skips_a_zero_length_leg():
    f = stair(x=2, y=2, w=2, d=2)
    corners = [np.array([[8, 2.2, 0.5], [9, 2.2, 0.5], [9, 3.8, 0.9], [8, 3.8, 0.9]])]   # feet at y = 2.2
    start = D._wall_point(f, "E", 1.0, -0.1)
    corners[0][:, 1] += start[1] - 2.2                                   # feet level with the start
    parts = D.inverter_parts([f], corners, SPEC, [Door("Stair1", 2, 2, 3, 2)])
    tray = [p for p in parts if p.group == "CableTray"]
    assert tray and all(np.all(np.isfinite(D.part_mesh(p)[0])) for p in tray)
    assert len([p for p in tray if p.material == "galv"]) == 3          # one straight leg


# ---------------------------------------------------------------- whole model and writers

def sample_drawing():
    from rooftop.dxf_read import read_dxf
    return read_dxf(Path(__file__).resolve().parents[1] / "data/sample_roof/roof.dxf")


def test_site_floors_defaults_and_reads_mm():
    assert D.site_floors({}) == (3, 3.0)
    assert D.site_floors({"FLOORS": 4.0, "FLOOR_H": 2900}) == (4, 2.9)


def test_build_has_every_group_and_every_material_is_known(sample_design):
    feats, lay, corners = sample_design
    parts = D.build(SPEC, feats, corners, [Door("Stair1", 1.8, 8.4, 2.8, 8.4)], DRAINS,
                    [Pipe(((14.0, 6.2), (14.0, 3.0)), 50)], 3, 3.2)
    groups = {p.group.split("_")[0] for p in parts}
    assert {"Slab", "Parapet", "Coping", "Upturn", "Screed", "Drain1", "Stair1", "Store1", "Tank1", "Tank2",
            "Pipe1", "Building", "Windows", "Ground", "Mounting", "Inverter", "CableTray"} <= groups
    mats = D.materials()
    assert {p.material for p in parts} <= set(mats)
    assert {"frame", "glass"} <= set(mats)
    lo, hi = bounds([p for p in parts if p.group == "Stair1"])
    assert np.isclose(hi[2], T + 2.7)
    door = [p for p in parts if p.group == "Stair1" and p.material == "door"]
    assert np.isclose(bounds(door)[0][1], 8.4 + D.WALL_T / 2 - 0.02)
    bare = D.build(SPEC)
    assert not {p.group.split("_")[0] for p in bare} & {"Mounting", "Inverter", "CableTray", "Stair1"}
    assert np.isclose(bounds([p for p in bare if p.group == "Building"])[0][2], -9.0)


def test_materials_textures_tile_and_whiten():
    m = D.materials({"concrete": b"PNG"})
    assert m["concrete"].texture == b"PNG" and m["concrete"].tile and m["concrete"].rgb == (1.0, 1.0, 1.0)
    assert not m["plaster"].tile and m["plaster"].rgb == D.COLOURS["plaster"][0]
    assert m["tank"].metallic == 0.0 and m["galv"].metallic == 0.8 and m["galv"].roughness == 0.45


def test_panel_meshes_frame_and_glass_normals():
    c = np.array([[0, 0, 1], [1.7, 0, 1], [1.7, 1.1, 1.3], [0, 1.1, 1.3]], float)
    (fv, ft, fuv, fk, fn), (gv, gt, guv, gk, gn) = D.panel_meshes([c])
    assert fk == "frame" and gk == "glass" and fuv is None
    assert fv.shape == (36, 3) and np.isclose(volume(fv, ft), np.linalg.norm(c[1] - c[0]) * np.linalg.norm(c[3] - c[0]) * D.PANEL_T)
    f = fv[ft]
    fnn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    assert np.all(np.einsum("ij,ij->i", fnn, fn[ft[:, 0]]) > 0)
    assert np.allclose(gn, D._normal(c)) and gn.shape == (4, 3) and guv.shape == (4, 2)


def test_glb_parts_are_y_up_with_normals():
    p = D.box("G", "concrete", (0, 0, 0), (1, 2, 3))
    out = D.glb_parts([p], [])
    (v, t, uv, k, n), = out
    assert k == "concrete" and np.isclose(v[:, 1].max(), 3) and np.isclose(v[:, 2].min(), -2)
    assert np.allclose(n, D.part_mesh(p)[2] @ Z_TO_Y.T)
    assert len(D.glb_parts([p], [np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)])) == 3


def read_glb(path):
    import struct
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


def test_write_glb_has_normals_and_every_material(tmp_path, sample_design):
    feats, lay, corners = sample_design
    parts = D.build(SPEC, feats, corners, [], DRAINS, [], 3, 3.2)
    path = D.write_glb(tmp_path / "x" / "m.glb", parts, corners)
    g = read_glb(path)
    names = {m["name"] for m in g["materials"]}
    assert names == {p.material for p in parts} | {"frame", "glass"}
    assert all("NORMAL" in p["attributes"] for p in g["meshes"][0]["primitives"])
    assert path.stat().st_size < 15e6
    mins = np.min([a["min"] for a in g["accessors"] if "min" in a], 0)
    assert np.isclose(mins[1], -9.7)                                   # ground under the building, y-up


def test_write_json(tmp_path, sample_design):
    feats, lay, corners = sample_design
    dwg = Drawing(SPEC, feats, north_deg=90.0, site={"LAT": 19.07, "LON": 72.88, "TZ": 5.5})
    p = D.write_json(tmp_path / "d" / "m.json", dwg, feats, lay, corners, 19.07, 270.0, 3, 3.2, {"kwh_year": 5})
    d = json.loads(p.read_text())
    assert d["detailed"] and d["panels"] == len(corners) and np.isclose(d["peak_kw"], round(len(corners) * 0.55, 2))
    assert d["longitude"] == 72.88 and d["tz"] == 5.5 and d["latitude"] == 19.07 and d["facing_deg"] == 270.0
    assert np.isclose(d["building_base"], -9.6) and d["floors"] == 3 and d["kwh_year"] == 5
    assert d["slab"] == 0.15 and d["parapet"] == 1.0 and len(d["features"]) == 4 and d["row_gap"] >= 0.6
    assert d["features"][0]["name"] == "Stair1" and d["cells"]


def test_step_reopens_in_cadquery_with_the_right_volume_size_and_names(tmp_path):
    """Integration: write a small detailed model to STEP, read it back with OpenCascade."""
    import cadquery as cq
    parts = [p for p in D.roof_parts(SPEC, DRAINS[:1]) if p.group in ("Slab", "Coping", "Screed")]
    parts += D.room_parts(store(), T, None) + D.tank_parts(tank(), T)
    parts += [D.box("Ground", "ground", (-1, -1, -1), (19, 13, 0), step=False)]
    corner = np.array([[2, 2, 0.5], [3.1, 2, 0.5], [3.1, 4.2, 0.9], [2, 4.2, 0.9]], float)
    path = D.export_step(tmp_path / "s" / "m.step", parts, [corner])
    txt = path.read_text(errors="ignore")
    assert "Store1_shutter" in txt and "Tank1_steel" in txt and "Panel_000" in txt and "Ground" not in txt
    shape = cq.importers.importStep(str(path)).val()
    want = sum(abs(D.part_solid(p).Volume()) for p in parts if p.step)
    from rooftop.panels import quad_solid
    want += quad_solid(corner, D.PANEL_T, 1000.0).Volume()
    assert shape.Volume() == pytest.approx(want, rel=1e-4)
    bb = shape.BoundingBox()
    assert (bb.xmin, bb.ymin, bb.zmin) == pytest.approx((-30, -30, 0), abs=0.01)
    assert (bb.xmax, bb.ymax) == pytest.approx((18030, 12030), abs=0.01)
    assert bb.zmax == pytest.approx((T + store().h) * 1000, abs=0.01)         # store roof, above the tank


def test_materials_take_albedo_or_albedo_and_normal_textures():
    from rooftop.textures import jpeg
    a, n = jpeg(np.full((4, 4, 3), 90, np.uint8)), jpeg(np.full((4, 4, 3), [128, 128, 255], np.uint8))
    m = D.materials({"concrete": (a, n), "plaster": a, "galv": (None, n)})
    assert m["concrete"].texture == a and m["concrete"].normal == n and m["concrete"].tile
    assert m["concrete"].rgb == (1.0, 1.0, 1.0)                       # the image carries the colour
    assert m["plaster"].texture == a and m["plaster"].normal is None and m["plaster"].tile
    assert m["galv"].texture is None and m["galv"].normal == n and m["galv"].tile
    assert m["galv"].rgb == D.COLOURS["galv"][0]                     # no albedo: keep the flat colour
    assert m["membrane"].texture is None and m["membrane"].normal is None and not m["membrane"].tile
    assert m["membrane"].rgb == D.COLOURS["membrane"][0]
    assert (m["concrete"].metallic, m["concrete"].roughness) == D.COLOURS["concrete"][1:]
    assert set(m) >= set(D.COLOURS) | {"frame", "glass"}
    plain = D.materials()
    assert all(plain[k].texture is None and plain[k].normal is None for k in D.COLOURS)


def test_textured_glb_stays_under_15_mb_with_jpeg_maps(tmp_path, sample_design):
    from rooftop.textures import texture_set
    feats, lay, corners = sample_design
    parts = D.build(SPEC, feats, corners, [], DRAINS, [], 3, 3.2)
    path = D.write_glb(tmp_path / "t.glb", parts, corners, texture_set(D.COLOURS, px=256))
    g = read_glb(path)
    from rooftop.textures import KINDS
    textured = {m["name"] for m in g["materials"] if "normalTexture" in m}
    assert textured == {p.material for p in parts} & set(KINDS)
    assert {"concrete", "plaster", "membrane", "galv"} <= textured
    assert sum(i["mimeType"] == "image/jpeg" for i in g["images"]) == 2 * len(textured)
    assert path.stat().st_size < 15e6
