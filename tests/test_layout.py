import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from rooftop import layout as L


def _grid_pts(w, h, step=0.1):
    x, y = np.meshgrid(np.arange(0, w + 1e-9, step), np.arange(0, h + 1e-9, step))
    return np.c_[x.ravel(), y.ravel()]


def test_roof_outline_rectangle_area():
    o = L.roof_outline(_grid_pts(10, 6), ratio=0.1)
    assert o.area == pytest.approx(60, rel=0.02)


def test_roof_outline_follows_l_shape():
    pts = _grid_pts(10, 10)
    pts = pts[~((pts[:, 0] > 5) & (pts[:, 1] > 5))]  # remove a 5x5 corner
    o = L.roof_outline(pts, ratio=0.05)
    assert o.area == pytest.approx(75, rel=0.05)  # convex hull would be ~87.5


def test_roof_outline_rejects_line():
    with pytest.raises(ValueError):
        L.roof_outline(np.c_[np.linspace(0, 1, 20), np.zeros(20)])


def test_keepouts_finds_obstacle_and_ignores_roof_and_sky():
    roof = _grid_pts(10, 6)
    outline = box(0, 0, 10, 6)
    chimney = np.c_[np.random.default_rng(0).uniform(4, 5, (200, 2)) - [0, 2]]
    uv = np.vstack([roof, chimney, [[1, 1]] * 50])
    h = np.r_[np.zeros(len(roof)), np.full(len(chimney), 1.0), np.full(50, 10.0)]
    k = L.keepouts(uv, h, outline, clearance=0.3)
    assert k.contains(box(4.1, 2.1, 4.9, 2.9))
    assert not k.intersects(box(0.8, 0.8, 1.2, 1.2))  # tree/sky 10 m up ignored
    assert k.area < 4


def test_keepouts_empty_when_flat():
    assert L.keepouts(_grid_pts(5, 5), np.zeros(len(_grid_pts(5, 5))), box(0, 0, 5, 5)).is_empty


def test_usable_area_applies_setback_and_obstacles():
    u = L.usable_area(box(0, 0, 10, 6), setback=0.5, obstacles=box(4, 2, 5, 3))
    assert u.area == pytest.approx(9 * 5 - 1)


def test_layout_counts_known_rectangle():
    # 10 x 6 area, 1 x 2 panels, no gap: portrait 10*3=30, landscape 5*6=30
    lay = L.layout_panels(box(0, 0, 10, 6), panel_w=1, panel_h=2, gap=0, angle=0)
    assert lay.count == 30
    for p in lay.panels:
        assert box(0, 0, 10, 6).buffer(1e-6).contains(p)


def test_layout_panels_do_not_overlap_and_respect_gap():
    lay = L.layout_panels(box(0, 0, 7.3, 5.1), panel_w=1.134, panel_h=1.722, gap=0.02)
    for i, a in enumerate(lay.panels):
        for b in lay.panels[i + 1:]:
            assert a.distance(b) >= 0.02 - 1e-6


def test_layout_picks_better_orientation():
    # 4.5 x 9 with 1x3 panels: portrait 4 x 3 = 12, landscape 1 x 9 = 9
    lay = L.layout_panels(box(0, 0, 4.5, 9), panel_w=1, panel_h=3, gap=0, angle=0)
    assert (lay.count, lay.orientation) == (12, "portrait")
    # 3.5 x 20: portrait 3 x 6 = 18, landscape 1 x 20 = 20
    lay = L.layout_panels(box(0, 0, 3.5, 20), panel_w=1, panel_h=3, gap=0, angle=0)
    assert (lay.count, lay.orientation) == (20, "landscape")


def test_layout_follows_rotated_roof():
    roof = affinity.rotate(box(0, 0, 10, 4), 30, origin=(0, 0))
    lay = L.layout_panels(roof, panel_w=1, panel_h=2, gap=0)
    assert lay.count == 20  # same as unrotated, because grid aligns to long edge
    assert lay.angle_deg % 180 == pytest.approx(30, abs=1e-6)


def test_layout_stays_inside_irregular_area_with_hole():
    area = L.usable_area(Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]),
                         setback=0, obstacles=box(1, 1, 3, 3))
    lay = L.layout_panels(area, panel_w=1, panel_h=1.5, gap=0.05)
    assert lay.count > 0
    for p in lay.panels:
        assert area.buffer(1e-5).contains(p)
        assert not p.intersects(box(1.01, 1.01, 2.99, 2.99))


def test_layout_empty_area():
    assert L.layout_panels(Polygon(), angle=0).count == 0


def test_min_row_gap_widens_tilted_rows_to_a_walkway():
    area = box(0, 0, 12, 12)
    kw = dict(tilt_deg=15, facing_deg=270.0, sun_alt_deg=L.winter_sun_altitude(19.0))
    plain = L.layout_panels(area, **kw)
    walk = L.layout_panels(area, min_row_gap=1.5, **kw)
    assert plain.row_gap < 1.5 and walk.row_gap == pytest.approx(1.5)
    assert walk.count < plain.count
    ys = sorted({round(p.bounds[1], 3) for p in walk.panels})
    depth = walk.panels[0].bounds[3] - walk.panels[0].bounds[1]
    assert min(b - a for a, b in zip(ys, ys[1:])) == pytest.approx(depth + 0.02 + 1.5, abs=2e-3)
    small = L.layout_panels(area, min_row_gap=0.01, **kw)
    assert small.row_gap == pytest.approx(plain.row_gap) and small.count == plain.count
