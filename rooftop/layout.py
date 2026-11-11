"""Roof outline, keep-out zones and solar panel grid layout (all in plane 2D, metres)."""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union


def roof_outline(uv, ratio=0.15) -> Polygon:
    """Concave hull of the roof points; ratio 0 = tight, 1 = convex hull."""
    hull = shapely.concave_hull(MultiPoint(np.asarray(uv)), ratio=ratio)
    if hull.geom_type == "MultiPolygon":
        hull = max(hull.geoms, key=lambda g: g.area)
    if hull.geom_type != "Polygon" or hull.area == 0:
        raise ValueError("Roof points do not span an area")
    return hull.buffer(0)


def keepouts(uv, heights, outline, min_height=0.15, max_height=3.0,
             clearance=0.3, cell=0.25, min_points=10):
    """Obstacles: points standing min_height..max_height above the roof inside the outline.

    Grid-bins the obstacle points (no extra deps), keeps dense cells, merges and
    buffers by the clearance.
    """
    uv, heights = np.asarray(uv, float), np.asarray(heights, float)
    m = (heights > min_height) & (heights < max_height)
    if not m.any():
        return Polygon()
    pts = uv[m]
    inside = shapely.contains_xy(outline.buffer(clearance), pts[:, 0], pts[:, 1])
    pts = pts[inside]
    if len(pts) == 0:
        return Polygon()
    keys, counts = np.unique(np.floor(pts / cell).astype(int), axis=0, return_counts=True)
    cells = [box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell)
             for (i, j), c in zip(keys, counts) if c >= min_points]
    if not cells:
        return Polygon()
    return unary_union(cells).buffer(clearance)


def usable_area(outline, setback=0.5, obstacles=None):
    area = outline.buffer(-setback)
    if obstacles is not None and not obstacles.is_empty:
        area = area.difference(obstacles)
    return area


def winter_sun_altitude(latitude_deg) -> float:
    """Solar altitude at solar noon on the winter solstice (worst case for row shading)."""
    alt = 90.0 - abs(latitude_deg) - 23.44
    if alt <= 0:
        raise ValueError("Sun does not rise at noon on the winter solstice at this latitude")
    return alt


def row_shadow_gap(slope_len, tilt_deg, sun_alt_deg) -> float:
    """Clear distance behind a tilted row so its shadow does not reach the next row."""
    if tilt_deg <= 0:
        return 0.0
    rise = slope_len * np.sin(np.radians(tilt_deg))
    return rise / np.tan(np.radians(sun_alt_deg))


@dataclass
class Layout:
    panels: list           # shapely Polygons: panel footprints in plane 2D
    orientation: str       # "portrait" or "landscape"
    angle_deg: float       # grid rotation in the plane (rows run along this angle)
    tilt_deg: float = 0.0  # panel tilt relative to the roof plane
    rise: float = 0.0      # back edge height above front edge (m)
    row_gap: float = 0.0   # shading clearance between rows (m)

    @property
    def count(self):
        return len(self.panels)

    @property
    def facing(self) -> np.ndarray:
        """2D unit vector the tilted panels face (towards the low front edge)."""
        a = np.radians(self.angle_deg)
        return np.array([np.sin(a), -np.cos(a)])


def _grid(area, w, h, gap, row_pitch_extra, angle, offsets):
    """Place a w x h grid rotated by angle; return footprints fully inside area.

    Rows run along local x; row pitch along local y is h + gap + row_pitch_extra.
    """
    if area.is_empty:
        return []
    c = area.centroid
    rot = affinity.rotate(area, -angle, origin=c)
    minx, miny, maxx, maxy = rot.bounds
    rot = rot.buffer(1e-6)  # absorb float error so exact fits are kept
    shapely.prepare(rot)
    best = []
    dy = h + gap + row_pitch_extra
    for fx, fy in offsets:
        xs = np.arange(minx + fx * (w + gap), maxx - w + 1e-9, w + gap)
        ys = np.arange(miny + fy * dy, maxy - h + 1e-9, dy)
        cand = [box(x, y, x + w, y + h) for x in xs for y in ys]
        keep = [p for p in cand if rot.contains(p)]
        if len(keep) > len(best):
            best = keep
    return [affinity.rotate(p, angle, origin=c) for p in best]


def dominant_angle(outline) -> float:
    """Angle (deg) of the long edge of the outline's minimum rotated rectangle."""
    mrr = outline.minimum_rotated_rectangle
    x, y = np.asarray(mrr.exterior.coords).T
    edges = [(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(2)]
    dx, dy = max(edges, key=lambda e: np.hypot(*e))
    return float(np.degrees(np.arctan2(dy, dx)))


def layout_panels(area, panel_w=1.134, panel_h=1.722, gap=0.02, angle=None, steps=4,
                  tilt_deg=0.0, facing_deg=None, sun_alt_deg=None, min_row_gap=0.0):
    """Fill area with panels; try portrait/landscape and grid offsets, keep the max.

    Flat (tilt 0): rows align with the roof's long edge unless angle is given.
    Tilted: panels face facing_deg (direction in the roof 2D frame, towards the
    equator), rows run perpendicular to it, each footprint shrinks to
    slope*cos(tilt) and rows are spaced so the winter-noon shadow (sun_alt_deg)
    clears the next row. min_row_gap: the extra space between rows is at least this
    (a maintenance walkway).
    """
    if tilt_deg > 0:
        if facing_deg is None or sun_alt_deg is None:
            raise ValueError("Tilted layout needs facing_deg and sun_alt_deg")
        angle = facing_deg + 90.0
    elif angle is None:
        angle = dominant_angle(area.convex_hull) if not area.is_empty else 0.0
    offsets = [(i / steps, j / steps) for i in range(steps) for j in range(steps)]
    cos_t = np.cos(np.radians(tilt_deg))
    best = Layout([], "portrait", angle, tilt_deg)
    for name, (w, slope) in (("portrait", (panel_w, panel_h)), ("landscape", (panel_h, panel_w))):
        gap_r = row_shadow_gap(slope, tilt_deg, sun_alt_deg) if tilt_deg > 0 else 0.0
        gap_r = max(gap_r, min_row_gap)
        panels = _grid(area, w, slope * cos_t, gap, gap_r, angle, offsets)
        if len(panels) > best.count:
            best = Layout(panels, name, angle, tilt_deg,
                          slope * np.sin(np.radians(tilt_deg)), gap_r)
    return best
