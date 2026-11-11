"""M7: measure the detailed 3D model and compare it with the drawing's DIMENSIONS and an
independent measurements.json (all mm). Read the model from its STEP file
(groups_from_step) or straight from details.build parts (groups_from_parts, fast, for tests).

A model is a dict "<Group>_<material>" -> list of axis-aligned boxes (lo, hi) in mm, one per
solid. Every check is a Row; a row passes when |model - drawing| <= tol.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, box as sbox
from shapely.ops import unary_union

TOL = 5.0          # mm
SNAP = 50.0        # mm: a drawing dimension point must have a model corner this close


@dataclass
class Row:
    name: str
    drawing: float     # mm, the drawing's (or tape's) value
    model: float       # mm, measured on the model; nan when it could not be measured
    tol: float = TOL

    @property
    def error(self):
        return self.model - self.drawing

    @property
    def ok(self):
        return bool(abs(self.error) <= self.tol)                         # NaN (missing) never passes


# ---------------------------------------------------------------- reading a model

def groups_from_parts(parts):
    """Model from details.Part list (only parts that go to STEP), using the meshed vertices."""
    from .details import part_mesh

    out = {}
    for p in parts:
        if p.step:
            v = part_mesh(p)[0] * 1000.0
            out.setdefault(f"{p.group}_{p.material}", []).append((v.min(0), v.max(0)))
    return out


def groups_from_step(path):
    """Model from a details.export_step file: each named child, one box per solid."""
    import cadquery as cq

    asm = cq.Assembly.importStep(str(path))
    out = {}
    for name, child in asm.objects.items():
        for obj in ([] if child.obj is None else child.obj.vals() if hasattr(child.obj, "vals") else [child.obj]):
            for s in obj.Solids():
                bb = s.BoundingBox()
                out.setdefault(name, []).append((np.array([bb.xmin, bb.ymin, bb.zmin]),
                                                 np.array([bb.xmax, bb.ymax, bb.zmax])))
    return out


def _boxes(model, *names):
    return [b for n in names for b in model.get(n, [])]


def extent(model, *names):
    """(lo, hi) over all solids of these groups, or None."""
    b = _boxes(model, *names)
    if not b:
        return None
    return np.min([lo for lo, _ in b], 0), np.max([hi for _, hi in b], 0)


def plan(model, *names):
    """Union of the plan footprints of these groups (shapely)."""
    return unary_union([sbox(lo[0], lo[1], hi[0], hi[1]) for lo, hi in _boxes(model, *names)])


def _nan():
    return float("nan")


# ---------------------------------------------------------------- checks

def snap_groups(model):
    """Groups whose plan corners drawing dimensions may point at: slab, parapet, room walls."""
    return ["Slab_concrete"] + [k for k in model if k.endswith("_plaster") and k != "Building_plaster"]


def _station(corners, p, u, snap):
    """The model corner that a dimension end p measures to, as on a drawing's extension line:
    among corners within snap of the line through p across u, the one nearest p. None if none."""
    if not len(corners):
        return None
    near = corners[np.abs((corners - p) @ u) <= snap]
    if not len(near):
        return None
    return near[np.argmin(np.linalg.norm(near - p, axis=1))]


def dimension_rows(model, dims, snap=SNAP, tol=TOL):
    """Drawing DIMENSIONS: snap each end to the nearest model plan corner and measure along the
    dimension direction. dims: dxf_read.Dimension (p1, p2 in m, value mm) or dicts in mm."""
    corners = np.array([(x, y) for lo, hi in _boxes(model, *snap_groups(model))
                        for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]).reshape(-1, 2)
    rows = []
    for i, d in enumerate(dims):
        if isinstance(d, dict):
            name, p1, p2, val = d.get("name", f"dim{i + 1}"), np.array(d["p1"], float), np.array(d["p2"], float), d["value"]
        else:
            name, p1, p2, val = f"dim{i + 1}", np.array(d.p1) * 1000, np.array(d.p2) * 1000, d.value
        u = (p2 - p1) / np.linalg.norm(p2 - p1)
        a, b = _station(corners, p1, u, snap), _station(corners, p2, u, snap)
        got = _nan() if a is None or b is None else float(abs(np.dot(b - a, u)))
        rows.append(Row(f"DIMENSIONS {name}", float(val), got, tol))
    return rows


def _opening(walls, door, centre):
    """(a0, a1) of the gap in the walls along a door line, as positions along the door."""
    p, q = np.asarray(door[:2], float), np.asarray(door[2:], float)
    u = (q - p) / np.linalg.norm(q - p)
    n = np.array([-u[1], u[0]])
    if np.dot(centre - p, n) < 0:
        n = -n
    o = p + n * 10.0                                  # 10 mm inside the wall
    line = LineString([o - u * 2000, o + u * 2000])
    gaps = line.difference(walls)
    mid = Point(*(o + (q - p) / 2))
    for g in getattr(gaps, "geoms", [gaps]):
        if g.distance(mid) < 1e-6:
            a, b = (np.dot(np.asarray(c)[:2] - o, u) for c in (g.coords[0], g.coords[-1]))
            return sorted((a, b))
    return None


def measurement_rows(model, m, tol=TOL):
    """measurements.json (mm) against the model."""
    rows = []
    add = lambda name, want, got: rows.append(Row(name, float(want), _nan() if got is None else float(got), tol))
    slab = extent(model, "Slab_concrete")
    t = slab[1][2] if slab else 0.0                  # slab top = roof level
    L = W = None
    if slab:
        L, W = slab[1][0] - slab[0][0], slab[1][1] - slab[0][1]
    for k, v in (("length", L), ("width", W)):
        if k in m:
            add(k, m[k], v)
    if "diagonal" in m:
        add("diagonal", m["diagonal"], None if L is None else np.hypot(L, W))
    out = plan(model, "Slab_concrete")
    if slab and not out.is_empty and ("notch_length" in m or "notch_width" in m):
        x0, y0, x1, y1 = out.bounds
        top = out.intersection(LineString([(x0 - 1, y1 - 1), (x1 + 1, y1 - 1)])).bounds
        right = out.intersection(LineString([(x1 - 1, y0 - 1), (x1 - 1, y1 + 1)])).bounds
        if "notch_length" in m:
            add("notch_length", m["notch_length"], x1 - top[2] if top else None)
        if "notch_width" in m:
            add("notch_width", m["notch_width"], y1 - right[3] if right else None)
    par = plan(model, "Parapet_plaster")
    if "parapet_thickness" in m:
        got = None
        if slab and not par.is_empty:
            inner = out.difference(par).buffer(-1, join_style=2).buffer(1, join_style=2)  # drop float slivers
            if not inner.is_empty:
                (a0, b0, a1, b1), (c0, d0, c1, d1) = out.bounds, inner.bounds
                sides = [c0 - a0, d0 - b0, a1 - c1, b1 - d1]
                got = max(sides, key=lambda s: abs(s - m["parapet_thickness"]))
        add("parapet_thickness", m["parapet_thickness"], got)
    if "parapet_height" in m:
        cop = extent(model, "Coping_concrete", "Parapet_plaster")
        add("parapet_height", m["parapet_height"], cop[1][2] - t if cop else None)
    for r in m.get("rooms", []):
        n = r["name"]
        walls = extent(model, f"{n}_plaster")
        roof = extent(model, f"{n}_concrete")
        for k, i in (("x", 0), ("y", 1)):
            add(f"{n} {k}", r[k], walls[0][i] if walls else None)
        add(f"{n} w", r["w"], walls[1][0] - walls[0][0] if walls else None)
        add(f"{n} d", r["d"], walls[1][1] - walls[0][1] if walls else None)
        add(f"{n} h", r["h"], roof[1][2] - t if roof else None)
        if "door" in r:
            dr = r["door"]
            low = [(lo, hi) for lo, hi in model.get(f"{n}_plaster", []) if abs(lo[2] - t) < 1.0]
            gap = None
            if walls and low:
                wall_plan = unary_union([sbox(lo[0], lo[1], hi[0], hi[1]) for lo, hi in low])
                gap = _opening(wall_plan, dr, (walls[0][:2] + walls[1][:2]) / 2)
            width = np.hypot(dr[2] - dr[0], dr[3] - dr[1])
            add(f"{n} door start", 0.0, gap[0] if gap else None)
            add(f"{n} door end", width, gap[1] if gap else None)
    for tk in m.get("tanks", []):
        n = tk["name"]
        e = extent(model, f"{n}_tank")
        add(f"{n} cx", tk["cx"], (e[0][0] + e[1][0]) / 2 if e else None)
        add(f"{n} cy", tk["cy"], (e[0][1] + e[1][1]) / 2 if e else None)
        add(f"{n} r", tk["r"], (e[1][0] - e[0][0]) / 2 if e else None)
        add(f"{n} h", tk["h"], e[1][2] - e[0][2] if e else None)
        add(f"{n} stand", tk["s"], e[0][2] - t if e else None)
    for i, (x, y) in enumerate(m.get("drains", [])):
        e = extent(model, f"Drain{i + 1}_drain")
        add(f"Drain{i + 1} x", x, (e[0][0] + e[1][0]) / 2 if e else None)
        add(f"Drain{i + 1} y", y, (e[0][1] + e[1][1]) / 2 if e else None)
    for i, pp in enumerate(m.get("pipes", [])):
        n, d = f"Pipe{i + 1}", pp["d"]
        e = extent(model, f"{n}_pvc")
        pts = np.asarray(pp["points"], float)
        lo, hi = pts.min(0) - d / 2, pts.max(0) + d / 2
        for k, want, got in (("x min", lo[0], e and e[0][0]), ("y min", lo[1], e and e[0][1]),
                             ("x max", hi[0], e and e[1][0]), ("y max", hi[1], e and e[1][1]),
                             ("d", d, e and e[1][2] - e[0][2])):
            add(f"{n} {k}", want, got)
    site = m.get("site", {})
    if "floors" in site and "floor_h" in site:
        b = extent(model, "Building_plaster")
        add("building height", site["floors"] * site["floor_h"], -b[0][2] if b else None)
    return rows


def table(rows):
    """Markdown table of the rows plus a summary line."""
    lines = ["| check | drawing mm | model mm | error mm | result |", "|---|---:|---:|---:|---|"]
    for r in rows:
        got = f"{r.model:.1f}" if np.isfinite(r.model) else "missing"
        err = f"{round(r.error, 1) + 0.0:+.1f}" if np.isfinite(r.model) else "-"
        lines.append(f"| {r.name} | {r.drawing:.1f} | {got} | {err} | {'pass' if r.ok else 'FAIL'} |")
    bad = sum(not r.ok for r in rows)
    lines.append("")
    lines.append(f"{len(rows) - bad}/{len(rows)} checks pass (tolerance {rows[0].tol:g} mm)" if rows else "no checks")
    return "\n".join(lines)
