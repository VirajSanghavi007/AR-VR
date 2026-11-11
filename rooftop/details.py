"""Phase 2 M3: the detailed rooftop model, one set of parts for both STEP and GLB.

Everything is metres, z up, origin at the roof outline's south-west corner, slab
bottom at z = 0 (so the slab top is spec.slab_thickness / 1000). Every part is a
box (any orientation), a cylinder (any axis) or a closed triangle mesh, so the same
part becomes a CadQuery solid for STEP (mm) and a triangle mesh with normals for GLB.

Dimensions the drawing states stay exact: the slab and parapet follow the outline,
the parapet top (coping) is slab + H, room tops are slab + H, tank tops are
slab + H + S with the drawn diameter at the ribs. Everything else (screed fall,
coping overhang, door frames, stands, mounting) is detail added around them.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, box as sbox
from shapely.geometry.polygon import orient

from . import phase2 as P2

COPING_T, COPING_OVER = 0.05, 0.03       # coping thickness, overhang each side
UPTURN_H, UPTURN_T = 0.30, 0.005         # waterproofing turned up the parapet's inner face
SCREED_MIN, FALL, GRID = 0.025, 0.01, 1.0  # screed at the drains, 1:100 fall, mesh spacing
WALL_T, ROOF_T, OVERHANG = 0.115, 0.12, 0.15  # rooftop room walls and roof slab
FRAME = 0.06                             # door frame section
TANK_GAP, RIB_H, RIB_STEP = 0.015, 0.03, 0.25
STEEL_STAND = 0.45                       # stands taller than this are steel, lower ones brick
SIDES = 24
PANEL_T = 0.035
RAIL = 0.04                              # mounting rail / leg section
LEG_STEP = 1.8                           # most space between legs along a rail
TRAY_W, TRAY_H, TRAY_Z = 0.10, 0.06, 0.20
WINDOW_W, WINDOW_H, SILL, WINDOW_STEP = 1.2, 1.3, 0.9, 2.6


@dataclass
class Part:
    """One solid. kind box: centre, axes (rows: right-handed unit vectors), half sizes;
    cyl: base, axis (unit), r, length; mesh: verts, tris (closed, outward)."""
    group: str
    material: str
    kind: str
    centre: np.ndarray = None
    axes: np.ndarray = None
    half: np.ndarray = None
    base: np.ndarray = None
    axis: np.ndarray = None
    r: float = 0.0
    length: float = 0.0
    verts: np.ndarray = None
    tris: np.ndarray = None
    step: bool = True        # False: GLB only (the ground)


def box(group, material, lo, hi, step=True):
    """Axis-aligned box between corners lo and hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.any(hi - lo <= 0):
        raise ValueError(f"{group}: empty box {lo} {hi}")
    return Part(group, material, "box", centre=(lo + hi) / 2, axes=np.eye(3), half=(hi - lo) / 2, step=step)


def beam(group, material, p, q, w, h, up=(0, 0, 1)):
    """Box from point p to point q with a w (sideways) x h (towards up) section."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = q - p
    n = np.linalg.norm(u)
    if n <= 0:
        raise ValueError(f"{group}: zero-length beam")
    u /= n
    s = np.cross(np.asarray(up, float), u)
    if np.linalg.norm(s) < 1e-9:                      # beam along up: any sideways axis
        s = np.cross([1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0], u)
    s /= np.linalg.norm(s)
    v = np.cross(u, s)
    return Part(group, material, "box", centre=(p + q) / 2, axes=np.array([u, s, v]),
                half=np.array([n / 2, w / 2, h / 2]))


def cyl(group, material, base, axis, r, length):
    a = np.asarray(axis, float)
    if r <= 0 or length <= 0:
        raise ValueError(f"{group}: cylinder needs a positive radius and length")
    return Part(group, material, "cyl", base=np.asarray(base, float), axis=a / np.linalg.norm(a),
                r=float(r), length=float(length))


def mesh(group, material, verts, tris, step=True):
    return Part(group, material, "mesh", verts=np.asarray(verts, float), tris=np.asarray(tris, int), step=step)


def pipe(group, material, p, q, d):
    """Cylinder of diameter d from p to q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return cyl(group, material, p, q - p, d / 2, np.linalg.norm(q - p))


# ---------------------------------------------------------------- part -> mesh / solid

def _perp(a):
    e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _uv(v, n):
    """World-metre UVs, projected along each vertex normal's main axis (tiling textures)."""
    k = np.abs(n).argmax(1)
    uv = np.empty((len(v), 2))
    for ax, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        m = k == ax
        uv[m] = v[m][:, [i, j]]
    return uv


def part_mesh(p, sides=SIDES):
    """(verts, tris, normals, uv) for one part; triangles wind outwards."""
    if p.kind == "box":
        V, T, N = [], [], []
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            a, b = p.axes[i] * p.half[i], p.axes[j] * p.half[j]
            for s in (-1, 1):
                n = p.axes[k] * s
                fc = p.centre + n * p.half[k]
                o = len(V)
                V += [fc - a - b, fc + a - b, fc + a + b, fc - a + b]
                N += [n] * 4
                T += ([[o, o + 1, o + 2], [o, o + 2, o + 3]] if s > 0 else [[o, o + 2, o + 1], [o, o + 3, o + 2]])
        V, T, N = np.array(V), np.array(T), np.array(N)
    elif p.kind == "cyl":
        e1, e2 = _perp(p.axis)
        ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        rad = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
        top = p.base + p.axis * p.length
        n = sides
        V = np.vstack([p.base + p.r * rad, top + p.r * rad,        # side
                       p.base + p.r * rad, top + p.r * rad,        # caps
                       [p.base], [top]])
        N = np.vstack([rad, rad, np.tile(-p.axis, (n, 1)), np.tile(p.axis, (n, 1)), [-p.axis], [p.axis]])
        T = []
        for i in range(n):
            j = (i + 1) % n
            T += [[i, j, n + j], [i, n + j, n + i],
                  [4 * n, 2 * n + j, 2 * n + i], [4 * n + 1, 3 * n + i, 3 * n + j]]
        T = np.array(T)
    else:
        f = p.verts[p.tris]
        fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        V, N = f.reshape(-1, 3), np.repeat(fn, 3, axis=0)
        T = np.arange(len(V)).reshape(-1, 3)
    return V, T, N, _uv(V, N)


def part_solid(p, to_mm=1000.0):
    """CadQuery solid in millimetres."""
    import cadquery as cq

    if p.kind == "box":
        h = p.half * to_mm
        s = cq.Solid.makeBox(*(2 * h), cq.Vector(*(-h)))
        c = p.centre * to_mm
        if np.allclose(p.axes, np.eye(3)):
            return s.translate(cq.Vector(*c))
        plane = cq.Plane(cq.Vector(*c), cq.Vector(*p.axes[0]), cq.Vector(*p.axes[2]))
        return s.moved(cq.Location(plane))
    if p.kind == "cyl":
        return cq.Solid.makeCylinder(p.r * to_mm, p.length * to_mm, cq.Vector(*(p.base * to_mm)),
                                     cq.Vector(*p.axis))
    v = p.verts * to_mm
    faces = [cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*v[i]) for i in t], close=True))
             for t in p.tris]
    return cq.Solid.makeSolid(cq.Shell.makeShell(faces))


# ---------------------------------------------------------------- roof

def screed_height(x, y, drains, spec):
    """Top of the screed at (x, y): SCREED_MIN at the nearest drain, rising 1:100 away from it.
    With no drains the roof falls 1:100 towards its low-y edge."""
    z = spec.slab_thickness / 1000 + SCREED_MIN
    if len(drains):
        d = np.min(np.hypot(np.asarray(drains)[:, 0] - x, np.asarray(drains)[:, 1] - y))
    else:
        d = y - P2.inner_outline(spec).bounds[1]
    return z + FALL * d


def _grid_lines(lo, hi, extra):
    xs = np.r_[np.arange(lo, hi, GRID), hi, [e for e in extra if lo < e < hi]]
    xs = np.unique(np.round(xs, 6))
    keep = [xs[0]]
    for x in xs[1:]:
        if x - keep[-1] > 0.05:
            keep.append(x)
    keep[-1] = hi
    return np.array(keep)


def screed_part(spec, drains):
    """The sloped screed (topped by the waterproofing membrane) as one closed mesh."""
    floor = P2.inner_outline(spec).buffer(-UPTURN_T, join_style=2)
    z0 = spec.slab_thickness / 1000
    xy = shapely.get_coordinates(floor)
    minx, miny, maxx, maxy = floor.bounds
    dr = np.asarray(drains, float).reshape(-1, 2)
    xs = _grid_lines(minx, maxx, list(xy[:, 0]) + list(dr[:, 0]))
    ys = _grid_lines(miny, maxy, list(xy[:, 1]) + list(dr[:, 1]))
    inside = np.array([[floor.contains(shapely.Point((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2))
                        for j in range(len(ys) - 1)] for i in range(len(xs) - 1)])
    V, T, idx = [], [], {}

    def vid(i, j, top):
        k = (i, j, top)
        if k not in idx:
            idx[k] = len(V)
            V.append((xs[i], ys[j], screed_height(xs[i], ys[j], dr, spec) if top else z0))
        return idx[k]

    def quad(a, b, c, d, out):
        """Two triangles over a, b, c, d (in order round the quad), wound to face out."""
        for tri in ((a, b, c), (a, c, d)):
            p = np.array([V[t] for t in tri])
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), out) < 0:
                tri = tri[::-1]
            T.append(tri)

    ni, nj = inside.shape
    for i in range(ni):
        for j in range(nj):
            if not inside[i, j]:
                continue
            for top, out in ((True, (0, 0, 1)), (False, (0, 0, -1))):
                quad(vid(i, j, top), vid(i + 1, j, top), vid(i + 1, j + 1, top), vid(i, j + 1, top), out)
            for (a, b), nb, out in ((((i, j), (i, j + 1)), (i - 1, j), (-1, 0, 0)),
                                    (((i + 1, j), (i + 1, j + 1)), (i + 1, j), (1, 0, 0)),
                                    (((i, j), (i + 1, j)), (i, j - 1), (0, -1, 0)),
                                    (((i, j + 1), (i + 1, j + 1)), (i, j + 1), (0, 1, 0))):
                if 0 <= nb[0] < ni and 0 <= nb[1] < nj and inside[nb]:
                    continue
                quad(vid(*a, False), vid(*b, False), vid(*b, True), vid(*a, True), out)
    return mesh("Screed", "membrane", V, T)


def roof_parts(spec, drains=()):
    """Slab, parapet with coping, membrane upturn, sloped screed, drain grates."""
    t, H = spec.slab_thickness / 1000, spec.parapet_height / 1000
    out, inner = P2.outline(spec), P2.inner_outline(spec)
    cells = P2.rect_cells
    parts = [box("Slab", "concrete", (x0, y0, 0), (x1, y1, t)) for x0, y0, x1, y1 in cells(out)]
    parts += [box("Parapet", "plaster", (x0, y0, t), (x1, y1, t + H - COPING_T))
              for x0, y0, x1, y1 in cells(out.difference(inner))]
    cop = out.buffer(COPING_OVER, join_style=2).difference(inner.buffer(-COPING_OVER, join_style=2))
    parts += [box("Coping", "concrete", (x0, y0, t + H - COPING_T), (x1, y1, t + H))
              for x0, y0, x1, y1 in cells(cop)]
    parts += [box("Upturn", "membrane", (x0, y0, t), (x1, y1, t + UPTURN_H))
              for x0, y0, x1, y1 in cells(inner.difference(inner.buffer(-UPTURN_T, join_style=2)))]
    parts.append(screed_part(spec, drains))
    for i, (x, y) in enumerate(drains):
        parts += drain_parts(f"Drain{i + 1}", x, y, screed_height(x, y, np.asarray(drains), spec))
    return parts


def drain_parts(g, x, y, z):
    """A 200 mm square grate: dark sump, metal frame, three bars."""
    parts = [box(g, "drain", (x - 0.09, y - 0.09, z - 0.004), (x + 0.09, y + 0.09, z + 0.002))]
    a, b, w, h = 0.1, 0.08, 0.02, 0.012
    parts += [box(g, "metal", (x - a, y - a, z), (x + a, y - b, z + h)),
              box(g, "metal", (x - a, y + b, z), (x + a, y + a, z + h)),
              box(g, "metal", (x - a, y - b, z), (x - b, y + b, z + h)),
              box(g, "metal", (x + b, y - b, z), (x + a, y + b, z + h))]
    parts += [box(g, "metal", (x + o - 0.004, y - b, z), (x + o + 0.004, y + b, z + h * 0.8))
              for o in (-0.04, 0.0, 0.04)]
    return parts


# ---------------------------------------------------------------- rooms

def _side(f, s):
    """(outer-face start point, along unit, inward unit, length) of room side S, N, W or E."""
    return {"S": ((f.x, f.y), (1, 0), (0, 1), f.w), "N": ((f.x, f.y + f.d), (1, 0), (0, -1), f.w),
            "W": ((f.x, f.y), (0, 1), (1, 0), f.d), "E": ((f.x + f.w, f.y), (0, 1), (-1, 0), f.d)}[s]


def side_box(g, mat, f, s, a0, a1, d0, d1, z0, z1):
    """Box on room side s: a0..a1 along the side, d0..d1 inwards from the outer face."""
    o, al, inw, _ = (np.asarray(v, float) if i < 3 else v for i, v in enumerate(_side(f, s)))
    pts = np.array([o + al * a + inw * d for a in (a0, a1) for d in (d0, d1)])
    return box(g, mat, (*pts.min(0), z0), (*pts.max(0), z1))


def door_on(f, door):
    """(side, a0, a1) of a drawn door on room f, or the default door mid +y side."""
    if door is None:
        dw = 1.0 if f.kind == "stair" else 0.9
        return "N", f.w / 2 - dw / 2, f.w / 2 + dw / 2
    tol = 0.01
    if abs(door.y0 - f.y) < tol and abs(door.y1 - f.y) < tol:
        s = "S"
    elif abs(door.y0 - f.y - f.d) < tol and abs(door.y1 - f.y - f.d) < tol:
        s = "N"
    elif abs(door.x0 - f.x) < tol and abs(door.x1 - f.x) < tol:
        s = "W"
    elif abs(door.x0 - f.x - f.w) < tol and abs(door.x1 - f.x - f.w) < tol:
        s = "E"
    else:
        raise ValueError(f"door of {f.name} is not on one of its sides")
    o, al, _, _ = _side(f, s)
    a = sorted(np.dot(np.subtract(p, o), al) for p in ((door.x0, door.y0), (door.x1, door.y1)))
    return s, a[0], a[1]


def room_parts(f, z0, door=None):
    """Hollow room: walls with a door opening, overhanging roof slab, a door (stair) or a
    rolling shutter (store), a vent on the back wall (stair)."""
    g, top = f.name, z0 + f.h
    wtop = top - ROOF_T
    s, a0, a1 = door_on(f, door)
    dh = min(2.1 if f.kind == "stair" else 2.2, wtop - z0 - 0.2)
    parts = [box(g, "concrete", (f.x - OVERHANG, f.y - OVERHANG, wtop), (f.x + f.w + OVERHANG, f.y + f.d + OVERHANG, top))]
    for side in "SNWE":
        L = _side(f, side)[3]
        lo, hi = (0.0, L) if side in "SN" else (WALL_T, L - WALL_T)
        if side != s:
            parts.append(side_box(g, "plaster", f, side, lo, hi, 0, WALL_T, z0, wtop))
            continue
        if a0 - lo > 1e-6:
            parts.append(side_box(g, "plaster", f, side, lo, a0, 0, WALL_T, z0, wtop))
        if hi - a1 > 1e-6:
            parts.append(side_box(g, "plaster", f, side, a1, hi, 0, WALL_T, z0, wtop))
        parts.append(side_box(g, "plaster", f, side, a0, a1, 0, WALL_T, z0 + dh, wtop))
    if f.kind == "stair":
        parts += [side_box(g, "door_frame", f, s, a0, a0 + FRAME, 0, WALL_T, z0, z0 + dh),
                  side_box(g, "door_frame", f, s, a1 - FRAME, a1, 0, WALL_T, z0, z0 + dh),
                  side_box(g, "door_frame", f, s, a0 + FRAME, a1 - FRAME, 0, WALL_T, z0 + dh - FRAME, z0 + dh),
                  side_box(g, "door", f, s, a0 + FRAME, a1 - FRAME, WALL_T / 2 - 0.02, WALL_T / 2 + 0.02,
                           z0 + 0.01, z0 + dh - FRAME),
                  side_box(g, "metal", f, s, a1 - FRAME - 0.12, a1 - FRAME - 0.06, WALL_T / 2 - 0.07,
                           WALL_T / 2 - 0.02, z0 + 1.0, z0 + 1.03)]
        back = {"S": "N", "N": "S", "W": "E", "E": "W"}[s]
        L = _side(f, back)[3]
        parts.append(side_box(g, "vent", f, back, L / 2 - 0.2, L / 2 + 0.2, -0.03, 0, wtop - 0.5, wtop - 0.3))
    else:
        parts += [side_box(g, "shutter", f, s, a0, a1, 0.01, 0.03, z0, z0 + dh),
                  side_box(g, "metal", f, s, a0 - 0.04, a0 + 0.02, -0.02, 0.04, z0, z0 + dh),
                  side_box(g, "metal", f, s, a1 - 0.02, a1 + 0.04, -0.02, 0.04, z0, z0 + dh),
                  side_box(g, "shutter", f, s, a0 - 0.05, a1 + 0.05, -0.25, 0, z0 + dh, z0 + dh + 0.3)]
        parts += [side_box(g, "shutter", f, s, a0 + 0.01, a1 - 0.01, 0.0, 0.01, z, z + 0.012)
                  for z in np.arange(z0 + 0.08, z0 + dh - 0.05, 0.1)]
    return parts


# ---------------------------------------------------------------- tanks

def tank_parts(f, z0):
    """Ribbed plastic tank (diameter f.w at the ribs, top at z0 + f.h) on a steel stand
    (stand over STEEL_STAND) or a brick plinth, with an inlet and an outlet pipe."""
    g, r = f.name, f.w / 2
    cx, cy, zb, top = f.x + r, f.y + r, z0 + f.stand, z0 + f.h
    if top - 0.1 - zb < 0.2:
        raise ValueError(f"{g}: tank is too short ({f.h - f.stand:.2f} m) for its stand")
    parts = [cyl(g, "tank", (cx, cy, zb), (0, 0, 1), r - TANK_GAP, top - 0.1 - zb),
             cyl(g, "tank", (cx, cy, top - 0.1), (0, 0, 1), 0.8 * r, 0.05),
             cyl(g, "tank", (cx, cy, top - 0.05), (0, 0, 1), 0.3 * r, 0.05)]
    ribs = np.arange(zb + 0.15, top - 0.15 - RIB_H, RIB_STEP)
    if not len(ribs):
        ribs = [(zb + top - 0.1 - RIB_H) / 2]
    parts += [cyl(g, "tank", (cx, cy, z), (0, 0, 1), r, RIB_H) for z in ribs]
    if f.stand > STEEL_STAND:
        a, leg = 0.8 * r, 0.05
        parts.append(box(g, "steel", (cx - 0.95 * r, cy - 0.95 * r, zb - 0.02), (cx + 0.95 * r, cy + 0.95 * r, zb)))
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * a, cy + sy * a
                parts.append(box(g, "steel", (x - leg / 2, y - leg / 2, z0), (x + leg / 2, y + leg / 2, zb - 0.02)))
        zm = z0 + f.stand / 2
        parts += [box(g, "steel", (cx - a, cy + sy * a - 0.02, zm - 0.02), (cx + a, cy + sy * a + 0.02, zm + 0.02))
                  for sy in (-1, 1)]
        parts += [box(g, "steel", (cx + sx * a - 0.02, cy - a, zm - 0.02), (cx + sx * a + 0.02, cy + a, zm + 0.02))
                  for sx in (-1, 1)]
    elif f.stand > 0:
        parts.append(box(g, "brick", (f.x, f.y, z0), (f.x + f.w, f.y + f.d, zb)))
    d, xi = 0.032, cx + r + 0.06
    zi, zo = top - 0.25, zb + 0.1
    parts += [pipe(g, "pvc", (xi, cy, z0 + 0.05), (xi, cy, zi + d / 2), d),
              pipe(g, "pvc", (xi + d / 2, cy, zi), (cx + r - 0.05, cy, zi), d),
              pipe(g, "pvc", (cx - r + 0.05, cy, zo), (cx - r - 0.25 - d / 2, cy, zo), d),
              pipe(g, "pvc", (cx - r - 0.25, cy, zo + d / 2), (cx - r - 0.25, cy, z0 + 0.05), d)]
    return parts


# ---------------------------------------------------------------- pipes and building

def pipe_parts(pipes, spec, drains):
    """Drawn pipes along their routes on small supports above the screed."""
    parts, z0 = [], spec.slab_thickness / 1000
    dr = np.asarray(drains, float).reshape(-1, 2)
    for i, pp in enumerate(pipes):
        g, d, pts = f"Pipe{i + 1}", pp.diameter / 1000, np.asarray(pp.points, float)
        zc = max(screed_height(x, y, dr, spec) for x, y in pts) + 0.05 + d / 2
        for a, b in zip(pts[:-1], pts[1:]):
            u = (b - a) / np.linalg.norm(b - a)
            parts.append(pipe(g, "pvc", (*(a - u * d / 2), zc), (*(b + u * d / 2), zc), d))
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / 1.5)) + 1)
            for t in np.linspace(0.1, 0.9, n):
                x, y = a + (b - a) * t
                parts.append(box(g, "concrete", (x - 0.05, y - 0.05, z0), (x + 0.05, y + 0.05, zc - d / 2)))
    return parts


def building_parts(spec, floors=3, floor_h=3.0, ground_margin=8.0):
    """The floors below the roof: plastered block, a band at each floor, windows on every
    outer wall, and the ground (GLB only)."""
    out = orient(P2.outline(spec))                       # counter-clockwise: outward = right
    base = -floors * floor_h
    parts = [box("Building", "plaster", (x0, y0, base), (x1, y1, 0)) for x0, y0, x1, y1 in P2.rect_cells(out)]
    band = out.buffer(0.03, join_style=2).difference(out)
    for k in range(floors):
        z = base + k * floor_h
        parts += [box("Building", "concrete", (x0, y0, z + floor_h - 0.15), (x1, y1, z + floor_h))
                  for x0, y0, x1, y1 in P2.rect_cells(band)]
    ring = np.asarray(out.exterior.coords)
    for a, b in zip(ring[:-1], ring[1:]):
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        n = np.array([u[1], -u[0]])
        count = int((L - 1.2) // WINDOW_STEP)
        for c in (a + u * (L / 2 + (i - (count - 1) / 2) * WINDOW_STEP) for i in range(count)):
            for k in range(floors):
                z = base + k * floor_h + SILL
                w = np.array([c - u * WINDOW_W / 2, c + u * WINDOW_W / 2, c - u * WINDOW_W / 2 + n * 0.015,
                              c + u * WINDOW_W / 2 + n * 0.015])
                parts.append(box("Windows", "window", (*w.min(0), z), (*w.max(0), z + WINDOW_H)))
                s = np.array([c - u * 0.65, c + u * 0.65 + n * 0.06])
                parts.append(box("Windows", "concrete", (*s.min(0), z - 0.05), (*s.max(0), z)))
    x0, y0, x1, y1 = out.bounds
    parts.append(box("Ground", "ground", (x0 - ground_margin, y0 - ground_margin, base - 0.1),
                     (x1 + ground_margin, y1 + ground_margin, base), step=False))
    return parts


# ---------------------------------------------------------------- panels

def _panel_edges(c):
    """Front (low) edge and the vector from front to back of one panel quad."""
    c = np.asarray(c, float)
    order = np.argsort(c[:, 2], kind="stable")
    f0, f1 = c[order[0]], c[order[1]]
    k = [i for i in range(4) if i not in order[:2]]
    b = [c[i] for i in k]
    b0 = min(b, key=lambda p: np.linalg.norm(p - f0))
    return f0, f1, b0 - f0


def _normal(c):
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n /= np.linalg.norm(n)
    return n if n[2] >= 0 else -n


def mounting_parts(corners, spec, drains=(), fractions=(0.2, 0.8)):
    """Rails under each run of touching panels (two per run, along the row) on vertical legs
    with base plates standing on the screed."""
    if not len(corners):
        return []
    dr = np.asarray(drains, float).reshape(-1, 2)
    info = []
    for c in corners:
        f0, f1, g = _panel_edges(c)
        e = (f1 - f0) / np.linalg.norm(f1 - f0)
        info.append((f0, f1, g, e, _normal(np.asarray(c, float))))
    e = info[0][3]
    rows = {}
    for f0, f1, g, _, n in info:
        across = np.round(np.dot(f0[:2], [-e[1], e[0]]), 2)
        rows.setdefault((across, round(f0[2], 3)), []).append((f0, f1, g, n))
    parts = []
    for r, key in enumerate(sorted(rows)):
        items = sorted(rows[key], key=lambda it: min(np.dot(it[0], e), np.dot(it[1], e)))
        runs, cur = [], [items[0]]
        for it in items[1:]:
            prev_end = max(np.dot(cur[-1][0], e), np.dot(cur[-1][1], e))
            if min(np.dot(it[0], e), np.dot(it[1], e)) - prev_end > 0.1:
                runs.append(cur)
                cur = []
            cur.append(it)
        runs.append(cur)
        for k, run in enumerate(runs):
            gname = f"Mounting_R{r + 1:02d}_{k + 1}"
            ends = [p for it in run for p in it[:2]]
            p0 = min(ends, key=lambda p: np.dot(p, e))
            p1 = max(ends, key=lambda p: np.dot(p, e))
            g, n = run[0][2], run[0][3]
            for fr in fractions:
                off = g * fr - n * (PANEL_T + RAIL / 2)
                a, b = p0 + off, p1 + off
                parts.append(beam(gname, "galv", a, b, RAIL, RAIL, up=n))
                span = np.linalg.norm(b - a)
                m = int(np.ceil((span - 0.2) / LEG_STEP)) + 1
                for t in np.linspace(0.1, span - 0.1, max(2, m)):
                    x, y, zr = a + (b - a) * t / span
                    zs = screed_height(x, y, dr, spec)
                    ztop = zr - RAIL / 2
                    if ztop - zs < 0.03:
                        continue
                    h = RAIL / 2
                    parts.append(box(gname, "galv", (x - 0.06, y - 0.06, zs), (x + 0.06, y + 0.06, zs + 0.008)))
                    parts.append(box(gname, "galv", (x - h, y - h, zs + 0.008), (x + h, y + h, ztop)))
    return parts


def _wall_point(f, s, along, depth):
    o, al, inw, _ = (np.asarray(v, float) if i < 3 else v for i, v in enumerate(_side(f, s)))
    return o + al * along + inw * depth


def inverter_parts(features, corners, spec, doors=()):
    """Inverter on the stair room's wall facing the array, a conduit down, and a cable tray
    on the roof to the nearest panel row (two straight legs, around the rooms and tanks)."""
    rooms = [f for f in features if f.kind == "stair"] or [f for f in features if f.kind != "tank"]
    if not rooms or not len(corners):
        return []
    f, z0 = rooms[0], spec.slab_thickness / 1000
    door = next((d for d in doors if d.room == f.name), None)
    ds = door_on(f, door)[0]
    target = np.mean([np.mean(np.asarray(c)[:, :2], 0) for c in corners], 0)
    centre = np.array([f.x + f.w / 2, f.y + f.d / 2])
    best = max((s for s in "SNWE" if s != ds),
               key=lambda s: np.dot(-np.asarray(_side(f, s)[2], float), target - centre))
    L = _side(f, best)[3]
    parts = [side_box("Inverter", "inverter", f, best, L / 2 - 0.225, L / 2 + 0.225, -0.2, 0, z0 + 1.0, z0 + 1.6),
             side_box("Inverter", "metal", f, best, L / 2 - 0.1, L / 2 + 0.1, -0.205, -0.2, z0 + 1.35, z0 + 1.5)]
    start = _wall_point(f, best, L / 2, -0.1)
    parts.append(pipe("Inverter", "pvc", (*start, z0 + 1.0), (*start, z0 + TRAY_Z), 0.04))
    feet = np.array([np.asarray(c)[np.argsort(np.asarray(c)[:, 2])[:2]].mean(0)[:2] for c in corners])
    end = feet[np.argmin(np.linalg.norm(feet - start, axis=1))]
    blocks = [g.footprint.buffer(0.05) for g in features]
    routes = [[start, (end[0], start[1]), end], [start, (start[0], end[1]), end]]

    def hits(rt):
        ln = LineString(rt)
        return sum(ln.intersects(b) for b in blocks if not b.contains(shapely.Point(start)))

    route = min(routes, key=hits)
    zt = z0 + TRAY_Z
    for a, b in zip(route[:-1], route[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.linalg.norm(b - a) < 0.05:
            continue
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        a, b = a - u * TRAY_W / 2, b + u * TRAY_W / 2

        def strip(o0, o1, z_0, z_1):
            p = np.array([a + n * o0, a + n * o1, b + n * o0, b + n * o1])
            return box("CableTray", "galv", (*p.min(0), z_0), (*p.max(0), z_1))

        parts += [strip(-TRAY_W / 2, TRAY_W / 2, zt, zt + 0.005),
                  strip(-TRAY_W / 2, -TRAY_W / 2 + 0.005, zt, zt + TRAY_H),
                  strip(TRAY_W / 2 - 0.005, TRAY_W / 2, zt, zt + TRAY_H)]
        steps = max(2, int(np.linalg.norm(b - a) // 1.5) + 1)
        for t in np.linspace(0.15, 0.85, steps):
            x, y = a + (b - a) * t
            parts.append(box("CableTray", "concrete", (x - 0.06, y - 0.06, z0), (x + 0.06, y + 0.06, zt)))
    return parts


# ---------------------------------------------------------------- the whole model

def build(spec, features=(), corners=(), doors=(), drains=(), pipes=(), floors=3, floor_h=3.0):
    """All detailed parts (not the panels themselves, which stay quads; see write_glb)."""
    z0 = spec.slab_thickness / 1000
    drains = [tuple(d) for d in drains]
    parts = roof_parts(spec, drains)
    by_room = {d.room: d for d in doors}
    for f in features:
        parts += tank_parts(f, z0) if f.kind == "tank" else room_parts(f, z0, by_room.get(f.name))
    parts += pipe_parts(pipes, spec, drains)
    parts += building_parts(spec, floors, floor_h)
    parts += mounting_parts(corners, spec, drains)
    parts += inverter_parts(features, corners, spec, doors)
    return parts


def site_floors(site):
    """(floors, floor height m) from a drawing's SITE_INFO (FLOOR_H in mm), default 3 x 3.0 m."""
    return int(site.get("FLOORS", 3)), float(site.get("FLOOR_H", 3000)) / 1000


# ---------------------------------------------------------------- writers

COLOURS = {   # name: (rgb linear, metallic, roughness); tiling textures: rooftop/textures.py
    "concrete": ((0.62, 0.61, 0.58), 0.0, 0.9), "plaster": ((0.86, 0.83, 0.76), 0.0, 0.85),
    "membrane": ((0.50, 0.50, 0.49), 0.0, 0.8), "galv": ((0.70, 0.72, 0.74), 0.8, 0.45),
    "steel": ((0.30, 0.31, 0.33), 0.7, 0.5), "metal": ((0.55, 0.56, 0.58), 0.9, 0.35),
    "tank": ((0.07, 0.18, 0.42), 0.0, 0.55), "brick": ((0.62, 0.33, 0.24), 0.0, 0.9),
    "door": ((0.40, 0.28, 0.18), 0.0, 0.65), "door_frame": ((0.30, 0.22, 0.15), 0.0, 0.6),
    "shutter": ((0.60, 0.62, 0.64), 0.7, 0.45), "pvc": ((0.88, 0.88, 0.86), 0.0, 0.5),
    "window": ((0.20, 0.27, 0.32), 0.3, 0.1), "inverter": ((0.92, 0.92, 0.90), 0.1, 0.5),
    "ground": ((0.36, 0.38, 0.30), 0.0, 1.0), "vent": ((0.35, 0.35, 0.35), 0.5, 0.6),
    "drain": ((0.08, 0.08, 0.08), 0.0, 0.9),
}


def materials(textures=None):
    """glTF materials for every key build() uses plus the panel's frame and glass.
    textures: optional {key: image bytes, or (albedo bytes, normal-map bytes)} of tiling textures
    (world-metre UVs), e.g. rooftop.textures.texture_set(COLOURS)."""
    from .ar_export import Material, _materials

    m = _materials()
    for k, (rgb, met, rough) in COLOURS.items():
        tex = (textures or {}).get(k)
        tex, nrm = tex if isinstance(tex, tuple) else (tex, None)
        m[k] = Material((1.0, 1.0, 1.0) if tex else rgb, met, rough, tex, tile=tex is not None or nrm is not None,
                        normal=nrm)
    return m


def panel_meshes(corners, thickness=PANEL_T):
    """(verts, tris, uv, key, normals) for each panel: aluminium body + textured glass."""
    from .ar_export import box_from_quad, laminate_quad, _normal as qn

    out = []
    for c in corners:
        v, t = box_from_quad(c, thickness)
        f = v[t]
        fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        out.append((f.reshape(-1, 3), np.arange(len(t) * 3).reshape(-1, 3), None, "frame", np.repeat(fn, 3, 0)))
        lv, lt, luv = laminate_quad(c)
        out.append((lv, lt, luv, "glass", np.tile(qn(np.asarray(c, float)), (4, 1))))
    return out


def glb_parts(parts, corners):
    """All parts as glb_bytes input, converted to glTF's y-up."""
    from .ar_export import Z_TO_Y

    out = []
    for p in parts:
        v, t, n, uv = part_mesh(p)
        out.append((v @ Z_TO_Y.T, t, uv, p.material, n @ Z_TO_Y.T))
    out += [(v @ Z_TO_Y.T, t, uv, k, n @ Z_TO_Y.T) for v, t, uv, k, n in panel_meshes(corners)]
    return out


def write_glb(path, parts, corners, textures=None):
    from pathlib import Path

    from .ar_export import glb_bytes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(glb_parts(parts, corners), materials(textures)))
    return path


def export_step(path, parts, corners, thickness=PANEL_T):
    """One STEP assembly (mm): a compound per group and material, coloured, plus Panel_### solids.
    Parts with step=False (the ground) are left out."""
    from pathlib import Path

    import cadquery as cq

    from .panels import quad_solid

    asm = cq.Assembly(name="RooftopDetailed")
    groups = {}
    for p in parts:
        if p.step:
            groups.setdefault((p.group, p.material), []).append(part_solid(p))
    for (g, mat), solids in groups.items():
        rgb = COLOURS.get(mat, ((0.5, 0.5, 0.5),))[0]
        asm.add(cq.Workplane().add(cq.Compound.makeCompound(solids)), name=f"{g}_{mat}", color=cq.Color(*rgb))
    for i, c in enumerate(corners):
        asm.add(cq.Workplane().add(quad_solid(c, thickness, 1000.0)), name=f"Panel_{i:03d}",
                color=cq.Color(0.1, 0.2, 0.8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    asm.export(str(path), exportType="STEP")
    return path


def write_json(path, dwg, features, lay, corners, latitude, facing_deg, floors, floor_h, extra=None):
    """Sidecar the customer page reads: site, structures, panels and figures (metres, plan x/y)."""
    import json
    from pathlib import Path

    spec = dwg.spec
    data = {"detailed": True, "slab": spec.slab_thickness / 1000, "parapet": spec.parapet_height / 1000,
            "cells": [list(c) for c in P2.rect_cells(P2.outline(spec))],
            "floors": floors, "floor_h": floor_h, "building_base": round(-floors * floor_h, 3),
            "latitude": latitude, "longitude": dwg.site.get("LON"), "tz": dwg.site.get("TZ"),
            "north_deg": dwg.north_deg, "facing_deg": facing_deg,
            "tilt_deg": lay.tilt_deg, "row_gap": lay.row_gap,
            "features": [{"name": f.name, "kind": f.kind, "x": f.x, "y": f.y, "w": f.w, "d": f.d, "h": f.h,
                          "stand": f.stand} for f in features],
            "panels": len(corners), "panel_w": 550, "peak_kw": round(len(corners) * 0.55, 2)}
    data.update(extra or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path
