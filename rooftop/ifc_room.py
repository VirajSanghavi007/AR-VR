"""One room cut out of an IFC building model (BIM) as a GLB for the AR page's Room model card.

The room is an IfcSpace, picked by its number (Name, e.g. "A102") or its name (LongName, e.g. "Living Room").
Everything in the building (walls, slabs, ceilings, doors, windows, furniture) is clipped to the room's
box grown by `margin`, so the GLB is the room's inside faces plus a thin skin of its walls, floor and
ceiling, like the sample room (rooftop/room.py). Coordinates are shifted so the room's inside starts at
the origin, then turned y up (metres): the IFC +y side (the room's "back") becomes the model's -Z side.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ar_export import Material, Z_TO_Y, glb_bytes
from .room import MATERIALS as ROOM_MATERIALS

KINDS = {
    "IfcWall": "wall", "IfcWallStandardCase": "wall", "IfcColumn": "wall", "IfcBeam": "wall",
    "IfcCurtainWall": "glass", "IfcRailing": "wall", "IfcStair": "wall", "IfcStairFlight": "wall",
    "IfcSlab": "slab", "IfcCovering": "slab", "IfcRoof": "ceiling",
    "IfcDoor": "door", "IfcWindow": "glass",
    "IfcFurnishingElement": "furniture", "IfcFlowTerminal": "furniture",
}
MATERIALS = dict(ROOM_MATERIALS, furniture=Material((0.62, 0.55, 0.48), 0.0, 0.7))


@dataclass
class RoomCut:
    name: str              # e.g. "A102 Living Room"
    size: tuple            # inside width (x), depth (y), height (z), metres
    parts: list            # glb_bytes parts: (vertices y-up metres, triangles, None, material key)
    openings: list         # (kind "door"/"window", side "front"/"back"/"left"/"right", width m)


def find_space(spaces, room):
    """The one space whose Name or LongName is `room` (case-insensitive); a clear error otherwise."""
    want = room.strip().lower()
    hits = [s for s in spaces if (s.Name or "").lower() == want]
    hits = hits or [s for s in spaces if (s.LongName or "").lower() == want]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(sorted(f"{s.Name} ({s.LongName})" for s in (hits or spaces)))
    if hits:
        raise ValueError(f'"{room}" matches {len(hits)} rooms, use the number: {names}')
    raise ValueError(f'no room "{room}"; rooms: {names}')


def clip_polygon(poly, axis, value, keep_above):
    """Sutherland-Hodgman: the part of a convex polygon (k, 3) on one side of the plane coord[axis] = value."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = a[axis] - value, b[axis] - value
        if not keep_above:
            da, db = -da, -db
        if da >= 0:
            out.append(a)
        if da * db < 0:                                          # a strict crossing: a vertex on the plane is kept once
            out.append(a + (b - a) * (da / (da - db)))
    return out


def clip_mesh(v, tri, lo, hi):
    """Triangles of (v, tri) clipped to the box [lo, hi]; returns new (vertices, triangles)."""
    v, tri = np.asarray(v, float), np.asarray(tri, int).reshape(-1, 3)
    p = v[tri]                                                   # (m, 3 corners, 3)
    inside = ((p >= lo) & (p <= hi)).all(axis=(1, 2))
    outside = ((p < lo).all(axis=1) | (p > hi).all(axis=1)).any(axis=1)
    verts, faces = [p[inside].reshape(-1, 3)], [np.arange(3 * inside.sum()).reshape(-1, 3)]
    n = 3 * int(inside.sum())
    for t in p[~inside & ~outside]:
        poly = list(t)
        for axis in range(3):
            poly = clip_polygon(poly, axis, lo[axis], True)
            if len(poly) < 3:
                break
            poly = clip_polygon(poly, axis, hi[axis], False)
            if len(poly) < 3:
                break
        if len(poly) < 3:
            continue
        verts.append(np.array(poly))
        faces.append(np.array([[n, n + i, n + i + 1] for i in range(1, len(poly) - 1)]))
        n += len(poly)
    return np.vstack(verts) if n else np.zeros((0, 3)), np.vstack(faces) if n else np.zeros((0, 3), int)


def side_of(centre, size):
    """Which inside face a point (room frame: inside is [0, size]) is nearest to: front (y=0), back, left (x=0), right."""
    d = {"front": centre[1], "back": size[1] - centre[1], "left": centre[0], "right": size[0] - centre[0]}
    return min(d, key=d.get)


def cut_room(meshes, box, name, margin=0.02):
    """meshes: [(ifc type, vertices world metres z up, triangles)]; box: (lo, hi) of the room's inside.

    Pure geometry (no ifcopenshell), so it can be tested with plain boxes.
    """
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    size = hi - lo
    mid_z = (lo[2] + hi[2]) / 2
    parts, openings = [], []
    for kind_ifc, v, tri in meshes:
        key = KINDS.get(kind_ifc)
        if key is None:
            continue
        cv, ct = clip_mesh(v, tri, lo - margin, hi + margin)
        if not len(ct):
            continue
        if key == "slab":
            key = "floor" if np.asarray(v)[:, 2].mean() < mid_z else "ceiling"
        cv = cv - lo                                             # the room's inside starts at the origin
        ext = np.sort(np.ptp(cv, axis=0))
        if key in ("door", "glass") and kind_ifc != "IfcCurtainWall" and ext[1] > 0.3:
            c = (cv.min(0) + cv.max(0)) / 2
            side = side_of(c, size)
            width = np.ptp(cv[:, 0] if side in ("front", "back") else cv[:, 1])
            openings.append(("door" if key == "door" else "window", side, round(float(width), 2)))
        parts.append((cv @ Z_TO_Y.T, ct, None, key))
    return RoomCut(name, tuple(round(float(s), 3) for s in size), parts, openings)


def load_room(path, room, margin=0.02):
    """Read an IFC file and cut one room out of it (needs ifcopenshell)."""
    import ifcopenshell
    import ifcopenshell.geom

    f = ifcopenshell.open(str(path))
    # ifcopenshell's geometry comes out in metres whatever the file's own length unit
    space = find_space(f.by_type("IfcSpace"), room)
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)

    def mesh(e):
        g = ifcopenshell.geom.create_shape(s, e).geometry
        return np.array(g.verts, float).reshape(-1, 3), np.array(g.faces, int).reshape(-1, 3)

    sv, _ = mesh(space)
    box = sv.min(0), sv.max(0)
    meshes = []
    for e in f.by_type("IfcProduct"):
        if e.is_a() not in KINDS or not e.Representation:
            continue
        try:
            v, t = mesh(e)
        except RuntimeError:
            continue                                             # no body geometry (e.g. an axis-only element)
        if len(t) and (v.min(0) <= box[1] + margin).all() and (v.max(0) >= box[0] - margin).all():
            meshes.append((e.is_a(), v, t))
    return cut_room(meshes, box, f"{space.Name} {space.LongName or ''}".strip(), margin)


def write_glb(path, cut):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(cut.parts, MATERIALS))
    return path
