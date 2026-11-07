"""A simple room as CAD: thin walls, floor and ceiling around an interior box, with a door and a window.

Used as the sample "room model" for the tabletop AR page (preview/tagar.html, Room model card): the page
lines the model's wall up with the AprilTags on the real wall, so the person can stand inside it.

Coordinates (mm, z up): the interior is x in [0, width], y in [0, depth], z in [0, height]. Walls, floor
and ceiling are `skin` thick and grow outward, so the inside faces are exactly the room's measurements.
Walls by side: front (y = 0), back (y = depth), left (x = 0), right (x = width). In the GLB (y up,
metres) the back wall becomes the model's -Z side, front +Z, left -X, right +X, matching model.js ROOM_WALLS.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ar_export import Material, Z_TO_Y, box_from_quad, glb_bytes

SIDES = ("front", "back", "left", "right")


@dataclass(frozen=True)
class Opening:
    side: str          # one of SIDES
    offset: float      # mm from the wall's start (x = 0 for front/back, y = 0 for left/right) to the opening's edge
    width: float
    sill: float        # mm above the floor (0 for a door)
    height: float
    kind: str = "door"  # "door" or "window": what fills it


@dataclass(frozen=True)
class RoomSpec:
    width: float = 4000.0
    depth: float = 3500.0
    height: float = 2800.0
    skin: float = 20.0
    openings: tuple = field(default_factory=lambda: (
        Opening("front", 600, 900, 0, 2100, "door"),
        Opening("left", 1000, 1500, 900, 1200, "window"),
    ))

    def wall_length(self, side):
        return self.width if side in ("front", "back") else self.depth


def check(spec):
    """Problems with a spec, as sentences (empty list = fine)."""
    bad = []
    for name in ("width", "depth", "height", "skin"):
        if not getattr(spec, name) > 0:
            bad.append(f"{name} must be positive")
    if bad:
        return bad
    for o in spec.openings:
        if o.side not in SIDES:
            bad.append(f"opening on unknown side {o.side!r}")
            continue
        if o.kind not in ("door", "window"):
            bad.append(f"opening kind {o.kind!r} is not door or window")
        if not (o.width > 0 and o.height > 0 and o.offset >= 0 and o.sill >= 0):
            bad.append(f"{o.kind} on the {o.side} wall needs a positive size and a non-negative offset and sill")
        elif o.offset + o.width > spec.wall_length(o.side) or o.sill + o.height > spec.height:
            bad.append(f"{o.kind} on the {o.side} wall does not fit inside the wall")
    for side in SIDES:
        spans = sorted((o.offset, o.offset + o.width) for o in spec.openings if o.side == side)
        if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
            bad.append(f"openings overlap on the {side} wall")
    return bad


def wall_pieces(u0, u1, z0, z1, holes):
    """Rectangles (u, z, du, dz) covering [u0, u1] x [z0, z1] minus the hole rectangles (u, z, du, dz).

    Holes must not overlap along u. Each hole leaves a strip below it (sill) and above it (lintel);
    the full-height parts between holes are single pieces.
    """
    out, u = [], u0
    for hu, hz, hdu, hdz in sorted(holes):
        if hu > u:
            out.append((u, z0, hu - u, z1 - z0))
        if hz > z0:
            out.append((hu, z0, hdu, hz - z0))
        if hz + hdz < z1:
            out.append((hu, hz + hdz, hdu, z1 - hz - hdz))
        u = hu + hdu
    if u < u1:
        out.append((u, z0, u1 - u, z1 - z0))
    return out


def boxes(spec):
    """Every solid as (origin xyz mm, size xyz mm, material key): walls, floor, ceiling, door and window fills."""
    bad = check(spec)
    if bad:
        raise ValueError("; ".join(bad))
    W, D, H, t = spec.width, spec.depth, spec.height, spec.skin
    out = [((-t, -t, -t), (W + 2 * t, D + 2 * t, t), "floor"),
           ((-t, -t, H), (W + 2 * t, D + 2 * t, t), "ceiling")]
    for side in SIDES:
        ops = [o for o in spec.openings if o.side == side]
        # front/back run along x and cover the corners; left/right run along y between them
        u0, u1 = (-t, W + t) if side in ("front", "back") else (0.0, D)
        for u, z, du, dz in wall_pieces(u0, u1, 0.0, H, [(o.offset, o.sill, o.width, o.height) for o in ops]):
            out.append(_wall_box(spec, side, u, z, du, dz) + ("wall",))
        for o in ops:
            out.append(_wall_box(spec, side, o.offset, o.sill, o.width, o.height) + ("glass" if o.kind == "window" else "door",))
    return out


def _wall_box(spec, side, u, z, du, dz):
    t = spec.skin
    if side == "front":
        return (u, -t, z), (du, t, dz)
    if side == "back":
        return (u, spec.depth, z), (du, t, dz)
    if side == "left":
        return (-t, u, z), (t, du, dz)
    return (spec.width, u, z), (t, du, dz)


MATERIALS = {
    "wall": Material((0.86, 0.84, 0.80), 0.0, 0.9),
    "floor": Material((0.55, 0.45, 0.35), 0.0, 0.8),
    "ceiling": Material((0.93, 0.93, 0.93), 0.0, 0.9),
    "door": Material((0.40, 0.26, 0.15), 0.0, 0.7),
    "glass": Material((0.45, 0.70, 0.90), 0.0, 0.1),
}


def glb_parts(spec):
    """Parts for glb_bytes: y up, metres."""
    parts = []
    for (x, y, z), (dx, dy, dz), key in boxes(spec):
        x, y, z, dx, dy, dz = (v / 1000 for v in (x, y, z, dx, dy, dz))
        quad = np.array([[x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]])
        v, tri = box_from_quad(quad, dz)
        parts.append((np.asarray(v) @ Z_TO_Y.T, tri, None, key))
    return parts


def write_glb(path, spec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(glb_parts(spec), MATERIALS))
    return path


def write_step(path, spec):
    import cadquery as cq  # heavy: only when writing STEP

    solids = [cq.Solid.makeBox(dx, dy, dz, pnt=cq.Vector(x, y, z)) for (x, y, z), (dx, dy, dz), _ in boxes(spec)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(solids)), str(path))
    return path
