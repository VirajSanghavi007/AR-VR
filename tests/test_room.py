"""Sample room CAD (rooftop/room.py, scripts/make_room_cad.py): unit, smoke and integration tests."""
import json
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rooftop import room
from rooftop.room import Opening, RoomSpec

ROOT = Path(__file__).resolve().parents[1]


def area(pieces):
    return sum(du * dz for _, _, du, dz in pieces)


# ---------- unit ----------

def test_wall_pieces_no_holes_is_the_whole_wall():
    assert room.wall_pieces(0, 4, 0, 3, []) == [(0, 0, 4, 3)]


def test_wall_pieces_door_leaves_sides_and_lintel_only():
    p = room.wall_pieces(0, 4, 0, 3, [(1, 0, 1, 2)])
    assert sorted(p) == [(0, 0, 1, 3), (1, 2, 1, 1), (2, 0, 2, 3)]


def test_wall_pieces_window_has_sill_and_lintel():
    p = room.wall_pieces(0, 4, 0, 3, [(1, 1, 1, 1)])
    assert (1, 0, 1, 1) in p and (1, 2, 1, 1) in p
    assert area(p) == pytest.approx(12 - 1)


def test_wall_pieces_two_holes_unsorted_and_at_the_ends():
    p = room.wall_pieces(0, 6, 0, 3, [(5, 0, 1, 3), (0, 1, 1, 1)])
    assert area(p) == pytest.approx(18 - 3 - 1)
    assert all(du > 0 and dz > 0 for _, _, du, dz in p)


def test_wall_pieces_keeps_thin_strips_at_both_ends():
    assert room.wall_pieces(0, 4, 0, 3, [(0.5, 0, 3, 3)]) == [(0, 0, 0.5, 3), (3.5, 0, 0.5, 3)]


def test_check_default_spec_is_fine():
    assert room.check(RoomSpec()) == []


@pytest.mark.parametrize("change, words", [
    (dict(width=0), "width must be positive"),
    (dict(skin=-1), "skin must be positive"),
    (dict(openings=(Opening("roof", 0, 1, 0, 1),)), "unknown side"),
    (dict(openings=(Opening("front", 0, 1, 0, 1, "hatch"),)), "not door or window"),
    (dict(openings=(Opening("front", 3500, 900, 0, 2000),)), "does not fit"),
    (dict(openings=(Opening("left", 0, 900, 0, 2000),)), None),        # depth 3500: fits
    (dict(openings=(Opening("left", 3000, 900, 0, 2000),)), "does not fit"),
    (dict(openings=(Opening("front", 0, 900, 2000, 900),)), "does not fit"),
    (dict(openings=(Opening("front", 0, 0, 0, 900),)), "positive size"),
    (dict(openings=(Opening("front", 0, 900, 0, 2000), Opening("front", 800, 900, 0, 2000))), "overlap"),
    (dict(openings=(Opening("front", 0, 900, 0, 2000), Opening("front", 900, 900, 0, 2000))), None),
])
def test_check_reports_bad_specs(change, words):
    bad = room.check(replace(RoomSpec(), **change))
    if words is None:
        assert bad == []
    else:
        assert any(words in b for b in bad), bad


def test_boxes_rejects_a_bad_spec():
    with pytest.raises(ValueError, match="width"):
        room.boxes(replace(RoomSpec(), width=0))


def box_bounds(bs, keys=None):
    lo = np.array([o for o, s, k in bs if keys is None or k in keys], float)
    hi = np.array([np.add(o, s) for o, s, k in bs if keys is None or k in keys], float)
    return lo.min(0), hi.max(0)


def test_boxes_inside_faces_are_the_room_measurements():
    s = RoomSpec(width=4000, depth=3500, height=2800, skin=20)
    bs = room.boxes(s)
    lo, hi = box_bounds(bs)
    assert lo.tolist() == [-20, -20, -20] and hi.tolist() == [4020, 3520, 2820]
    # every wall piece sits outside the interior; floor top at 0, ceiling bottom at height
    for (x, y, z), (dx, dy, dz), k in bs:
        inside = (x < 4000 and x + dx > 0) and (y < 3500 and y + dy > 0) and (z < 2800 and z + dz > 0)
        assert not inside, (k, x, y, z, dx, dy, dz)
    floor = [b for b in bs if b[2] == "floor"][0]
    ceiling = [b for b in bs if b[2] == "ceiling"][0]
    assert floor[0][2] + floor[1][2] == 0 and ceiling[0][2] == 2800


def test_boxes_wall_area_matches_openings():
    s = RoomSpec()
    bs = room.boxes(s)
    wall = sum(sz[0] * sz[2] if sz[1] == s.skin else sz[1] * sz[2] for _, sz, k in bs if k == "wall")
    full = 2 * (s.width + 2 * s.skin) * s.height + 2 * s.depth * s.height
    holes = sum(o.width * o.height for o in s.openings)
    assert wall == pytest.approx(full - holes)
    kinds = [k for *_, k in bs]
    assert kinds.count("door") == 1 and kinds.count("glass") == 1


def test_door_on_front_window_on_left():
    s = RoomSpec()
    bs = room.boxes(s)
    (dx, dy, dz), _, _ = [b for b in bs if b[2] == "door"][0]
    assert dy == -s.skin and dx == 600 and dz == 0            # front wall: y in [-skin, 0], 600 mm from x = 0
    (wx, wy, wz), (wdx, wdy, _), _ = [b for b in bs if b[2] == "glass"][0]
    assert wx == -s.skin and wy == 1000 and wz == 900 and wdx == s.skin and wdy == 1500


def test_glb_parts_are_y_up_metres():
    parts = room.glb_parts(RoomSpec())
    v = np.vstack([p[0] for p in parts])
    assert v.min(0) == pytest.approx([-0.02, -0.02, -3.52])
    assert v.max(0) == pytest.approx([4.02, 2.82, 0.02])


# ---------- glb reader for the integration tests ----------

def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A
    doc = json.loads(data[20:20 + jlen])
    lo = np.min([a["min"] for a in doc["accessors"] if "min" in a], 0)
    hi = np.max([a["max"] for a in doc["accessors"] if "max" in a], 0)
    return doc, lo, hi


# ---------- smoke ----------

def test_script_builds_glb_only(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_room_cad.py"), "--out", str(tmp_path), "--no-step",
                        "--width", "3000"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "3.00 x 3.50 m" in r.stdout
    _, lo, hi = read_glb(tmp_path / "sample_room.glb")
    assert hi[0] - lo[0] == pytest.approx(3.04, abs=1e-4)
    assert not (tmp_path / "sample_room.step").exists()


def test_script_rejects_a_room_too_small_for_the_door(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_room_cad.py"), "--out", str(tmp_path), "--no-step",
                        "--width", "1000"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "does not fit" in r.stderr     # a clean usage error, not a crash
    assert "Traceback" not in r.stderr and not (tmp_path / "sample_room.glb").exists()


# ---------- integration ----------

def test_published_sample_room_matches_the_spec():
    doc, lo, hi = read_glb(ROOT / "preview/models/sample_room.glb")
    assert (hi - lo) == pytest.approx([4.04, 2.84, 3.54], abs=1e-4)
    assert hi[2] == pytest.approx(0.02, abs=1e-4)         # back wall (y = depth in CAD) is the -Z side
    assert len(doc["materials"]) == 5


def test_step_reads_back_with_the_same_size(tmp_path):
    cq = pytest.importorskip("cadquery")
    p = room.write_step(tmp_path / "r.step", RoomSpec())
    bb = cq.importers.importStep(str(p)).val().BoundingBox()
    assert (bb.xmin, bb.ymin, bb.zmin) == pytest.approx((-20, -20, -20), abs=1e-3)
    assert (bb.xmax, bb.ymax, bb.zmax) == pytest.approx((4020, 3520, 2820), abs=1e-3)
