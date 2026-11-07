"""One room cut out of an IFC model (rooftop/ifc_room.py, scripts/ifc_room_to_glb.py): unit, smoke, integration."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rooftop import ifc_room
from rooftop.ar_export import box_from_quad

ROOT = Path(__file__).resolve().parents[1]
DUPLEX = ROOT / "data/duplex/Duplex_A_20110907.ifc"
SCRIPT = ROOT / "scripts/ifc_room_to_glb.py"


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return box_from_quad(np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float), z1 - z0)


def tri_area(v, t):
    p = np.asarray(v)[np.asarray(t)]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()


@pytest.fixture
def duplex_available():
    if not DUPLEX.is_file():
        pytest.skip(f"{DUPLEX} not downloaded")
    pytest.importorskip("ifcopenshell")


# ---------- unit ----------

def sp(name, long):
    return SimpleNamespace(Name=name, LongName=long)


def test_find_space_by_number_then_name():
    spaces = [sp("A102", "Living Room"), sp("B102", "Living Room"), sp("A103", "Kitchen")]
    assert ifc_room.find_space(spaces, "a102") is spaces[0]
    assert ifc_room.find_space(spaces, " Kitchen ") is spaces[2]
    with pytest.raises(ValueError, match="matches 2 rooms, use the number: A102 .*B102"):
        ifc_room.find_space(spaces, "living room")
    with pytest.raises(ValueError, match='no room "Garage"; rooms: A102 .*A103 .*B102'):
        ifc_room.find_space(spaces, "Garage")


def test_clip_polygon_keeps_one_side():
    sq = [np.array(p, float) for p in ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))]
    above = np.array(ifc_room.clip_polygon(sq, 0, 0.5, True))
    below = np.array(ifc_room.clip_polygon(sq, 0, 0.5, False))
    assert above[:, 0].min() == pytest.approx(0.5) and above[:, 0].max() == 2 and len(above) == 4
    assert below[:, 0].max() == pytest.approx(0.5) and below[:, 0].min() == 0 and len(below) == 4
    assert ifc_room.clip_polygon(sq, 1, 3, True) == []
    on = np.array(ifc_room.clip_polygon(sq, 0, 0, True))     # an edge lying on the plane is kept
    assert len(on) == 4 and on[:, 0].min() == 0


def test_clip_polygon_keeps_a_vertex_on_the_plane_once():
    t = [np.array(p, float) for p in ((1, 0, 0), (0, 0, 0), (2, 1, 0))]
    out = np.array(ifc_room.clip_polygon(t, 0, 1, True))
    assert len(out) == 3 and len(np.unique(out.round(9), axis=0)) == 3
    assert out[:, 0].min() == pytest.approx(1)


def test_clip_mesh_cuts_a_box_to_the_room():
    v, t = box((-1, -1, -1), (3, 3, 3))                      # 4 m cube, faces outside the 2 m room
    cv, ct = ifc_room.clip_mesh(v, t, np.zeros(3), np.full(3, 2.0))
    assert len(ct) == 0                                      # every face lies outside: nothing kept
    v, t = box((-1, 0.5, 0.5), (3, 1.5, 1.5))                # a 1 m beam right through the room
    cv, ct = ifc_room.clip_mesh(v, t, np.zeros(3), np.full(3, 2.0))
    assert cv.min(0) == pytest.approx([0, 0.5, 0.5]) and cv.max(0) == pytest.approx([2, 1.5, 1.5])
    assert tri_area(cv, ct) == pytest.approx(4 * 2 * 1)      # four 2 x 1 sides; end caps are outside
    v, t = box((0.5, 0.5, 0.5), (1, 1, 1))                   # wholly inside: unchanged
    cv, ct = ifc_room.clip_mesh(v, t, np.zeros(3), np.full(3, 2.0))
    assert len(ct) == 12 and tri_area(cv, ct) == pytest.approx(6 * 0.25)


@pytest.mark.parametrize("c, side", [((1, 0.1, 1), "front"), ((1, 2.9, 1), "back"), ((0.1, 1.5, 1), "left"),
                                     ((3.9, 1.5, 1), "right")])
def test_side_of(c, side):
    assert ifc_room.side_of(np.array(c, float), np.array([4, 3, 2.5])) == side


def test_cut_room_keeps_skins_shifts_and_names_parts():
    lo, hi = (10, 20, 0), (14, 23, 2.5)                       # inside 4 x 3 x 2.5 m
    meshes = [
        ("IfcWallStandardCase", *box((9.8, 19.8, 0), (10, 23.2, 2.5))),     # left wall, 0.2 thick
        ("IfcSlab", *box((9, 19, -0.2), (15, 24, 0))),                       # floor
        ("IfcCovering", *box((9, 19, 2.5), (15, 24, 2.55))),                 # ceiling
        ("IfcWindow", *box((11, 19.8, 1), (12.5, 20, 2))),                   # 1.5 m window in the front wall
        ("IfcDoor", *box((13, 23, 0), (13.9, 23.2, 2.1))),                   # 0.9 m door in the back wall
        ("IfcFurnishingElement", *box((11, 21, 0), (12, 22, 0.8))),
        ("IfcSpace", *box(lo, hi)),                                           # not a kind: skipped
        ("IfcWall", *box((30, 30, 0), (31, 31, 2))),                         # far away: clipped to nothing
    ]
    cut = ifc_room.cut_room(meshes, (lo, hi), "R1 Test", margin=0.02)
    assert cut.size == (4.0, 3.0, 2.5) and cut.name == "R1 Test"
    keys = [p[3] for p in cut.parts]
    assert keys == ["wall", "floor", "ceiling", "glass", "door", "furniture"]
    allv = np.vstack([p[0] for p in cut.parts])
    # y up, inside starts at 0: x in [-0.02, 4.02], z = -(cad y) in [-3.02, 0.02] (the walls' 2 cm skin); the
    # floor and ceiling slabs' sides lie outside the box, so only their faces at 0 and 2.5 m are kept
    assert allv.min(0) == pytest.approx([-0.02, 0, -3.02]) and allv.max(0) == pytest.approx([4.02, 2.5, 0.02])
    assert cut.openings == [("window", "front", 1.5), ("door", "back", 0.9)]


def test_cut_room_ignores_thin_slivers_of_openings_next_door():
    lo, hi = (0, 0, 0), (4, 3, 2.5)
    window_next_door = box((-0.3, 3.0, 1), (0, 4.5, 2))     # only a 2 cm corner of it reaches the margin
    cut = ifc_room.cut_room([("IfcWindow", *window_next_door)], (lo, hi), "R", margin=0.02)
    assert cut.openings == [] and len(cut.parts) == 1


def test_cut_room_measures_side_windows_along_the_wall_and_skips_curtain_walls():
    lo, hi = (0, 0, 0), (4, 3, 2.5)
    meshes = [("IfcWindow", *box((-0.2, 0.5, 1), (0, 1.7, 2))),           # 1.2 m window in the left wall
              ("IfcCurtainWall", *box((1, 2.9, 0), (3, 3.1, 2.5)))]         # glass, but not counted as a window
    cut = ifc_room.cut_room(meshes, (lo, hi), "R", margin=0.02)
    assert cut.openings == [("window", "left", 1.2)] and [p[3] for p in cut.parts] == ["glass", "glass"]


# ---------- smoke ----------

def test_script_rejects_a_missing_file(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "none.ifc"), "--room", "A102"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "no file" in r.stderr and "Traceback" not in r.stderr


def test_script_cuts_the_living_room(tmp_path, duplex_available):
    r = subprocess.run([sys.executable, str(SCRIPT), str(DUPLEX), "--room", "A102", "--out", str(tmp_path),
                        "--name", "lr"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "A102 Living Room: inside 5.78 x 4.78 m, 2.58 m high" in r.stdout
    assert "window 4.83 m wide on the front wall (the model's +Z side)" in r.stdout
    assert (tmp_path / "lr.glb").stat().st_size > 10_000


def test_script_explains_an_ambiguous_room(tmp_path, duplex_available):
    r = subprocess.run([sys.executable, str(SCRIPT), str(DUPLEX), "--room", "Living Room", "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "use the number" in r.stderr and "A102" in r.stderr and "Traceback" not in r.stderr


# ---------- integration ----------

def test_load_room_reads_the_duplex_living_room(duplex_available):
    cut = ifc_room.load_room(DUPLEX, "A102")
    assert cut.size == pytest.approx((5.783, 4.783, 2.581), abs=0.002)
    assert {"wall", "floor", "ceiling", "glass", "door", "furniture"} <= {p[3] for p in cut.parts}
    assert ("door", "front") in [(k, s) for k, s, _ in cut.openings]
    v = np.vstack([p[0] for p in cut.parts])
    assert v.min(0) == pytest.approx([-0.02, -0.02, -4.803], abs=0.003)
    assert v.max(0) == pytest.approx([5.803, 2.601, 0.02], abs=0.003)


def test_published_duplex_room_has_the_room_size():
    from tests.test_room import read_glb
    doc, lo, hi = read_glb(ROOT / "preview/models/duplex_living_room.glb")
    assert (hi - lo) == pytest.approx([5.823, 2.621, 4.823], abs=0.003)
    assert "CC BY 4.0" in (ROOT / "preview/models/CREDITS.md").read_text(encoding="utf8")
