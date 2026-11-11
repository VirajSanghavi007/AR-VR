"""Laser-scanned room -> SketchUp exports (rooftop/scan_room.py, scripts/scan_room_to_sketchup.py).
Unit (frame, faces, parts, writer), smoke (the CLI on a synthetic two-scan box room), integration (the real
ETH3D office scan through the whole chain, the published .dae, and tagar.html's room card in Chrome)."""
import functools
import http.server
import json
import shutil
import sys
import threading
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scan_room_to_sketchup  # noqa: E402
from rooftop import scan_room as R  # noqa: E402
from rooftop import sketchup_sample as S  # noqa: E402

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
PREVIEW = ROOT / "preview"
OFFICE = ROOT / "data/eth3d/office/dslr_scan_eval"
PUBLISHED = PREVIEW / "models" / "office_scan_room.dae"
OFFICE_SIZE = (3.14, 4.48, 4.05)          # x, y, height in metres, as measured by this chain on the full scan
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)


def box_room(L=3.0, W=4.0, H=2.5, step=0.05, desk=True):
    """Points + inward normals on the 6 faces of an L x W x H box (corner at the origin), and a desk top."""
    g = lambda a, b: (np.arange(step / 2, a, step), np.arange(step / 2, b, step))
    pts, nrm = [], []

    def face(u, v, fix, axis, value, normal):
        uu, vv = np.meshgrid(u, v, indexing="ij")
        p = np.zeros((uu.size, 3))
        others = [i for i in range(3) if i != axis]
        p[:, others[0]], p[:, others[1]], p[:, axis] = uu.ravel(), vv.ravel(), value
        pts.append(p)
        nrm.append(np.tile(normal, (len(p), 1)))
    face(*g(L, W), None, 2, 0.0, [0, 0, 1])
    face(*g(L, W), None, 2, H, [0, 0, -1])
    face(*g(W, H), None, 0, 0.0, [1, 0, 0])
    face(*g(W, H), None, 0, L, [-1, 0, 0])
    face(*g(L, H), None, 1, 0.0, [0, 1, 0])
    face(*g(L, H), None, 1, W, [0, -1, 0])
    if desk:
        u, v = np.arange(0.8, 1.6, step), np.arange(1.0, 1.8, step)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        pts.append(np.c_[uu.ravel(), vv.ravel(), np.full(uu.size, 0.75)])
        nrm.append(np.tile([0, 0, 1.0], (uu.size, 1)))
    return np.vstack(pts), np.vstack(nrm).astype(float)


def tilt_yaw(tilt_deg, yaw_deg):
    a, b = np.radians(tilt_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return R.rot_z(b) @ rx


# ---------- unit: frame ----------

@pytest.mark.parametrize("n", [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.1, -0.2, 0.97], [0.3, 0.4, -0.5]])
def test_rotation_to_z_is_a_proper_rotation_onto_up(n):
    m = R.rotation_to_z(n)
    u = np.asarray(n, float) / np.linalg.norm(n)
    assert m @ u == pytest.approx([0, 0, 1], abs=1e-9)
    assert m @ m.T == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(m) == pytest.approx(1)


def test_rot_z_turns_counter_clockwise():
    assert R.rot_z(np.pi / 2) @ [1, 0, 0] == pytest.approx([0, 1, 0])


@pytest.mark.parametrize("deg", [0, 10, -30, 44])
def test_wall_yaw_finds_the_angle_of_square_walls(deg):
    a = np.radians([deg, deg + 90, deg + 180, deg + 270])
    n = np.c_[np.cos(a), np.sin(a), np.zeros(4)]
    floor = np.tile([0, 0, 1.0], (50, 1))                            # many horizontal surfaces: ignored
    assert np.degrees(R.wall_yaw(np.vstack([n, floor]))) == pytest.approx(deg, abs=1e-6)


def test_wall_yaw_wraps_into_plus_minus_45_and_weights_by_horizontal_length():
    a = np.radians(50)
    assert np.degrees(R.wall_yaw([[np.cos(a), np.sin(a), 0]])) == pytest.approx(-40)
    a1, a2 = np.radians(0), np.radians(20)
    n = [[np.cos(a1), np.sin(a1), 0], [np.cos(a2), np.sin(a2), 0],
         [0.1 * np.cos(a2), 0.1 * np.sin(a2), 0.15]]                 # tilted a little, short: counts, weakly
    w1, w2 = 1.0, 1.0 + 0.1                                          # 4*20 deg = 80 deg; the short one adds 0.1
    want = np.degrees(np.arctan2(w2 * np.sin(np.radians(80)), w1 + w2 * np.cos(np.radians(80)))) / 4
    assert np.degrees(R.wall_yaw(n)) == pytest.approx(want, abs=1e-6)
    with pytest.raises(ValueError):
        R.wall_yaw([[0, 0, 1], [0.3, 0, 0.95]])
    a = np.radians(45)                                                # the edge: half-open range [-45, 45)
    assert np.degrees(R.wall_yaw([[np.cos(a), np.sin(a), 0]])) == pytest.approx(-45)


def test_floor_ceiling_ignores_a_desk_and_stray_hits():
    rng = np.random.default_rng(0)
    floor = rng.normal(0.0, 0.005, 400)
    desk = np.full(400, 0.75)
    ceil = rng.normal(2.6, 0.005, 400)
    z = np.r_[floor, desk, ceil, [-0.5], [3.5]]
    nz = np.r_[np.ones(800), -np.ones(400), [1.0], [-1.0]]
    f, c = R.floor_ceiling(z, nz)
    assert f == pytest.approx(0.0, abs=0.01) and c == pytest.approx(2.6, abs=0.01)
    # a cabinet top just under the ceiling faces up, a desk underside faces down: neither is the ceiling
    z2 = np.r_[z, np.full(600, 2.5), np.full(2000, 0.72)]
    nz2 = np.r_[nz, np.ones(600), -np.ones(2000)]
    assert R.floor_ceiling(z2, nz2)[1] == pytest.approx(2.6, abs=0.01)
    walls = np.linspace(0, 2.6, 20000)                                # sideways normals: walls, not the ceiling
    assert R.floor_ceiling(np.r_[z, walls], np.r_[nz, np.zeros(20000)])[1] == pytest.approx(2.6, abs=0.01)


def test_floor_ceiling_needs_both_and_ceiling_above():
    with pytest.raises(ValueError):
        R.floor_ceiling([0, 1], [1, 1])
    with pytest.raises(ValueError):
        R.floor_ceiling([2.0, 0.0], [1, -1])


def test_plan_box_drops_the_few_points_seen_through_a_window():
    rng = np.random.default_rng(1)
    xy = np.c_[rng.uniform(0, 3, 5000), rng.uniform(0, 4, 5000)]
    xy = np.vstack([xy, [[9.0, 2.0], [-6.0, 1.0]]])
    x0, y0, x1, y1 = R.plan_box(xy)
    assert (x0, y0) == pytest.approx((0, 0), abs=0.05) and (x1, y1) == pytest.approx((3, 4), abs=0.05)
    assert R.plan_box(xy, q=0)[2] == pytest.approx(9.0)


@pytest.mark.parametrize("tilt,yaw", [(0, 0), (2, 17), (-3, -35)])
def test_room_frame_levels_squares_and_puts_the_corner_at_the_origin(tilt, yaw):
    p, n = box_room()
    m = tilt_yaw(tilt, yaw)
    off = np.array([5.0, -2.0, -1.4])
    door = np.c_[np.random.default_rng(2).uniform(3.0, 5.0, (3000, 1)), np.full((3000, 1), 2.0), np.zeros((3000, 1))]
    p, n = np.vstack([p, door]), np.vstack([n, np.tile([0, 0, 1.0], (3000, 1))])  # floor seen through a door
    rot, t, size = R.room_frame(p @ m.T + off, n @ m.T)
    back = (p[:-3000] @ m.T + off) @ rot.T + t
    assert size == pytest.approx((3.0, 4.0, 2.5), abs=0.06) or size == pytest.approx((4.0, 3.0, 2.5), abs=0.06)
    assert back.min(0) == pytest.approx([0, 0, 0], abs=0.06)
    assert back.max(0) == pytest.approx(size, abs=0.06)
    assert rot @ m @ [0, 0, 1] == pytest.approx([0, 0, 1], abs=1e-6)       # levelled


def test_crop_box_keeps_the_room_and_a_margin():
    p = np.array([[0, 0, 0], [-0.04, 1, 1], [-0.06, 1, 1], [3.0, 4.0, 2.5], [3.06, 1, 1], [1, 1, 2.54]])
    assert R.crop_box(p, (3, 4, 2.5)).tolist() == [True, True, False, True, False, True]
    assert R.crop_box(p, (3, 4, 2.5), margin=0.1).all()


# ---------- unit: faces and parts ----------

def test_classify_faces_by_normal_and_place():
    size = (3.0, 4.0, 2.5)
    c = np.array([[1, 1, 0.02], [1, 1, 0.75], [1, 1, 2.45], [0.1, 2, 1], [1.5, 3.85, 1], [1.5, 2, 1], [1, 1, 0.2]])
    n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 0, -1]], float)
    assert R.classify_faces(c, n, size).tolist() == ["Floor", "Furniture", "Ceiling", "Wall", "Wall", "Furniture", "Furniture"]
    # a near-vertical face 0.25 m off the wall: furniture by default, wall with a wider band
    c2, n2 = np.array([[0.25, 2, 1]]), np.array([[1.0, 0, 0]])
    assert R.classify_faces(c2, n2, size)[0] == "Furniture" and R.classify_faces(c2, n2, size, wall_band=0.3)[0] == "Wall"
    # a floor-facing face just above the tolerance is not floor; a wall leaning 30 deg still counts
    assert R.classify_faces([[1, 1, 0.13]], [[0, 0, 1.0]], size)[0] == "Furniture"
    assert R.classify_faces([[0.05, 2, 1]], [[0.92, 0, 0.39]], size)[0] == "Wall"
    assert R.classify_faces([[1.5, 2, 2.4]], [[0, 0, -1.0]], size)[0] == "Ceiling"
    assert R.classify_faces([[1.5, 2, 2.37]], [[0, 0, -1.0]], size)[0] == "Furniture"


def test_mesh_parts_reindexes_each_material_in_palette_order():
    v = np.arange(18, dtype=float).reshape(6, 3)
    f = np.array([[0, 1, 2], [3, 4, 5], [2, 3, 4]])
    parts = R.mesh_parts(v, f, ["Furniture", "Floor", "Furniture"])
    assert [p.material for p in parts] == ["Floor", "Furniture"]
    assert parts[0].faces.tolist() == [[0, 1, 2]] and parts[0].vertices.tolist() == v[3:6].tolist()
    assert parts[1].vertices.tolist() == v[[0, 1, 2, 3, 4]].tolist()
    assert parts[1].faces.tolist() == [[0, 1, 2], [2, 3, 4]]


def test_plane_parts_cover_the_room_and_face_inside():
    fl, ce = R.plane_parts((3.0, 4.0, 2.5))
    assert (fl.material, ce.material) == ("Floor", "Ceiling")
    for p, z, nz in ((fl, 0.0, 1), (ce, 2.5, -1)):
        n = S.face_normals(p.vertices, p.faces)
        assert np.allclose(n, [0, 0, nz]) and np.allclose(p.vertices[:, 2], z)
        t = p.vertices[p.faces]
        assert np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2 == pytest.approx(12.0)
        assert p.vertices.max(0)[:2].tolist() == [3.0, 4.0]


def test_room_parts_swaps_scanned_floor_and_ceiling_for_planes():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], float)
    f = np.array([[0, 1, 2], [0, 2, 4], [1, 2, 3]])
    lab = ["Floor", "Wall", "Ceiling"]
    parts = R.room_parts(v, f, lab, (3, 4, 2.5))
    assert [p.material for p in parts] == ["Floor", "Wall", "Ceiling"]
    assert parts[0].vertices.max(0)[:2].tolist() == [3, 4] and len(parts[2].faces) == 2
    assert parts[1].faces.tolist() == [[0, 1, 2]]
    raw = R.room_parts(v, f, lab, (3, 4, 2.5), planes=False)
    assert [len(p.faces) for p in raw] == [1, 1, 1] and raw[0].vertices.max() == 1
    only_furniture = R.room_parts(v, f[:1], ["Furniture"], (3, 4, 2.5))
    assert [p.material for p in only_furniture] == ["Floor", "Ceiling", "Furniture"]


def test_drop_degenerate_removes_flat_and_repeated_faces():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 0, 2], [2, 1, 0]])
    assert R.drop_degenerate(v, f).tolist() == [[0, 1, 2], [2, 1, 0]]
    tiny = np.array([[0, 0, 0], [1e-6, 0, 0], [0, 1e-6, 0]])
    assert len(R.drop_degenerate(tiny, np.array([[0, 1, 2]]))) == 0
    assert len(R.drop_degenerate(tiny, np.array([[0, 1, 2]]), min_area=1e-13)) == 1
    assert len(R.drop_degenerate(v, np.array([[1, 1, 1], [3, 1, 3]]), min_area=0)) == 0


def test_face_normals_centers():
    n, c = R.face_normals_centers([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.array([[0, 1, 2], [0, 2, 1]]))
    assert n.tolist() == [[0, 0, 1], [0, 0, -1]] and c[0] == pytest.approx([2 / 3, 2 / 3, 0])


# ---------- unit: writers ----------

def test_sketchup_writers_take_a_palette_and_a_tool_name():
    part = S.Part("Wall", np.eye(3), np.array([[0, 1, 2]]))
    dae = S.dae_text([part], materials=R.ROOM_MATERIALS, tool="Scan & co")
    root = ET.fromstring(dae)
    assert root.find("c:asset/c:contributor/c:authoring_tool", NS).text == "Scan & co"
    assert [m.get("name") for m in root.findall(".//c:material", NS)] == ["Wall"]
    assert root.find(".//c:color", NS).text == "0.86 0.85 0.81 1"
    assert S.obj_text([part], "w.mtl", tool="Scan").splitlines()[0] == "# Scan"
    assert S.mtl_text([part], materials=R.ROOM_MATERIALS).startswith("newmtl Wall\nKd 0.86 0.85 0.81\n")
    assert S.mtl_text([part]) == ""                                     # default palette has no Wall
    assert "sketchup_sample.py" in S.dae_text(S.pergola()) and S.obj_text(S.pergola(), "x.mtl").startswith("# SketchUp-style sample")


def test_write_room_writes_sketchup_files_and_a_summary(tmp_path):
    parts = R.plane_parts((3.0, 4.0, 2.5)) + [S.box([1, 1, 0], [2, 2, 0.75], "Furniture")]
    files, info = R.write_room(tmp_path / "o", "room", parts, (3.0, 4.0, 2.5), "src", planes=True)
    assert sorted(f.name for f in files.values()) == ["room.dae", "room.json", "room.mtl", "room.obj"]
    root = ET.fromstring(files["dae"].read_text(encoding="utf-8"))
    assert root.find("c:asset/c:unit", NS).get("name") == "inch" and root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    assert root.find("c:asset/c:contributor/c:authoring_tool", NS).text == R.TOOL
    assert [g.get("name") for g in root.findall(".//c:geometry", NS)] == ["Floor", "Ceiling", "Furniture"]
    pos = np.array(root.find(".//c:float_array", NS).text.split(), float).reshape(-1, 3) * S.INCH
    assert pos.max(0)[:2] == pytest.approx([3.0, 4.0], abs=1e-4)         # inches in the file
    assert files["obj"].read_text().startswith(f"# {R.TOOL}\n# File units = inches\nmtllib room.mtl")
    assert "newmtl Furniture" in files["mtl"].read_text() and "newmtl Wall" not in files["mtl"].read_text()
    saved = json.loads(files["json"].read_text())
    assert saved == info and info["size_m"] == [3.0, 4.0, 2.5] and info["source"] == "src"
    assert info["faces"] == {"Floor": 2, "Ceiling": 2, "Furniture": 12} and info["floor_ceiling"] == "fitted planes"
    assert info["units_in_files"] == "inch" and info["up"] == "z"
    _, info2 = R.write_room(tmp_path / "o", "room2", parts, (3, 4, 2.5))
    assert info2["floor_ceiling"] == "scanned mesh"


# ---------- smoke: the CLI on a synthetic two-scan room ----------

@pytest.fixture(scope="module")
def synthetic_scans(tmp_path_factory):
    """data/<site>/dslr_scan_eval with two PLY scans of a box room seen from two scanner spots, each in its
    own frame (scanner at the origin), and the MeshLab file that puts them back together."""
    o3d = pytest.importorskip("open3d")
    data = tmp_path_factory.mktemp("data")
    d = data / "fake" / "room" / "dslr_scan_eval"
    d.mkdir(parents=True)
    p, _ = box_room(step=0.03)
    p = np.vstack([p, np.c_[np.full(10, 4.5), np.linspace(1, 3, 10), np.linspace(0.5, 2, 10)]])  # seen through a window
    world = tilt_yaw(1.5, 12)
    pw = p @ world.T
    mlp = ['<MeshLabProject><MeshGroup>']
    for i, spot in enumerate(([1.0, 1.5, 1.5], [2.0, 2.5, 1.5]), 1):
        s = world @ np.array(spot)
        m = np.eye(4)
        m[:3, 3] = s
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw - s))
        o3d.io.write_point_cloud(str(d / f"scan{i}.ply"), pc)
        rows = "\n".join(" ".join(f"{x:.9g}" for x in r) for r in m)
        mlp.append(f'<MLMesh label="scan{i}.ply" filename="scan{i}.ply"><MLMatrix44>\n{rows}\n</MLMatrix44></MLMesh>')
    (d / "scan_alignment.mlp").write_text("".join(mlp) + "</MeshGroup></MeshLabProject>")
    return data


def test_cli_turns_a_scanned_box_room_into_sketchup_files(synthetic_scans, tmp_path, capsys):
    files, info = scan_room_to_sketchup.main(["fake/room", "--data", str(synthetic_scans), "--out", str(tmp_path),
                                              "--name", "box", "--voxel", "0.05", "--depth", "6", "--triangles", "4000"])
    out = capsys.readouterr().out
    for k in range(1, 6):
        assert f"[{k}/5]" in out
    assert "elapsed" in out and out.count("wrote ") == 4
    assert all(f.is_file() for f in files.values()) and (tmp_path / "box_cloud.ply").is_file()
    assert sorted(info["size_m"][:2]) == pytest.approx([3.0, 4.0], abs=0.1) and info["size_m"][2] == pytest.approx(2.5, abs=0.1)
    assert info["faces"]["Floor"] == 2 and info["faces"]["Ceiling"] == 2 and info["faces"]["Wall"] > 100
    assert sum(info["faces"].values()) - 4 <= 4000                                  # decimated to --triangles
    o3d = pytest.importorskip("open3d")
    cloud = np.asarray(o3d.io.read_point_cloud(str(tmp_path / "box_cloud.ply")).points)
    assert (cloud.min(0) >= -0.051).all() and (cloud.max(0) <= np.array(info["size_m"]) + 0.051).all()  # window points cropped
    assert "ETH3D" in info["source"]


def test_cli_can_keep_the_scanned_floor(synthetic_scans, tmp_path, capsys):
    _, info = scan_room_to_sketchup.main(["fake/room", "--data", str(synthetic_scans), "--out", str(tmp_path),
                                          "--voxel", "0.05", "--depth", "6", "--triangles", "4000", "--scanned-floor"])
    assert info["faces"]["Floor"] > 2 and info["floor_ceiling"] == "scanned mesh"


def test_cli_explains_missing_scans(tmp_path):
    with pytest.raises(SystemExit) as e:
        scan_room_to_sketchup.main(["eth3d/nothere", "--data", str(tmp_path)])
    assert "nothere_dslr_scan_eval.7z" in str(e.value) and "eth3d.net" in str(e.value)


def test_cli_publish_copies_to_preview_models(synthetic_scans, tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    (root / "preview" / "models").mkdir(parents=True)
    monkeypatch.setattr(scan_room_to_sketchup, "ROOT", root)
    scan_room_to_sketchup.main(["fake/room", "--data", str(synthetic_scans), "--name", "pub", "--voxel", "0.05",
                                "--depth", "6", "--triangles", "2000", "--publish"])
    assert sorted(p.name for p in (root / "preview" / "models").iterdir()) == ["pub.dae", "pub.mtl", "pub.obj"]
    assert (root / "outputs" / "fake" / "room" / "pub.json").is_file()
    assert "copied to preview/models" in capsys.readouterr().out


# ---------- integration: the real ETH3D office scan ----------

@pytest.fixture(scope="module")
def office_available():
    pytest.importorskip("open3d")
    if not (OFFICE / "scan_alignment.mlp").is_file():
        pytest.skip("ETH3D office scan not downloaded (data/eth3d/office/dslr_scan_eval)")


def test_office_scan_becomes_a_room_of_the_right_size(office_available, tmp_path):
    files, info = R.scan_to_room(OFFICE, tmp_path, "office", voxel=0.05, depth=7, triangles=6000, log=lambda *_: None)
    assert info["size_m"] == pytest.approx(OFFICE_SIZE, abs=0.08)
    assert info["faces"]["Wall"] > 1000 and info["faces"]["Furniture"] > 500
    ET.fromstring(files["dae"].read_text(encoding="utf-8"))


def test_published_office_dae_is_a_sketchup_export_of_the_room():
    root = ET.parse(PUBLISHED).getroot()
    assert root.find("c:asset/c:unit", NS).get("meter") == "0.0254" and root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    geos = {g.get("name"): g for g in root.findall(".//c:geometry", NS)}
    assert list(geos) == list(R.ROOM_MATERIALS)
    v = np.vstack([np.array(g.find(".//c:float_array", NS).text.split(), float).reshape(-1, 3) for g in geos.values()]) * S.INCH
    assert v.min(0) == pytest.approx([0, 0, 0], abs=0.05) and v.max(0) == pytest.approx(OFFICE_SIZE, abs=0.05)
    tris = sum(int(g.find(".//c:triangles", NS).get("count")) for g in geos.values())
    assert 30_000 < tris <= 60_004 and PUBLISHED.stat().st_size < 8_000_000
    for ext in ("obj", "mtl"):
        assert PUBLISHED.with_suffix("." + ext).is_file()
    assert "office_scan_room.dae" in (PREVIEW / "models" / "CREDITS.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page():
    pytest.importorskip("playwright.sync_api")
    if not CHROME:
        pytest.skip("Chrome not installed")
    try:
        urllib.request.urlopen("https://cdn.jsdelivr.net/npm/three@0.160.0/package.json", timeout=10).close()
    except OSError:
        pytest.skip("cdn.jsdelivr.net (three.js) unreachable")
    from playwright.sync_api import sync_playwright
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PREVIEW))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 800, "height": 450})
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/tagar.html")
        pg.wait_for_function("window.__tagar && window.AR", timeout=30000)
        yield pg
        b.close()
    srv.shutdown()


def test_room_card_button_loads_the_scanned_office_in_metres(page):
    page.evaluate("document.getElementById('r-units').value = 'cm'")         # the button resets the units menu
    page.evaluate("document.getElementById('r-scan').click()")
    page.wait_for_function("window.__tagar.roomState().name === 'office_scan_room.dae'", timeout=60000)
    s = page.evaluate("__tagar.roomState()")
    assert page.evaluate("document.getElementById('r-units').value") == "auto"
    assert s["loaded"] and s["units"] == "m"
    L, W, H = OFFICE_SIZE                                                     # three.js: x, y up, z = -file y
    assert s["sizeM"] == pytest.approx([L, H, W], abs=0.12)          # walls run into the 5 cm crop margin
    assert "(as the file says)" in page.evaluate("document.getElementById('r-info').textContent")
    assert "CC BY-NC-SA 4.0" in page.content()
