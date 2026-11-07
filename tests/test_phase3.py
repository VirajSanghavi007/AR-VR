"""rooftop.phase3 + scripts/phase3_preview.py: the scanned-roof model for vr.html?phase=3.
Unit (frames, rings, crop, thin, PLY), smoke (CLI on a tiny fake site), integration (the real ETH3D terrace)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rooftop import phase3 as P  # noqa: E402

IDENT = {"unit_scale": 1.0, "R_level": np.eye(3).tolist(), "roof_origin": [0, 0, 0]}


# ---------- unit ----------

def test_cloud_to_gltf_is_z_up_to_y_up_about_the_roof_origin():
    t = dict(IDENT, roof_origin=[1, 2, 3])
    out = P.cloud_to_gltf([[1, 2, 3], [2, 2, 3], [1, 3, 3], [1, 2, 4]], t)
    assert np.allclose(out, [[0, 0, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0]])


def test_cloud_to_gltf_scales_before_levelling():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)            # 90 deg about z
    t = {"unit_scale": 2.0, "R_level": r.tolist(), "roof_origin": [0, 2, 0]}
    assert np.allclose(P.cloud_to_gltf([[1, 0, 0]], t), [[0, 0, 0]])   # (2,0,0) -> (0,2,0) -> origin


def test_plan_ring_flips_gltf_z_to_plan_y():
    assert P.plan_ring([[1.23456, 9, -2.5], [0, 0, 3]]) == [[1.235, 2.5], [0.0, -3.0]]


def test_crop_keeps_points_near_the_outline_only():
    pts = np.array([[0, 5, 0], [12, 0, 0], [0, 0, -12], [2.9, 0, -1], [3.1, 0, -1], [0, 0, 3.1]], float)  # plan y = -z
    col = np.arange(18).reshape(6, 3)
    kept, kc = P.crop_to_outline(pts, col, [[[-1, -1], [1, -1], [1, 1]]], margin=2)
    assert kept.tolist() == [[0, 5, 0], [2.9, 0, -1]] and kc.tolist() == [[0, 1, 2], [9, 10, 11]]
    kept, kc = P.crop_to_outline(pts, None, [[[-1, -1], [1, 1]]], margin=2.0)
    assert len(kept) == 2 and kc is None


def test_crop_margin_grows_the_box_on_every_side():
    ring = [[[0, 0], [2, 0], [2, 1]]]
    pts = np.array([[-0.5, 0, 0], [2.5, 0, 0], [1, 0, 0.5], [1, 0, -1.5], [-0.6, 0, 0], [1, 0, -1.6]], float)
    kept, _ = P.crop_to_outline(pts, None, ring, margin=0.55)
    assert kept.tolist() == pts[:4].tolist()                            # low x, high x, low y, high y all kept
    assert len(P.crop_to_outline(pts, None, ring, margin=0)[0]) == 0


def test_thin_is_repeatable_ordered_and_capped():
    pts = np.arange(300, dtype=float).reshape(100, 3)
    a, ca = P.thin(pts, pts, 10)
    b, _ = P.thin(pts, None, 10)
    assert len(a) == 10 and np.array_equal(a, b) and np.array_equal(a, ca)
    assert np.all(np.diff(a[:, 0]) > 0)
    same, c = P.thin(pts, None, 100)
    assert same is pts and c is None


def test_height_colors_ramp():
    c = P.height_colors([[0, 0, 0], [0, 5, 0], [0, 10, 0]])
    assert c.shape == (3, 3) and c[0, 0] < c[1, 0] < c[2, 0]
    assert np.isclose(c[0, 0], 0.35) and np.isclose(c[2, 0], 0.9) and (c <= 1).all()
    assert np.allclose(P.height_colors([[0, 1, 0], [5, 1, 0]])[:, 0], 0.35)   # flat: no divide by zero


def test_ply_round_trip(tmp_path):
    pts = np.array([[0, 1, 2], [-3.5, 4.25, 1e-3]], np.float32)
    col = np.array([[0, 0.5, 1], [1, 1, 0]])
    path = P.write_points_ply(tmp_path / "a" / "s.ply", pts, col)
    head = path.read_bytes()[:200].decode(errors="ignore")
    assert "binary_little_endian" in head and "element vertex 2" in head and "property uchar red" in head
    p2, c2 = P.read_points_ply(path)
    assert np.array_equal(p2, pts) and c2.tolist() == [[0, 128, 255], [255, 255, 0]]
    with pytest.raises(ValueError):
        P.write_points_ply(tmp_path / "b.ply", pts, col[:1])


def fake_anchor():
    return {"outline": [[[0, 0.1, 0], [4, 0.1, 0], [4, 0.1, -3], [0, 0.1, -3]]],
            "keepouts": [[[1, 0.1, -1], [2, 0.1, -1], [2, 0.1, -2]]]}


def fake_summary():
    return {"site": "x/y", "panels": 4, "kwp": 1.6, "outline_area_m2": 12.04, "usable_area_m2": 6.96, "transform": IDENT}


def test_sidecar_fields():
    d = P.sidecar(fake_summary(), fake_anchor(), 7)
    assert d["phase"] == 3 and d["site"] == "x/y" and d["panels"] == 4 and d["kwp"] == 1.6
    assert d["outline_area_m2"] == 12.0 and d["usable_area_m2"] == 7.0 and d["roof_y"] == 0.1
    assert d["outline"] == [[[0, 0], [4, 0], [4, 3], [0, 3]]] and d["keepouts"][0][2] == [2, 2]
    assert d["scan_points"] == 7
    assert P.sidecar(fake_summary(), {"outline": []})["keepouts"] == []


def fake_site(tmp_path):
    out = tmp_path / "outputs" / "x" / "y"
    out.mkdir(parents=True)
    (out / "panels_summary.json").write_text(json.dumps(fake_summary()))
    (out / "panels_anchor.json").write_text(json.dumps(fake_anchor()))
    (out / "roof_panels.glb").write_bytes(b"glTF-fake")
    return out


def test_build_with_scan_writes_three_files(tmp_path):
    out = fake_site(tmp_path)
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-10, 14, 500), rng.uniform(-10, 14, 500), rng.uniform(0, 5, 500)]  # z-up cloud
    files, data = P.build(out, tmp_path / "m", "p3", (pts, None), max_points=50, margin=1.0)
    assert (tmp_path / "m" / "p3.glb").read_bytes() == b"glTF-fake"
    j = json.loads((tmp_path / "m" / "p3.json").read_text())
    assert j == data and j["scan"] == "p3_scan.ply" and 0 < j["scan_points"] <= 50
    p, c = P.read_points_ply(files["scan"])
    assert len(p) == j["scan_points"] and c.std() > 0                     # height colours, not flat grey
    assert p[:, 0].min() >= -1 - 1e-6 and p[:, 0].max() <= 5 + 1e-6      # cropped to outline + 1 m
    assert (-p[:, 2]).min() >= -1 - 1e-6 and (-p[:, 2]).max() <= 4 + 1e-6


def test_build_keeps_real_scan_colours(tmp_path):
    out = fake_site(tmp_path)
    pts = np.array([[1, 1, 0], [2, 2, 1]], float)
    col = np.array([[1, 0, 0], [0, 0, 1]], float)
    files, _ = P.build(out, tmp_path / "m", "p3", (pts, col))
    assert P.read_points_ply(files["scan"])[1].tolist() == [[255, 0, 0], [0, 0, 255]]


def test_build_thins_to_max_points(tmp_path):
    out = fake_site(tmp_path)
    pts = np.c_[np.linspace(0.1, 3.9, 40), np.linspace(0.1, 2.9, 40), np.linspace(0, 2, 40)]
    files, data = P.build(out, tmp_path / "m", "p3", (pts, None), max_points=7)
    assert data["scan_points"] == 7 and len(P.read_points_ply(files["scan"])[0]) == 7


def test_build_replaces_flat_grey_colours_with_height_colours(tmp_path):
    out = fake_site(tmp_path)
    pts = np.array([[1, 1, 0], [2, 2, 1], [3, 2, 2]], float)
    files, _ = P.build(out, tmp_path / "m", "p3", (pts, np.full((3, 3), 0.5)))
    c = P.read_points_ply(files["scan"])[1]
    assert c[0, 0] < c[1, 0] < c[2, 0]                                   # a scan in one grey gets the height ramp


def test_build_without_scan(tmp_path):
    files, data = P.build(fake_site(tmp_path), tmp_path / "m", "p3")
    assert set(files) == {"glb", "json"} and "scan" not in data and data["scan_points"] == 0


# ---------- smoke ----------

def test_cli_runs_on_a_tiny_site(tmp_path, monkeypatch):
    import phase3_preview as cli
    fake_site(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    files = cli.main(["x/y", "--name", "t", "--models", str(tmp_path / "m"), "--no-scan"])
    assert json.loads(Path(files["json"]).read_text())["panels"] == 4


def test_cli_scan_flags(tmp_path, monkeypatch):
    import phase3_preview as cli
    fake_site(tmp_path)
    (tmp_path / "data" / "x" / "y" / "dslr_scan_eval").mkdir(parents=True)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    pts = np.c_[np.linspace(0.1, 3.9, 20), np.linspace(0.1, 2.9, 20), np.linspace(0, 2, 20)]
    monkeypatch.setattr(cli, "load_scan", lambda d, voxel: (pts, None))
    m = str(tmp_path / "m")
    assert "scan" not in cli.main(["x/y", "--models", m, "--no-scan"])  # --no-scan wins over a scan folder
    files = cli.main(["x/y", "--models", m, "--max-points", "5"])
    assert json.loads(Path(files["json"]).read_text())["scan_points"] == 5


# ---------- integration: the real ETH3D terrace ----------

TERRACE = ROOT / "outputs" / "eth3d" / "terrace"
SCANS = ROOT / "data" / "eth3d" / "terrace" / "dslr_scan_eval"


@pytest.fixture(scope="module")
def terrace_available():
    if not (TERRACE / "panels_summary.json").is_file() or not (SCANS / "scan_alignment.mlp").is_file():
        pytest.skip("ETH3D terrace outputs or laser scans not present")
    pytest.importorskip("open3d")


def test_terrace_scan_lines_up_with_the_roof_cad(terrace_available, tmp_path):
    import phase3_preview as cli
    files = cli.main(["eth3d/terrace", "--name", "t", "--models", str(tmp_path), "--voxel", "0.08",
                      "--max-points", "40000"])
    data = json.loads(Path(files["json"]).read_text())
    assert data["panels"] == 39 and data["kwp"] == 15.6 and data["scan_points"] == 40000
    p, _ = P.read_points_ply(files["scan"])
    from shapely.geometry import Point, Polygon
    roof = Polygon(data["outline"][0]).buffer(-0.5)
    on = np.array([roof.contains(Point(x, -z)) for x, _, z in p[::20]])
    y = p[::20][on, 1]
    assert len(y) > 100
    assert abs(np.median(y) - data["roof_y"]) < 0.05                   # laser points sit on the CAD roof plane
    assert np.mean(np.abs(y - data["roof_y"]) < 0.1) > 0.7


def test_shipped_phase3_model_matches_the_outputs(terrace_available):
    models = ROOT / "preview" / "models"
    j = json.loads((models / "phase3_terrace.json").read_text())
    s = json.loads((TERRACE / "panels_summary.json").read_text())
    assert j["panels"] == s["panels"] and j["kwp"] == s["kwp"] and j["scan"] == "phase3_terrace_scan.ply"
    assert (models / "phase3_terrace.glb").read_bytes() == (TERRACE / "roof_panels.glb").read_bytes()
    assert len(P.read_points_ply(models / j["scan"])[0]) == j["scan_points"]
