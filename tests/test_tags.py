"""AprilTag scale: unit (math, model IO, undistortion), smoke (CLIs), integration
(real OpenCV detection on synthetic photos with known poses, real PDF render)."""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from rooftop import tags as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_tags  # noqa: E402
import tag_scale  # noqa: E402


def rot(ax, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]


def look_at(eye, target):
    """World->camera R, t for a camera at eye looking at target (OpenCV: x right, y down, z forward)."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ eye


def R_to_quat(R):
    qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return qw, (R[2, 1] - R[1, 2]) / (4 * qw), (R[0, 2] - R[2, 0]) / (4 * qw), (R[1, 0] - R[0, 1]) / (4 * qw)


# ---------- unit ----------

def test_quat_roundtrip_and_triangulate_exact():
    R = rot("z", 30) @ rot("x", 70)
    assert np.allclose(T.quat_to_R(*R_to_quat(R)), R)
    X = np.array([0.3, -0.2, 0.1])
    obs = []
    for eye in ([0, -3, 2], [2, -2, 2.5], [-2, -1, 3]):
        Rc, t = look_at(np.array(eye, float), np.zeros(3))
        c = Rc @ X + t
        obs.append((Rc, t, c[:2] / c[2]))
    assert np.allclose(T.triangulate(obs), X, atol=1e-9)
    assert T.reprojection_error(X, obs) < 1e-12


@pytest.mark.parametrize("model,params", [
    ("SIMPLE_RADIAL", [1500, 800, 600, -0.08]),
    ("RADIAL", [1500, 800, 600, -0.08, 0.02]),
    ("OPENCV", [1500, 1490, 810, 590, -0.1, 0.03, 0.001, -0.0005]),
    ("OPENCV_FISHEYE", [900, 900, 800, 600, 0.05, -0.01, 0.002, 0.0]),
    ("PINHOLE", [1500, 1490, 810, 590]),
])
def test_normalize_points_inverts_projection(model, params):
    xy = np.array([[0.1, -0.2], [-0.3, 0.25], [0.0, 0.0]])
    p = np.array(params, float)
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
        dist = np.array([p[3], p[4] if model == "RADIAL" else 0, 0, 0])
    else:
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        dist = p[4:8] if len(p) > 4 else np.zeros(4)
    pts3 = np.hstack([xy, np.ones((3, 1))]).reshape(-1, 1, 3)
    if model == "OPENCV_FISHEYE":
        px = cv2.fisheye.projectPoints(pts3, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    else:
        px = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    back = T.normalize_points(px, T.Camera(model, 1600, 1200, p))
    assert np.allclose(back, xy, atol=1e-6)


def test_unsupported_camera_model_is_explicit():
    with pytest.raises(ValueError, match="image_undistorter"):
        T.normalize_points([[1, 2]], T.Camera("THIN_PRISM_FISHEYE", 10, 10, np.zeros(12)))


def _write_text_model(d, cams, imgs):
    d.mkdir(parents=True, exist_ok=True)
    (d / "cameras.txt").write_text("# c\n" + "".join(
        f"{cid} {c.model} {c.width} {c.height} " + " ".join(map(str, c.params)) + "\n" for cid, c in cams.items()))
    lines = ["# i"]
    for k, im in enumerate(imgs, 1):
        q = R_to_quat(im.R)
        lines += [f"{k} {q[0]} {q[1]} {q[2]} {q[3]} {im.t[0]} {im.t[1]} {im.t[2]} {im.camera_id} {im.name}", ""]
    (d / "images.txt").write_text("\n".join(lines) + "\n")


def test_read_binary_model_matches_text(tmp_path):
    R = rot("x", 120) @ rot("z", 15)
    cams = {1: T.Camera("SIMPLE_RADIAL", 1600, 1200, np.array([1500.0, 800, 600, -0.05]))}
    imgs = [T.Image("a/img 1.png", R, np.array([0.1, 0.2, 3.0]), 1), T.Image("b.png", R.T, np.array([1.0, 0, 2]), 1)]
    _write_text_model(tmp_path / "txt", cams, imgs)
    b = tmp_path / "bin"; b.mkdir()
    with open(b / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 2, 1600, 1200) + struct.pack("<4d", 1500, 800, 600, -0.05))
    with open(b / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for k, im in enumerate(imgs, 1):
            f.write(struct.pack("<i7di", k, *R_to_quat(im.R), *im.t, 1) + im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", 2) + struct.pack("<ddq", 1.5, 2.5, -1) * 2)   # 2D points, skipped
    for c_, i_ in (T.read_model(tmp_path / "txt"), T.read_model(b)):
        assert c_[1].model == "SIMPLE_RADIAL" and np.allclose(c_[1].params, [1500, 800, 600, -0.05])
        assert [im.name for im in i_] == ["a/img 1.png", "b.png"]
        assert np.allclose(i_[0].R, R, atol=1e-6) and np.allclose(i_[1].t, [1, 0, 2])
    with pytest.raises(FileNotFoundError):
        T.read_model(tmp_path)


def _square(centre, size, scale=1.0):
    h = size / 2
    return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]) / scale + np.asarray(centre) / scale


def test_scale_from_tags_and_distances():
    s = 0.25                                                  # true metres per model unit
    tags3d = {0: {"corners": _square([0, 0, 0], 0.16, s), "views": 5, "reproj": 0.0},
              3: {"corners": _square([2, 1, 0], 0.16 * 1.01, s), "views": 4, "reproj": 0.0}}  # 1 % off
    r = T.scale_from_tags(tags3d, 0.16)
    assert r["n_tags"] == 2 and r["spread_pct"] == pytest.approx(1.0, rel=0.02)
    assert r["metres_per_unit"] == pytest.approx(s, rel=0.006)
    assert r["tags"][0]["squareness"] == pytest.approx(1.0)
    assert T.tag_distances(tags3d, s)["0-3"] == pytest.approx(np.hypot(2, 1))
    with pytest.raises(ValueError, match="2 photos"):
        T.scale_from_tags({}, 0.16)


def test_tag_image_detects_its_own_id():
    for tid in (0, 7, 42):
        img = cv2.copyMakeBorder(T.tag_image(tid, cell_px=20), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        found = T.detect_tags(img)
        assert list(found) == [tid]
        side = np.linalg.norm(found[tid][1] - found[tid][0])
        assert side == pytest.approx(T.CELLS * 20, abs=1.5)   # corners are the black square's


# ---------- integration: real detection on synthetic photos ----------

def render_site(root, site="synth", s_true=0.37, size=0.16, n_views=7, model="PINHOLE"):
    """Photos of 3 flat tags from known cameras; COLMAP model written in arbitrary units."""
    rng = np.random.default_rng(1)
    W, H, f = 1600, 1200, 1400.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    tags = {0: [0.0, 0.0], 5: [0.9, 0.3], 9: [0.2, 0.8]}
    centre = np.array([0.4, 0.35, 0.0])
    imgs = []
    img_dir = root / "data" / site / "images"
    img_dir.mkdir(parents=True)
    src = T.tag_image(0, cell_px=20)
    for v in range(n_views):
        ang = 2 * np.pi * v / n_views
        eye = centre + [1.3 * np.cos(ang), 1.3 * np.sin(ang), 1.7 + 0.2 * np.sin(3 * ang)]
        R, t = look_at(eye, centre + rng.normal(0, 0.05, 3))
        im = np.full((H, W), 140, np.uint8)
        for tid, (x, y) in tags.items():
            src = T.tag_image(tid, cell_px=20)                 # 200 px: 20 px margin + 160 px square
            m = 20 / 160 * size                               # margin in metres
            corners = _square([x, y, 0], size + 2 * m)        # outer white quiet zone
            # image rows go down; walk the tag's corners so its top-left is corner 0
            corners = corners[[3, 2, 1, 0]]
            cam = (R @ corners.T).T + t
            px = (K @ (cam / cam[:, 2:3]).T).T[:, :2].astype(np.float32)
            Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [199, 0], [199, 199], [0, 199]]), px)
            warped = cv2.warpPerspective(src, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpPerspective(np.full_like(src, 255), Hm, (W, H), flags=cv2.INTER_NEAREST)
            im[mask > 0] = warped[mask > 0]
        im = np.clip(im + rng.normal(0, 2, im.shape), 0, 255).astype(np.uint8)
        name = f"img_{v:02d}.png"
        cv2.imwrite(str(img_dir / name), im)
        imgs.append(T.Image(name, R, t / s_true, 1))       # model units = metres / s_true
    cams = {1: T.Camera(model, W, H, np.array([f, f, W / 2, H / 2]))}
    _write_text_model(root / "data" / site / "colmap" / "sparse" / "0", cams, imgs)
    return s_true, tags


def test_tag_scale_recovers_true_scale_from_photos(tmp_path):
    s_true, tags = render_site(tmp_path)
    res = tag_scale.run("synth", 0.16, root=tmp_path)
    assert res["n_tags"] == 3
    assert res["metres_per_unit"] == pytest.approx(s_true, rel=0.005)     # within 0.5 %
    assert res["spread_pct"] < 1.0
    d = res["tag_distances_m"]
    assert d["0-5"] == pytest.approx(np.hypot(0.9, 0.3), abs=0.01)        # 1 cm on a ~1 m baseline
    saved = json.loads((tmp_path / "outputs/synth/tag_scale.json").read_text())
    assert saved["metres_per_unit"] == pytest.approx(res["metres_per_unit"])
    assert tag_scale.load_scale("synth", root=tmp_path) == pytest.approx(s_true, rel=0.005)


def test_load_scale_missing_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="tag_scale.py"):
        tag_scale.load_scale("nowhere", root=tmp_path)


def test_printed_sheet_has_exact_size_and_is_detectable(tmp_path):
    import pymupdf
    path = make_tags.main(["--size-mm", "160", "--ids", "3", "11", "--out", str(tmp_path / "t.pdf")])
    doc = pymupdf.open(path)
    assert doc.page_count == 2
    page = doc[0]
    assert page.rect.width == pytest.approx(210 / 25.4 * 72, abs=0.5)       # A4 in points
    dpi = 300
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    found = T.detect_tags(g)
    assert list(found) == [3]
    side_px = np.mean([np.linalg.norm(found[3][(i + 1) % 4] - found[3][i]) for i in range(4)])
    assert side_px / dpi * 25.4 == pytest.approx(160, abs=0.5)                # black square = 160 mm
    with pytest.raises(ValueError):
        T.write_tag_sheet(tmp_path / "x.pdf", [0], size_mm=250)


@pytest.fixture
def tagged_site_available():
    p = ROOT / "data" / "rooftop_01" / "images"
    if not p.exists() or not any(p.iterdir()):
        pytest.skip("no real tagged capture in data/rooftop_01 yet")
    try:
        tag_scale.find_model(ROOT / "data" / "rooftop_01")
    except FileNotFoundError:
        pytest.skip("data/rooftop_01 has no COLMAP reconstruction yet")
    return "rooftop_01"


@pytest.mark.integration
def test_real_capture_tags(tagged_site_available):
    res = tag_scale.run(tagged_site_available, 0.16)
    assert res["n_tags"] >= 1 and res["metres_per_unit"] > 0


def test_design_panels_unit_scale_tags(tmp_path, monkeypatch):
    import design_panels
    (tmp_path / "outputs" / "s").mkdir(parents=True)
    (tmp_path / "outputs" / "s" / "tag_scale.json").write_text(json.dumps({"metres_per_unit": 0.123}))
    seen = {}
    monkeypatch.setattr(design_panels, "ROOT", tmp_path)
    monkeypatch.setattr(design_panels, "run", lambda site, unit_scale, **kw: seen.update(u=unit_scale) or {})
    design_panels.main(["s", "--unit-scale", "tags"])
    assert seen["u"] == 0.123
    design_panels.main(["s", "--unit-scale", "2.5"])
    assert seen["u"] == 2.5
