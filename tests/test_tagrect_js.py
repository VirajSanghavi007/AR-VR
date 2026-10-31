"""Tabletop AR on 4 AprilTags: unit (preview/tagrect.js through node), smoke (tagar.html wiring),
integration (a synthetic camera photo of 4 tags on a table goes through the page's real chain:
bottom-up readPixels buffer, downsample, js-aruco2 decode, pixel rays, table plane, rectangle fit)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from rooftop import tags as T

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
JS = (PREVIEW / "tagrect.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as R from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


# ---------- a pinhole camera in WebXR conventions ----------

W, H, F = 1280, 960, 1000.0


def cam_to_world(eye, target):
    eye, target = np.array(eye, float), np.array(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def projection(near=0.05, far=50.0):
    p = np.zeros((4, 4))
    p[0, 0], p[1, 1] = 2 * F / W, 2 * F / H
    p[2, 2], p[2, 3], p[3, 2] = -(far + near) / (far - near), -2 * far * near / (far - near), -1
    return p


def to_pixel(c2w, pt):
    X, Y, Z = (np.linalg.inv(c2w) @ np.append(pt, 1))[:3]
    return [W / 2 + F * X / -Z, H / 2 - F * Y / -Z]


col = lambda m: m.T.ravel().tolist()                          # column-major, like XRView matrices
EYE, TARGET = [0.1, 1.0, 0.75], [0, 0, 0]


# ---------- unit ----------

def test_invert4_and_transform(node_available):
    m = cam_to_world(EYE, TARGET) @ np.diag([2, 1, 1, 1])
    inv = np.array(run(f"R.invert4({col(m)})")).reshape(4, 4).T
    assert np.allclose(inv, np.linalg.inv(m), atol=1e-9)
    p = run(f"R.transform({col(projection())}, [0.1, 0.2, -2])")
    h = projection() @ [0.1, 0.2, -2, 1]
    assert np.allclose(p, h[:3] / h[3])
    with pytest.raises(RuntimeError, match="not invertible"):
        run(f"R.invert4({[0] * 16})")


def test_pixel_ray_hits_the_point_it_came_from(node_available):
    c2w = cam_to_world(EYE, TARGET)
    for pt in ([0.3, 0, -0.2], [-0.4, 0, 0.25], [0, 0, 0]):
        u, v = to_pixel(c2w, pt)
        hit = run(f"R.rayPlane(R.pixelRay({u}, {v}, {W}, {H}, {col(projection())}, {col(c2w)}), [0, 0, 0], [0, 1, 0])")
        assert np.allclose(hit, pt, atol=1e-6)
    ray = run(f"R.pixelRay({W / 2}, {H / 2}, {W}, {H}, {col(projection())}, {col(c2w)})")
    assert np.allclose(ray["dir"], -c2w[:3, 2], atol=1e-9)    # the image centre looks straight ahead


def test_ray_plane_parallel_and_behind(node_available):
    assert run("R.rayPlane({origin: [0, 1, 0], dir: [1, 0, 0]}, [0, 0, 0], [0, 1, 0])") is None
    assert run("R.rayPlane({origin: [0, 1, 0], dir: [0, 1, 0]}, [0, 0, 0], [0, 1, 0])") is None
    assert run("R.rayPlane({origin: [0, 1, 0], dir: [0, -1, 0]}, [0, 0.5, 0], [0, 1, 0])") == [0, 0.5, 0]
    # parallel from below the plane: no hit, not a point at infinity
    assert run("R.rayPlane({origin: [0, -1, 0], dir: [1, 0, 0]}, [0, 0, 0], [0, 1, 0])") is None
    assert run("R.rayPlane({origin: [0, -1, 0], dir: [1, 1e-9, 0]}, [0, 0, 0], [0, 1, 0])") is None


def test_fit_rectangle_any_order_and_rotation(node_available):
    a = np.radians(35)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pts = (np.array([[0.4, 0, 0.25], [-0.4, 0, -0.25], [0.4, 0, -0.25], [-0.4, 0, 0.25]]) @ R.T + [1, 0.7, 2]).tolist()
    r = run(f"R.fitRectangle({pts}, [0, 2, 0])")
    assert r["width"] == pytest.approx(0.8) and r["height"] == pytest.approx(0.5)
    assert np.allclose(r["center"], [1, 0.7, 2]) and r["angleErrDeg"] < 1e-6 and r["diagErrPct"] < 1e-9
    assert np.allclose(np.abs(r["u"]), np.abs(R @ [1, 0, 0]), atol=1e-9)   # long side
    assert np.allclose(np.cross(r["u"], r["n"]), r["v"]) and np.allclose(r["n"], [0, 1, 0])
    # tall input: the long side still comes out as u
    tall = run("R.fitRectangle([[0,0,0],[0.3,0,0],[0.3,0,0.9],[0,0,0.9]], [0,1,0])")
    assert tall["width"] == pytest.approx(0.9) and np.allclose(np.abs(tall["u"]), [0, 0, 1])


def test_fit_rectangle_flags_skew_and_bad_input(node_available):
    skew = run("R.fitRectangle([[0,0,0],[1,0,0],[1.3,0,0.5],[0.3,0,0.5]], [0,1,0])")
    assert 25 < skew["angleErrDeg"] < 35
    d1, d2 = np.hypot(1.3, 0.5), np.hypot(0.7, 0.5)
    assert skew["diagErrPct"] == pytest.approx(200 * (d1 - d2) / (d1 + d2))
    with pytest.raises(RuntimeError, match="need 4"):
        run("R.fitRectangle([[0,0,0],[1,0,0],[1,0,1]], [0,1,0])")
    with pytest.raises(RuntimeError, match="in a line"):
        run("R.fitRectangle([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], [0,1,0])")


def test_fit_rectangle_long_axis_averages_both_long_sides(node_available):
    # not quite parallel long sides (slopes 0.1 and 0.2): u is their mean direction, flat in the plane
    r = run("R.fitRectangle([[0,0,0],[1,0,0.1],[1,0,0.6],[0,0,0.4]], [0,1,0])")
    assert np.allclose(np.abs(r["u"]), np.abs(np.array([2, 0, 0.3]) / np.hypot(2, 0.3)), atol=1e-9)


def test_tag_edge(node_available):
    assert run("R.tagEdge([[0,0,0],[0.16,0,0],[0.16,0,0.16],[0,0,0.16]])") == pytest.approx(0.16)
    assert run("R.tagEdge([[0,0,0],[0.2,0,0],[0.2,0,0.1],[0,0,0.1]])") == pytest.approx(0.15)


def test_fit_scale_and_best_turns(node_available):
    assert run("R.fitScale(16, 12, 0.8, 0.5, 0, 1)") == pytest.approx(0.5 / 12)
    assert run("R.fitScale(16, 12, 0.8, 0.5, 1, 1)") == pytest.approx(0.5 / 16)
    assert run("R.fitScale(16, 12, 0.8, 0.5)") == pytest.approx(0.95 * 0.5 / 12)
    assert run("R.bestTurns(16, 12, 0.8, 0.5)") == 0
    assert run("R.bestTurns(12, 16, 0.8, 0.5)") == 1


def test_scale_label_and_nice_steps(node_available):
    assert run("R.scaleLabel(1 / 20.43)") == "1:20.4"
    assert run("R.scaleLabel(1 / 251.6)") == "1:252"
    assert run("R.scaleLabel(0.05)") == "1:20"
    assert run("1 / R.nextNiceScale(1 / 20.4, 1)") == pytest.approx(20)
    assert run("1 / R.nextNiceScale(1 / 20.4, -1)") == pytest.approx(25)
    assert run("1 / R.nextNiceScale(1 / 20, 1)") == pytest.approx(15)      # already round: next one
    assert run("1 / R.nextNiceScale(1 / 20, -1)") == pytest.approx(25)
    assert run("1 / R.nextNiceScale(1, 1)") == pytest.approx(1)            # clamped at the ends
    assert run("1 / R.nextNiceScale(1 / 1000, -1)") == pytest.approx(1000)


def rect(c, w=0.8, h=0.5):
    return f"{{center: {c}, width: {w}, height: {h}}}"


def test_is_stable(node_available):
    good = ",".join(rect([0.001 * i, 0, 0]) for i in range(3))
    assert run(f"R.isStable([{good}])") is True
    assert run(f"R.isStable([{rect([0, 0, 0])}, {rect([0, 0, 0])}])") is False          # too few
    assert run(f"R.isStable([{rect([0, 0, 0])},{rect([0, 0, 0])},{rect([0.05, 0, 0])}])") is False
    assert run(f"R.isStable([{rect([0, 0, 0])},{rect([0, 0, 0])},{rect([0, 0, 0], w=0.85)}])") is False
    assert run(f"R.isStable([{rect([0.5, 0, 0])},{good}])") is True                      # only the last 3


def test_downsample_averages_blocks_and_flips(node_available):
    # 4x2 image, red channel = 10*row + col; rows stored bottom-up like gl.readPixels
    expr = ("(() => { const a = new Uint8Array(4 * 2 * 4);"
            " for (let y = 0; y < 2; y++) for (let x = 0; x < 4; x++) { a[(y * 4 + x) * 4] = 10 * y + x; a[(y * 4 + x) * 4 + 2] = 40 * y; }"
            " const f = R.downsample(a, 4, 2, true, 2), n = R.downsample(a, 4, 2, false, 8), u = R.downsample(a, 4, 2, true, 8);"
            " const red = (d) => Array.from(d.data.filter((_, i) => i % 4 === 0));"
            " return [f.width, f.height, f.factor, red(f), f.data[2], f.data[3], n.factor, red(n), red(u)]; })()")
    # 2x2 blocks: (0+1+10+11)/4 = 5.5 -> 6, (2+3+12+13)/4 = 7.5 -> 8; blue (0+0+40+40)/4 = 20
    assert run(expr) == [2, 1, 2, [6, 8], 20, 255, 1, [0, 1, 2, 3, 10, 11, 12, 13], [10, 11, 12, 13, 0, 1, 2, 3]]
    # 3 pixels over a budget of 2: the step rounds up to 2, never down to 1 (which would blow the budget)
    assert run("(() => { const d = R.downsample(new Uint8Array(12), 3, 1, false, 2); return [d.factor, d.width]; })()") == [2, 1]


def test_pick_tags(node_available):
    m = lambda i: f"{{id: {i}, corners: [[0,0],[1,0],[1,1],[0,1]]}}"
    ok = run(f"R.pickTags([{m(3)},{m(9)},{m(1)},{m(0)},{m(2)}], [0,1,2,3])")
    assert ok["ok"] and [t["id"] for t in ok["tags"]] == [0, 1, 2, 3]
    miss = run(f"R.pickTags([{m(0)},{m(2)}], [0,1,2,3])")
    assert not miss["ok"] and miss["found"] == 2 and "missing id 1, 3" in miss["why"]
    assert run(f"R.pickTags([], [0,1,2,3])")["found"] == 0
    dup = run(f"R.pickTags([{m(0)},{m(0)},{m(1)},{m(2)},{m(3)}], [0,1,2,3])")
    assert not dup["ok"] and "id 0 seen twice" in dup["why"]


# ---------- smoke ----------

def test_tagar_page_wiring():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for src in re.findall(r'(?:src|from)="\.?/?([\w./-]+?\.js)(?:\?[^"]*)?"', html):
        assert (PREVIEW / src).exists(), src
    js = (PREVIEW / "tagrect.js").read_text(encoding="utf-8")
    names = re.search(r'import \{([^}]+)\} from "\./tagrect\.js', html).group(1)
    for name in re.findall(r"\w+", names):
        assert re.search(rf"export (?:function|const) {name}\b", js), name
    for feature in ('"hit-test"', '"camera-access"', '"anchors"', "createAnchor", "getCameraImage"):
        assert feature in html, feature
    assert 'href="tagar.html"' in (PREVIEW / "index.html").read_text(encoding="utf-8")


# ---------- integration: synthetic table photo through the real chain ----------

CHAIN = """
require(process.argv[1] + "/vendor/js-aruco2/cv.js");
const AR = require(process.argv[1] + "/tag36h11.js");
const fs = require("fs");
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const raw = fs.readFileSync(process.argv[3]);
import(require("url").pathToFileURL(process.argv[1] + "/tagrect.js").href).then((R) => {
  const img = R.downsample(new Uint8Array(raw.buffer, raw.byteOffset, raw.length), job.w, job.h, true, job.maxPixels);
  const found = new AR.Detector({ dictionaryName: "APRILTAG_36h11" }).detect(img);
  const pick = R.pickTags(found.map((m) => ({ id: m.id, corners: m.corners.map((c) => [c.x * img.factor, c.y * img.factor]) })), [0, 1, 2, 3]);
  if (!pick.ok) { console.log(JSON.stringify(pick)); return; }
  const n = [job.hit[4], job.hit[5], job.hit[6]], p0 = [job.hit[12], job.hit[13], job.hit[14]];
  const centres = [], edges = [];
  for (const t of pick.tags) {
    const w = t.corners.map(([x, y]) => R.rayPlane(R.pixelRay(x, y, job.w, job.h, job.proj, job.pose), p0, n));
    centres.push(w.reduce((a, p) => a.map((v, i) => v + p[i] / 4), [0, 0, 0]));
    edges.push(R.tagEdge(w));
  }
  console.log(JSON.stringify({ ok: true, factor: img.factor, rect: R.fitRectangle(centres, n), edges }));
});
"""

CENTRES = {0: [-0.4, -0.25], 1: [0.4, -0.25], 2: [0.4, 0.25], 3: [-0.4, 0.25]}


def table_photo(tmp_path, ids=(0, 1, 2, 3), max_pixels=2.6e6):
    """4 tags (0.16 m black square) on the plane y = 0, photographed from EYE; RGBA bottom-up."""
    c2w = cam_to_world(EYE, TARGET)
    img = np.full((H, W), 170, np.uint8)
    for k, tid in zip(CENTRES, ids):
        t = T.tag_image(tid, cell_px=20)                    # 10 cells: black square 8 cells = 0.16 m
        s = t.shape[0]
        cx, cz = CENTRES[k]
        world = [[cx - 0.1, 0, cz - 0.1], [cx + 0.1, 0, cz - 0.1], [cx + 0.1, 0, cz + 0.1], [cx - 0.1, 0, cz + 0.1]]
        dst = np.float32([to_pixel(c2w, p) for p in world])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]), dst)
        mask = cv2.warpPerspective(np.full_like(t, 255), M, (W, H)) > 0
        img[mask] = cv2.warpPerspective(t, M, (W, H))[mask]
    rgba = cv2.cvtColor(img[::-1], cv2.COLOR_GRAY2RGBA)     # gl.readPixels rows start at the bottom
    (tmp_path / "frame.bin").write_bytes(rgba.tobytes())
    hit = np.eye(4)                                          # hit-test pose: table plane, y up
    job = {"w": W, "h": H, "proj": col(projection()), "pose": col(c2w), "hit": col(hit), "maxPixels": max_pixels}
    (tmp_path / "job.json").write_text(json.dumps(job))


def chain(tmp_path):
    out = subprocess.run(["node", "-e", CHAIN, str(PREVIEW), str(tmp_path / "job.json"), str(tmp_path / "frame.bin")],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_table_photo_gives_the_true_rectangle(tmp_path, node_available):
    table_photo(tmp_path)
    r = chain(tmp_path)
    assert r["ok"] and r["factor"] == 1                     # 1.2 MP frame: kept whole
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01)
    assert r["rect"]["height"] == pytest.approx(0.5, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0], atol=0.01)
    assert r["rect"]["angleErrDeg"] < 2
    assert np.allclose(r["edges"], 0.16, atol=0.008)        # tag size check agrees with the print


def test_table_photo_with_a_wrong_tag_is_refused(tmp_path, node_available):
    table_photo(tmp_path, ids=(0, 1, 2, 7))
    r = chain(tmp_path)
    assert r["ok"] is False and "missing id 3" in r["why"]
