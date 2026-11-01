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
    far = "{id: 0, corners: [[9,9],[10,9],[10,10],[9,10]]}"            # the same id printed twice, far apart
    dup = run(f"R.pickTags([{m(0)},{far},{m(1)},{m(2)},{m(3)}], [0,1,2,3])")
    assert not dup["ok"] and "id 0 seen twice" in dup["why"]


def test_tag_axis_follows_the_top_edge(node_available):
    flat = run("R.tagAxis([[0,0,0],[0.16,0,0],[0.16,0,0.16],[0,0,0.16]])")
    assert np.allclose(flat, [1, 0, 0])
    # a slightly sheared quad: mean of the top and bottom edges
    assert np.allclose(run("R.tagAxis([[0,0,0],[0.16,0,0.02],[0.16,0,0.16],[0,0,0.16]])"),
                       np.array([0.32, 0, 0.02]) / np.hypot(0.32, 0.02))


def test_fit_from_two_opposite_corners(node_available):
    r = run("R.fitFromTwo([[-0.4,0.1,-0.25],[0.4,0.1,0.25]], [[1,0,0],[1,0,0]], [0,1,0])")
    assert r["width"] == pytest.approx(0.8) and r["height"] == pytest.approx(0.5)
    assert np.allclose(r["center"], [0, 0.1, 0]) and np.allclose(r["u"], [1, 0, 0])
    assert np.allclose(r["v"], np.cross(r["u"], r["n"])) and r["angleErrDeg"] == pytest.approx(0)
    assert np.allclose(sorted(map(tuple, r["corners"])), sorted([(-0.4, 0.1, -0.25), (0.4, 0.1, -0.25), (0.4, 0.1, 0.25), (-0.4, 0.1, 0.25)]))
    # tall: the tags' axis is the short side, so u swaps to the long one, still right-handed
    t = run("R.fitFromTwo([[0,0,0],[0.3,0,0.9]], [[1,0,0],[1,0,0]], [0,1,0])")
    assert t["width"] == pytest.approx(0.9) and t["height"] == pytest.approx(0.3)
    assert np.allclose(np.abs(t["u"]), [0, 0, 1]) and np.allclose(t["v"], np.cross(t["u"], t["n"]))
    # rotated 30 deg, axes with a tilt out of the plane and a 4 deg disagreement
    a = np.radians(30)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c = (np.array([[-0.4, 0, -0.25], [0.4, 0, 0.25]]) @ R.T).tolist()
    ax = [(R @ [np.cos(np.radians(d)), 0.1, np.sin(np.radians(d))]).tolist() for d in (2, -2)]
    r = run(f"R.fitFromTwo({c}, {ax}, [0, 1, 0])")
    assert r["angleErrDeg"] == pytest.approx(4, abs=1e-6) and np.allclose(r["u"], R @ [1, 0, 0], atol=1e-9)
    assert r["width"] == pytest.approx(0.8) and r["height"] == pytest.approx(0.5)
    # opposite axes fall back to the first tag's
    assert np.allclose(run("R.fitFromTwo([[0,0,0],[0.8,0,0.5]], [[1,0,0],[-1,0,0]], [0,1,0])")["u"], [1, 0, 0])


def test_fit_from_two_refuses_bad_input(node_available):
    with pytest.raises(RuntimeError, match="need 2"):
        run("R.fitFromTwo([[0,0,0]], [[1,0,0]], [0,1,0])")
    with pytest.raises(RuntimeError, match="opposite corners"):
        run("R.fitFromTwo([[0,0,0],[0.8,0,0.01]], [[1,0,0],[1,0,0]], [0,1,0])")
    with pytest.raises(RuntimeError, match="opposite corners"):
        run("R.fitFromTwo([[0,0,0],[0.01,0,0.8]], [[1,0,0],[1,0,0]], [0,1,0])")


def test_align_axes_and_rect_moved(node_available):
    a = "{center: [0,0,0], u: [1,0,0], v: [0,0,-1], width: 0.8, height: 0.5}"
    flipped = run(f"R.alignAxes({{center: [0,0,0], u: [-1,0,0], v: [0,0,1], width: 0.8, height: 0.5}}, {a})")
    assert flipped["u"] == [1, 0, 0] and flipped["v"] == [0, 0, -1]
    assert run(f"R.alignAxes({a}, {a})")["u"] == [1, 0, 0]
    assert run(f"R.alignAxes({a}, null)")["u"] == [1, 0, 0]
    same = "{center: [0.01,0,0], u: [-1,0,0], v: [0,0,1], width: 0.81, height: 0.49}"
    assert run(f"R.rectMoved({a}, {same})") is False            # small jitter, u sign flip is not a move
    for moved in ("{center: [0.03,0,0], u: [1,0,0], width: 0.8, height: 0.5}",
                  "{center: [0,0,0], u: [1,0,0], width: 0.83, height: 0.5}",
                  "{center: [0,0,0], u: [1,0,0], width: 0.8, height: 0.47}",
                  f"{{center: [0,0,0], u: [{np.cos(np.radians(5))},0,{np.sin(np.radians(5))}], width: 0.8, height: 0.5}}"):
        assert run(f"R.rectMoved({a}, {moved})") is True, moved


# ---------- smoke ----------

def test_tagar_page_wiring():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for src in re.findall(r'(?:src|from)="\.?/?([\w./-]+?\.js)(?:\?[^"]*)?"', html):
        assert (PREVIEW / src).exists(), src
    for mod in ("tagrect", "panel"):
        js = (PREVIEW / f"{mod}.js").read_text(encoding="utf-8")
        names = re.search(rf'import \{{([^}}]+)\}} from "\./{mod}\.js', html).group(1)
        for name in re.findall(r"\w+", names):
            assert re.search(rf"export (?:function|const) {name}\b", js), name
    for feature in ('"hit-test"', '"camera-access"', '"anchors"', "createAnchor", "getCameraImage", 'id="touch"',
                    "pointerdown", "pointermove", "pointerup", "fitFromTwo", "rectMoved(rect, r)", "lock(frame, r, true)",
                    "averageTwo(history)", "tagsMoved(rect.tags, r.tags)"):
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
  const pick = R.pickTags(found.map((m) => ({ id: m.id, corners: m.corners.map((c) => [c.x * img.factor, c.y * img.factor]) })), job.ids);
  if (!pick.ok) { console.log(JSON.stringify(pick)); return; }
  const n = [job.hit[4], job.hit[5], job.hit[6]], p0 = [job.hit[12], job.hit[13], job.hit[14]];
  const centres = [], edges = [], axes = [];
  for (const t of pick.tags) {
    const w = t.corners.map(([x, y]) => R.rayPlane(R.pixelRay(x, y, job.w, job.h, job.proj, job.pose), p0, n));
    centres.push(w.reduce((a, p) => a.map((v, i) => v + p[i] / 4), [0, 0, 0]));
    edges.push(R.tagEdge(w));
    axes.push(R.tagAxis(w));
  }
  const rect = job.ids.length === 2 ? R.fitFromTwo(centres, axes, n) : R.fitRectangle(centres, n);
  console.log(JSON.stringify({ ok: true, factor: img.factor, rect, edges }));
});
"""

CENTRES = {0: [-0.4, -0.25], 1: [0.4, -0.25], 2: [0.4, 0.25], 3: [-0.4, 0.25]}


def table_photo(tmp_path, ids=(0, 1, 2, 3), max_pixels=2.6e6, corners=(0, 1, 2, 3), want=(0, 1, 2, 3)):
    """Tags (0.16 m black square) at the given rectangle corners on the plane y = 0, upright along +x,
    photographed from EYE; RGBA bottom-up. `want` = the ids the page is told to look for."""
    c2w = cam_to_world(EYE, TARGET)
    img = np.full((H, W), 170, np.uint8)
    for k, tid in zip(corners, ids):
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
    job = {"w": W, "h": H, "proj": col(projection()), "pose": col(c2w), "hit": col(hit), "maxPixels": max_pixels, "ids": list(want)}
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


def test_table_photo_with_two_tags_gives_the_true_rectangle(tmp_path, node_available):
    table_photo(tmp_path, ids=(0, 1), corners=(0, 2), want=(0, 1))     # opposite corners
    r = chain(tmp_path)
    assert r["ok"]
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01)
    assert r["rect"]["height"] == pytest.approx(0.5, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0], atol=0.01)
    assert np.allclose(np.abs(r["rect"]["u"]), [1, 0, 0], atol=0.02)
    assert r["rect"]["angleErrDeg"] < 2


def test_two_tags_along_one_side_are_refused(tmp_path, node_available):
    table_photo(tmp_path, ids=(0, 1), corners=(0, 1), want=(0, 1))     # same long side
    with pytest.raises(RuntimeError, match="opposite corners"):
        chain(tmp_path)


def test_pick_tags_takes_any_two_in_two_tag_mode(node_available):
    m = lambda i: f"{{id: {i}, corners: [[{i},0],[1,0],[1,1],[0,1]]}}"
    r = run(f"R.pickTags([{m(2)},{m(0)}], [0,1])")                 # diagonal pair of a 4-tag sheet
    assert r["ok"] and [t["id"] for t in r["tags"]] == [0, 2]
    assert run(f"R.pickTags([{m(1)},{m(0)}], [0,1])")["tags"][0]["id"] == 0
    assert not run(f"R.pickTags([{m(0)}], [0,1])")["ok"]
    assert not run(f"R.pickTags([{m(0)},{m(5)},{m(7)}], [0,1])")["ok"]   # 3 tags: which 2 is unclear
    assert not run(f"R.pickTags([{m(5)},{m(5)}], [0,1])")["ok"]          # same id twice
    assert not run(f"R.pickTags([{m(0)},{m(2)}], [0,1,2,3])")["ok"]      # 4-tag mode stays strict


def test_average_two_waits_for_agreement_then_averages(node_available):
    o = lambda dx, deg: (f"{{centers: [[{dx},0,0],[0.8,0,0.5]], axes: [[{np.cos(np.radians(deg))},0,"
                         f"{np.sin(np.radians(deg))}],[2,0,0]]}}")
    assert run(f"R.averageTwo([{o(0, 0)},{o(0, 0)},{o(0, 0)}])") is None             # too few
    a = run(f"R.averageTwo([{o(0.004, 3)},{o(-0.004, -3)},{o(0.002, 2)},{o(-0.002, -2)}])")
    assert np.allclose(a["centers"], [[0, 0, 0], [0.8, 0, 0.5]], atol=1e-12)
    assert np.allclose(a["axes"], [[1, 0, 0], [1, 0, 0]], atol=1e-12)                 # unit mean
    assert run(f"R.averageTwo([{o(0.012, 0)},{o(-0.004, 0)},{o(-0.004, 0)},{o(-0.004, 0)}])") is None
    assert run(f"R.averageTwo([{o(0, 9)},{o(0, -3)},{o(0, -3)},{o(0, -3)}])") is None  # one axis 9 deg off
    assert run(f"R.averageTwo([{o(0.5, 40)},{o(0, 0)},{o(0, 0)},{o(0, 0)},{o(0, 0)}])") is not None  # only the last 4


def test_tags_moved(node_available):
    assert run("R.tagsMoved([[0,0,0],[0.8,0,0.5]], [[0.01,0,0],[0.8,0,0.51]])") is False
    assert run("R.tagsMoved([[0,0,0],[0.8,0,0.5]], [[0.03,0,0],[0.8,0,0.5]])") is True
    assert run("R.tagsMoved([[0,0,0],[0.8,0,0.5]], [[0,0,0],[0.8,0,0.53]])") is True


def test_two_tag_fit_is_steady_under_detection_noise(node_available):
    """Phone-like noise (3 mm centres, 3 deg tag edges): the averaged fit locks often and its size holds."""
    res = run("""(() => {
      let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const g = () => { let s = 0; for (let i = 0; i < 6; i++) s += rnd(); return s - 3; };
      const ax = () => { const a = (g() * 3 * Math.PI) / 180; return [Math.cos(a), 0, Math.sin(a)]; };
      const H = [], w = []; let locks = 0;
      for (let f = 0; f < 400; f++) {
        H.push({ centers: [[-0.4 + g() * 0.003, 0, -0.25 + g() * 0.003], [0.4 + g() * 0.003, 0, 0.25 + g() * 0.003]], axes: [ax(), ax()] });
        if (H.length > 8) H.shift();
        const a = R.averageTwo(H);
        if (a) { locks++; w.push(R.fitFromTwo(a.centers, a.axes, [0, 1, 0]).width); }
      }
      return { locks, w };
    })()""")
    assert res["locks"] > 100
    assert np.std(res["w"]) < 0.008 and abs(np.mean(res["w"]) - 0.8) < 0.005


def test_merge_nested_keeps_the_biggest_outline(node_available):
    big = "{id: 4, corners: [[0,0],[10,0],[10,10],[0,10]]}"
    inner = "{id: 4, corners: [[1,1],[9,1],[9,9],[1,9]]}"           # inner outline of the same tag
    other = "{id: 5, corners: [[2,2],[8,2],[8,8],[2,8]]}"           # a different id at the same spot
    far = "{id: 4, corners: [[30,0],[40,0],[40,10],[30,10]]}"       # the same id elsewhere: a real duplicate
    out = run(f"R.mergeNested([{inner},{other},{big},{far}])")
    assert sorted((m["id"], m["corners"][0][0]) for m in out) == [(4, 0), (4, 30), (5, 2)]
    near = "{id: 4, corners: [[8,0],[18,0],[18,10],[8,10]]}"       # centre 8 away, inside one tag edge (10)
    assert len(run(f"R.mergeNested([{big},{near}])")) == 1
    assert run("R.mergeNested([])") == []
    assert "c" not in run(f"R.mergeNested([{big}])")[0]


def test_pick_tags_ignores_a_nested_outline(node_available):
    m = lambda i, a, b: f"{{id: {i}, corners: [[{a},{a}],[{b},{a}],[{b},{b}],[{a},{b}]]}}"
    r = run(f"R.pickTags([{m(0, 0, 10)},{m(0, 1, 9)},{m(1, 30, 40)}], [0,1])")
    assert r["ok"] and [t["id"] for t in r["tags"]] == [0, 1]


def test_level_normal_snaps_a_near_vertical_normal(node_available):
    assert run("R.levelNormal([0.02, 1, -0.03])") == [0, 1, 0]
    t = np.radians(9.9); assert run(f"R.levelNormal([{np.sin(t)}, {np.cos(t)}, 0])") == [0, 1, 0]
    t = np.radians(10.1); n = [float(np.sin(t)), float(np.cos(t)), 0]
    assert np.allclose(run(f"R.levelNormal({n})"), n)                  # a real slope is kept
    assert np.allclose(run("R.levelNormal([0, 5, 0])"), [0, 1, 0])
    assert np.allclose(run("R.levelNormal([1, 0, 0])"), [1, 0, 0])
    assert run("R.levelNormal([0.1, 1, 0], 3)") != [0, 1, 0]           # custom tolerance


def test_fit_from_two_accepts_any_quarter_turn(node_available):
    c = "[[-0.4,0.1,-0.25],[0.4,0.1,0.25]]"
    base = run(f"R.fitFromTwo({c}, [[1,0,0],[1,0,0]], [0,1,0])")
    for ax1 in ([0, 0, 1], [-1, 0, 0], [0, 0, -1]):
        r = run(f"R.fitFromTwo({c}, [[1,0,0],{json.dumps(ax1)}], [0,1,0])")
        assert r["angleErrDeg"] == pytest.approx(0, abs=1e-6)
        assert r["width"] == pytest.approx(base["width"]) and r["height"] == pytest.approx(base["height"])
        assert np.allclose(r["u"], base["u"])
    # a quarter turn plus 4 degrees of disagreement is still reported as 4
    r = run(f"R.fitFromTwo({c}, [[1,0,0],[{np.sin(np.radians(4))},0,{np.cos(np.radians(4))}]], [0,1,0])")
    assert r["angleErrDeg"] == pytest.approx(4, abs=1e-6)
    # 45 degrees is as wrong as it gets
    r = run(f"R.fitFromTwo({c}, [[1,0,0],[1,0,1]], [0,1,0])")
    assert r["angleErrDeg"] == pytest.approx(45, abs=1e-6)


def test_turn_toward_picks_the_closest_quarter_turn(node_available):
    for b in ([1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]):
        assert np.allclose(run(f"R.turnToward({json.dumps(b)},[1,0,0],[0,1,0])"), [1, 0, 0])
