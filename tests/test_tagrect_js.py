"""Tabletop AR on 2, 3 or 4 AprilTags: unit (preview/tagrect.js through node), smoke (tagar.html wiring),
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
    return [float(W / 2 + F * X / -Z), float(H / 2 - F * Y / -Z)]


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


def sq(i, x=0, size=1):
    return f"{{id: {i}, corners: [[{x},0],[{x + size},0],[{x + size},{size}],[{x},{size}]]}}"


def test_choose_tags_takes_2_3_or_4_of_any_ids(node_available):
    for ids in ((7, 2), (30, 4, 11), (9, 0, 5, 3)):
        r = run(f"R.chooseTags([{','.join(sq(i, 5 * k) for k, i in enumerate(ids))}])")
        assert r["ok"] and r["found"] == len(ids) and [t["id"] for t in r["tags"]] == sorted(ids)
    one = run(f"R.chooseTags([{sq(4)}])")
    assert not one["ok"] and one["found"] == 1 and "1 tag in view" in one["why"]
    none = run("R.chooseTags([])")
    assert not none["ok"] and none["found"] == 0 and none["why"] == "No tags in view"


def test_choose_tags_keeps_the_biggest_four(node_available):
    five = [sq(1, 0, 3), sq(2, 10, 1), sq(3, 20, 5), sq(4, 30, 4), sq(5, 40, 2)]   # id 2 is the smallest
    r = run(f"R.chooseTags([{','.join(five)}])")
    assert r["ok"] and [t["id"] for t in r["tags"]] == [1, 3, 4, 5]
    assert [t["id"] for t in run(f"R.chooseTags([{','.join(five)}], 2)")["tags"]] == [3, 4]


def test_choose_tags_refuses_a_repeated_id(node_available):
    dup = run(f"R.chooseTags([{sq(0)},{sq(0, 9)},{sq(1, 20)}])")      # the same id printed twice, far apart
    assert not dup["ok"] and dup["found"] == 3 and "id 0 seen twice" in dup["why"]


def test_most_seen_counts_only_recent_frames(node_available):
    recent = "[{at: 0, n: 4}, {at: 1000, n: 2}, {at: 1600, n: 3}]"
    assert run(f"R.mostSeen({recent}, 1500)") == 4
    assert run(f"R.mostSeen({recent}, 1501)") == 3                    # the 4 is 1501 ms old
    assert run(f"R.mostSeen({recent}, 2000, 400)") == 3
    assert run("R.mostSeen([], 0)") == 0


def test_fit_from_three_any_order_and_rotation(node_available):
    a = np.radians(25)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    full = np.array([[-0.4, 0, -0.25], [0.4, 0, -0.25], [0.4, 0, 0.25], [-0.4, 0, 0.25]]) @ R.T + [1, 0.7, 2]
    for drop in range(4):
        for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)):
            three = np.delete(full, drop, axis=0)[list(order)].tolist()
            r = run(f"R.fitFromThree({three}, [0, 1, 0])")
            assert r["width"] == pytest.approx(0.8) and r["height"] == pytest.approx(0.5)
            assert np.allclose(r["center"], [1, 0.7, 2]) and r["angleErrDeg"] < 1e-6
            assert np.allclose(sorted(map(tuple, np.round(r["corners"], 9))), sorted(map(tuple, np.round(full, 9))))


def test_fit_from_three_reports_a_skewed_corner(node_available):
    r = run("R.fitFromThree([[0,0,0],[1,0,0],[0.2,0,0.5]], [0,1,0])")    # best corner ~80 deg, not 90
    assert 5 < r["angleErrDeg"] < 20
    with pytest.raises(RuntimeError, match="need 3"):
        run("R.fitFromThree([[0,0,0],[1,0,0]], [0,1,0])")


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
    with pytest.raises(RuntimeError, match="need 2"):
        run("R.fitFromTwo([[0,0,0],[1,0,0]], [[1,0,0]], [0,1,0])")
    with pytest.raises(RuntimeError, match="on top of each other"):
        run("R.fitFromTwo([[0,0,0],[0.01,0,0.01]], [[1,0,0],[1,0,0]], [0,1,0])")


def test_fit_from_two_side_by_side_gives_the_square_on_their_line(node_available):
    ax = "[[1,0,0],[1,0,0]]"
    for c in ("[[0,0,0],[0.8,0,0.01]]", "[[0,0,0],[0.01,0,0.8]]", "[[0.8,0,0],[0,0,0]]"):
        r = run(f"R.fitFromTwo({c}, {ax}, [0,1,0])")
        L = np.linalg.norm(np.subtract(*json.loads(c)))
        assert r["side"] is True and r["width"] == pytest.approx(L) and r["height"] == pytest.approx(L)
        assert np.allclose(r["center"], np.mean(json.loads(c), axis=0))            # no toward: on the line
        assert np.allclose(np.cross(r["u"], r["n"]), r["v"])
    # toward picks the side: the square grows away from the line toward that point
    for tz, cz in ((1, 0.4), (-2, -0.4)):
        r = run(f"R.fitFromTwo([[-0.4,0,0],[0.4,0,0]], {ax}, [0,1,0], [0.1,0.5,{tz}])")
        assert np.allclose(r["center"], [0, 0, cz])
        assert np.allclose(sorted(map(tuple, np.round(r["corners"], 9))),
                           sorted([(-0.4, 0, 0), (-0.4, 0, 2 * cz), (0.4, 0, 0), (0.4, 0, 2 * cz)]))
    # toward exactly on the line still picks a side
    r = run(f"R.fitFromTwo([[-0.4,0,0],[0.4,0,0]], {ax}, [0,1,0], [3,0,0])")
    assert abs(r["center"][2]) == pytest.approx(0.4)


def test_side_ratio_is_the_cutoff(node_available):
    k = run("R.SIDE_RATIO")
    assert run(f"R.fitFromTwo([[0,0,0],[1,0,{k * 0.99}]], [[1,0,0],[1,0,0]], [0,1,0])").get("side") is True
    assert "side" not in run(f"R.fitFromTwo([[0,0,0],[1,0,{k * 1.01}]], [[1,0,0],[1,0,0]], [0,1,0])")
    assert "side" not in run("R.fitFromTwo([[0,0,0],[1,0,0.2]], [[1,0,0],[1,0,0]], [0,1,0])")   # a 1 x 0.2 strip is a diagonal


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
                    "averageTwo(history)", "tagsMoved(rect.tags, r.tags)", "chooseTags(", "fitFromThree(",
                    "mostSeen(", "renderer.resetState()", "shadowMap.enabled = true", "ShadowMaterial",
                    "buildRack(", "mountParts("):
        assert feature in html, feature
    assert 'url=tagar.html' in (PREVIEW / "index.html").read_text(encoding="utf-8")   # the site root opens the AR


def test_camera_read_hands_the_xr_target_back():
    # the blink: resetState drops the XR layer target, so the frame rendered to the hidden canvas every camera read
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    body = html[html.index("function readCamera("):html.index("function detectRect(")]
    save = body.index("target = renderer.getRenderTarget()")
    raw = body.index("gl.bindFramebuffer(gl.FRAMEBUFFER, fb)")
    reset = body.index("renderer.resetState();")
    back = body.index("renderer.setRenderTarget(target);")
    assert save < raw < reset < back < body.index("return { w, h };")


def test_page_locks_on_a_wall():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    body = html[html.index("function detectRect("):]
    body = body[:body.index("\nfunction ")]
    # before the lock the plane comes from the tags' printed size, checked against the hit-test plane
    assert "surfacePlane(pick.tags.map((t) => tagCorners3D(t.corners, tagSize, cam.w, cam.h, proj, pose)), lastHit, pose.slice(12, 15))" in body
    assert 'catch (e) { return "Hold still so the tags are sharp"; }' in body
    assert "if (rect) { n = rect.n; p0 = rect.center; }" in body
    assert body.index("r = plumbAxes(r);") < body.index("r.ids = key;")
    assert "lastHit[" not in body and "hp[" not in body                 # the hit pose is no longer read directly
    # no detected surface is fine: a plain wall rarely gets one
    assert "} else if (!lastHit) {" not in html and "Move the phone slowly to find the table" not in html
    assert "on a table or a wall: aim between the tags" in html


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
  const pick = R.chooseTags(found.map((m) => ({ id: m.id, corners: m.corners.map((c) => [c.x * img.factor, c.y * img.factor]) })));
  if (!pick.ok) { console.log(JSON.stringify(pick)); return; }
  const n = [job.hit[4], job.hit[5], job.hit[6]], p0 = [job.hit[12], job.hit[13], job.hit[14]];
  const centres = [], edges = [], axes = [];
  for (const t of pick.tags) {
    const w = t.corners.map(([x, y]) => R.rayPlane(R.pixelRay(x, y, job.w, job.h, job.proj, job.pose), p0, n));
    centres.push(w.reduce((a, p) => a.map((v, i) => v + p[i] / 4), [0, 0, 0]));
    edges.push(R.tagEdge(w));
    axes.push(R.tagAxis(w));
  }
  const k = centres.length;
  const rect = k === 2 ? R.fitFromTwo(centres, axes, n, job.toward) : k === 3 ? R.fitFromThree(centres, n) : R.fitRectangle(centres, n);
  console.log(JSON.stringify({ ok: true, ids: pick.tags.map((t) => t.id), factor: img.factor, rect, edges }));
});
"""

CENTRES = {0: [-0.4, -0.25], 1: [0.4, -0.25], 2: [0.4, 0.25], 3: [-0.4, 0.25]}


def table_photo(tmp_path, ids=(0, 1, 2, 3), max_pixels=2.6e6, corners=(0, 1, 2, 3), toward=None):
    """Tags (0.16 m black square) at the given rectangle corners on the plane y = 0, upright along +x,
    photographed from EYE; RGBA bottom-up. `toward` = the side point for 2 tags side by side."""
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
    job = {"w": W, "h": H, "proj": col(projection()), "pose": col(c2w), "hit": col(hit), "maxPixels": max_pixels, "toward": toward}
    (tmp_path / "job.json").write_text(json.dumps(job))


def chain(tmp_path):
    out = subprocess.run(["node", "-e", CHAIN, str(PREVIEW), str(tmp_path / "job.json"), str(tmp_path / "frame.bin")],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


WALL_CHAIN = """
require(process.argv[1] + "/vendor/js-aruco2/cv.js");
const AR = require(process.argv[1] + "/tag36h11.js");
const fs = require("fs");
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const raw = fs.readFileSync(process.argv[3]);
import(require("url").pathToFileURL(process.argv[1] + "/tagrect.js").href).then((R) => {
  const img = R.downsample(new Uint8Array(raw.buffer, raw.byteOffset, raw.length), job.w, job.h, true, job.maxPixels);
  const found = new AR.Detector({ dictionaryName: "APRILTAG_36h11" }).detect(img);
  const pick = R.chooseTags(found.map((m) => ({ id: m.id, corners: m.corners.map((c) => [c.x * img.factor, c.y * img.factor]) })));
  if (!pick.ok) { console.log(JSON.stringify(pick)); return; }
  const worlds = pick.tags.map((t) => R.tagCorners3D(t.corners, 0.16, job.w, job.h, job.proj, job.pose));
  const { n, p0, fromHit } = R.surfacePlane(worlds, job.hit, job.pose.slice(12, 15));
  const eye = job.pose.slice(12, 15), avg = (w) => w.reduce((a, p) => a.map((v, i) => v + p[i] / 4), [0, 0, 0]);
  const centres = worlds.map((w) => R.rayPlane({ origin: eye, dir: avg(w).map((v, i) => v - eye[i]) }, p0, n));
  const rect = R.plumbAxes(R.fitRectangle(centres, n));
  console.log(JSON.stringify({ ok: true, fromHit, n, rect, edges: worlds.map((w) => R.tagEdge(w)) }));
});
"""


def test_wall_photo_locks_from_the_tags_alone(tmp_path, node_available):
    """4 tags on the wall z = 0 (0.8 x 0.5 m apart), the hit-test plane on the floor below or absent: the
    tags' own plane (from their printed size) gives the true rectangle, plumb, with v pointing down."""
    eye = [0.15, 1.35, 1.1]
    c2w = cam_to_world(eye, [0, 1.2, 0])
    img = np.full((H, W), 170, np.uint8)
    for tid, (cx, cy) in zip((0, 1, 2, 3), ([-0.4, 1.45], [0.4, 1.45], [0.4, 0.95], [-0.4, 0.95])):
        t = T.tag_image(tid, cell_px=20)
        s = t.shape[0]
        world = [[cx - 0.1, cy + 0.1, 0], [cx + 0.1, cy + 0.1, 0], [cx + 0.1, cy - 0.1, 0], [cx - 0.1, cy - 0.1, 0]]
        dst = np.float32([to_pixel(c2w, p) for p in world])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]), dst)
        mask = cv2.warpPerspective(np.full_like(t, 255), M, (W, H)) > 0
        img[mask] = cv2.warpPerspective(t, M, (W, H))[mask]
    (tmp_path / "frame.bin").write_bytes(cv2.cvtColor(img[::-1], cv2.COLOR_GRAY2RGBA).tobytes())
    for hit in (None, hit_pose([0, 0, 0.8], [0, 1, 0])):               # nothing found, or the floor below
        job = {"w": W, "h": H, "proj": col(projection()), "pose": col(c2w), "hit": hit, "maxPixels": 2.6e6}
        (tmp_path / "job.json").write_text(json.dumps(job))
        out = subprocess.run(["node", "-e", WALL_CHAIN, str(PREVIEW), str(tmp_path / "job.json"), str(tmp_path / "frame.bin")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        r = json.loads(out.stdout)
        assert r["ok"] and r["fromHit"] is False and np.allclose(r["n"], [0, 0, 1], atol=0.03)   # within 1.7 degrees
        assert r["rect"]["width"] == pytest.approx(0.8, abs=0.015)
        assert r["rect"]["height"] == pytest.approx(0.5, abs=0.015)
        assert np.allclose(r["rect"]["center"], [0, 1.2, 0], atol=0.015)
        assert np.allclose(r["rect"]["v"], [0, -1, 0], atol=0.02) and np.allclose(r["rect"]["u"], [1, 0, 0], atol=0.02)
        assert np.allclose(r["edges"], 0.16, atol=0.008)


def test_table_photo_gives_the_true_rectangle(tmp_path, node_available):
    table_photo(tmp_path)
    r = chain(tmp_path)
    assert r["ok"] and r["factor"] == 1                     # 1.2 MP frame: kept whole
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01)
    assert r["rect"]["height"] == pytest.approx(0.5, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0], atol=0.01)
    assert r["rect"]["angleErrDeg"] < 2
    assert np.allclose(r["edges"], 0.16, atol=0.008)        # tag size check agrees with the print


def test_table_photo_with_any_ids_and_three_tags(tmp_path, node_available):
    table_photo(tmp_path, ids=(21, 5, 13), corners=(1, 2, 3))           # corner 0 left empty
    r = chain(tmp_path)
    assert r["ok"] and r["ids"] == [5, 13, 21]
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01)
    assert r["rect"]["height"] == pytest.approx(0.5, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0], atol=0.01)


def test_table_photo_with_a_repeated_tag_is_refused(tmp_path, node_available):
    table_photo(tmp_path, ids=(0, 1, 2, 0))
    r = chain(tmp_path)
    assert r["ok"] is False and "id 0 seen twice" in r["why"]


def test_table_photo_with_two_tags_gives_the_true_rectangle(tmp_path, node_available):
    table_photo(tmp_path, ids=(8, 3), corners=(0, 2))                  # opposite corners, any ids
    r = chain(tmp_path)
    assert r["ok"]
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01)
    assert r["rect"]["height"] == pytest.approx(0.5, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0], atol=0.01)
    assert np.allclose(np.abs(r["rect"]["u"]), [1, 0, 0], atol=0.02)
    assert r["rect"]["angleErrDeg"] < 2


def test_two_tags_along_one_side_give_the_square_away_from_the_camera(tmp_path, node_available):
    # tags at the two front corners (z = 0.25, nearer EYE); the page passes toward = 2 * mid - eye
    mid = np.array([0, 0, 0.25])
    table_photo(tmp_path, ids=(0, 1), corners=(3, 2), toward=(2 * mid - EYE).tolist())
    r = chain(tmp_path)
    assert r["ok"] and r["rect"]["side"] is True
    assert r["rect"]["width"] == pytest.approx(0.8, abs=0.01) and r["rect"]["height"] == pytest.approx(0.8, abs=0.01)
    assert np.allclose(r["rect"]["center"], [0, 0, 0.25 - 0.4], atol=0.01)


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


def test_choose_tags_ignores_a_nested_outline(node_available):
    m = lambda i, a, b: f"{{id: {i}, corners: [[{a},{a}],[{b},{a}],[{b},{b}],[{a},{b}]]}}"
    r = run(f"R.chooseTags([{m(0, 0, 10)},{m(0, 1, 9)},{m(1, 30, 40)}])")
    assert r["ok"] and [t["id"] for t in r["tags"]] == [0, 1]


def test_level_normal_snaps_a_near_vertical_normal(node_available):
    assert run("R.levelNormal([0.02, 1, -0.03])") == [0, 1, 0]
    t = np.radians(9.9); assert run(f"R.levelNormal([{np.sin(t)}, {np.cos(t)}, 0])") == [0, 1, 0]
    t = np.radians(10.1); n = [float(np.sin(t)), float(np.cos(t)), 0]
    assert np.allclose(run(f"R.levelNormal({n})"), n)                  # a real slope is kept
    assert np.allclose(run("R.levelNormal([0, 5, 0])"), [0, 1, 0])
    assert np.allclose(run("R.levelNormal([1, 0, 0])"), [1, 0, 0])
    assert run("R.levelNormal([0.1, 1, 0], 3)") != [0, 1, 0]           # custom tolerance


def test_level_normal_plumbs_a_near_horizontal_normal(node_available):
    t = np.radians(9.9); n = run(f"R.levelNormal([{np.cos(t)}, {np.sin(t)}, 0])")
    assert np.allclose(n, [1, 0, 0])                                     # a wall leaning 9.9 degrees: plumb
    t = np.radians(9.9); n = run(f"R.levelNormal([0, {-np.sin(t)}, {2 * np.cos(t)}])")
    assert np.allclose(n, [0, 0, 1])                                     # leaning the other way, not unit length
    t = np.radians(10.1); n = [float(np.cos(t)), float(np.sin(t)), 0]
    assert np.allclose(run(f"R.levelNormal({n})"), n)                    # a real slope is kept
    assert np.allclose(run("R.levelNormal([0.6, 0.08, 0.8])"), [0.6, 0, 0.8])   # the heading is kept


def test_on_wall_splits_at_45_degrees(node_available):
    for n, want in (([0, 1, 0], False), ([0, 0, 1], True), ([1, 0, 0], True), ([0, -1, 0], True),
                    ([0, 1.01, 1], False), ([0, 0.99, 1], True), ([0, 5, 0.1], False)):
        assert run(f"R.onWall({n})") is want, n


def test_homography_maps_the_four_points(node_available):
    src = [[-1, 1], [1, 1], [1, -1], [-1, -1]]
    Hm = np.array([[0.9, 0.2, 0.1], [-0.1, 1.1, 0.3], [0.05, -0.02, 1]])
    dst = [((Hm @ [x, y, 1])[:2] / (Hm @ [x, y, 1])[2]).tolist() for x, y in src]
    h = np.array(run(f"R.homography({src}, {dst})")).reshape(3, 3)
    assert np.allclose(h, Hm, atol=1e-9) and h[2, 2] == 1
    p = h @ [0.3, -0.4, 1]
    assert np.allclose(p[:2] / p[2], (Hm @ [0.3, -0.4, 1])[:2] / (Hm @ [0.3, -0.4, 1])[2])   # any point, not just the 4
    # a pivot is needed: the first column of the system starts with zeros
    assert np.allclose(run("R.homography([[0,0],[1,0],[1,1],[0,1]], [[0,0],[2,0],[2,2],[0,2]])"),
                       [2, 0, 0, 0, 2, 0, 0, 0, 1], atol=1e-12)
    with pytest.raises(RuntimeError, match="degenerate"):
        run("R.homography([[0,0],[1,1],[2,2],[3,3]], [[0,0],[1,0],[1,1],[0,1]])")


def wall_tag(center, size, yaw_deg=0):
    """World corners (printed TL, TR, BR, BL) of an upright tag on the wall z = 0 facing +z, as lists."""
    a, h = np.radians(yaw_deg), size / 2
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return [[*(np.array(center) + R @ c).tolist(), 0.0] for c in ([-h, h], [h, h], [h, -h], [-h, -h])]


def test_tag_corners_3d_recovers_the_tag_from_its_pixels(node_available):
    eye, size = [0.25, 1.35, 0.9], 0.16
    c2w = cam_to_world(eye, [0, 1.2, 0])
    for center, yaw in (([0.3, 1.4], 0), ([-0.3, 1.0], 30), ([0.05, 1.2], -90)):
        world = wall_tag(center, size, yaw)
        px = [to_pixel(c2w, p) for p in world]
        got = np.array(run(f"R.tagCorners3D({px}, {size}, {W}, {H}, {col(projection())}, {col(c2w)})"))
        assert np.allclose(got, world, atol=1e-6), (center, yaw)
    # the scale comes from the printed size: the same pixels with a tag twice as big lie twice as far
    world = wall_tag([0.3, 1.4], size)
    px = [to_pixel(c2w, p) for p in world]
    big = np.array(run(f"R.tagCorners3D({px}, {2 * size}, {W}, {H}, {col(projection())}, {col(c2w)})"))
    assert np.allclose(big - eye, 2 * (np.array(world) - eye), atol=1e-6)
    # on a table too, seen from above
    c2w = cam_to_world(EYE, TARGET)
    table = [[x, 0, z] for x, z in ([0.1, -0.2], [0.26, -0.2], [0.26, -0.04], [0.1, -0.04])]
    got = run(f"R.tagCorners3D({[to_pixel(c2w, p) for p in table]}, {size}, {W}, {H}, {col(projection())}, {col(c2w)})")
    assert np.allclose(got, table, atol=1e-6)


def test_fit_plane(node_available):
    rng = np.random.default_rng(3)
    for n in ([0, 0, 1], [0, 1, 0], [1, 0, 0], [0.3, 0.8, -0.5]):
        n = np.array(n) / np.linalg.norm(n)
        a = np.cross(n, [0.6, 0.0, 0.8] if abs(n[1]) > 0.5 else [0, 1, 0]); a /= np.linalg.norm(a)
        b = np.cross(n, a)
        pts = [(np.array([1, 2, 3]) + s * a + t * b).tolist() for s, t in rng.uniform(-1, 1, (12, 2))]
        r = run(f"R.fitPlane({pts})")
        assert abs(np.dot(r["n"], n)) == pytest.approx(1, abs=1e-9), n
        assert np.allclose(r["center"], np.mean(pts, axis=0))
    with pytest.raises(RuntimeError, match="need 3 points"):
        run("R.fitPlane([[0,0,0],[1,0,0]])")
    with pytest.raises(RuntimeError, match="in a line"):
        run("R.fitPlane([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])")


def hit_pose(pos, normal):
    """A hit-test pose (16 numbers, column-major) whose y axis is the surface normal."""
    m = np.eye(4); m[:3, 1] = np.array(normal) / np.linalg.norm(normal); m[:3, 3] = pos
    return col(m)


def test_surface_plane_takes_the_tags_plane_on_a_wall(node_available):
    tags = [wall_tag(c, 0.16) for c in ([-0.3, 1.0], [0.3, 1.4])]
    eye = [0.1, 1.3, 1.2]
    for hit in ("null", hit_pose([0, 0, 0.8], [0, 1, 0])):                 # no surface, or the floor below
        r = run(f"R.surfacePlane({tags}, {hit}, {eye})")
        assert r["fromHit"] is False and np.allclose(r["n"], [0, 0, 1]) and r["p0"][2] == pytest.approx(0, abs=1e-12)
    r = run(f"R.surfacePlane({tags}, null, [0.1, 1.3, -1.2])")           # seen from behind: faces the eye
    assert np.allclose(r["n"], [0, 0, -1])
    # a hit on the wall itself agrees, and wins (it is steadier), turned to face the eye
    r = run(f"R.surfacePlane({tags}, {hit_pose([0, 1.2, 0.001], [0, 0, -1])}, {eye})")
    assert r["fromHit"] is True and np.allclose(r["n"], [0, 0, 1]) and np.allclose(r["p0"], [0, 1.2, 0.001])


def test_surface_plane_keeps_the_hit_on_a_table(node_available):
    flat = lambda x, z: [[x - 0.08, 0.002, z - 0.08], [x + 0.08, 0, z - 0.08], [x + 0.08, -0.002, z + 0.08], [x - 0.08, 0, z + 0.08]]
    tags = [flat(-0.3, 0.1), flat(0.3, -0.1)]
    hit = hit_pose([0.01, 0, 0.02], [0.05, 1, 0])                         # 3 degrees off: still level
    r = run(f"R.surfacePlane({tags}, {hit}, {EYE})")
    assert r["fromHit"] is True and r["n"] == [0, 1, 0] and np.allclose(r["p0"], [0.01, 0, 0.02])
    hit = hit_pose([0, 0, 0], [0.5, 1, 0])                                # 27 degrees off: the tags win
    r = run(f"R.surfacePlane({tags}, {hit}, {EYE})")
    assert r["fromHit"] is False and r["n"] == [0, 1, 0]
    assert run(f"R.surfacePlane({tags}, {hit}, {EYE}, 30)")["fromHit"] is True   # custom tolerance


def test_plumb_axes_points_v_down_the_wall(node_available):
    base = {"n": [0, 0, 1], "center": [0, 1.2, 0], "width": 0.6, "height": 0.4}
    for u, swap in (([1, 0, 0], False), ([0, 1, 0], True), ([-1, 0, 0], False), ([0, -1, 0], True)):
        r = run(f"R.plumbAxes({json.dumps({**base, 'u': u})})")
        assert np.allclose(r["v"], [0, -1, 0]) and np.allclose(r["u"], [1, 0, 0]), u
        assert (r["width"], r["height"]) == ((0.4, 0.6) if swap else (0.6, 0.4)), u
        assert r["center"] == base["center"]
    t = {"n": [0, 1, 0], "u": [0, 0, 1], "v": [1, 0, 0], "width": 0.6, "height": 0.4}
    assert run(f"R.plumbAxes({json.dumps(t)})") == t                       # a table is left as it was


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
