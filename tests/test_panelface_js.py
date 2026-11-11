"""preview/panelface.js (M4): the Standard panel face shared by tagar.html and roof.html, drawn into a
recording 2D context under node."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
JS = (PREVIEW / "panelface.js").as_uri()
PANEL = (PREVIEW / "panel.js").as_uri()

# a 2D context that records every call; gradients record their colour stops
REC = """
const calls = [];
const grad = (...a) => { const st = []; calls.push(["grad", a, st]); return { addColorStop: (o, c) => st.push([o, c]) }; };
const g = new Proxy({}, {
  get: (o, k) => k === "createLinearGradient" ? grad : k in o ? o[k] : (...a) => calls.push([k, a]),
  set: (o, k, v) => { calls.push(["set " + String(k), typeof v === "string" ? v : "gradient"]); return true; },
});
"""


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(body, expr):
    code = (f'import * as F from "{JS}"; import * as P from "{PANEL}";' + REC + body
            + f"; console.log(JSON.stringify({expr}));")
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def srgb_to_linear(v):
    c = v / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def test_to_linear_matches_the_srgb_curve(node_available):
    got = run("", 'F.toLinear("#1a2232").concat(F.toLinear("ffffff"), F.toLinear("#000000"), F.toLinear("#0a0b80"))')
    want = [srgb_to_linear(v) for v in (0x1a, 0x22, 0x32, 255, 255, 255, 0, 0, 0, 0x0a, 0x0b, 0x80)]
    assert got == pytest.approx(want, abs=1e-12)
    assert got[3:6] == [1, 1, 1] and got[0] < 0x1a / 255                  # linear is darker than the hex


def test_shade_scales_and_clamps(node_available):
    assert run("", "F.shade([0.5, 0.1, 1], 1)") == "rgb(128,26,255)"
    assert run("", "F.shade([0.5, 0.1, 0.2], 3)") == "rgb(255,77,153)"   # 1.5 clamps to white
    assert run("", "F.shade([0.2, 0.2, 0.2], 0)") == "rgb(0,0,0)"


def test_face_size_keeps_the_module_proportions(node_available):
    assert run("", 'F.faceSize(P.lookByKey("standard"))') == {"W": 1024, "H": round(1024 * 1.755 / 1.038)}
    assert run("", "F.faceSize({w: 1, d: 2}, 300)") == {"W": 300, "H": 600}


def draw(look_expr="P.lookByKey('standard')", W=1024, H=None):
    h = H if H is not None else f"F.faceSize({look_expr}).H"
    return run(f"const look = {look_expr}; const n = F.drawStandardFace(g, look, {W}, {h})", "{n, calls}")


def test_standard_face_draws_every_cell_with_busbars_fingers_and_frame(node_available):
    r = draw()
    look = run("", "P.lookByKey('standard')")
    calls = r["calls"]
    assert r["n"] == look["cols"] * look["rows"] == 120
    H = round(1024 * look["d"] / look["w"])
    assert calls[0] == ["set fillStyle", look["gap"]] and calls[1] == ["fillRect", [0, 0, 1024, H]]   # backsheet
    assert sum(c[0] == "fill" for c in calls) == 120 and sum(c[0] == "clip" for c in calls) == 120
    assert sum(c[0] == "save" for c in calls) == sum(c[0] == "restore" for c in calls) == 120
    # 9 busbars per cell, each 1.2 px wide and the full cell height, in the busbar colour
    bars = [c for c in calls if c[0] == "fillRect" and c[1][2] == 1.2]
    assert len(bars) == 9 * 120
    assert sum(c == ["set fillStyle", look["busbar"]] for c in calls) == 120
    # fingers: 0.6 px strips every 3.2 px down each cell
    fingers = [c for c in calls if c[0] == "fillRect" and c[1][3] == 0.6]
    cells = run("", "P.cellLayout(P.lookByKey('standard'))")
    per = [len([0 for k in range(1000) if cl["y"] * H + 2 + 3.2 * k < (cl["y"] + cl["h"]) * H]) for cl in cells]
    assert len(fingers) == sum(per)
    # frame: four bevelled strips round the edge, each f = 30 mm of the 1038 mm width
    f = 30 / 1000 / look["w"] * 1024
    frame = [c[1] for c in calls if c[0] == "fillRect"][-4:]
    want = [[0, 0, 1024, f], [0, H - f, 1024, f], [0, f, f, H - 2 * f], [1024 - f, f, f, H - 2 * f]]
    assert [v for r in frame for v in r] == pytest.approx([v for r in want for v in r])


def test_cells_follow_the_layout_and_half_cut_chamfers(node_available):
    r = draw()
    calls = r["calls"]
    look = run("", "P.lookByKey('standard')")
    H = round(1024 * look["d"] / look["w"])
    cells = run("", "P.cellLayout(P.lookByKey('standard'))")
    paths, cur = [], None
    for c in calls:
        if c[0] == "moveTo":
            cur = [c[1]]
        elif c[0] == "lineTo":
            cur.append(c[1])
        elif c[0] == "closePath":
            paths.append(cur)
    assert len(paths) == 120
    for i, (cl, p) in enumerate(zip(cells, paths)):
        xs, ys = [q[0] for q in p], [q[1] for q in p]
        assert min(xs) == pytest.approx(cl["x"] * 1024) and max(xs) == pytest.approx((cl["x"] + cl["w"]) * 1024)
        assert min(ys) == pytest.approx(cl["y"] * H) and max(ys) == pytest.approx((cl["y"] + cl["h"]) * H)
        assert len(p) == 6                               # half cut: one edge chamfered (2 corners), one square
        row = i // look["cols"]
        top_y = cl["y"] * H
        on_top = [q for q in p if abs(q[1] - top_y) < 1e-9]
        # even rows: wafer edge on top, so the top corners are cut (the top edge is shorter than the cell)
        span = max(q[0] for q in on_top) - min(q[0] for q in on_top)
        assert (span < cl["w"] * 1024 - 1) == (row % 2 == 0)
        if row % 2 == 0:                                 # the cut is 7 % of the cell width at each corner
            assert span == pytest.approx(cl["w"] * 1024 * (1 - 2 * 0.07))
    full = run("const look = Object.assign({}, P.lookByKey('standard'), {halfCut: false}); F.drawStandardFace(g, look, 1024, 1700)",
               "calls.filter(c => c[0] === 'lineTo').length")
    assert full == 120 * 7                               # whole cells: all 4 corners cut, 8 points


def test_cell_sheen_uses_the_cell_shade_and_linear_colour(node_available):
    r = draw()
    grads = [c for c in r["calls"] if c[0] == "grad"]
    assert len(grads) == 120 + 4
    look = run("", "P.lookByKey('standard')")
    lin = [srgb_to_linear(int(look["cell"][i:i + 2], 16)) for i in (1, 3, 5)]
    for i in (0, 7, 119):
        k = run("", f"P.cellShade({i})")
        stops = grads[i][2]
        assert [s[0] for s in stops] == [0, 0.5, 1]
        for (_, col), m in zip(stops, (0.92, 1.12, 0.95)):
            want = [round(min(1, v * k * m) * 255) for v in lin]
            assert [int(v) for v in re.findall(r"\d+", col)] == want
        x, y, w, h = grads[i][1][0], grads[i][1][1], grads[i][1][2] - grads[i][1][0], grads[i][1][3] - grads[i][1][1]
        assert w > 0 and h > 0                           # diagonal across the cell
    # the frame's lit edges are brighter than its shaded ones
    fr = [srgb_to_linear(int(look["frame"][i:i + 2], 16)) for i in (1, 3, 5)]
    lit, dark = grads[120][2][0][1], grads[121][2][0][1]
    assert lit == "rgb(" + ",".join(str(round(min(1, v * 1.12) * 255)) for v in fr) + ")"
    assert dark == "rgb(" + ",".join(str(round(min(1, v * 0.8) * 255)) for v in fr) + ")"
    assert grads[120][1] == [0, 0, 0, pytest.approx(30 / 1038 * 1024)]
    assert grads[121][1][1] > grads[121][1][3]           # bottom strip: lit from the outer edge inwards


def test_frameless_and_gapless_looks_draw_only_cells(node_available):
    r = draw("Object.assign({}, P.lookByKey('standard'), {frameMm: 0, gap: null, cols: 2, rows: 2, halfCut: false})", 200, 300)
    rects = [c for c in r["calls"] if c[0] == "fillRect"]
    assert r["n"] == 4 and not any(c[1][:2] == [0, 0] and c[1][2] == 200 for c in rects)
    assert len([c for c in r["calls"] if c[0] == "grad"]) == 4


def test_tagar_uses_the_shared_face():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert 'from "./panelface.js' in html and "drawStandardFace(c.getContext(\"2d\"), look, W, H)" in html
    assert "createLinearGradient" not in html.split("function panelTexture")[1].split("\n}\n")[0]
