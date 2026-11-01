"""panel.js: the one movable solar panel in tagar.html (looks, cell layout, touch math), run through node."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
JS = (PREVIEW / "panel.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as P from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_looks_are_real_module_sizes(node_available):
    looks = run("P.PANEL_LOOKS")
    assert [l["key"] for l in looks] == ["standard", "bipv"]
    for l in looks:
        assert 0.9 < l["w"] < 1.2 and 1.4 < l["d"] < 2.4 and l["d"] > l["w"]    # portrait, metres
    assert run('P.lookByKey("bipv").name') == "BIPV brick"
    assert [l["key"] for l in looks if l.get("texture")] == ["bipv"]
    assert run('P.lookByKey("nope").key') == looks[0]["key"]


def test_next_look_wraps_both_ways(node_available):
    keys = [l["key"] for l in run("P.PANEL_LOOKS")]
    assert run(f'P.nextLook("{keys[0]}")') == keys[1]
    assert run(f'P.nextLook("{keys[-1]}", 1)') == keys[0]
    assert run(f'P.nextLook("{keys[0]}", -1)') == keys[-1]
    assert run('P.nextLook("nope", 1)') == keys[1]


@pytest.mark.parametrize("key", ["standard"])
def test_cell_layout_fills_inside_the_frame(key, node_available):
    look = run(f'P.lookByKey("{key}")')
    cells = run(f'P.cellLayout(P.lookByKey("{key}"))')
    assert len(cells) == look["cols"] * look["rows"]
    fx, fy = look["frameMm"] / 1000 / look["w"], look["frameMm"] / 1000 / look["d"]
    xs = [c["x"] for c in cells] + [c["x"] + c["w"] for c in cells]
    ys = [c["y"] for c in cells] + [c["y"] + c["h"] for c in cells]
    assert min(xs) == pytest.approx(fx) and max(xs) == pytest.approx(1 - fx)    # flush with the frame
    assert min(ys) == pytest.approx(fy) and max(ys) == pytest.approx(1 - fy)
    gap_y = 0.002 / look["d"]
    rows = sorted({round(c["y"], 9) for c in cells})
    steps = np.diff(rows) - cells[0]["h"]
    mid = 0.012 / look["d"] if look["halfCut"] else 0
    assert np.allclose(steps[: look["rows"] // 2 - 1], gap_y)
    assert steps[look["rows"] // 2 - 1] == pytest.approx(gap_y + mid)        # the half-cut split
    assert np.allclose(np.diff(sorted({round(c["x"], 9) for c in cells})) - cells[0]["w"], 0.002 / look["w"])


def test_bipv_is_one_frameless_textured_face_in_the_pictures_proportions(node_available):
    from PIL import Image
    look = run('P.lookByKey("bipv")')
    img = Image.open(PREVIEW / look["texture"])
    assert look["frameMm"] == 0 and (look["cols"], look["rows"]) == (1, 1)
    assert look["w"] / look["d"] == pytest.approx(img.width / img.height, abs=0.005)      # bricks not stretched
    assert run('P.cellLayout(P.lookByKey("bipv"))') == [{"x": 0, "y": 0, "w": 1, "h": 1}]


def test_to_panel_frame_and_hit(node_available):
    pan = "{x: 1, z: 2, yaw: 0}"
    assert np.allclose(run(f"P.toPanelFrame([1.3, 2.4], {pan})"), [0.3, 0.4])
    # yaw 90 deg: rotation.y takes local +u to world -z
    assert np.allclose(run(f"P.toPanelFrame([1, 1.5], {{x: 1, z: 2, yaw: Math.PI / 2}})"), [0.5, 0], atol=1e-12)
    look = '{w: 1, d: 2}'
    assert run(f"P.panelHit([1.45, 2.9], {pan}, {look})") is True
    assert run(f"P.panelHit([1.55, 2], {pan}, {look})") is False
    assert run(f"P.panelHit([1.55, 2], {pan}, {look}, 0.1)") is True        # slop widens the target
    assert run(f"P.panelHit([1, 3.05], {pan}, {look})") is False
    assert run(f"P.panelHit([1, 3.05], {pan}, {look}, 0.1)") is True
    # turned 90 deg the long side lies along x
    assert run(f"P.panelHit([1.9, 2], {{x: 1, z: 2, yaw: Math.PI / 2}}, {look})") is True
    assert run(f"P.panelHit([1, 2.9], {{x: 1, z: 2, yaw: Math.PI / 2}}, {look})") is False


def test_twist_delta_matches_rotation_y(node_available):
    # fingers turn 90 deg counter-clockwise seen from above (x right, z toward the viewer is down the page)
    d = run("P.twistDelta([0,0], [1,0], [0,0], [0,-1])")
    assert d == pytest.approx(np.pi / 2)
    # rotation.y by d takes the old finger direction to the new one
    c, s = np.cos(d), np.sin(d)
    assert np.allclose([c * 1 + s * 0, -s * 1 + c * 0], [0, -1])
    assert run("P.twistDelta([5,5], [6,5], [7,7], [8,7])") == pytest.approx(0)   # pure drag: no turn
    # across the +-180 seam it takes the short way round
    assert run("P.twistDelta([0,0], [-1,0.01], [0,0], [-1,-0.01])") == pytest.approx(-0.02, abs=1e-4)


def test_long_press(node_available):
    assert run("P.isLongPress(0, 450, 0)") is True
    assert run("P.isLongPress(0, 449, 0)") is False
    assert run("P.isLongPress(0, 900, 12)") is True
    assert run("P.isLongPress(0, 900, 13)") is False
    assert run("P.isLongPress(0, 200, 0, 150, 5)") is True


def test_clamp_and_yaw_deg(node_available):
    box = "{minX: 0, maxX: 10, minZ: -2, maxZ: 2}"
    assert run(f"P.clampToBox([5, 0], {box})") == [5, 0]
    assert run(f"P.clampToBox([-1, 3], {box})") == [0, 2]
    assert run(f"P.clampToBox([11, -3], {box})") == [10, -2]
    assert run("P.yawDeg(Math.PI / 2)") == 90
    assert run("P.yawDeg(-Math.PI / 12)") == 345
    assert run("P.yawDeg(4 * Math.PI)") == 0


def test_page_uses_one_panel_not_the_grid():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert "GLTFLoader" not in html and "terrace_tilted_ar.glb" not in html
    for hook in ("TextureLoader", "look.texture", "buildPanel", "panelTexture", "cellLayout", "isLongPress", "twistDelta", 'id="o-look"', 'id="looks"'):
        assert hook in html, hook


def grey(vals, w, h):
    return [v for x in vals for v in (x, x, x, 255)]


def hm(vals, w, h, blur=0, rim=0):
    return run(f"Array.from(P.heightMap({json.dumps(grey(vals, w, h))}, {w}, {h}, {blur}, {rim}))")[0::4]


def test_height_map_stretches_dark_joints_low_and_bright_faces_high(node_available):
    assert hm([50, 100, 150, 100], 2, 2) == [0, 128, 255, 128]          # 127.5 rounds to 128 in a clamped byte


def test_height_map_blur_bevels_an_edge(node_available):
    vals = [0, 0, 60, 60, 60]
    assert hm(vals, 5, 1) == [0, 0, 255, 255, 255]
    assert hm(vals, 5, 1, blur=1) == [0, 85, 170, 255, 255]            # box blur: the step becomes a slope


def test_height_map_rim_is_forced_low(node_available):
    out = hm([100, 200] * 8, 4, 4, rim=1)                                # columns alternate 100 / 200
    rows = [out[r * 4:(r + 1) * 4] for r in range(4)]
    assert rows[0] == [0, 0, 0, 0] and rows[3] == [0, 0, 0, 0]
    assert rows[1] == [0, 255, 0, 0] and rows[2] == [0, 255, 0, 0]


def test_height_map_flat_image_does_not_divide_by_zero(node_available):
    assert hm([77] * 4, 2, 2) == [0, 0, 0, 0]


def test_height_map_rejects_wrong_size(node_available):
    with pytest.raises(RuntimeError, match="w x h"):
        run("P.heightMap([0,0,0,255], 2, 2)")


def test_border_px_scales_with_face_width(node_available):
    assert run("P.borderPx(14, 1.1, 512)") == 7
    assert run("P.borderPx(0, 1.1, 512)") == 0


def test_bipv_is_glass_laminate_with_ceramic_border_and_relief(node_available):
    b = run('P.lookByKey("bipv")')
    assert 9 <= b["thickMm"] <= 13 and 8 <= b["borderMm"] <= 25 and b["relief"] > 0 and b["edge"].startswith("#")
    assert "thickMm" not in run('P.lookByKey("standard")')


def test_page_wires_glass_laminate_look():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ("MeshPhysicalMaterial", "heightMap", "borderPx", "RoomEnvironment", "clearcoat", "bumpMap", "closeUp", "panelWorld"):
        assert hook in html, hook
