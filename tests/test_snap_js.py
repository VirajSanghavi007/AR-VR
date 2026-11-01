"""snap.js: the Photo button in tagar.html (camera + rendered panel composite, caption, file name), run through node."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
SNAP = (PREVIEW / "snap.js").as_uri()
PANEL = (PREVIEW / "panel.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, pre=""):
    code = f'import * as S from "{SNAP}"; import * as P from "{PANEL}"; {pre} console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def px(rows):
    """Flat RGBA list from rows of (r, g, b, a) pixels."""
    return [v for row in rows for p in row for v in p]


def comp(cam, scene, w, h, flip=True):
    return run(f"Array.from(S.composite({json.dumps(cam)}, {json.dumps(scene)}, {w}, {h}, {str(flip).lower()}))")


def test_clear_scene_shows_the_camera_flipped_upright(node_available):
    cam = px([[(10, 20, 30, 255)], [(40, 50, 60, 255)]])          # bottom-up: row 0 is the bottom
    clear = px([[(0, 0, 0, 0)], [(0, 0, 0, 0)]])
    assert comp(cam, clear, 1, 2) == px([[(40, 50, 60, 255)], [(10, 20, 30, 255)]])


def test_camera_already_top_down_is_kept(node_available):
    cam = px([[(10, 20, 30, 255)], [(40, 50, 60, 255)]])
    clear = px([[(0, 0, 0, 0)], [(0, 0, 0, 0)]])
    assert comp(cam, clear, 1, 2, flip=False) == cam


def test_opaque_scene_covers_the_camera_and_is_flipped(node_available):
    cam = px([[(200, 200, 200, 255)], [(200, 200, 200, 255)]])
    scene = px([[(1, 2, 3, 255)], [(4, 5, 6, 255)]])
    assert comp(cam, scene, 1, 2) == px([[(4, 5, 6, 255)], [(1, 2, 3, 255)]])


def test_half_transparent_scene_blends_premultiplied(node_available):
    cam = px([[(200, 100, 0, 255)]])
    scene = px([[(50, 50, 50, 128)]])                              # premultiplied: colour already x alpha
    k = 1 - 128 / 255
    assert comp(cam, scene, 1, 1) == [round(50 + 200 * k), round(50 + 100 * k), 50, 255]


def test_columns_stay_in_place(node_available):
    cam = px([[(1, 0, 0, 255), (2, 0, 0, 255), (3, 0, 0, 255)]])
    clear = px([[(0, 0, 0, 0)] * 3])
    assert comp(cam, clear, 3, 1)[0::4] == [1, 2, 3]


def test_size_mismatch_is_an_error(node_available):
    with pytest.raises(RuntimeError, match="w x h"):
        comp([0] * 8, [0] * 4, 1, 2)
    with pytest.raises(RuntimeError, match="w x h"):
        comp([0] * 4, [0] * 8, 1, 2)


def test_caption_names_type_size_tilt_and_power(node_available):
    look = {"name": "Standard", "w": 1.038, "d": 1.755, "watts": 400}
    assert run(f"S.caption({json.dumps(look)}, 15)") == "Standard panel · 1.04 × 1.75 m · 15° tilt · ~400 W typical"


def test_caption_for_an_array_counts_panels_and_peak_power(node_available):
    look = {"name": "Standard", "w": 1.038, "d": 1.755, "watts": 400}
    assert run(f"S.caption({json.dumps(look)}, 15, 6)") == "6 × Standard panels · 1.04 × 1.75 m each · 15° tilt · 2.4 kW peak"
    # with sun hours: 6 x 400 W x 5 h x 0.8 = 9.6 kWh a day
    assert run(f"S.caption({json.dumps(look)}, 15, 6, 5)").endswith("2.4 kW peak · ~9.6 kWh/day")
    assert run(f"S.caption({json.dumps(look)}, 0, 1, 4)") == "Standard panel · 1.04 × 1.75 m · 0° tilt · ~400 W typical · ~1.3 kWh/day"
    assert run(f"S.caption({json.dumps(look)}, 0, 2, 5, 1)") == \
        "2 × Standard panels · 1.04 × 1.75 m each · 0° tilt · 0.8 kW peak · ~4.0 kWh/day"     # no losses
    assert run(f"S.caption({json.dumps(look)}, 0, 1, 0.5)").endswith("~0.2 kWh/day")       # a dull site still shows


def test_caption_energy_matches_panel_energy(node_available):
    # integration: the photo caption and the HUD (panel.js energy) give the same kWh a day
    for n, sun in ((1, 5), (7, 4.5), (12, 6)):
        cap = run(f'S.caption(P.lookByKey("standard"), 15, {n}, {sun})')
        kwh = run(f'P.energy({n}, P.lookByKey("standard").watts, {sun}).kwhDay')
        assert cap.endswith(f"~{kwh:.1f} kWh/day"), (cap, kwh)


def test_shot_name_is_local_time_zero_padded(node_available):
    assert run("S.shotName(new Date(2026, 0, 2, 3, 4, 5))") == "solar-preview-20260102-030405.png"
    assert run("S.shotName(new Date(2026, 10, 21, 13, 45, 59))") == "solar-preview-20261121-134559.png"


def test_watts_are_plausible_for_the_area(node_available):
    # integration with panel.js: typical rated power per m2 for each real look
    looks = run("P.PANEL_LOOKS")
    per_m2 = {l["key"]: l["watts"] / (l["w"] * l["d"]) for l in looks}
    assert 200 <= per_m2["standard"] <= 240            # mono PERC around 21-23 %
    assert 90 <= per_m2["bipv"] <= 150                  # patterned BIPV face gives up a lot
    assert run('S.caption(P.lookByKey("bipv"), 0)') == "BIPV shingle panel · 1.10 × 1.54 m · 0° tilt · ~200 W typical"


def test_composite_full_phone_frame_smoke(node_available):
    # a phone-sized frame runs through in one go and keeps the camera where the scene is clear
    w, h = 360, 780
    pre = (f"const w = {w}, h = {h}, cam = new Uint8Array(w*h*4).fill(90), sc = new Uint8Array(w*h*4);"
           "for (let i = 0; i < w*4*10; i += 4) { sc[i] = 255; sc[i+3] = 255; }")   # bottom 10 rows red
    out = run("(() => { const o = S.composite(cam, sc, w, h); return [o.length, o[0], o[(h-1)*w*4], o[(h-1)*w*4+1]]; })()", pre)
    assert out == [w * h * 4, 90, 255, 0]


def test_page_wires_the_photo_button():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf8")
    for hook in ['id="o-shot"', '"o-shot"', 'from "./snap.js', "takeShot(frame, view)", "emuShot()",
                 'id="shot-save"', 'id="shot-share"', "navigator.canShare", "readRenderTargetPixels",
                 "renderer.xr.enabled = false", "composite(camPx, scenePx, w, h, flipY)"]:
        assert hook in html, hook
    assert re.search(r"if \(view && wantShot\) takeShot\(frame, view\);", html)
    assert "~${l.watts} W" in html
    assert "caption(lookByKey(look), tiltNow(), panels.length, sunHours())" in html
