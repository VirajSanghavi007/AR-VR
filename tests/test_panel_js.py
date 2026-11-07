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
    assert run('P.lookByKey("bipv").name') == "BIPV shingle"
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
    assert look["w"] / look["d"] == pytest.approx(img.width / img.height, abs=0.005)      # shingles not stretched
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
    assert "terrace_tilted_ar.glb" not in html                   # GLTFLoader is only for the designer's model now
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


def test_page_draws_only_the_panel_not_the_roof_slab():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert "0x9aa3ad" not in html and "ShapeGeometry" not in html
    assert "design.keepouts.forEach" not in html


def test_move_box_keeps_the_panel_inside_the_tag_rectangle(node_available):
    # 0.8 x 0.5 m tags at 1:2 are 1.6 x 1.0 m of panel space; a 1 x 0.6 m panel can move 0.3 / 0.2 either way
    b = run("P.moveBox(0.8, 0.5, 0, 0.5, 1, 0.6)")
    assert b == pytest.approx({"minX": -0.3, "maxX": 0.3, "minZ": -0.2, "maxZ": 0.2})
    # an odd turn swaps which rectangle side runs along the panel's x
    b = run("P.moveBox(0.8, 0.5, 1, 0.5, 0.6, 1)")
    assert b == pytest.approx({"minX": -0.2, "maxX": 0.2, "minZ": -0.3, "maxZ": 0.3})
    assert run("P.moveBox(0.8, 0.5, 2, 0.5, 1, 0.6)") == pytest.approx(run("P.moveBox(0.8, 0.5, 0, 0.5, 1, 0.6)"))
    # panel bigger than the rectangle: pinned to the centre, never an inverted box
    assert run("P.moveBox(0.8, 0.5, 0, 1, 1, 0.6)") == pytest.approx({"minX": 0, "maxX": 0, "minZ": 0, "maxZ": 0})


def test_page_fits_the_panel_not_the_roof():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert "terrace_anchor.json" not in html and "m roof" not in html
    for hook in ("setDesign()", "arrayBounds(panels, l.w, l.d)", "moveBox(rect.width, rect.height, turns, scale, l.w, l.d)",
                 "b.minX + design.cx", "b.minZ + design.cz", "clampToBox([t.roof[0] + grabOffset[0]", "panels shown", "./panel.js?v=9"):
        assert hook in html, hook


# ---------- several panels ----------
W, D, G = 1.0, 1.6, 0.02


def test_turn_array_swings_the_rows_round_their_centre(node_available):
    assert run("P.turnArray([], 1)") == []
    row = [{"x": 0, "z": 0, "yaw": 0}, {"x": 2, "z": 0, "yaw": 0}, {"x": 4, "z": 0.6, "yaw": 0.1, "k": 7}]
    out = run(f"P.turnArray({json.dumps(row)}, Math.PI / 2)")
    # mean centre (2, 0.2) stays put; a quarter turn (rotation.y) sends +x to -z, +z to +x
    assert [v for p in out for v in (p["x"], p["z"])] == pytest.approx([1.8, 2.2, 1.8, 0.2, 2.4, -1.8])
    assert [p["yaw"] for p in out] == pytest.approx([np.pi / 2, np.pi / 2, 0.1 + np.pi / 2])
    assert out[2]["k"] == 7                                           # other fields kept
    assert row[0] == {"x": 0, "z": 0, "yaw": 0}                       # input untouched (JSON copy, so check in node)
    assert run("(() => { const a = [{x: 1, z: 0, yaw: 0}, {x: 3, z: 0, yaw: 0}]; P.turnArray(a, 1); return a[0].x; })()") == 1
    # a neighbour stays a neighbour: same offset in each panel's own frame, so a turned row stays one row
    two = [{"x": 0, "z": 0, "yaw": 0}, {"x": W + G, "z": 0, "yaw": 0}]
    t = run(f"P.turnArray({json.dumps(two)}, 0.7)")
    assert run(f"P.toPanelFrame([{t[1]['x']}, {t[1]['z']}], {json.dumps(t[0])})") == pytest.approx([W + G, 0], abs=1e-9)
    assert run(f"P.overlaps({json.dumps(t[1])}, {json.dumps(t[0])}, {W}, {D})") is False
    for a, b in zip(run(f"P.turnArray({json.dumps(two)}, 0)"), two):
        assert a == pytest.approx(b)


def test_page_twists_the_array_when_nothing_is_picked():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert "if (touches.size === 2) {" in html and "touches.size === 2 && sel >= 0" not in html
    assert "if (sel >= 0) panels[sel].yaw += da; else panels = turnArray(panels, da);" in html
    assert "if (sel >= 0) panels[sel].yaw -= PANEL_STEP; else panels = turnArray(panels, -PANEL_STEP);" in html
    assert "turn the picked panel, or the whole array when none is picked" in html


def test_overlaps_is_strict_so_touching_neighbours_are_free(node_available):
    a = {"x": 0, "z": 0, "yaw": 0}
    assert run(f"P.overlaps({json.dumps(a)}, {json.dumps(a)}, {W}, {D})")
    assert run(f'P.overlaps({{x: 0.99, z: 0, yaw: 0}}, {json.dumps(a)}, {W}, {D})')
    assert not run(f'P.overlaps({{x: 1.0, z: 0, yaw: 0}}, {json.dumps(a)}, {W}, {D})')      # edge to edge
    assert not run(f'P.overlaps({{x: 0, z: 1.6, yaw: 0}}, {json.dumps(a)}, {W}, {D})')
    assert run(f'P.overlaps({{x: 0, z: 1.5, yaw: 0}}, {json.dumps(a)}, {W}, {D})')


def test_grid_layout_is_centred_rows_by_columns(node_available):
    g = run(f"P.gridLayout(2, 3, {W}, {D})")
    assert len(g) == 6 and all(p["yaw"] == 0 for p in g)
    # row by row: columns along x one panel + gap apart, rows along z
    assert [p["x"] for p in g] == pytest.approx([-(W + G), 0, W + G] * 2)
    assert [p["z"] for p in g] == pytest.approx([-(D + G) / 2] * 3 + [(D + G) / 2] * 3)
    assert run(f"P.gridLayout(1, 1, {W}, {D})") == [{"x": 0, "z": 0, "yaw": 0}]
    assert run(f"P.gridLayout(3, 1, {W}, {D}, 0.5)")[2]["z"] == pytest.approx(D + 0.5)
    b = run(f"P.arrayBounds(P.gridLayout(4, 5, {W}, {D}), {W}, {D})")
    assert b == pytest.approx({"cx": 0, "cz": 0, "w": 5 * W + 4 * G, "d": 4 * D + 3 * G})


def test_clamp_count_reads_typed_numbers(node_available):
    assert run('P.clampCount("3", 1)') == 3
    assert run('P.clampCount("2.6", 1)') == 3
    assert run('P.clampCount("", 4)') == 4 and run('P.clampCount("abc", 2)') == 2
    assert run('P.clampCount("0", 1)') == 1 and run('P.clampCount("-5", 7)') == 1
    assert run('P.clampCount("99", 1)') == 10 and run('P.clampCount("10", 1)') == 10
    assert run('P.clampCount("50", 1, 1, 20)') == 20 and run('P.clampCount("1", 5, 2)') == 2


def test_array_bounds_cover_every_corner(node_available):
    assert run(f"P.arrayBounds([{{x: 0, z: 0, yaw: 0}}], {W}, {D})") == pytest.approx({"cx": 0, "cz": 0, "w": W, "d": D})
    two = [{"x": 0, "z": 0, "yaw": 0}, {"x": W + G, "z": 0, "yaw": 0}]
    assert run(f"P.arrayBounds({json.dumps(two)}, {W}, {D})") == pytest.approx({"cx": (W + G) / 2, "cz": 0, "w": 2 * W + G, "d": D})
    turned = run(f"P.arrayBounds([{{x: 1, z: -1, yaw: Math.PI / 2}}], {W}, {D})")
    assert turned == pytest.approx({"cx": 1, "cz": -1, "w": D, "d": W})
    diag = run(f"P.arrayBounds([{{x: 0, z: 0, yaw: Math.PI / 4}}], 1, 1)")
    assert diag["w"] == pytest.approx(2 ** 0.5) and diag["d"] == pytest.approx(2 ** 0.5)


def test_relayout_keeps_neighbours_one_gap_apart(node_available):
    a, b = {"w": 1.0, "d": 1.6}, {"w": 1.2, "d": 1.4}
    grid = [{"x": i * (a["w"] + G), "z": j * (a["d"] + G), "yaw": 0.3} for i in range(3) for j in range(2)]
    out = run(f"P.relayout({json.dumps(grid)}, {json.dumps(a)}, {json.dumps(b)})")
    assert [p["x"] for p in out] == pytest.approx([i * (b["w"] + G) for i in range(3) for j in range(2)])
    assert [p["z"] for p in out] == pytest.approx([j * (b["d"] + G) for i in range(3) for j in range(2)])
    assert all(p["yaw"] == 0.3 for p in out)


def test_energy_from_sun_hours_and_losses(node_available):
    e = run("P.energy(6, 400)")                     # 5 sun hours, 20 % losses
    assert e == pytest.approx({"peakW": 2400, "kwhDay": 9.6, "avgW": 400, "dayAvgW": 800})
    assert run("P.energy(1, 200, 4, 1, 10)") == pytest.approx({"peakW": 200, "kwhDay": 0.8, "avgW": 800 / 24, "dayAvgW": 80})
    assert run("P.energy(0, 400)") == pytest.approx({"peakW": 0, "kwhDay": 0, "avgW": 0, "dayAvgW": 0})
    # the 24 h average can never beat the peak: 24 sun hours with no losses is the limit
    assert run("P.energy(3, 400, 24, 1).avgW") == pytest.approx(1200)


def test_pattern_smoke_builds_a_clean_grid(node_available):
    # smoke: lay out a 3 x 4 array from typed numbers, turn it, and check nothing overlaps and the bounds are tight
    code = f"""(() => {{
      const a = P.gridLayout(P.clampCount("3", 1), P.clampCount("4", 1), {W}, {D});
      const t = P.turnArray(a, Math.PI / 2);
      const clash = (x) => x.some((p, i) => x.some((q, j) => i !== j && P.overlaps(p, q, {W}, {D})));
      return {{ n: a.length, clash: clash(a) || clash(t), b: P.arrayBounds(a, {W}, {D}), bt: P.arrayBounds(t, {W}, {D}) }};
    }})()"""
    out = run(code)
    assert out["n"] == 12 and not out["clash"]
    assert out["b"]["w"] == pytest.approx(4 * W + 3 * G) and out["b"]["d"] == pytest.approx(3 * D + 2 * G)
    assert out["bt"]["w"] == pytest.approx(out["b"]["d"]) and out["bt"]["d"] == pytest.approx(out["b"]["w"])


def test_page_wires_the_panel_array_and_power():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ('id="rows"', 'id="cols"', 'id="o-rows"', 'id="o-cols"', 'id="o-grid"', 'id="sun"', '"o-msave", "o-grid", "o-minus",',
                 'const gridSize = () => [clampCount($("rows").value, 1), clampCount($("cols").value, 1)];',
                 "panels = turnArray(gridLayout(r, c, l.w, l.d), yaw);",
                 '[["rows", "o-rows"], ["cols", "o-cols"]]', "$(a).oninput = () => { $(b).value = $(a).value; regrid(); };",
                 "$(b).oninput = () => { $(a).value = $(b).value; regrid(); };",
                 "  $(b).value = $(a).value;                                     // a reload may have kept the typed numbers\n}\nregrid();",
                 "relayout(panels, lookByKey(look), lookByKey(key))", "energy(panels.length, l.watts, sunHours())",
                 "kWh/day", "W per hour over 24 h", 'caption(lookByKey(look), tiltNow(), panels.length, sunHours())',
                 "if (tap && !grabbed) sel = panelAt(t.roof)", "ring.visible = i === sel", "./snap.js?v=2", "pick(i)",
                 ".bar .grid[hidden] { display: none; }"):
        assert hook in html, hook
    for gone in ('id="o-add"', 'id="o-del"', 'id="o-row"', 'id="o-unrow"', "addSlot", "addRow", "removeRow", "lastRow",
                 "+ Panel", "+ Row"):
        assert gone not in html, gone
    assert "const panel = " not in html and "panelObj " not in html


def test_page_has_one_fixed_tilt_no_toggle():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert "const TILT_DEG = 15," in html and "const deg = tiltNow()," in html
    # on a wall the array mounts flush, the HUD says so and the sun comes from above (+z is down there)
    assert "const tiltNow = () => (rect && onWall(rect.n) ? 0 : TILT_DEG);" in html
    assert '`${tiltNow()}° tilt` : "flush on the wall"' in html
    assert "(onWall(rect.n) ? -0.7 : 0.7) * R" in html
    for gone in ('id="b-tilted"', 'id="b-flat"', "Tilted 15°", ">Flat<", "chooseTilt", "TILTS", "tiltKey"):
        assert gone not in html, gone


# ---------- mountParts: the rack under the panel ----------

def mount(look, tilt, opts=""):
    return run(f'P.mountParts(P.lookByKey("{look}"), {tilt}{", " + opts if opts else ""})')


def test_mount_flat_standard_sits_on_two_rails(node_available):
    look = run('P.lookByKey("standard")')
    m = mount("standard", 0)
    assert m["y"] == pytest.approx(0.04) and len(m["parts"]) == 2
    xs = sorted(p["x"] for p in m["parts"])
    assert xs == pytest.approx([-(look["w"] / 2 - 0.2 * look["w"]), look["w"] / 2 - 0.2 * look["w"]])
    for p in m["parts"]:
        assert p["y"] - p["sy"] / 2 == pytest.approx(0)             # on the table
        assert p["sz"] == pytest.approx(0.95 * look["d"]) and p["rotX"] == 0


def test_mount_flat_bipv_is_flush(node_available):
    assert mount("bipv", 0) == {"y": 0.002, "parts": []}


def test_mount_tilted_legs_reach_the_rails_and_the_table(node_available):
    look = run('P.lookByKey("standard")')
    for tilt in (10, 15, 30):
        a = np.radians(tilt)
        m = mount("standard", tilt)
        parts = m["parts"]
        assert len(parts) == 9                                         # 2 x (rail, 2 legs, foot) + brace
        rail, front, rear, foot = parts[:4]
        assert rail["rotX"] == pytest.approx(a) and rail["sz"] == pytest.approx(0.95 * look["d"])
        for leg in (front, rear):
            assert leg["y"] - leg["sy"] / 2 == pytest.approx(0)       # stands on the table
        assert front["sy"] == pytest.approx(0.15)                      # front leg top = lift
        zf = look["d"] / 2 - 0.1 * look["d"]
        assert rear["sy"] - front["sy"] == pytest.approx(2 * zf * np.sin(a))   # rear leg is taller
        assert front["z"] > rear["z"] and front["z"] - rear["z"] == pytest.approx(2 * zf * np.cos(a))
        assert front["z"] == pytest.approx(zf * np.cos(a) - 0.04 * np.sin(a))   # under the rail, not the panel face
        # leg tops touch the rail underside: a point zt along the tilted panel, bar below it
        for leg, zt in ((front, zf), (rear, -zf)):
            under_y = m["y"] - 0.04 * np.cos(a) - zt * np.sin(a)
            assert leg["sy"] == pytest.approx(under_y)
        assert foot["sz"] == pytest.approx(front["z"] - rear["z"] + 0.04)
        brace = parts[8]
        assert brace["x"] == 0 and brace["sx"] == pytest.approx(2 * (look["w"] / 2 - 0.2 * look["w"]))
        assert brace["z"] == pytest.approx(rear["z"]) and 0 < brace["y"] < rear["sy"]


def test_mount_options_and_bipv_tilted_gets_a_rack(node_available):
    m = mount("standard", 15, "{lift: 0.3, bar: 0.05}")
    assert m["parts"][1]["sy"] == pytest.approx(0.3) and m["parts"][1]["sx"] == pytest.approx(0.05)
    assert len(mount("bipv", 15)["parts"]) == 9


def test_page_builds_the_rack_from_mount_parts():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ("mountParts(", "function buildRack(", "obj.userData.rackTilt !== deg", "castShadow = true"):
        assert hook in html, hook


# ---------- realism: face texture helpers and the light ----------

def test_cell_shade_varies_a_little_and_repeats(node_available):
    k = run("Array.from({length: 120}, (_, i) => P.cellShade(i))")
    assert all(0.94 - 1e-9 <= v <= 1.06 + 1e-9 for v in k)
    assert max(k) - min(k) > 0.08                       # really varies, not one flat tint
    assert len({round(v, 6) for v in k}) > 100
    assert run("P.cellShade(7)") == run("P.cellShade(7)")
    assert all(0.8 - 1e-9 <= v <= 1.2 + 1e-9 for v in run("Array.from({length: 50}, (_, i) => P.cellShade(i, 0.2))"))
    assert run("P.cellShade(3, 0)") == 1


def test_chamfer_cell_cuts_only_the_wafer_corners(node_available):
    full = run("P.chamferCell(0, 0, 10, 20, 2)")
    assert full == [[2, 0], [8, 0], [10, 2], [10, 18], [8, 20], [2, 20], [0, 18], [0, 2]]
    assert run("P.chamferCell(0, 0, 10, 20, 2, false, true)") == [[0, 0], [10, 0], [10, 18], [8, 20], [2, 20], [0, 18]]
    assert run("P.chamferCell(0, 0, 10, 20, 2, true, false)") == [[2, 0], [8, 0], [10, 2], [10, 20], [0, 20], [0, 2]]
    assert run("P.chamferCell(0, 0, 10, 20, 2, false, false)") == [[0, 0], [10, 0], [10, 20], [0, 20]]
    assert run("P.chamferCell(5, 7, 10, 20, 2)")[0] == [7, 7]
    # the cut never passes the middle of a side
    assert run("P.chamferCell(0, 0, 4, 20, 9)")[:2] == [[2, 0], [2, 0]]
    assert run("P.chamferCell(0, 0, 20, 4, 9)")[2] == [20, 2]


def test_smooth_light_eases_and_clamps(node_available):
    prev = {"color": [1, 1, 1], "intensity": 1.8}
    one = run(f"P.smoothLight({json.dumps(prev)}, [3, 3, 1.5], 1)")
    assert one["color"] == pytest.approx([1, 1, 0.5]) and one["intensity"] == pytest.approx(2.5)     # clamped to hi
    assert run(f"P.smoothLight({json.dumps(prev)}, [0.1, 0.2, 0.2], 1)")["intensity"] == pytest.approx(0.6)   # lo
    assert run(f"P.smoothLight({json.dumps(prev)}, [1.2, 1.2, 1.2], 1)")["intensity"] == pytest.approx(1.2)
    step = run(f"P.smoothLight({json.dumps(prev)}, [2, 1, 1])")             # default k = 0.08 of the way
    assert step["intensity"] == pytest.approx(1.8 + 0.08 * 0.2)
    assert step["color"] == pytest.approx([1, 1 - 0.08 * 0.5, 1 - 0.08 * 0.5])
    # no light, or nonsense: keep what we had
    assert run(f"P.smoothLight({json.dumps(prev)}, [0, 0, 0])") == prev
    assert run(f"P.smoothLight({json.dumps(prev)}, [NaN, NaN, NaN])") == prev
    # repeated steps settle on the target without overshoot
    s = run(f"(() => {{ let l = {json.dumps(prev)}; for (let i = 0; i < 200; i++) l = P.smoothLight(l, [1, 1, 1]); return l; }})()")
    assert s["intensity"] == pytest.approx(1.0, abs=1e-4)


def test_page_lights_and_draws_the_panels_realistically():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ("renderer.toneMapping = THREE.ACESFilmicToneMapping;", '"dom-overlay", "light-estimation"]',
                 "session.requestLightProbe(lightProbeOptions(session)).then((p) => { lightProbe = p; useRoomReflections(p); })", "lightProbe = null; light = { ...SUN };",
                 "if (lightProbe) matchLight(frame.getLightEstimate(lightProbe));",
                 "light = smoothLight(light, [i.x, i.y, i.z]);", "sun.color.setRGB(...light.color);",
                 "sun.intensity = light.intensity;", "hemi.intensity = light.intensity * SKY_PER_SUN;",
                 "const W = 1024, H = Math.round((W * look.d) / look.w)",
                 "const top = !look.halfCut || row % 2 === 0, bottom = !look.halfCut || row % 2 === 1;",
                 "chamferCell(x, y, w, h, w * 0.07, top, bottom), k = cellShade(i);", "for (let k2 = 1; k2 <= 9; k2++)",
                 "cameraLook(new THREE.MeshStandardMaterial({ map, roughness: 0.45, metalness: 0, envMapIntensity: 0.7"):
        assert hook in html, hook
    # the sun keeps its place over the tags: the estimate never moves it (a swinging glint was the flicker)
    assert "primaryLightDirection" not in html


# ---------- blending into the camera image: contact shadow, grain + vignette, room reflections ----------

STD = {"w": 1.7, "d": 1.1}


def test_contact_shadow_covers_the_footprint_with_a_fade(node_available):
    flat = run(f"P.contactShadow({json.dumps(STD)}, 0, 0.06)")
    assert flat["w"] == pytest.approx(1.82) and flat["d"] == pytest.approx(1.22)
    assert flat["core"] == pytest.approx([1.7 / 1.82, 1.1 / 1.22]) and flat["opacity"] == pytest.approx(0.45)
    tilted = run(f"P.contactShadow({json.dumps(STD)}, 15)")                  # footprint shrinks by cos(tilt)
    assert tilted["d"] == pytest.approx(1.1 * np.cos(np.radians(15)) + 0.12)
    assert tilted["core"][1] == pytest.approx(1.1 * np.cos(np.radians(15)) / tilted["d"])
    assert tilted["opacity"] < flat["opacity"]                               # raised on a rack: lighter
    flush = run(f"P.contactShadow({json.dumps({**STD, 'flush': True})}, 0)")
    assert flush["opacity"] > flat["opacity"]                                # glass on the surface: darkest


def test_contact_alpha_is_solid_in_the_core_and_fades_to_the_edge(node_available):
    n = 32
    a = np.array(run(f"Array.from(P.contactAlpha({n}, [0.5, 0.5]))")).reshape(n, n, 4)
    assert (a[..., 0] == a[..., 3]).all() and (a[..., 1] == a[..., 3]).all()
    alpha = a[..., 3]
    assert alpha[n // 2, n // 2] == 255 and alpha[8:24, 8:24].min() == 255     # core = middle half
    assert alpha[0, 0] == 0 and alpha[n // 2, 0] < 10                           # corners and edges fade out
    row = alpha[n // 2, : n // 2]
    assert (np.diff(row.astype(int)) >= 0).all()                                # no rim: rises monotonically
    assert (alpha == alpha[::-1]).all() and (alpha == alpha[:, ::-1]).all()     # symmetric
    wide = np.array(run(f"Array.from(P.contactAlpha({n}, [0.9, 0.2]))")).reshape(n, n, 4)[..., 3]
    assert wide[n // 2, 2] == 255 and wide[2, n // 2] < 255                     # core per side
    assert run("Array.from(P.contactAlpha(4, [1, 1]))")[3] == 255               # core = all: no divide by zero


def test_grain_rises_as_the_light_drops(node_available):
    assert run("P.grainFor(0.6)") == pytest.approx(0.04) and run("P.grainFor(2.5)") == pytest.approx(0.012)
    assert run("P.grainFor(0.1)") == pytest.approx(0.04) and run("P.grainFor(9)") == pytest.approx(0.012)   # clamped
    assert run("P.grainFor(1.55)") == pytest.approx((0.04 + 0.012) / 2)


FRAG = "uniform vec3 diffuse;\nvoid main() {\n  vec4 c = vec4(1.0);\n  #include <dithering_fragment>\n}"


def test_camera_look_shader_adds_grain_and_vignette_at_the_end(node_available):
    out = run(f"P.cameraLookShader({json.dumps(FRAG)})")
    assert "uniform float grainAmt, grainSeed, vignetteAmt;\nuniform vec2 grainRes;\nvoid main() {" in out
    head, tail = out.split("#include <dithering_fragment>")
    assert "gl_FragColor" not in head                                           # after three's own colour
    assert "gl_FragCoord.xy / grainRes - 0.5" in tail and "1.0 - vignetteAmt * dot(lookUv, lookUv)" in tail
    assert "grainAmt * (fract(sin(dot(gl_FragCoord.xy + grainSeed" in tail and tail.rstrip().endswith("}")
    with pytest.raises(RuntimeError, match="not a three.js fragment shader"):
        run('P.cameraLookShader("void main() {}")')


def test_light_probe_options_ask_for_the_room_cube_map(node_available):
    assert run('P.lightProbeOptions({ preferredReflectionFormat: "srgba8" })') == {"reflectionFormat": "srgba8"}
    assert run("P.lightProbeOptions({}) === undefined") is True
    assert run("P.lightProbeOptions(null) === undefined") is True


def test_page_blends_the_panels_into_the_camera_image():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ("renderer.xr.setFramebufferScaleFactor(0.8);", "scene.environment = reflections();",
                 "const cube = binding.getReflectionCubeMap(probe);", "renderer.properties.get(realEnv).__webglTexture = cube;",
                 "realEnv.needsPMREMUpdate = true;", "scene.environment = realEnv;",
                 "s.fragmentShader = cameraLookShader(s.fragmentShader);", "camLook.grainAmt.value = grainFor(light.intensity);",
                 "lookStep(t, layer ? layer.framebufferWidth : innerWidth, layer ? layer.framebufferHeight : innerHeight);",
                 "camLook.grainRes.value.set(w, h);", "camLook.grainRes.value.copy(res);",
                 "const c = contactShadow(l, deg);", "contact.scale.set(c.w, 1, c.d);", "contact.material.opacity = c.opacity;",
                 "contact.material.alphaMap = contactMap(c.core);", "new THREE.DataTexture(contactAlpha(64, core), 64, 64)",
                 "metals.set(color, cameraLook(new THREE.MeshStandardMaterial("):
        assert hook in html, hook
    assert "envMap: reflections()" not in html          # every material takes the scene's (real) environment
    assert html.count("cameraLook(new THREE.") == 5     # face (both looks), glass edge, glass back, aluminium
