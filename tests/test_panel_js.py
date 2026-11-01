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
                 "b.minX + design.cx", "b.minZ + design.cz", "clampToBox([t.roof[0] + grabOffset[0]", "panels shown", "./panel.js?v=6"):
        assert hook in html, hook


# ---------- several panels ----------
W, D, G = 1.0, 1.6, 0.02


def test_overlaps_is_strict_so_touching_neighbours_are_free(node_available):
    a = {"x": 0, "z": 0, "yaw": 0}
    assert run(f"P.overlaps({json.dumps(a)}, {json.dumps(a)}, {W}, {D})")
    assert run(f'P.overlaps({{x: 0.99, z: 0, yaw: 0}}, {json.dumps(a)}, {W}, {D})')
    assert not run(f'P.overlaps({{x: 1.0, z: 0, yaw: 0}}, {json.dumps(a)}, {W}, {D})')      # edge to edge
    assert not run(f'P.overlaps({{x: 0, z: 1.6, yaw: 0}}, {json.dumps(a)}, {W}, {D})')
    assert run(f'P.overlaps({{x: 0, z: 1.5, yaw: 0}}, {json.dumps(a)}, {W}, {D})')


def test_add_slot_goes_beside_the_panel_then_further_out(node_available):
    assert run(f"P.addSlot([], {W}, {D})") == {"x": 0, "z": 0, "yaw": 0}
    one = [{"x": 0, "z": 0, "yaw": 0}]
    assert run(f"P.addSlot({json.dumps(one)}, {W}, {D})") == pytest.approx({"x": W + G, "z": 0, "yaw": 0})
    # right taken: left; both taken: behind; then in front; then two along
    row = one + [{"x": W + G, "z": 0, "yaw": 0}]
    assert run(f"P.addSlot({json.dumps(row)}, {W}, {D}, 0)") == pytest.approx({"x": -(W + G), "z": 0, "yaw": 0})
    row3 = row + [{"x": -(W + G), "z": 0, "yaw": 0}]
    assert run(f"P.addSlot({json.dumps(row3)}, {W}, {D}, 0)") == pytest.approx({"x": 0, "z": D + G, "yaw": 0})
    ring = row3 + [{"x": 0, "z": D + G, "yaw": 0}, {"x": 0, "z": -(D + G), "yaw": 0}]
    assert run(f"P.addSlot({json.dumps(ring)}, {W}, {D}, 0)") == pytest.approx({"x": 2 * (W + G), "z": 0, "yaw": 0})
    # from = which panel to add beside; out of range is clamped
    assert run(f"P.addSlot({json.dumps(row)}, {W}, {D}, 1)") == pytest.approx({"x": 2 * (W + G), "z": 0, "yaw": 0})
    assert run(f"P.addSlot({json.dumps(row)}, {W}, {D}, 9)") == pytest.approx({"x": 2 * (W + G), "z": 0, "yaw": 0})
    assert run(f"P.addSlot({json.dumps(row)}, {W}, {D}, -3)") == pytest.approx({"x": -(W + G), "z": 0, "yaw": 0})


def test_add_slot_follows_a_turned_panel(node_available):
    # a panel turned 90 degrees: its own +x (across) now runs along roof -z (rotation.y turns +x toward -z)
    q = run(f"P.addSlot([{{x: 1, z: 2, yaw: Math.PI / 2}}], {W}, {D})")
    assert q == pytest.approx({"x": 1, "z": 2 - (W + G), "yaw": np.pi / 2})
    assert not run(f"P.overlaps({json.dumps(q)}, {{x: 1, z: 2, yaw: Math.PI / 2}}, {W}, {D})")


def test_rows_copy_and_drop_the_back_row(node_available):
    row = [{"x": i * (W + G), "z": 0, "yaw": 0} for i in range(3)]
    assert run(f"P.lastRow({json.dumps(row)}, {D})") == [0, 1, 2]
    two = run(f"P.addRow({json.dumps(row)}, {D})")
    assert len(two) == 6 and two[:3] == row
    assert [p["z"] for p in two[3:]] == pytest.approx([D + G] * 3) and [p["x"] for p in two[3:]] == pytest.approx([0, W + G, 2 * (W + G)])
    assert run(f"P.lastRow({json.dumps(two)}, {D})") == [3, 4, 5]
    # a ragged back row (one panel) is what gets copied
    ragged = row + [{"x": 0, "z": D + G + 0.3, "yaw": 0}]
    assert run(f"P.lastRow({json.dumps(ragged)}, {D})") == [3]
    assert len(run(f"P.addRow({json.dumps(ragged)}, {D})")) == 5
    # a panel nudged less than half a depth back still counts as the back row
    nudged = row[:2] + [{"x": 2 * (W + G), "z": 0.3, "yaw": 0}]
    assert run(f"P.lastRow({json.dumps(nudged)}, {D})") == [0, 1, 2]
    # but more than half a depth back is a row of its own
    assert run(f"P.lastRow([{{x: 0, z: 0, yaw: 0}}, {{x: 0, z: {0.6 * D}, yaw: 0}}], {D})") == [1]
    assert run(f"P.removeRow({json.dumps(two)}, {D})") == row
    assert run(f"P.removeRow({json.dumps(row)}, {D})") == row          # the only row stays
    # pure: the input is not changed
    assert run(f"(() => {{ const a = {json.dumps(row)}; P.addRow(a, {D}); P.removeRow(a, {D}); return a.length; }})()") == 3


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
    # smoke: grow a 4 x 3 array the way the buttons do and check nothing overlaps and the bounds are tight
    code = f"""(() => {{
      let a = [{{x: 0, z: 0, yaw: 0}}];
      for (let i = 0; i < 3; i++) a.push(P.addSlot(a, {W}, {D}, a.length - 1));
      a = P.addRow(P.addRow(a, {D}), {D});
      const clash = a.some((p, i) => a.some((q, j) => i !== j && P.overlaps(p, q, {W}, {D})));
      return {{ n: a.length, clash, b: P.arrayBounds(a, {W}, {D}) }};
    }})()"""
    out = run(code)
    assert out["n"] == 12 and not out["clash"]
    assert out["b"]["w"] == pytest.approx(4 * W + 3 * G) and out["b"]["d"] == pytest.approx(3 * D + 2 * G)


def test_page_wires_the_panel_array_and_power():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ('id="o-add"', 'id="o-del"', 'id="o-row"', 'id="o-unrow"', 'id="sun"', '"o-add", "o-del", "o-row", "o-unrow"',
                 "panels.push(addSlot(panels, l.w, l.d, sel >= 0 ? sel : panels.length - 1))",
                 "panels.splice(sel >= 0 ? sel : panels.length - 1, 1)", "panels = addRow(panels", "panels = removeRow(panels",
                 "relayout(panels, lookByKey(look), lookByKey(key))", "energy(panels.length, l.watts, sunHours())",
                 "kWh/day", "W per hour over 24 h", 'caption(lookByKey(look), TILTS[tiltKey], panels.length, sunHours())',
                 '$("o-del").disabled = panels.length < 2', "if (tap && !grabbed) sel = panelAt(t.roof)",
                 "ring.visible = i === sel", "./snap.js?v=2", "pick(i)"):
        assert hook in html, hook
    assert "const panel = " not in html and "panelObj " not in html


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
