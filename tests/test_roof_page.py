"""roof.html (M6 customer roof viewer) in a real headless Chrome, and scripts/roof_stills.py.
Unit (static page checks, CLI argument errors), smoke (the page loads the detailed GLB, the stills
script writes PNGs), integration (Chrome + three.js from the CDN + the shipped model)."""
import functools
import http.server
import json
import math
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

from rooftop import sun as S

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
HTML = (PREVIEW / "roof.html").read_text(encoding="utf-8")
INFO = json.loads((PREVIEW / "models" / "rooftop_detailed.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
import roof_stills  # noqa: E402

CHROME = roof_stills.find_chrome()


# ---------- unit ----------

def test_page_has_the_customer_controls_and_loads_the_detailed_model():
    for i in ("sMonth", "sHour", "bOrbit", "bBefore", "bNeighbour", "bAR", "bArSize", "bArMove", "bArExit", "hudSize", "hudSun"):
        assert f'id="{i}"' in HTML, i
    assert '"beforexrselect"' in HTML
    assert "models/rooftop_detailed.glb" in HTML and (PREVIEW / "models" / "rooftop_detailed.glb").is_file()
    assert "toneMapping = THREE.ACESFilmicToneMapping" in HTML and "PCFSoftShadowMap" in HTML


def test_page_url_builds_the_query():
    assert roof_stills.page_url("http://h") == "http://h/roof.html?still=1&month=12"
    assert roof_stills.page_url("http://h", 6, 9.5, True, "models/x.glb") == \
        "http://h/roof.html?still=1&month=6&hour=9.5&neighbour=1&glb=models/x.glb"
    assert roof_stills.page_url("http://h", 3, 0) == "http://h/roof.html?still=1&month=3&hour=0"


def test_find_chrome_returns_an_existing_file_or_none(monkeypatch):
    c = roof_stills.find_chrome()
    assert c is None or Path(c).is_file()

    class Missing(type(Path())):
        def is_file(self):
            return False
    monkeypatch.setattr(roof_stills, "Path", Missing)
    monkeypatch.setattr(roof_stills.shutil, "which", lambda name: "C:/nowhere/chrome.exe")
    assert roof_stills.find_chrome() is None                       # a path that is not a file is skipped


def test_serve_is_quiet_and_serves_the_preview(capsys):
    srv, base = roof_stills.serve(PREVIEW)
    try:
        body = urllib.request.urlopen(f"{base}/roof.html", timeout=10).read()
        assert body == (PREVIEW / "roof.html").read_bytes()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/no_such_file.html", timeout=10)
    finally:
        srv.shutdown()
    assert capsys.readouterr().err == ""                                   # no per-request log lines


def test_cli_rejects_an_unknown_shot(capsys):
    with pytest.raises(SystemExit) as e:
        roof_stills.main(["--shots", "aerial,top"])
    assert e.value.code == 2 and "unknown shot(s) ['top']" in capsys.readouterr().err


def test_render_without_chrome_says_so(monkeypatch, tmp_path):
    pytest.importorskip("playwright.sync_api")
    monkeypatch.setattr(roof_stills, "find_chrome", lambda: None)
    with pytest.raises(RuntimeError, match="Chrome not found"):
        roof_stills.render(tmp_path)


# ---------- smoke + integration ----------

@pytest.fixture(scope="module")
def chrome_available():
    pytest.importorskip("playwright.sync_api")
    if not CHROME:
        pytest.skip("Chrome not installed")
    try:
        urllib.request.urlopen("https://cdn.jsdelivr.net/npm/three@0.160.0/package.json", timeout=10).close()
    except OSError:
        pytest.skip("cdn.jsdelivr.net (three.js) unreachable")


@pytest.fixture(scope="module")
def page(chrome_available):
    from playwright.sync_api import sync_playwright
    handler = functools.partial(roof_stills.QuietHandler, directory=str(PREVIEW))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=roof_stills.SOFT_ARGS)
        pg = b.new_page(viewport={"width": 640, "height": 360})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/roof.html?month=12&hour=10", timeout=90000)
        pg.wait_for_function("window.__roof && __roof.state.ready", timeout=90000)
        pg.errors = errors
        pg.initial = pg.evaluate("({mh: [__roof.state.month, __roof.state.hour], sun: document.getElementById('hudSun').textContent, polar: __roof.orbit.maxPolarAngle})")
        yield pg
        b.close()
    srv.shutdown()


@pytest.fixture
def fresh(page):
    """Reset the shared page: December 10:00, After, no neighbour, aerial, roof at full size at the origin."""
    page.evaluate("__roof.setBefore(false); __roof.setNeighbour(false); __roof.world.scale.setScalar(1);"
                  " __roof.world.position.set(0, 0, 0); __roof.setSun(12, 10); __roof.shot('aerial')")
    return page


def test_page_loads_the_model_with_all_panels(fresh):
    s = fresh.evaluate("({ panels: __roof.state.panels, watts: __roof.state.watts, parts: Object.keys(__roof.parts),"
                       " frames: __roof.frames(), hud: document.getElementById('hudSize').textContent })")
    assert s["panels"] == INFO["panels"] == 41
    assert s["watts"] == pytest.approx(INFO["peak_kw"] * 1000 / 41)
    for n in ("glass", "frame", "galv", "inverter", "concrete", "plaster", "tank"):
        assert n in s["parts"], n
    assert s["frames"] > 0 and fresh.errors == []
    assert s["hud"].startswith("41 panels · 22.55 kWp · about ")


def test_url_month_and_hour_set_the_sun(page):
    assert page.initial["mh"] == [12, 10]
    assert page.initial["sun"].startswith("21 Dec 10:00 · sun ")
    assert page.initial["polar"] == pytest.approx(math.pi * 0.495)       # orbit stays above the ground at load



def test_orbit_stays_above_the_ground_when_the_model_fails_to_load(page):
    pg = page.context.browser.new_page(viewport={"width": 320, "height": 180})
    try:
        pg.goto(page.url.split("?")[0] + "?glb=models/missing.glb", timeout=90000)
        pg.wait_for_function("document.getElementById('hudSize').textContent.startsWith('Could not load')", timeout=90000)
        assert pg.evaluate("__roof.orbit.maxPolarAngle") == pytest.approx(math.pi * 0.495)   # no shot ran, the load-time limit holds
    finally:
        pg.close()

def test_set_sun_moves_the_light_like_sun_py(fresh):
    for month, hour in ((12, 9.0), (12, 12.5), (6, 15.5)):
        got = fresh.evaluate(f"(() => {{ __roof.setSun({month}, {hour}); const l = __roof.sun;"
                             " const d = l.position.clone().sub(l.target.position).normalize();"
                             " return {d: d.toArray(), i: l.intensity, v: l.visible,"
                             " m: document.getElementById('sMonth').value, h: document.getElementById('sHour').value,"
                             " sun: document.getElementById('hudSun').textContent}; })()")
        alt, az = S.sun_at(INFO["latitude"], INFO["longitude"], INFO["tz"], month, hour)
        x, y, z = S.sun_dir(alt, az, INFO["north_deg"])
        assert got["d"] == pytest.approx([x, z, -y], abs=1e-6)
        assert got["v"] is True and got["i"] > 1
        assert (got["m"], float(got["h"])) == (str(month), hour)
        assert f"sun {alt:.0f}° up" in got["sun"]
    night = fresh.evaluate("(() => { __roof.setSun(12, 19.5); return [__roof.sun.visible, __roof.sun.intensity,"
                           " document.getElementById('hudSun').textContent]; })()")
    assert night[0] is False and night[1] == 0 and night[2].endswith("sun below the horizon")


def test_shadow_tip_matches_the_hand_calculation(fresh):
    for hour in (9.0, 12.5, 15.5):
        tip = fresh.evaluate(f"(__roof.setSun(12, {hour}), __roof.shadowOf(2))")
        alt, az = S.sun_at(INFO["latitude"], INFO["longitude"], INFO["tz"], 12, hour)
        dx, dy = S.shadow(2, alt, az, INFO["north_deg"])
        assert tip == pytest.approx([dx, -dy], abs=1e-6)                   # plan (dx, dy) -> world (x, z)
        assert math.hypot(*tip) == pytest.approx(2 / math.tan(math.radians(alt)), rel=1e-6)


def test_sliders_drive_the_sun(fresh):
    fresh.evaluate("(() => { const m = document.getElementById('sMonth'); m.value = 6; m.dispatchEvent(new Event('input'));"
                   " const h = document.getElementById('sHour'); h.value = 8.25; h.dispatchEvent(new Event('input')); })()")
    assert fresh.evaluate("[__roof.state.month, __roof.state.hour]") == [6, 8.25]
    assert "21 Jun 08:15" in fresh.evaluate("document.getElementById('hudSun').textContent")


def test_before_hides_the_solar_parts_only(fresh):
    on = fresh.evaluate("(() => { document.getElementById('bBefore').click();"
                        " return {s: ['glass','frame','galv','inverter'].map(__roof.visible), r: ['concrete','plaster','tank'].map(__roof.visible),"
                        " b: document.getElementById('bBefore').textContent, p: document.getElementById('bBefore').getAttribute('aria-pressed')}; })()")
    assert on == {"s": [False] * 4, "r": [True] * 3, "b": "After", "p": "true"}
    off = fresh.evaluate("(() => { document.getElementById('bBefore').click(); return [['glass','frame','galv','inverter'].map(__roof.visible),"
                         " document.getElementById('bBefore').textContent]; })()")
    assert off == [[True] * 4, "Before"]


def test_neighbour_block_stands_on_the_sunny_side(fresh):
    n = fresh.evaluate("(() => { document.getElementById('bNeighbour').click(); const b = __roof.world.getObjectByName('neighbour');"
                       " return {p: b.position.toArray(), v: b.visible,"
                       " s: [b.geometry.parameters.width, b.geometry.parameters.height, b.geometry.parameters.depth],"
                       " pressed: document.getElementById('bNeighbour').getAttribute('aria-pressed')}; })()")
    assert n["v"] is True and n["pressed"] == "true"
    assert n["s"] == pytest.approx([22, 18, 10])
    assert n["p"] == pytest.approx([9, -9.6 + 9, 13])                      # plan (9, -13), world z = 13
    off = fresh.evaluate("(document.getElementById('bNeighbour').click(), __roof.world.getObjectByName('neighbour').visible)")
    assert off is False
    again = fresh.evaluate("(__roof.setNeighbour(true), __roof.world.children.filter((c) => c.name === 'neighbour').length)")
    assert again == 1                                                       # built once, then shown and hidden


def test_shots_place_the_camera(fresh):
    for name in ("aerial", "eye", "front", "closeup"):
        got = fresh.evaluate(f"(__roof.shot({json.dumps(name)}), {{c: __roof.camera.position.toArray(), t: __roof.orbit.target.toArray(),"
                             f" f: __roof.camera.fov, want: __roof.shots()[{json.dumps(name)}]}})")
        assert got["c"] == pytest.approx(got["want"]["pos"], abs=1e-3)
        assert got["t"] == pytest.approx(got["want"]["target"], abs=1e-3) and got["f"] == got["want"]["fov"]
    # the eye shot looks up from the street; the orbit's floor limit must not pull it back up
    fresh.evaluate("__roof.shot('eye')")
    fresh.wait_for_function(f"__roof.frames() > {fresh.evaluate('__roof.frames()') + 3}")
    assert fresh.evaluate("__roof.camera.position.y") == pytest.approx(-9.6 + 1.6, abs=1e-3)
    fresh.evaluate("__roof.shot('aerial')")
    assert fresh.evaluate("__roof.orbit.maxPolarAngle") == pytest.approx(math.pi * 0.495)
    p = fresh.evaluate("__roof.panelAt()")
    assert 0 < p[0] < 18 and -12 < p[2] < 0 and 0 < p[1] < 2.5             # on the roof, under 2.5 m


def test_panel_glass_carries_the_panel_face_texture(fresh):
    t = fresh.evaluate("(() => { const m = __roof.parts.glass[0].material; const i = m.map.image;"
                       " return {type: m.type, w: i.width, h: i.height, flip: m.map.flipY, cc: m.clearcoat}; })()")
    assert t["type"] == "MeshPhysicalMaterial" and t["flip"] is False and t["cc"] > 0
    assert t["w"] > t["h"] and t["h"] == 1024                                # canvas turned: long side along x


def test_place_at_scales_and_moves_the_roof(fresh):
    r = fresh.evaluate("__roof.placeAt([1, 0.7, -2], 0.03)")
    assert r["scale"] == 0.03
    pos = fresh.evaluate("__roof.world.position.toArray()")
    assert pos == pytest.approx([1 - 9 * 0.03, 0.7, -2 + 6 * 0.03])
    assert fresh.evaluate("document.getElementById('bArSize').textContent") == "Full size"
    full = fresh.evaluate("__roof.placeAt([0, 0, 0], 1)")
    assert r["size"] == pytest.approx([v * 0.03 for v in full["size"]], rel=1e-6)
    assert full["size"][0] > 18 and full["size"][2] > 12 and fresh.evaluate("document.getElementById('bArSize').textContent") == "Table size"


def test_ar_button_explains_when_ar_is_missing(fresh):
    fresh.evaluate("document.getElementById('bAR').click()")
    fresh.wait_for_function("document.getElementById('msg').textContent.length > 0", timeout=10000)
    assert "ARCore" in fresh.evaluate("document.getElementById('msg').textContent")


def test_stills_script_writes_pngs(chrome_available, tmp_path):
    # its own process: the page fixture already holds Playwright's sync loop in this one
    from PIL import Image
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "roof_stills.py"), "--out", str(tmp_path), "--width", "320",
                        "--software", "--shots", "aerial,closeup", "--month", "6", "--hour", "10"],
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "1/2 (50%) aerial" in out and "2/2 (100%) closeup" in out and "41 panels" in out
    for n in ("aerial", "closeup"):
        im = Image.open(tmp_path / f"{n}.png")
        assert im.size == (320, 180)
        assert len(set(im.convert("L").getdata())) > 20                     # a picture, not a blank frame
    assert not (tmp_path / "eye.png").exists()
