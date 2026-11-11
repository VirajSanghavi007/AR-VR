"""vr.html (Phase 2 and Phase 3 viewer) in a real headless Chrome: both phases load and can be seen in
Orbit, Cardboard VR and AR (table size or full size). Unit (static), smoke (page loads), integration
(Chrome + three.js from the CDN + the shipped models)."""
import functools
import http.server
import json
import re
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
HTML = (PREVIEW / "vr.html").read_text(encoding="utf-8")
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)


# ---------- unit ----------

def test_both_phases_point_at_shipped_models():
    for n in ("2", "3"):
        block = re.search(rf"\n  {n}: \{{(.*?)\}},", HTML).group(1)
        for f in re.findall(r'"(models/[^"]+)"', block):
            assert (PREVIEW / f).is_file(), f
    j = json.loads((PREVIEW / "models" / "phase3_terrace.json").read_text())
    assert (PREVIEW / "models" / j["scan"]).is_file()


def test_phase_links_and_ar_controls_exist():
    assert 'href="?phase=2"' in HTML and 'href="?phase=3"' in HTML
    for b in ("bOrbit", "bVR", "bAR", "bScan", "bArSize", "bArMove", "bArExit"):
        assert f'id="{b}"' in HTML, b
    assert '"beforexrselect"' in HTML                    # overlay taps must not also place the model


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
def browser(chrome_available):
    from playwright.sync_api import sync_playwright
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PREVIEW))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield b, f"http://127.0.0.1:{srv.server_address[1]}/vr.html"
        b.close()
    srv.shutdown()


@pytest.fixture(scope="module", params=[2, 3])
def page(request, browser):
    b, url = browser
    pg = b.new_page(viewport={"width": 800, "height": 450})
    pg.goto(f"{url}?phase={request.param}", timeout=90000)
    pg.wait_for_function("window.__vr && __vr.world.children.length > 0 && /Phase/.test(document.getElementById('info').textContent)"
                         + (" && __vr.scanCount() > 0" if request.param == 3 else ""), timeout=60000)
    yield request.param, pg
    pg.close()


def test_page_loads_its_phase(page):
    phase, pg = page
    s = pg.evaluate("({ phase: __vr.PHASE, info: document.getElementById('info').textContent,"
                    "  on: document.querySelector('#phases a.on').id, scan: !document.getElementById('bScan').hidden })")
    assert s["phase"] == phase and s["on"] == f"p{phase}" and s["info"].startswith(f"Phase {phase}")
    assert s["scan"] == (phase == 3)
    if phase == 3:
        assert "39 panels" in s["info"] and "15.6 kWp" in s["info"] and "LiDAR" in s["info"]



def test_phase2_view_is_sized_from_the_roof_cells(page):
    phase, pg = page
    if phase == 3:
        pytest.skip("Phase 3 has its own fixed frame")
    cells = json.loads((PREVIEW / "models" / "rooftop_solar.json").read_text(encoding="utf-8"))["cells"]
    sx, sy = max(c[2] for c in cells), max(c[3] for c in cells)
    s = pg.evaluate("""(() => {
      const kids = __vr.surround.children;
      const flat = kids.filter((o) => o.geometry && o.geometry.type === 'PlaneGeometry')
        .map((o) => [o.position.x, o.position.z, o.geometry.parameters.width, o.geometry.parameters.height]);
      const trees = kids.filter((o) => o.isGroup).map((t) => {           // plan box [x0, x1, z0, z1] of each tree
        t.updateMatrixWorld(true);
        const r = [Infinity, -Infinity, Infinity, -Infinity];
        t.traverse((m) => {
          if (!m.isMesh) return;
          m.geometry.computeBoundingBox();
          const b = m.geometry.boundingBox.clone().applyMatrix4(m.matrixWorld);
          r[0] = Math.min(r[0], b.min.x); r[1] = Math.max(r[1], b.max.x);
          r[2] = Math.min(r[2], b.min.z); r[3] = Math.max(r[3], b.max.z);
        });
        return r;
      });
      const body = kids.filter((o) => o.geometry && o.geometry.type === 'BoxGeometry' && o.geometry.parameters.height === 9
                        && Math.abs(o.position.y + 4.501) < 1e-9)            // not the 9 m neighbour block
        .map((o) => [o.position.x, o.position.z, o.geometry.parameters.width, o.geometry.parameters.depth]);
      return { centre: __vr.CENTRE.toArray(), target: __vr.orbit.target.toArray(), table: __vr.AR_TABLE, flat, trees, body };
    })()""")
    # the 9 m building under the roof: one block per roof cell, at the cell's place and size
    want = [[(c[0] + c[2]) / 2, -(c[1] + c[3]) / 2, c[2] - c[0], c[3] - c[1]] for c in cells]
    assert len(s["body"]) == len(want)
    for got, w in zip(s["body"], want):
        assert got == pytest.approx(w)
    assert s["centre"] == pytest.approx([sx / 2, 0.15, -sy / 2]) and s["target"] == pytest.approx(s["centre"])
    assert s["table"] == pytest.approx(0.5 / max(sx, sy))                    # about 50 cm on the long side
    assert [sx / 2, -sy / 2, 900, 900] in [pytest.approx(f) for f in s["flat"]]          # grass under the roof
    assert [sx / 2, -sy / 2, sx + 10, sy + 10] in [pytest.approx(f) for f in s["flat"]]  # paving 5 m round it
    assert len(s["trees"]) == 8
    for x0, x1, z0, z1 in s["trees"]:                                        # no tree canopy reaches the walls
        assert x1 < -0.5 or x0 > sx + 0.5 or z1 < -sy - 0.5 or z0 > 0.5, (x0, x1, z0, z1)


def test_ar_places_a_table_model_and_a_full_size_one(page):
    phase, pg = page
    cells = json.loads((PREVIEW / "models" / "rooftop_solar.json")
                       .read_text(encoding="utf-8"))["cells"]
    full = {2: max(max(c[2], c[3]) for c in cells), 3: 16.69}[phase]     # Phase 2: the published roof's long side
    t = pg.evaluate("__vr.placeAt([0.5, 0.8, -1], __vr.AR_TABLE)")
    assert t["placed"] and abs(max(t["size"]) - full * t["scale"]) < 0.02 and max(t["size"]) < 0.6
    assert abs(t["centre"][0] - 0.5) < 0.05 and abs(t["centre"][2] + 1) < 0.05
    f = pg.evaluate("__vr.toggleArSize()")
    assert f["scale"] == 1 and abs(max(f["size"]) - full) < 0.05 and f["at"] == [0.5, 0.8, -1]
    assert abs(f["centre"][0] - 0.5) < 0.1 and abs(f["centre"][2] + 1) < 0.1
    assert pg.evaluate("__vr.toggleArSize().scale") == pytest.approx(t["scale"])
    assert pg.evaluate("document.getElementById('bArSize').textContent") == "Full size"


def test_ar_size_toggle_before_placing_only_sets_the_size(browser):
    b, url = browser
    pg = b.new_page(viewport={"width": 800, "height": 450})
    pg.goto(url, timeout=90000)
    pg.wait_for_function("window.__vr && __vr.world.children.length > 0", timeout=60000)
    s = pg.evaluate("__vr.toggleArSize()")
    assert s["scale"] == 1 and not s["placed"] and s["at"] is None
    assert pg.evaluate("[__vr.world.scale.x, document.getElementById('bArSize').textContent]") == [1, "Table size"]
    pg.close()


def test_scan_layer_shrinks_with_the_model_and_toggles(page):
    phase, pg = page
    if phase == 2:
        pytest.skip("Phase 2 has no scan")
    assert pg.evaluate("__vr.scanCount()") == 120000
    size = pg.evaluate("(__vr.placeAt([0, 0, 0], 1), __vr.world.getObjectByName('scan').material.size)")
    small = pg.evaluate("(__vr.placeAt([0, 0, 0], __vr.AR_TABLE), __vr.world.getObjectByName('scan').material.size)")
    assert size == pytest.approx(0.035) and small == pytest.approx(0.003)
    pg.evaluate("__vr.showScan(false)")
    assert pg.evaluate("[__vr.world.getObjectByName('scan').visible, document.getElementById('bScan').textContent]") == [False, "Scan hidden"]
    pg.evaluate("document.getElementById('bScan').click()")
    assert pg.evaluate("__vr.world.getObjectByName('scan').visible")


def test_height_colours_go_blue_to_red(page):
    _, pg = page
    low, high, mid = pg.evaluate("[0, 12, 6].map((y) => __vr.heightColour(y).toArray())")
    assert low[2] > low[0] and high[0] > high[2] and mid[1] > mid[0] and mid[1] > mid[2]


def test_cardboard_vr_stands_on_the_roof_and_returns(page):
    phase, pg = page
    s = pg.evaluate("""async () => { await __vr.startVR(); const a = { mode: __vr.mode(), pos: __vr.camera.position.toArray(),
                       stand: __vr.STAND.toArray(), bar: document.getElementById('bar').className };
                       __vr.stopVR(); return { ...a, after: __vr.mode(), bar2: document.getElementById('bar').className }; }""")
    assert s["mode"] == "vr" and s["pos"] == s["stand"] and s["bar"] == "hide"
    assert s["after"] == "orbit" and s["bar2"] == ""
    roof_y = {2: 0.15, 3: 0.0}[phase]
    assert s["stand"][1] == pytest.approx(roof_y + 1.6)                 # eye height above the roof


def test_ar_button_without_webxr_explains(page):
    _, pg = page
    pg.evaluate("document.getElementById('bAR').click()")
    pg.wait_for_function("document.getElementById('msg').textContent.length > 0", timeout=5000)
    assert "ARCore" in pg.evaluate("document.getElementById('msg').textContent")
