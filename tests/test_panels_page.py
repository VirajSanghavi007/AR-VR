"""panels.html (solar panels in AR on up to 8 AprilTags, walk all round) in a real headless Chrome with the phone
emulator. Unit (static: imports, tag limit, hooks), smoke (page loads, emuStart runs), integration (Chrome +
three.js from the CDN + js-aruco2 detection on rendered tags: 8 tags lock on the 4 corners, then a full
circle with 1 to 3 tags in view and drifting tracking keeps the panels on the tags)."""
import functools
import http.server
import re
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
HTML = (PREVIEW / "panels.html").read_text(encoding="utf-8")
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)
EIGHT = "[{id:0,at:[-0.5,-0.3]},{id:1,at:[0.5,-0.3]},{id:2,at:[0.5,0.3]},{id:3,at:[-0.5,0.3]},"\
        "{id:4,at:[0,-0.6]},{id:5,at:[0,0.6]},{id:6,at:[-0.9,0]},{id:7,at:[0.9,0]}]"


# ---------- unit ----------

def test_imports_resolve_and_versions_match():
    for f in re.findall(r'from "\./([a-z0-9]+\.js)\?v=\d+"', HTML):
        assert (PREVIEW / f).is_file(), f
    # one version of each shared module across the pages and modules that import it
    for mod in ("tagrect.js", "tagmap.js"):
        vers = set()
        for f in list(PREVIEW.glob("*.html")) + list(PREVIEW.glob("*.js")):
            vers |= set(re.findall(rf"{re.escape(mod)}\?v=(\d+)", f.read_text(encoding="utf-8")))
        assert len(vers) == 1, (mod, vers)


def test_up_to_8_tags_and_the_walk_hooks():
    assert "chooseTags(found, MAX_TAGS)" in HTML and "bestRectangle(" in HTML
    assert "walkStep(walk, seen)" in HTML and "lockPoseNow(walk)" in HTML
    for hook in ("emuStart(", "emuRun(", "emuWalk(", "emuShow("):
        assert f"  {hook}" in HTML, hook
    assert "Not all the locked tags are in view" not in HTML          # one mapped tag in view is enough now
    assert "requiredFeatures" in HTML and "immersive-ar" in HTML


def test_menu_pages_link_it():
    assert 'href="panels.html"' in (PREVIEW / "index.html").read_text(encoding="utf-8")
    assert 'href="panels.html"' in (PREVIEW / "phase1.html").read_text(encoding="utf-8")


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
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PREVIEW))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 960, "height": 600})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/panels.html")
        pg.wait_for_function("window.__tagar && window.AR", timeout=30000)
        pg.errors = errors
        yield pg
        b.close()
    srv.shutdown()


def test_page_loads_and_emulator_starts(page):
    s = page.evaluate("__tagar.emuStart({seed: 1})")
    assert s["locked"] is False and s["walk"] is None
    assert page.title() == "Solar Panels AR"
    assert page.errors == []


def test_8_tags_lock_on_the_corners_and_a_full_circle_holds(page):
    page.evaluate("__tagar.emuStart({seed: 2})")
    r = page.evaluate(f"__tagar.emuRun({{tags: {EIGHT}, steps: 14}})")
    assert r["locked"] and r["rect"]["ids"] == "0,1,2,3"
    assert sorted([r["rect"]["width"], r["rect"]["height"]]) == [pytest.approx(0.6, abs=0.01), pytest.approx(1.0, abs=0.01)]
    assert r["walk"]["origin"] == [0, 1, 2, 3] and 4 in r["walk"]["mapped"]   # the extra tag in view joins the map
    # 2 laps round the area, looking at its near edge (1 to 3 tags in view), tracking drifting 25 mm and 1 degree per metre
    w = page.evaluate(f"__tagar.emuWalk({{tags: {EIGHT}, radius: 1.2, eyeY: 0.6, lookIn: 0.75, laps: 2, mmPerM: 25, degPerM: 1.0}})")
    assert w["mapped"] == list(range(8))
    assert w["held"] >= w["stops"] - 2 and w["stops"] == 48                  # the stop seeing only tag 5 at the frame edge may miss
    assert max(len(r["seen"]) for r in w["rows"]) <= 4 and min(len(r["seen"]) for r in w["rows"]) == 1
    assert w["maxRawMm"] > 50                                                 # tracking alone would drift this far
    assert w["maxMm"] < 25 and w["maxMm"] < w["maxRawMm"] / 2
    assert max(r["mm"] for r in w["rows"] if r["lap"] == 1) < 25             # no growth on the second lap
    assert "8 tags mapped" in w["hud"]
    assert page.errors == []
