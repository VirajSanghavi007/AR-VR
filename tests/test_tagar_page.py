"""tagar.html in a real headless Chrome: the desktop checks (simulate, emuStart) show the canvas at window size
on a high-DPI screen. Unit (static), smoke (page loads, hooks run), integration (Chrome + three.js from the CDN)."""
import functools
import http.server
import re
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
HTML = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)
W, H, DPR = 800, 450, 1.75


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
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=DPR)
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/tagar.html?samples=0")
        pg.wait_for_function("window.__tagar && window.AR", timeout=30000)
        yield pg
        b.close()
    srv.shutdown()


def canvas(pg):
    return pg.evaluate("""() => { const c = document.querySelector('body > canvas'); const r = c.getBoundingClientRect();
                                 return { w: r.width, h: r.height, left: r.left, top: r.top, px: c.width,
                                          pos: getComputedStyle(c).position, z: getComputedStyle(c).zIndex }; }""")


# ---------- unit ----------

def test_desktop_checks_keep_the_canvas_size():
    assert "style.cssText" not in HTML                  # wiping the style drops setSize's CSS width/height
    body = re.search(r"function fillScreen\(\) \{(.*?)\n\}", HTML, re.S).group(1)
    assert "setSize(innerWidth, innerHeight)" in body and 'position: "fixed"' in body
    for hook in ("simulate(", "emuStart("):
        start = HTML.index(f"  {hook}")
        assert "fillScreen();" in HTML[start:HTML.index("return this.state();", start)], hook


# ---------- smoke + integration ----------

@pytest.mark.parametrize("call", ["window.__tagar.emuStart()", "window.__tagar.simulate()"])
def test_canvas_fills_the_window_on_a_high_dpi_screen(page, call):
    page.evaluate(call)
    c = canvas(page)
    assert (c["w"], c["h"], c["left"], c["top"]) == (W, H, 0, 0)
    assert c["px"] == round(W * DPR)                     # still drawn at full device resolution
    assert c["pos"] == "fixed" and c["z"] == "0"
