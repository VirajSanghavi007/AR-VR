"""Pictures for the phase pages (preview/phase1.html, phase2.html, phase3.html), taken from the live viewers.

Each shot opens a page of preview/ in headless Chrome, runs its setup (statements, may await; the pages' own __tagar / __vr hooks),
waits until the scene is ready and saves a JPEG to preview/phases/. The Phase 3 roof plan is copied from
design_panels.py's panels_plan.png when that exists.
"""
import contextlib
import functools
import http.server
import shutil
import threading
from pathlib import Path

import cv2

CHROME_PATHS = (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe")
CHROME_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
VIEWPORT = {"width": 1200, "height": 750}

# Phase 1: a 2 x 3 panel array locked to four tags on a table. Phase 2 / 3: the vr.html viewers.
# Phase 3 room: the office scan room behind tags on its back wall, seen from high in the far corner.
SHOTS = [
    {"name": "phase1_ar.jpg", "page": "tagar.html?samples=0",
     "ready": "window.__tagar",
     "setup": "__tagar.simulate([[-0.5, 0, -0.3], [0.5, 0, -0.3], [0.5, 0, 0.3], [-0.5, 0, 0.3]]);"
              "for (const [id, v] of [['rows', 2], ['cols', 3]]) { const e = document.getElementById(id);"
              " e.value = v; e.dispatchEvent(new Event('input')); }",
     "settle_ms": 1500},
    {"name": "phase2_roof.jpg", "page": "vr.html?phase=2",
     "ready": "window.__vr && __vr.world.children.length > 0 && /Phase/.test(document.getElementById('info').textContent)",
     "setup": "", "settle_ms": 2500},
    {"name": "phase3_terrace.jpg", "page": "vr.html?phase=3",
     "ready": "window.__vr && __vr.world.children.length > 0 && __vr.scanCount() > 0",
     "setup": "", "settle_ms": 2500},
    {"name": "phase3_room.jpg", "page": "tagar.html?room=1",
     "ready": "window.__tagar",
     "setup": "await __tagar.samples(); __tagar.simulate([[-0.5, 1.6, -1], [0.5, 1.6, -1], [0.5, 1.0, -1],"
              " [-0.5, 1.0, -1]], [0, 0, 1]); __tagar.setRoom({ seeThrough: false });"
              " document.getElementById('overlay').style.visibility = 'hidden';"
              " __tagar.roomView([0.4, 4.3, -2.6], [0.3, 0.3, 0.6]);",
     "keep_view": "__tagar.roomView([0.4, 4.3, -2.6], [0.3, 0.3, 0.6])",
     "settle_ms": 1500},
]


def find_chrome():
    return next((p for p in (*CHROME_PATHS, shutil.which("chrome"), shutil.which("google-chrome"))
                 if p and Path(p).is_file()), None)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextlib.contextmanager
def serve(directory):
    """Static server over `directory` on a free port; yields its base URL (ending in /)."""
    handler = functools.partial(_QuietHandler, directory=str(directory))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
    finally:
        srv.shutdown()


def to_jpeg(src, dst, max_w=1200, quality=82):
    """Re-encode an image as JPEG no wider than max_w px. Returns dst."""
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return Path(dst)


def capture(shots, preview_dir, out_dir, chrome, quality=82, timeout_ms=90000, log=print):
    """Take each shot in headless Chrome; returns the written paths."""
    from playwright.sync_api import sync_playwright
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with serve(preview_dir) as base, sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True, args=CHROME_ARGS)
        try:
            for i, s in enumerate(shots, 1):
                pg = browser.new_page(viewport=VIEWPORT)
                pg.goto(base + s["page"])
                pg.wait_for_function(s["ready"], timeout=timeout_ms)
                pg.evaluate(f"(async () => {{ {s['setup']} }})()")
                pg.wait_for_timeout(s.get("settle_ms", 1000))
                if s.get("keep_view"):
                    pg.evaluate(s["keep_view"])
                png = out_dir / (Path(s["name"]).stem + ".png")
                pg.screenshot(path=str(png))
                pg.close()
                paths.append(to_jpeg(png, out_dir / s["name"], quality=quality))
                png.unlink()
                log(f"{i}/{len(shots)} ({100 * i // len(shots)}%) {s['name']}")
        finally:
            browser.close()
    return paths
