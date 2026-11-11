"""M6 proposal stills: render preview/roof.html in headless Chrome and save the four customer views
(aerial, eye level, front, panel close-up) as PNGs, 1920 px wide by default.

    python scripts/roof_stills.py                       # outputs/sample_roof/stills/*.png
    python scripts/roof_stills.py --month 6 --hour 10 --neighbour --width 1280 --out somewhere

Serves preview/ on a local port, opens roof.html?still=1 (no HUD), sets the sun and each camera shot
through window.__roof, waits a few frames and screenshots the page. Uses the GPU (ANGLE on D3D11)
unless --software (SwiftShader, slower; what the tests use).
"""
import argparse
import functools
import http.server
import json
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
SHOTS = ("aerial", "eye", "front", "closeup")
GPU_ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]
SOFT_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


def find_chrome():
    for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
              r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
              shutil.which("chrome"), shutil.which("google-chrome")):
        if p and Path(p).is_file():
            return p
    return None


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that does not log every request to stderr."""

    def log_message(self, *args):
        pass


def serve(directory):
    """Start a quiet static server on a free port; returns (server, base url)."""
    handler = functools.partial(QuietHandler, directory=str(directory))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def page_url(base, month=12, hour=None, neighbour=False, glb=None):
    q = [f"still=1", f"month={month}"]
    if hour is not None:
        q.append(f"hour={hour}")
    if neighbour:
        q.append("neighbour=1")
    if glb:
        q.append(f"glb={glb}")
    return f"{base}/roof.html?" + "&".join(q)


def render(out, width=1920, height=None, month=12, hour=None, neighbour=False, shots=SHOTS,
           software=False, chrome=None, glb=None, timeout=90):
    """Render each shot to out/<shot>.png; returns {shot: path} and the page's state."""
    from playwright.sync_api import sync_playwright

    height = height or round(width * 9 / 16)
    chrome = chrome or find_chrome()
    if not chrome:
        raise RuntimeError("Chrome not found")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    srv, base = serve(PREVIEW)
    saved, t0 = {}, time.time()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=chrome, headless=True, args=SOFT_ARGS if software else GPU_ARGS)
            pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            pg.goto(page_url(base, month, hour, neighbour, glb))
            pg.wait_for_function("window.__roof && __roof.state.ready", timeout=timeout * 1000)
            info = pg.evaluate("({gl: (() => { const g = __roof.scene && document.querySelector('canvas').getContext('webgl2');"
                               " const d = g && g.getExtension('WEBGL_debug_renderer_info');"
                               " return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : ''; })(),"
                               " panels: __roof.state.panels, sun: __roof.state.sun})")
            for i, name in enumerate(shots, 1):
                pg.evaluate(f"__roof.shot({json.dumps(name)})")
                f0 = pg.evaluate("__roof.frames()")
                pg.wait_for_function(f"__roof.frames() > {f0 + 4}", timeout=timeout * 1000)
                path = out / f"{name}.png"
                pg.screenshot(path=str(path))
                saved[name] = path
                el = time.time() - t0
                print(f"{i}/{len(shots)} ({100 * i // len(shots)}%) {name} - {el:.0f}s, eta {el / i * (len(shots) - i):.0f}s",
                      flush=True)
            b.close()
    finally:
        srv.shutdown()
    return saved, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sample_roof" / "stills"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--month", type=int, default=12)
    ap.add_argument("--hour", type=float, default=None, help="local clock hour, default solar noon")
    ap.add_argument("--neighbour", action="store_true")
    ap.add_argument("--software", action="store_true", help="SwiftShader instead of the GPU")
    ap.add_argument("--shots", default=",".join(SHOTS))
    a = ap.parse_args(argv)
    shots = [s for s in a.shots.split(",") if s]
    bad = [s for s in shots if s not in SHOTS]
    if bad:
        ap.error(f"unknown shot(s) {bad}; choose from {list(SHOTS)}")
    saved, info = render(a.out, a.width, None, a.month, a.hour, a.neighbour, shots, a.software)
    print(f"renderer: {info['gl'] or 'unknown'}; {info['panels']} panels")
    for k, v in saved.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
