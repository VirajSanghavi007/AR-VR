"""Smoke test: run_preview.bat serves the site: the root sends you to the tabletop AR, and every
file the AR and tag-check pages load is there."""
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch file")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_preview_bat_serves_the_ar_pages():
    port = _free_port()
    env = {**os.environ, "PORT": str(port), "NO_BROWSER": "1"}
    proc = subprocess.Popen(["cmd", "/c", str(ROOT / "run_preview.bat")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}/"
        for _ in range(100):
            try:
                html = urllib.request.urlopen(base, timeout=1).read().decode()
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail(f"server never answered: {proc.stdout.read1(4000) if proc.poll() is not None else ''}")
        assert 'href="phase1.html"' in html and 'href="tagar.html"' in html   # the root is the phase menu
        for page in ("tagar.html", "tags.html", "site.html", "viewer.html"):
            src = urllib.request.urlopen(base + page, timeout=2).read().decode()
            for ref in re.findall(r'(?:src|from)="\.?/?([\w./-]+?\.js)(?:\?[^"]*)?"', src):
                assert urllib.request.urlopen(base + ref, timeout=2).status == 200, ref
        assert urllib.request.urlopen(base + "bipv_shingle.jpg", timeout=2).read()[:2] == b"\xff\xd8"
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def test_site_has_no_legacy_roof_pages():
    for name in ("ar.html", "roofgrid.js", "placement.js", "terrace_anchor.json", "terrace_tilted.glb", "panel_cells.png"):
        assert not (PREVIEW / name).exists(), name
