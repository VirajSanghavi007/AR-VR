"""site.html (Site Designer AR) and viewer.html in a real headless Chrome, with the phone emulator standing in
for the phone. Unit (static: the page wires every button and hook), smoke (pages load, hooks run), integration
(Chrome + three.js from the CDN + the AprilTag detector + tagmap.js + blocks.js: lock, draw, size check,
options, undo, tape check, save/open, photo, video, GLB, model import and alignment, the viewer)."""
import functools
import hashlib
import http.server
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.request
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
SITE = (PREVIEW / "site.html").read_text(encoding="utf-8")
VIEWER = (PREVIEW / "viewer.html").read_text(encoding="utf-8")
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)
EMPTY = json.dumps({"kind": "arvr-site-design", "version": 1})
BACK = {"eye": [0, 1.4, -1.5], "look": [0, 1.3, -2.5]}         # facing the back wall's two tags (ids 0, 1)
FLOOR = {"eye": [0, 1.55, 0.9], "look": [0, 0, -0.9]}          # looking down at the floor in front of it
CDN_CACHE = Path(tempfile.gettempdir()) / "arvr_test_cdn"       # three.js etc. fetched once, shared by xdist workers


# ---------- unit (static) ----------

def test_every_button_the_hooks_click_exists_once():
    ids = ["b-lock", "b-map", "b-floor", "b-reset", "b-outline", "b-line", "b-done", "b-cancel", "b-left", "b-right",
           "b-drag", "b-w-", "b-w+", "b-h-", "b-h+", "b-mat", "b-del", "b-before", "b-A", "b-B", "b-C", "b-copy",
           "b-labels", "b-photo", "b-video", "b-save", "b-open-btn", "b-open", "b-glb", "b-import", "b-model",
           "b-align", "b-tape", "b-check", "b-undo", "b-exit"]
    for i in ids:
        assert len(re.findall(rf'id="{re.escape(i)}"', SITE)) == 1, i


def test_hooks_and_shared_modules():
    hooks = re.search(r"window\.__site = \{(.*?)\n\};", SITE, re.S).group(1)
    for h in ("emuStart(", "emuLook(", "lock()", "emuWalk(", "sizeCheck(", "trueEnds(", "tapRoom(", "addBlock(",
              "option(", "photo()", "async video(", "record:", "open:", "glb:", "importModel:", "align(", "tapeCheck:",
              "undo()", "state()"):
        assert h in hooks, h
    for mod in ("./tagmap.js", "./blocks.js", "./build3d.js", "./phoneemu.js"):
        assert mod in SITE, mod
    assert "./build3d.js" in VIEWER and "samples/site_design_sample.json" in VIEWER


# ---------- browser fixtures ----------

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
        yield b, f"http://127.0.0.1:{srv.server_address[1]}"
        b.close()
    srv.shutdown()


def cdn_body(url):
    """A CDN file from the disk cache, fetched on a miss (written to a temp name, then renamed, so workers never
    read half a file)."""
    key = hashlib.sha1(url.encode()).hexdigest()
    body, ctype = CDN_CACHE / key, CDN_CACHE / f"{key}.type"
    if not (body.is_file() and ctype.is_file()):
        CDN_CACHE.mkdir(exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "arvr-tests"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data, t = r.read(), r.headers.get("Content-Type", "application/javascript")
        for path, value in ((body, data), (ctype, t.encode())):
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(value)
            try:
                os.replace(tmp, path)
            except OSError:                       # Windows: another worker has it open; theirs is the same file
                tmp.unlink(missing_ok=True)
        return data, t
    return body.read_bytes(), ctype.read_text()


def new_page(b):
    """A page whose CDN requests come from the shared disk cache: 20 workers loading three.js at once from the
    network made the first page load time out."""
    pg = b.new_page(viewport={"width": 800, "height": 450})

    def serve(route):
        try:
            data, t = cdn_body(route.request.url)
        except OSError:
            return route.continue_()
        route.fulfill(status=200, body=data, headers={"Content-Type": t, "Access-Control-Allow-Origin": "*"})
    pg.route("https://cdn.jsdelivr.net/**", serve)
    return pg


@pytest.fixture(scope="module")
def page(browser):
    b, base = browser
    pg = new_page(b)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(f"{base}/site.html?samples=0")
    pg.wait_for_function("window.__site", timeout=60000)
    pg.errors = errors
    yield pg
    pg.close()


def S(pg, call, arg=None):
    return pg.evaluate(f"(a) => window.__site.{call}", arg) if arg is not None else pg.evaluate(f"() => window.__site.{call}")


@pytest.fixture
def site(page):
    """A fresh start for every test: empty design, new emulated room, locked on the back wall, floor in view."""
    S(page, "emuStart({seed: 1})")                           # builds the scene on first use
    S(page, f"open({json.dumps(EMPTY)})")
    S(page, "emuStart({seed: 1})")
    S(page, "emuLook(a)", {**BACK, "steps": 6})
    assert S(page, "lock()") is not None
    S(page, "emuLook(a)", {**FLOOR, "steps": 2})
    S(page, "setMode('view')")
    return page


def draw_outline(pg, pts):
    S(pg, "setMode('draw')")
    S(pg, "startDraw('slab')")
    for p in pts:
        S(pg, "tapRoom(a)", p)
    return S(pg, "finish()")


SQUARE = [[-0.6, 0.9], [0.6, 0.9], [0.6, 1.7], [-0.6, 1.7]]


# ---------- smoke ----------

def test_page_starts_and_locks_on_the_first_walls_tags(site):
    st = S(site, "state()")
    assert st["locked"] and sorted(st["tags"]) == [0, 1] and st["objects"] == [] and st["counts"] == [0, 0, 0]
    assert st["floorY"] < -0.8                               # the floor found well below the tags (1.3 m up)
    assert site.errors == []


# ---------- integration ----------

def test_size_check_draws_three_metres_where_it_belongs(site):
    r = S(site, "sizeCheck({len: 3})")
    assert r["drawn"] == pytest.approx(3.0, abs=0.06)       # taps snap to 5 cm
    assert r["trueLength"] == pytest.approx(r["drawn"], abs=0.02)
    assert max(r["endErrMm"]) < 60
    st = S(site, "state()")
    assert [o["type"] for o in st["objects"]] == ["walls"] and st["objects"][0]["label"].endswith("m high")


def test_size_check_over_seeds_length_exact_place_within_lock_error(page):
    """Locked on the first wall's two tags only: the drawn length is always right, but where the wall sits
    carries the lock's pose error (8 seeds, 2026-11-11: 5 to 222 mm, 5 of 8 under 20 mm). This pins that."""
    worst = []
    for seed in range(1, 9):
        S(page, f"emuStart({{seed: {seed}}})")
        S(page, f"open({json.dumps(EMPTY)})")
        S(page, "emuLook(a)", {**BACK, "steps": 6})
        assert S(page, "lock()") == [0, 1]
        r = S(page, "sizeCheck({len: 3})")
        assert r["drawn"] == pytest.approx(3.0, abs=1e-9) and r["trueLength"] == pytest.approx(3.0, abs=0.005), seed
        worst.append(max(r["endErrMm"]))
    worst.sort()
    assert worst[len(worst) // 2] < 30 and worst[-1] < 300, worst


def test_outline_drawing_gives_the_tapped_slab(site):
    st = draw_outline(site, SQUARE)
    assert st["tool"] is None and len(st["objects"]) == 1
    o = st["objects"][0]
    assert o["type"] == "slab" and o["w"] == pytest.approx(1.2, abs=0.06) and o["d"] == pytest.approx(0.8, abs=0.06)
    assert (o["x"], o["z"]) == (pytest.approx(0, abs=0.05), pytest.approx(1.3, abs=0.05))


def test_blocks_edit_buttons_and_undo(site):
    S(site, "addBlock('bench', 0, 1.2)")
    st = S(site, "addBlock('planter', 0.8, 1.2)")
    assert [o["type"] for o in st["objects"]] == ["bench", "planter"] and st["sel"] == 1
    S(site, "select(0)")
    w0 = S(site, "state()")["objects"][0]["w"]
    assert S(site, "click('b-w+')")["objects"][0]["w"] > w0
    assert S(site, "click('b-left')")["objects"][0]["yaw"] > 0
    assert S(site, "click('b-mat')")["objects"][0]["mat"] == "glass"          # wood -> glass
    h0 = S(site, "state()")["objects"][0]["h"]
    assert S(site, "dragHeight(100)")["objects"][0]["h"] == pytest.approx(h0 + 0.5)
    st = S(site, "click('b-del')")
    assert [o["type"] for o in st["objects"]] == ["planter"]
    st = S(site, "undo()")
    assert [o["type"] for o in st["objects"]] == ["bench", "planter"] and st["objects"][0]["h"] == pytest.approx(h0 + 0.5)
    assert S(site, "undo()")["objects"][0]["h"] == pytest.approx(h0)


def test_options_copy_before_and_size_labels(site):
    S(site, "addBlock('pergola', 0, 1.5)")
    st = S(site, "click('b-copy')")
    assert st["active"] == "B" and st["counts"] == [1, 1, 0]
    S(site, "addBlock('bench', 0, 1)")
    assert S(site, "option('A')")["counts"] == [1, 2, 0]
    assert [o["type"] for o in S(site, "state()")["objects"]] == ["pergola"]
    assert S(site, "option('C')")["objects"] == []
    assert S(site, "click('b-before')")["before"] is True and S(site, "click('b-before')")["before"] is False
    assert S(site, "click('b-labels')")["labels"] is True


def test_tape_check_save_and_reopen_in_place(site):
    r = S(site, "sizeCheck({len: 3})")
    S(site, "select(0)")
    c = S(site, "tapeCheck(3.0)")
    assert c["what"] == "Wall line" and c["tape"] == 3.0 and c["errMm"] == round((c["drawn"] - 3.0) * 1000)
    assert set(c["tags"]) <= {0, 1}
    S(site, "addBlock('door', 1, 0.2)")
    rec = S(site, "record()")
    assert rec["kind"] == "arvr-site-design" and len(rec["checks"]) == 1 and len(rec["options"]["A"]) == 2
    assert sorted(map(int, rec["map"]["tags"])) == [0, 1]

    # a new visit: new room session, the saved file opened, then any mapped tag seen puts the design back
    S(site, "emuStart({seed: 4})")
    S(site, "tapeCheck(2.5)")                                    # a check made after saving is not in the file
    out = S(site, "open(a)", json.dumps(rec))
    assert out == {"objects": [2, 0, 0], "tags": 2}
    assert S(site, "state()")["locked"] is False
    S(site, "emuLook(a)", {**BACK, "steps": 4})
    st = S(site, "state()")
    assert st["locked"] and [o["type"] for o in st["objects"]] == ["walls", "door"]
    ends = S(site, "endsOf(0)")
    assert max(ends["endErrMm"]) < max(60, max(r["endErrMm"]) + 40)
    assert S(site, "record()")["checks"] == rec["checks"]


def test_tape_check_needs_a_selection_and_a_length(site):
    assert S(site, "tapeCheck(3)") is None
    S(site, "addBlock('bench', 0, 1)")
    S(site, "select(-1)")
    assert S(site, "tapeCheck(3)") is None
    S(site, "select(0)")
    assert S(site, "tapeCheck(0)") is None and S(site, "tapeCheck(NaN)") is None
    assert S(site, "tapeCheck(1.6)")["errMm"] == 0
    assert S(site, "tapeCheck(1.5)")["errMm"] == 100 and S(site, "tapeCheck(1.7)")["errMm"] == -100   # drawn minus tape


def test_photo_video_and_glb(site):
    S(site, "addBlock('panel', 0, 1.3)")
    S(site, "photo()")
    site.wait_for_function("!document.getElementById('shot').hidden", timeout=10000)
    name = site.evaluate("() => document.getElementById('shot-save').download")
    assert re.fullmatch(r"site-design-[\d-]+\.png", name), name
    site.evaluate("() => document.getElementById('shot-close').click()")
    v = site.evaluate("() => window.__site.video(30)")
    assert v and v["bytes"] > 1000 and v["frames"] >= 25 and v["name"].startswith("site-design-")
    site.wait_for_function("document.getElementById('shot-vid').videoWidth > 0", timeout=15000)
    decoded = site.evaluate("""() => new Promise((done) => {          // play it through: the file holds many frames,
      const el = document.getElementById('shot-vid');                  // not one still (frames are pushed by hand)
      el.muted = true; el.currentTime = 0; el.playbackRate = 4;
      el.onended = () => done(el.getVideoPlaybackQuality().totalVideoFrames);
      el.play();
      setTimeout(() => done(el.getVideoPlaybackQuality().totalVideoFrames), 20000);
    })""")
    assert decoded >= 10, decoded
    g = site.evaluate("() => window.__site.glb()")
    assert g["bytes"] > 1000 and g["name"].endswith(".glb")


def test_imported_model_aligns_two_corners_to_the_site(site):
    m = site.evaluate("() => window.__site.importModel('models/sample_sketchup_pergola.dae', 'sample_sketchup_pergola.dae')")
    assert m["size"] == pytest.approx([3.0, 2.4, 2.7], abs=0.15) or sorted(m["size"]) == pytest.approx(sorted([3.0, 2.4, 2.7]), abs=0.15)
    box = S(site, "modelBox()")
    lo, hi = box["min"], box["max"]
    m1, m2 = [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]]
    r1, r2 = [-1.0, 0, 1.0], [-1.0, 0, 4.0]                 # the model's long side runs away from the back wall
    a = site.evaluate("([m1, r1, m2, r2]) => window.__site.align(m1, r1, m2, r2)", [m1, r1, m2, r2])
    assert a["lengthModel"] == pytest.approx(hi[0] - lo[0]) and a["lengthReal"] == pytest.approx(3.0)
    rec = S(site, "record()")
    assert rec["model"]["name"] == "sample_sketchup_pergola.dae" and rec["model"]["align"]["yaw"] == pytest.approx(a["yaw"])


def test_viewer_shows_the_sample_and_switches_options(browser):
    b, base = browser
    pg = new_page(b)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(f"{base}/viewer.html")
    pg.wait_for_function("window.__viewer && window.__viewer.state().objects > 0", timeout=30000)
    st = pg.evaluate("() => window.__viewer.state()")
    sample = json.loads((PREVIEW / "samples" / "site_design_sample.json").read_text(encoding="utf-8"))
    assert st["name"] == "site_design_sample.json" and st["opt"] == sample.get("active", "A")
    assert st["objects"] == len(sample["options"][st["opt"]])
    for k in ("A", "B", "C"):
        if sample["options"].get(k):
            assert pg.evaluate(f"() => window.__viewer.option('{k}')")["objects"] == len(sample["options"][k])
    assert errors == []
    pg.close()
