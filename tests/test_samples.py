"""Sample files the pages load by themselves while testing: tagar.html's sample model + room (model.js SAMPLES,
samplesOn, the newest-pick-wins guard) and tags.html's sample photo set (rooftop/sample_photos.py,
scripts/make_sample_tag_photos.py). Unit (node + numpy), smoke (the script, page wiring), integration (OpenCV
reads the photos; both pages in headless Chrome)."""
import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
sys.path.insert(0, str(ROOT))
from rooftop import sample_photos as sp  # noqa: E402
from rooftop.tags import CELLS  # noqa: E402

JS = (PREVIEW / "model.js").as_uri()
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)
DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as M from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def detect(img):
    _, ids, _ = DETECTOR.detectMarkers(img)
    return sorted(ids.ravel().tolist()) if ids is not None else []


# ---------- unit: model.js ----------

@pytest.mark.parametrize("search, on", [("", True), ("?", True), ("?phase=3", True), ("?samples=1", True),
                                        ("?samples=yes", True), ("?samples=0", False), ("?samples=off", False),
                                        ("?samples=FALSE", False), ("?samples=no", False), ("?x=1&samples=%200%20", False)])
def test_samples_on(node_available, search, on):
    assert run(f"M.samplesOn({json.dumps(search)})") is on


@pytest.mark.parametrize("search, on", [("", False), ("?", False), ("?samples=0", False), ("?room=0", False),
                                        ("?room=no", False), ("?room=", False), ("?room=1", True), ("?room=on", True),
                                        ("?room=TRUE", True), ("?x=1&room=%20yes%20", True)])
def test_room_on_only_when_asked(node_available, search, on):
    # the room card is Phase 3: tagar.html (the Phase 1 table demo) hides it unless ?room=1
    assert run(f"M.roomOn({json.dumps(search)})") is on


def test_room_on_without_a_search_string(node_available):
    assert run("[M.roomOn(undefined), M.roomOn(null)]") == [False, False]


def test_samples_on_without_a_search_string(node_available):
    assert run("[M.samplesOn(), M.samplesOn(null)]") == [True, True]


def test_sample_files_are_published(node_available):
    s = run("M.SAMPLES")
    for k in ("model", "room"):
        assert (PREVIEW / s[k]["url"].split("?")[0]).is_file(), s[k]
        assert s[k]["url"].split("?")[0].endswith(s[k]["name"])
        assert s[k]["units"] == "auto"
    assert s["model"]["name"] == "sample_sketchup_pergola.dae"
    assert s["room"]["name"] == "office_scan_room.dae"
    assert (s["room"]["wall"], s["room"]["along"], s["room"]["height"]) == ("back", "1.20", "1.10")


def test_sample_tags_fit_on_the_office_back_wall(node_available):
    # the prefilled along/height must pass the page's own check on the office's back wall (3.15 m wide, 4.05 m high)
    assert run('M.roomCheck(3.15, 4.05, Number(M.SAMPLES.room.along), Number(M.SAMPLES.room.height))') == []


# ---------- unit: sample photos ----------

def test_table_scene_puts_each_tag_at_a_corner():
    img, corners = sp.table_scene()
    assert img.shape == (820, 1220, 3)
    assert sorted(corners) == [0, 1, 2, 3]
    centres = {t: c.mean(0) for t, c in corners.items()}
    assert centres[0] == pytest.approx([110, 110], abs=1) and centres[1] == pytest.approx([1110, 110], abs=1)
    assert centres[2] == pytest.approx([1110, 710], abs=1) and centres[3] == pytest.approx([110, 710], abs=1)
    for c in corners.values():                                             # 160 mm black square at 1 px/mm
        assert np.ptp(c[:, 0]) == np.ptp(c[:, 1]) == 160
        x0, y0 = c[0].astype(int)
        assert img[y0 + 2, x0 + 2].max() < 50                               # just inside: black border
        assert img[y0 - 3, x0 - 3].min() > 200                              # just outside: white margin
    assert detect(img) == [0, 1, 2, 3]


def test_table_scene_scales_with_px_per_mm_and_ids():
    img, corners = sp.table_scene(ids=(5, 9, 11, 20), px_per_mm=0.5)
    assert img.shape[:2] == (410, 610)
    assert sorted(corners) == [5, 9, 11, 20]
    assert np.ptp(corners[5][:, 0]) == CELLS * 10
    assert detect(img) == [5, 9, 11, 20]


def test_view_homography_maps_the_scene_corners():
    q = [(0.1, 0.2), (0.9, 0.1), (0.8, 0.95), (0.05, 0.8)]
    H = sp.view_homography((1000, 500), q, (200, 100))
    got = sp.project(H, [[0, 0], [1000, 0], [1000, 500], [0, 500]])
    assert got == pytest.approx(np.array([[20, 20], [180, 10], [160, 95], [10, 80]]), abs=1e-6)


def test_in_frame():
    assert sp.in_frame([[2, 2], [1597, 1197]])
    assert not sp.in_frame([[1, 50]])
    assert not sp.in_frame([[50, 1198]])
    assert not sp.in_frame([[1598, 50]])
    assert not sp.in_frame([[50, -3]])
    assert not sp.in_frame([[50, 1]]) and sp.in_frame([[50, 2]])
    assert sp.in_frame([[0, 0]], pad=0) and not sp.in_frame([[10, 0]], (10, 10), pad=0)


def test_take_photo_noise_blur_and_floor():
    scene = np.full((100, 200, 3), 200, np.uint8)
    q = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    clean, H = sp.take_photo(scene, q, (400, 200), noise=0, blur=0)
    assert clean.shape == (200, 400, 3)
    assert (clean[100, 200] == 200).all() and tuple(clean[5, 5]) == (70, 74, 80)   # table inside, floor outside
    a, _ = sp.take_photo(scene, q, (400, 200), seed=3)
    b, _ = sp.take_photo(scene, q, (400, 200), seed=3)
    c, _ = sp.take_photo(scene, q, (400, 200), seed=4)
    assert (a == b).all() and (a != c).any()                               # seeded noise
    assert 1 < np.abs(a[90:110, 180:220].astype(float) - 200).mean() < 8
    soft, _ = sp.take_photo(scene, q, (400, 200), noise=0, blur=2)
    assert 70 < soft[50, 100].mean() < 200                                 # the table edge is blurred


def test_views_cover_tags_and_one_drops_id3():
    scene, corners = sp.table_scene()
    seen = {}
    for name, q in sp.VIEWS.items():
        H = sp.view_homography((scene.shape[1], scene.shape[0]), q)
        seen[name] = [t for t, c in corners.items() if sp.in_frame(sp.project(H, c))]
    assert len(seen) == 6
    assert sum(v == [0, 1, 2, 3] for v in seen.values()) == 5
    assert seen["tag_photo_06_id3_cut_off.jpg"] == [0, 1, 2]


# ---------- smoke: the script ----------

def test_script_writes_photos_and_index(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_sample_tag_photos
    paths, index = make_sample_tag_photos.main(["--out", str(tmp_path)])
    assert [p.name for p in paths] == list(sp.VIEWS)
    for p in paths:
        img = cv2.imread(str(p))
        assert img.shape == (1200, 1600, 3) and 100_000 < p.stat().st_size < 1_000_000
    on_disk = json.loads((tmp_path / "tag_photos.json").read_text(encoding="utf-8"))
    assert on_disk == index and index["tag_mm"] == 160 and index["ids"] == [0, 1, 2, 3]
    assert [p["name"] for p in index["photos"]] == list(sp.VIEWS)
    assert [p["ids"] for p in index["photos"]] == [[0, 1, 2, 3]] * 5 + [[0, 1, 2]]


# ---------- integration: the published photos ----------

def test_published_photos_read_as_the_index_says():
    index = json.loads((PREVIEW / "samples" / "tag_photos.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in index["photos"]] == list(sp.VIEWS)
    for p in index["photos"]:
        img = cv2.imread(str(PREVIEW / "samples" / p["name"]))
        assert detect(img) == p["ids"], p["name"]
    assert index["photos"][-1]["ids"] == [0, 1, 2]


def test_pages_wire_the_samples():
    tagar = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ["SAMPLES, samplesOn, roomOn } from \"./model.js?v=5\";", 'id="samples-note" hidden', 'href="?samples=0"',
                 'id="samples-room" hidden', "if (!roomMode) return Promise.all([loadModel(m.url, m.name), null]);",
                 "samplesOn(location.search) ? loadSamples()", "if (n !== modelLoads) return null;", "if (n !== roomLoads) return null;"]:
        assert hook in tagar, hook
    tags = (PREVIEW / "tags.html").read_text(encoding="utf-8")
    for hook in ['import { samplesOn } from "./model.js?v=4";', 'id="sampleBtn"', 'fetch("samples/tag_photos.json?v=1")',
                 "samplesOn(location.search)) checkSamples();"]:
        assert hook in tags, hook


# ---------- integration: both pages in Chrome ----------

@pytest.fixture(scope="module")
def browser():
    pytest.importorskip("playwright.sync_api")
    if not CHROME:
        pytest.skip("Chrome not installed")
    try:
        urllib.request.urlopen("https://cdn.jsdelivr.net/npm/three@0.160.0/package.json", timeout=10).close()
    except OSError:
        pytest.skip("cdn.jsdelivr.net (three.js) unreachable")
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


@pytest.fixture
def page(browser):
    b, base = browser
    pg = b.new_page(viewport={"width": 800, "height": 450})                 # a fresh page per test: no shared state
    pg.base = base
    yield pg
    pg.close()


def test_tagar_loads_only_the_sample_model_by_default(page):
    # Phase 1 table demo: the pergola loads, the room card stays hidden and no room is fetched
    page.goto(f"{page.base}/tagar.html")
    page.wait_for_function("window.__tagar", timeout=30000)
    got = page.evaluate("__tagar.samples()")                               # [model state, no room]
    assert got[0]["name"] == "sample_sketchup_pergola.dae" and got[1] is None
    hidden = lambda i: page.evaluate(f"document.getElementById('{i}').hidden")
    assert page.evaluate("__tagar.modelState().name") == "sample_sketchup_pergola.dae"
    assert page.evaluate("__tagar.roomState()") == {"loaded": False}
    assert (hidden("room-card"), hidden("samples-note"), hidden("samples-room")) == (True, False, True)
    assert page.evaluate("getComputedStyle(document.getElementById('room-card')).display") == "none"   # really not drawn
    assert page.evaluate("document.getElementById('r-along').value") == ""


def test_tagar_loads_the_sample_model_and_room(page):
    page.goto(f"{page.base}/tagar.html?room=1")
    page.wait_for_function("window.__tagar", timeout=30000)
    page.evaluate("__tagar.samples()")                                     # waits for both loads
    m, r = page.evaluate("__tagar.modelState()"), page.evaluate("__tagar.roomState()")
    assert m["name"] == "sample_sketchup_pergola.dae" and r["name"] == "office_scan_room.dae"
    assert r["loaded"] and r["units"] == "m" and r["wall"] == "back"
    val = lambda i: page.evaluate(f"document.getElementById('{i}').value")
    assert (val("r-along"), val("r-height"), val("r-wall"), val("m-units"), val("r-units")) == ("1.20", "1.10", "back", "auto", "auto")
    assert page.evaluate("document.getElementById('samples-note').hidden") is False
    assert page.evaluate("[document.getElementById('room-card').hidden, document.getElementById('samples-room').hidden]") == [False, False]
    assert page.evaluate("getComputedStyle(document.getElementById('room-card')).display") == "flex"
    info = page.evaluate("document.getElementById('m-info').textContent")
    assert "sample_sketchup_pergola.dae" in info and "Loading" not in info


def test_tagar_samples_off_starts_empty(page):
    page.goto(f"{page.base}/tagar.html?samples=0")
    page.wait_for_function("window.__tagar", timeout=30000)
    assert page.evaluate("__tagar.samples()") is None
    assert page.evaluate("document.getElementById('m-info').textContent") == "No model loaded."
    assert page.evaluate("__tagar.roomState()") == {"loaded": False}
    assert page.evaluate("document.getElementById('r-along').value") == ""
    assert page.evaluate("document.getElementById('samples-note').hidden") is True


def test_tagar_newest_pick_wins(page):
    # the slow 4 MB office loads first, the small sample room second: the second must stay
    page.goto(f"{page.base}/tagar.html?samples=0")
    page.wait_for_function("window.__tagar", timeout=30000)
    got = page.evaluate("""Promise.all([__tagar.loadRoom('models/office_scan_room.dae?v=1', 'office_scan_room.dae'),
                                         __tagar.loadRoom('models/sample_room.glb?v=1', 'sample_room.glb')])""")
    assert got[0] is None and got[1]["name"] == "sample_room.glb"
    assert page.evaluate("__tagar.roomState().name") == "sample_room.glb"
    assert "sample_room.glb" in page.evaluate("document.getElementById('r-info').textContent")
    got = page.evaluate("""Promise.all([__tagar.loadModel('models/sample_sketchup_pergola.dae?v=1', 'a.dae'),
                                         __tagar.loadModel('models/sample_sketchup_pergola.obj?v=1', 'b.obj')])""")
    assert got[0] is None and got[1]["name"] == "b.obj"
    assert page.evaluate("__tagar.modelState().name") == "b.obj"
    # a stale failure must not overwrite the newest pick's message either
    page.evaluate("""Promise.all([__tagar.loadRoom('models/missing.glb', 'missing.glb'),
                                  __tagar.loadRoom('models/sample_room.glb?v=1', 'sample_room.glb')])""")
    assert "Could not load" not in page.evaluate("document.getElementById('r-info').textContent")
    page.evaluate("""Promise.all([__tagar.loadModel('models/missing.glb', 'missing.glb'),
                                  __tagar.loadModel('models/sample_sketchup_pergola.dae?v=1', 'c.dae')])""")
    assert "Could not load" not in page.evaluate("document.getElementById('m-info').textContent")


def test_tags_page_checks_the_sample_photos_on_first_visit(page):
    page.goto(f"{page.base}/tags.html")
    page.wait_for_function("window.__tags", timeout=30000)
    page.evaluate("document.getElementById('modePhotos').click()")
    page.wait_for_function("window.__tags.results().length === 6 && document.getElementById('progress').textContent.startsWith('6/6')",
                           timeout=120000)
    res = page.evaluate("__tags.results()")
    seen = {p["name"]: sorted(t["id"] for t in p["tags"]) for p in res}
    index = json.loads((PREVIEW / "samples" / "tag_photos.json").read_text(encoding="utf-8"))
    assert seen == {p["name"]: p["ids"] for p in index["photos"]}
    assert all(not p["error"] and all(not t["problems"] for t in p["tags"]) for p in res)
    assert page.evaluate("__tags.verdict()") == "Ready: 4 of 4 tags usable by tag_scale.py."
    assert page.evaluate("document.querySelectorAll('#rows tr').length") == 6
    assert page.evaluate("document.getElementById('ids').value") == "0, 1, 2, 3"
    page.evaluate("document.getElementById('modeLive').click(); document.getElementById('modePhotos').click()")
    page.wait_for_timeout(300)
    assert page.evaluate("document.getElementById('progress').textContent").startswith("6/6")   # a second visit does not re-run


def test_tags_page_samples_off_waits_for_photos_and_button_still_works(page):
    page.goto(f"{page.base}/tags.html?samples=0")
    page.wait_for_function("window.__tags", timeout=30000)
    page.evaluate("document.getElementById('modePhotos').click()")
    page.wait_for_timeout(500)
    assert page.evaluate("__tags.results()") == []
    assert page.evaluate("document.getElementById('summary').hidden") is True
    page.fill("#ids", "7"); page.dispatch_event("#ids", "input")
    assert page.evaluate("__tags.checkSamples()") is True                  # the sample set brings its own ids
    assert len(page.evaluate("__tags.results()")) == 6
    assert page.evaluate("__tags.verdict()") == "Ready: 4 of 4 tags usable by tag_scale.py."


def test_tags_page_newer_run_stops_an_older_one(page):
    page.goto(f"{page.base}/tags.html?samples=0")
    page.wait_for_function("window.__tags", timeout=30000)
    got = page.evaluate("""(async () => {
        const names = ["tag_photo_01_overhead.jpg", "tag_photo_02_from_front.jpg"];
        const items = await Promise.all(names.map(async (n) => ({ name: n, blob: await (await fetch("samples/" + n)).blob() })));
        return Promise.all([__tags.checkPhotos(items), __tags.checkPhotos(items.slice(0, 1))]);
    })()""")
    assert got == [False, True]
    assert [p["name"] for p in page.evaluate("__tags.results()")] == ["tag_photo_01_overhead.jpg"]
    assert page.evaluate("document.querySelectorAll('#rows tr').length") == 1


def test_tags_page_unreadable_photo_of_an_older_run_is_dropped(page):
    page.goto(f"{page.base}/tags.html?samples=0")
    page.wait_for_function("window.__tags", timeout=30000)
    got = page.evaluate("""(async () => {
        const good = { name: "tag_photo_01_overhead.jpg", blob: await (await fetch("samples/tag_photo_01_overhead.jpg")).blob() };
        return Promise.all([__tags.checkPhotos([{ name: "broken.jpg", blob: new Blob(["not a jpeg"]) }]), __tags.checkPhotos([good])]);
    })()""")
    assert got == [False, True]
    assert [p["name"] for p in page.evaluate("__tags.results()")] == ["tag_photo_01_overhead.jpg"]
    one = page.evaluate("""__tags.checkPhotos([{ name: "broken.jpg", blob: new Blob(["not a jpeg"]) }])""")
    assert one is True and page.evaluate("__tags.results()")[0]["error"].startswith("could not read")


def test_tags_page_reports_a_missing_sample(page):
    page.route("**/samples/tag_photo_03_from_left.jpg*", lambda r: r.fulfill(status=404, body="no"))
    page.goto(f"{page.base}/tags.html?samples=0")
    page.wait_for_function("window.__tags", timeout=30000)
    assert page.evaluate("__tags.checkSamples()") is False
    assert "tag_photo_03_from_left.jpg: HTTP 404" in page.evaluate("document.getElementById('progress').textContent")
