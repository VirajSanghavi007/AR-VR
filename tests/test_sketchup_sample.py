"""SketchUp exports in tagar.html: the sample writer (rooftop/sketchup_sample.py) and loading .dae / .obj (+ .mtl)
through the page in a real headless Chrome. Unit (writer pieces), smoke (the CLI), integration (three.js's
ColladaLoader / OBJLoader / MTLLoader in Chrome, through the hook and through the file picker)."""
import functools
import http.server
import re
import shutil
import sys
import threading
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import make_sketchup_sample  # noqa: E402
from rooftop import sketchup_sample as S  # noqa: E402

PREVIEW = ROOT / "preview"
MODELS = PREVIEW / "models"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
CHROME = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                           r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                           shutil.which("chrome"), shutil.which("google-chrome")) if p and Path(p).is_file()), None)
SIZE = [3.0, 2.72, 2.4]                  # the pergola in metres, three.js axes: x across, y up, z depth


# ---------- unit ----------

def outward(part):
    """Every face normal points away from the part's centre."""
    v, f = part.vertices, part.faces
    n = S.face_normals(v, f)
    return bool(np.all(np.einsum("ij,ij->i", v[f].mean(1) - v.mean(0), n) > 0))


def test_box_spans_its_corners_and_faces_point_out():
    b = S.box([0, 0, 0], [1, 2, 3], "Wood_Cedar")
    assert b.material == "Wood_Cedar" and b.faces.shape == (12, 3)
    assert b.vertices.min(0).tolist() == [0, 0, 0] and b.vertices.max(0).tolist() == [1, 2, 3]
    assert outward(b)


def test_slab_on_a_slope_is_pushed_along_its_normal():
    s = S.slab([[0, 0, 1], [1, 0, 1], [1, 1, 2], [0, 1, 2]], 0.1, "Solar_Glass")
    n = np.array([0, -1, 1]) / np.sqrt(2)
    assert np.allclose(s.vertices[4:] - s.vertices[:4], n * 0.1)
    assert outward(s)


def test_face_normals_are_unit_and_follow_winding():
    n = S.face_normals([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.array([[0, 1, 2], [0, 2, 1]]))
    assert n.tolist() == [[0, 0, 1], [0, 0, -1]]


def test_pergola_parts_size_and_panels():
    parts = S.pergola()
    mats = [p.material for p in parts]
    assert mats.count("Wood_Cedar") == 6 and mats.count("Aluminium") == 2 and mats.count("Solar_Glass") == 6
    lo, hi = S.bounds(parts)
    assert np.allclose(lo, [0, 0, 0]) and np.allclose(hi - lo, [3.0, 2.4, 2.72], atol=0.01)
    assert all(outward(p) for p in parts)
    panels = [p for p in parts if p.material == "Solar_Glass"]
    front, back = panels[0].vertices[:2, 2].mean(), panels[0].vertices[2:4, 2].mean()
    assert front > back                                   # the roof falls toward the back
    assert all(p.vertices[:, 2].min() > 2.2 for p in panels)
    assert len(S.pergola(cols=4, rows=3)) == 8 + 12


def test_dae_is_inches_z_up_with_one_geometry_per_material():
    parts = S.pergola()
    root = ET.fromstring(S.dae_text(parts))
    unit = root.find("c:asset/c:unit", NS)
    assert unit.get("meter") == "0.0254" and unit.get("name") == "inch"
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    geos = root.findall(".//c:geometry", NS)
    assert [g.get("name") for g in geos] == ["Wood_Cedar", "Aluminium", "Solar_Glass"]
    pos = np.array(geos[0].find(".//c:float_array", NS).text.split(), float).reshape(-1, 3)
    assert pos[:, 2].max() == pytest.approx(2.65 / S.INCH, rel=1e-5)    # the front beam top (z up), in inches
    tri = geos[2].find(".//c:triangles", NS)
    idx = np.array(tri.find("c:p", NS).text.split(), int).reshape(-1, 6)
    assert int(tri.get("count")) == len(idx) == 6 * 12
    assert idx[:, ::2].max() == 6 * 8 - 1 and idx[:, 1::2].max() == 6 * 12 - 1
    colour = root.find(".//c:effect[@id='Solar_Glass-fx']//c:color", NS).text.split()
    assert [float(x) for x in colour] == [*S.MATERIALS["Solar_Glass"], 1]
    assert root.find(".//c:instance_material", NS).get("target") == "#Wood_Cedar-mat"


def test_dae_units_can_be_metres():
    root = ET.fromstring(S.dae_text([S.box([0, 0, 0], [1, 1, 2], "Aluminium")], unit=1.0, unit_name="meter"))
    pos = np.array(root.find(".//c:float_array", NS).text.split(), float)
    assert pos.max() == 2 and root.find("c:asset/c:unit", NS).get("name") == "meter"


def test_obj_is_y_up_inches_and_indexes_stay_in_range():
    parts = S.pergola()
    text = S.obj_text(parts, "p.mtl")
    lines = text.splitlines()
    assert "# File units = inches" in lines and "mtllib p.mtl" in lines
    v = np.array([l.split()[1:] for l in lines if l.startswith("v ")], float)
    vn = np.array([l.split()[1:] for l in lines if l.startswith("vn ")], float)
    f = [[tuple(map(int, c.split("//"))) for c in l.split()[1:]] for l in lines if l.startswith("f ")]
    assert len(v) == 14 * 8 and len(vn) == len(f) == 14 * 12
    assert np.allclose(v.max(0) - v.min(0), np.array([3.0, 2.72, 2.4]) / S.INCH, atol=0.5)
    assert v[:, 1].min() == pytest.approx(0) and v[:, 2].max() == pytest.approx(0)    # y up, depth along -z
    for face in f:
        assert all(1 <= a <= len(v) for a, _ in face) and len({b for _, b in face}) == 1
        a, b, c = (v[i - 1] for i, _ in face)
        n = np.cross(b - a, c - a)
        assert np.allclose(n / np.linalg.norm(n), vn[face[0][1] - 1], atol=1e-4)             # normal matches the face
    assert [l.split()[1] for l in lines if l.startswith("usemtl")] == [p.material for p in parts]
    assert sum(l.startswith("g ") for l in lines) == len(parts)


def test_obj_units_line_names_other_units():
    assert "# File units = meters" in S.obj_text([S.box([0, 0, 0], [1, 1, 1], "Aluminium")], "a.mtl", 1.0, "meters")


def test_mtl_lists_only_the_materials_used():
    t = S.mtl_text([S.box([0, 0, 0], [1, 1, 1], "Aluminium")])
    assert t.count("newmtl") == 1 and "newmtl Aluminium\nKd 0.78 0.8 0.83\n" in t
    full = S.mtl_text(S.pergola())
    assert re.findall(r"newmtl (\w+)", full) == list(S.MATERIALS)


def test_write_sample_writes_three_files(tmp_path):
    files = S.write_sample(tmp_path / "out", "x")
    assert sorted(f.name for f in files.values()) == ["x.dae", "x.mtl", "x.obj"]
    assert "mtllib x.mtl" in files["obj"].read_text()
    ET.fromstring(files["dae"].read_text(encoding="utf-8"))


def test_shipped_sample_matches_the_writer():
    parts = S.pergola()
    assert (MODELS / "sample_sketchup_pergola.dae").read_text(encoding="utf-8") == S.dae_text(parts)
    assert (MODELS / "sample_sketchup_pergola.obj").read_text(encoding="utf-8") == S.obj_text(parts, "sample_sketchup_pergola.mtl")
    assert (MODELS / "sample_sketchup_pergola.mtl").read_text(encoding="utf-8") == S.mtl_text(parts)


# ---------- smoke ----------

def test_cli_writes_the_sample(tmp_path, capsys):
    files = make_sketchup_sample.main(["--out", str(tmp_path), "--name", "s"])
    out = capsys.readouterr().out
    assert all(f.is_file() for f in files.values()) and out.count("wrote ") == 3
    assert sorted(f.name for f in files.values()) == ["s.dae", "s.mtl", "s.obj"]
    assert "size (m, x y z): 3.00 x 2.40 x 2.72" in out


# ---------- integration: the page loads them in Chrome ----------

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
        pg = b.new_page(viewport={"width": 800, "height": 450})
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/tagar.html")
        pg.wait_for_function("window.__tagar && window.AR", timeout=30000)
        yield pg
        b.close()
    srv.shutdown()


def info(pg):
    return pg.evaluate("document.getElementById('m-info').textContent")


def test_dae_loads_in_metres_from_its_own_units(page):
    s = page.evaluate("__tagar.loadModel('models/sample_sketchup_pergola.dae', 'sample_sketchup_pergola.dae')")
    assert s["loaded"] and s["units"] == "m"
    assert s["sizeM"] == pytest.approx(SIZE, abs=0.01)                  # inches + Z_UP turned into metres, y up
    assert "read as metres (as the file says)" in info(page)


def test_obj_loads_in_inches_from_its_header_with_mtl_colours(page):
    s = page.evaluate("__tagar.loadModel('models/sample_sketchup_pergola.obj', 'sample_sketchup_pergola.obj')")
    assert s["loaded"] and s["units"] == "in" and s["sizeM"] == pytest.approx(SIZE, abs=0.01)
    assert "read as inches (as the file says)" in info(page)
    cols = page.evaluate("""() => { const out = {}; window.__tagar.modelObj().traverse((o) => { if (o.isMesh)
                              for (const m of [o.material].flat()) { const c = m.color.clone().convertLinearToSRGB(); out[m.name] = [c.r, c.g, c.b, m.side]; } }); return out; }""")  # .mtl Kd is sRGB
    assert set(cols) == set(S.MATERIALS)
    assert cols["Solar_Glass"][:3] == pytest.approx(S.MATERIALS["Solar_Glass"], abs=0.02) and cols["Solar_Glass"][3] == 2  # DoubleSide


def test_units_menu_still_overrides_the_file(page):
    s = page.evaluate("__tagar.loadModel('models/sample_sketchup_pergola.obj', 'sample_sketchup_pergola.obj', 'cm')")
    assert s["units"] == "cm" and s["sizeM"][0] == pytest.approx(3.0 / S.INCH * 0.01, abs=0.01)
    assert "(as the file says)" not in info(page)


def test_file_picker_takes_the_obj_with_its_mtl(page):
    page.evaluate("__tagar.loadModel('models/sample_sketchup_pergola.dae', 'sample_sketchup_pergola.dae')")   # a different model first
    page.evaluate("document.getElementById('m-units').value = 'auto'")
    page.set_input_files("#m-file", [str(MODELS / "sample_sketchup_pergola.mtl"), str(MODELS / "sample_sketchup_pergola.obj")])
    page.wait_for_function("window.__tagar.modelState().name === 'sample_sketchup_pergola.obj'", timeout=20000)
    s = page.evaluate("__tagar.modelState()")
    assert s["units"] == "in" and s["sizeM"] == pytest.approx(SIZE, abs=0.01)
    names = page.evaluate("""() => { const n = new Set(); window.__tagar.modelObj().traverse((o) => { if (o.isMesh)
                              for (const m of [o.material].flat()) n.add(m.color.getHexString()); }); return [...n]; }""")
    assert len(names) == 3                                                      # the .mtl colours, not OBJLoader grey


def test_file_picker_explains_a_skp_file(page, tmp_path):
    f = tmp_path / "roof.skp"
    f.write_bytes(b"SketchUp Model")
    page.set_input_files("#m-file", [str(f)])
    page.wait_for_function("/COLLADA/.test(document.getElementById('m-info').textContent)", timeout=5000)
    assert "File > Export > 3D Model" in info(page)


def test_sample_button_loads_the_dae(page):
    page.evaluate("__tagar.loadModel('models/sample_sketchup_pergola.obj', 'sample_sketchup_pergola.obj')")
    page.evaluate("document.getElementById('m-units').value = 'cm'")              # the button resets the units menu
    page.evaluate("document.getElementById('m-sketchup').click()")
    page.wait_for_function("window.__tagar.modelState().name === 'sample_sketchup_pergola.dae'", timeout=20000)
    assert page.evaluate("document.getElementById('m-units').value") == "auto"
    assert page.evaluate("__tagar.modelState().sizeM") == pytest.approx(SIZE, abs=0.01)


def png_bytes(w=2, h=2, rgb=(200, 30, 30)):
    """A tiny solid-colour PNG."""
    import struct, zlib
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_file_picker_finds_a_texture_the_mtl_names_in_a_folder(page, tmp_path):
    (tmp_path / "Wood.PNG").write_bytes(png_bytes())
    (tmp_path / "t.mtl").write_text("newmtl wood\nKd 1 1 1\nmap_Kd Textures\\Wood.PNG\n")
    (tmp_path / "t.obj").write_text("# File units = meters\nmtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl wood\nf 1 2 3\n")
    page.set_input_files("#m-file", [str(tmp_path / f) for f in ("t.obj", "t.mtl", "Wood.PNG")])
    page.wait_for_function("window.__tagar.modelState().name === 't.obj'", timeout=20000)
    page.wait_for_function("""() => { let ok = false; window.__tagar.modelObj().traverse((o) => { if (o.isMesh)
                              for (const m of [o.material].flat()) ok = ok || !!(m.map && m.map.image && m.map.image.width === 2); });
                              return ok; }""", timeout=10000)   # the folder path is dropped, the name matched case-blind


def test_room_picker_reads_a_sketchup_dae(page):
    s = page.evaluate("__tagar.loadRoom('models/sample_sketchup_pergola.dae', 'sample_sketchup_pergola.dae')")
    assert s["loaded"] and s["units"] == "m" and s["sizeM"] == pytest.approx(SIZE, abs=0.01)
    assert "(as the file says)" in page.evaluate("document.getElementById('r-info').textContent")
