"""Phase 2 M1 + M2: the sample drawing and the DXF reader."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from rooftop import dxf_read as R
from rooftop import phase2, sample_dxf
from rooftop.dxf_read import DrawingError, read_dxf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DXF = ROOT / "data/sample_roof/roof.dxf"
SAMPLE_JSON = ROOT / "data/sample_roof/measurements.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(doc, tmp_path, name="d.dxf"):
    p = tmp_path / name
    doc.saveas(p)
    return p


def _layer(doc, name):
    return [e for e in doc.modelspace() if e.dxf.layer == name]


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    p = sample_dxf.write_sample(tmp_path_factory.mktemp("s") / "roof.dxf")
    return read_dxf(p)


# ---------------- unit: labels, units, mapping ----------------

@pytest.mark.parametrize("text,want", [
    ("H=1000", {"H": 1000.0}),
    ("H=1500 S=600", {"H": 1500.0, "S": 600.0}),
    ("lat = 19.07  FLOORS=3 FLOOR_H=3200", {"LAT": 19.07, "FLOORS": 3.0, "FLOOR_H": 3200.0}),
    ("LAT=-33.9", {"LAT": -33.9}),
    ("no labels here", {}),
    ("H=10 H=10", {"H": 10.0}),
])
def test_parse_labels(text, want):
    assert R.parse_labels(text) == want


def test_parse_labels_conflict():
    with pytest.raises(DrawingError, match="twice"):
        R.parse_labels("H=10 H=20")


class _Doc:
    def __init__(self, code):
        self.header = {} if code is None else {"$INSUNITS": code}


@pytest.mark.parametrize("code,mm", [(4, 1.0), (5, 10.0), (6, 1000.0), (1, 25.4), (2, 304.8)])
def test_unit_scale_from_header(code, mm):
    assert R.unit_scale(_Doc(code)) == mm


@pytest.mark.parametrize("code", [None, 0, 3, 99])
def test_unit_scale_missing_or_unknown_unit(code):
    with pytest.raises(DrawingError, match="no usable unit"):
        R.unit_scale(_Doc(code))


def test_unit_scale_override_wins_and_rejects_unknown():
    assert R.unit_scale(_Doc(None), "m") == 1000.0
    assert R.unit_scale(_Doc(6), "mm") == 1.0
    with pytest.raises(DrawingError, match="unknown unit"):
        R.unit_scale(_Doc(4), "yd")


def test_load_mapping(tmp_path):
    p = tmp_path / "layers.json"
    p.write_text(json.dumps({"ROOF_OUTLINE": "A-ROOF", "TANK": ["T1", "T2"]}))
    assert R.load_mapping(p) == {"ROOF_OUTLINE": ["A-ROOF"], "TANK": ["T1", "T2"]}
    p.write_text(json.dumps({"ROOF": "A-ROOF"}))
    with pytest.raises(DrawingError, match="unknown layers"):
        R.load_mapping(p)


# ---------------- unit: outline -> spec ----------------

def _L(L, W, nl, nw):
    return Polygon(sample_dxf.outline_points(dict(length=L, width=W, notch_length=nl, notch_width=nw)))


def _inner(L, W, nl, nw, t):
    return Polygon(sample_dxf.parapet_points(dict(length=L, width=W, notch_length=nl, notch_width=nw,
                                                  parapet_thickness=t)))


def test_spec_from_rectangle():
    s = R.spec_from_outline(shapely.box(0, 0, 10000, 6000), shapely.box(200, 200, 9800, 5800), 900)
    assert (s.length, s.width, s.parapet_thickness, s.parapet_height, s.notch_length) == (10000, 6000, 200, 900, 0)
    assert s.slab_thickness == 150


def test_spec_from_L():
    s = R.spec_from_outline(_L(18000, 12000, 6000, 4000), _inner(18000, 12000, 6000, 4000, 230), 1000, 180)
    assert (s.notch_length, s.notch_width, s.parapet_thickness, s.slab_thickness) == (6000, 4000, 230, 180)


def test_spec_rejects_notch_elsewhere():
    sw_notch = shapely.box(0, 0, 10000, 6000).difference(shapely.box(0, 0, 3000, 2000))
    with pytest.raises(DrawingError, match="north-east"):
        R.spec_from_outline(sw_notch, sw_notch.buffer(-200, join_style=2), 900)


@pytest.mark.parametrize("inner", [(200, 200, 9500, 5800), (200, 200, 9900, 5800)])
def test_spec_rejects_uneven_parapet(inner):
    with pytest.raises(DrawingError, match="even distance"):
        R.spec_from_outline(shapely.box(0, 0, 10000, 6000), shapely.box(*inner), 900)


@pytest.mark.parametrize("cut", [(7000, 0, 10000, 2000), (0, 4000, 3000, 6000)])   # south-east, north-west
def test_spec_rejects_notch_at_other_corners(cut):
    out = shapely.box(0, 0, 10000, 6000).difference(shapely.box(*cut))
    with pytest.raises(DrawingError, match="north-east"):
        R.spec_from_outline(out, out.buffer(-200, join_style=2), 900)


def test_spec_rejects_outline_off_origin():
    with pytest.raises(DrawingError, match="origin"):
        R.spec_from_outline(shapely.box(5, 0, 10000, 6000), shapely.box(205, 200, 9800, 5800), 900)


def test_facing_is_equator_side():
    d = R.Drawing(spec=None, features=[], north_deg=90.0, site={"LAT": 19.0})
    assert d.facing_deg() == 270.0
    d.site["LAT"] = -30.0
    assert d.facing_deg() == 90.0
    d.north_deg, d.site = 120.0, {}
    assert d.latitude is None and d.facing_deg() == 300.0


def test_door_width():
    assert R.Door("S", 0, 0, 0.6, 0.8).width == pytest.approx(1.0)


# ---------------- M2 done-when: the sample reads back within 1 mm ----------------

def test_sample_reads_back_within_1mm(sample):
    """Compared with measurements.json, which is written from SAMPLE, not from the reader."""
    m = sample_dxf.measurements_json()
    s = sample.spec
    for k in ("length", "width", "notch_length", "notch_width", "parapet_thickness", "parapet_height"):
        assert getattr(s, k) == pytest.approx(m[k], abs=1.0), k
    feats = {f.name: f for f in sample.features}
    assert len(feats) == len(m["rooms"]) + len(m["tanks"])
    for r in m["rooms"]:
        f = feats[r["name"]]
        assert f.kind == r["kind"]
        got = np.array([f.x, f.y, f.w, f.d, f.h]) * 1000
        assert got == pytest.approx([r[k] for k in "xywdh"], abs=1.0)
    for t in m["tanks"]:
        f = feats[t["name"]]
        assert f.kind == "tank"
        got = np.array([f.x + f.w / 2, f.y + f.d / 2, f.w / 2, f.h, f.stand]) * 1000
        assert got == pytest.approx([t["cx"], t["cy"], t["r"], t["h"] + t["s"], t["s"]], abs=1.0)
    doors = {d.room: d for d in sample.doors}
    for r in m["rooms"]:
        d = doors[r["name"]]
        assert np.array([d.x0, d.y0, d.x1, d.y1]) * 1000 == pytest.approx(r["door"], abs=1.0)
    assert np.array(sample.drains) * 1000 == pytest.approx(np.array(m["drains"]), abs=1.0)
    assert len(sample.pipes) == 1 and sample.pipes[0].diameter == m["pipes"][0]["d"]
    assert np.array(sample.pipes[0].points) * 1000 == pytest.approx(np.array(m["pipes"][0]["points"]), abs=1.0)
    assert sample.north_deg == pytest.approx(m["north_deg"])
    assert sample.site == {"LAT": 19.07, "LON": 72.88, "TZ": 5.5, "FLOORS": 3, "FLOOR_H": 3200}
    assert len(sample.dimensions) == len(m["dimensions"])
    for got, want in zip(sample.dimensions, m["dimensions"]):
        assert got.value == pytest.approx(want["value"], abs=1.0)
        assert np.array([got.p1, got.p2]) * 1000 == pytest.approx(np.array([want["p1"], want["p2"]]), abs=1.0)


def test_checked_in_sample_matches_generator():
    assert SAMPLE_DXF.exists() and SAMPLE_JSON.exists()
    assert json.loads(SAMPLE_JSON.read_text()) == json.loads(json.dumps(sample_dxf.measurements_json()))
    d = read_dxf(SAMPLE_DXF)
    assert d.spec.length == 18000 and len(d.features) == 4


def test_measurements_json_is_a_valid_spec(tmp_path):
    from rooftop.cad import RooftopSpec
    p = sample_dxf.write_measurements(tmp_path / "x" / "m.json")
    s = RooftopSpec.from_json(p)
    assert (s.length, s.notch_width, s.parapet_height) == (18000, 4000, 1000)
    assert json.loads(p.read_text())["diagonal"] == pytest.approx(21633.3, abs=0.1)


def test_reader_moves_origin_to_south_west_corner(tmp_path, sample):
    d = read_dxf(sample_dxf.write_sample(tmp_path / "off.dxf", offset=(250000.0, -73000.0)))
    assert d.spec == sample.spec and d.features == sample.features and d.doors == sample.doors
    assert d.dimensions[0].p1 == pytest.approx((0.0, 0.0))
    assert np.array(d.drains) == pytest.approx(np.array(sample.drains))
    assert d.dimensions[0].p2 == pytest.approx((18.0, 0.0))
    assert np.array(d.pipes[0].points) == pytest.approx(np.array(sample.pipes[0].points))


def test_reader_converts_units(tmp_path, sample):
    s = json.loads(json.dumps(sample_dxf.SAMPLE))
    cm = sample_dxf.build_doc(s, units=5)       # same numbers, now read as cm: 10x larger
    d = read_dxf(_save(cm, tmp_path))
    assert d.spec.length == pytest.approx(180000) and d.features[0].w == pytest.approx(30.0)
    assert d.features[2].w == pytest.approx(12.0) and d.features[2].x == pytest.approx(134.0)
    assert d.dimensions[0].value == pytest.approx(180000)
    assert d.pipes[0].diameter == 50                       # labels are always mm
    d = read_dxf(_save(cm, tmp_path, "b.dxf"), units="mm")
    assert d.spec.length == pytest.approx(18000)


def test_reader_layer_mapping(tmp_path, sample):
    doc = sample_dxf.build_doc()
    ren = {"ROOF_OUTLINE": "A-ROOF-OTLN", "TANK": "P-TANK", "SITE_INFO": "x-site"}
    for e in doc.modelspace():
        if e.dxf.layer in ren:
            e.dxf.layer = ren[e.dxf.layer]
    p = _save(doc, tmp_path)
    with pytest.raises(DrawingError, match="ROOF_OUTLINE"):
        read_dxf(p)
    m = tmp_path / "layers.json"
    m.write_text(json.dumps({"ROOF_OUTLINE": "A-ROOF-OTLN", "TANK": ["P-TANK"], "SITE_INFO": "X-SITE"}))
    d = read_dxf(p, m)
    assert d.spec == sample.spec and d.features == sample.features and d.site == sample.site


def test_reader_without_optional_layers(tmp_path):
    doc = sample_dxf.build_doc()
    msp = doc.modelspace()
    for e in list(msp):
        if e.dxf.layer not in ("ROOF_OUTLINE", "PARAPET"):
            msp.delete_entity(e)
    d = read_dxf(_save(doc, tmp_path))
    assert d.features == [] and d.doors == [] and d.dimensions == [] and d.pipes == []
    assert d.north_deg == 90.0 and d.site == {} and d.latitude is None


def test_reader_mtext_old_polyline_and_rotated_north(tmp_path):
    doc = sample_dxf.build_doc()
    msp = doc.modelspace()
    old = _layer(doc, "PARAPET")
    pts = [(p[0], p[1]) for p in old[0].get_points("xy")]
    for e in old:
        msp.delete_entity(e)
    msp.add_polyline2d(pts, close=True, dxfattribs={"layer": "PARAPET"})
    msp.add_mtext("H=1100", dxfattribs={"layer": "PARAPET", "insert": (500, 500)})
    for e in _layer(doc, "SITE_INFO"):
        msp.delete_entity(e)
    msp.add_mtext("LAT=-12.5\\PFLOORS=2", dxfattribs={"layer": "SITE_INFO", "insert": (0, -3000)})
    n = _layer(doc, "NORTH")
    n = [e for e in n if e.dxftype() == "LINE"][0]
    n.dxf.end = (n.dxf.start.x + 1000, n.dxf.start.y)          # north = +x
    d = read_dxf(_save(doc, tmp_path))
    assert d.spec.parapet_height == 1100 and d.spec.parapet_thickness == 230   # MTEXT label wins nearest
    assert d.site == {"LAT": -12.5, "FLOORS": 2.0}
    assert d.north_deg == pytest.approx(0.0) and d.facing_deg() == pytest.approx(0.0)


# ---------------- errors on bad drawings ----------------

def _bad(tmp_path, change, match, **kw):
    doc = sample_dxf.build_doc()
    change(doc, doc.modelspace())
    with pytest.raises(DrawingError, match=match):
        read_dxf(_save(doc, tmp_path), **kw)


def test_error_missing_file(tmp_path):
    with pytest.raises(DrawingError, match="no such drawing"):
        read_dxf(tmp_path / "nope.dxf")


def test_error_not_a_dxf(tmp_path):
    p = tmp_path / "x.dxf"
    p.write_text("hello")
    with pytest.raises(DrawingError, match="not a readable DXF"):
        read_dxf(p)


def test_error_missing_unit(tmp_path):
    _bad(tmp_path, lambda doc, m: doc.header.__setitem__("$INSUNITS", 0), "no usable unit")


def test_error_open_outline(tmp_path):
    def f(doc, m):
        _layer(doc, "ROOF_OUTLINE")[0].closed = False
    _bad(tmp_path, f, "open")


def test_closing_vertex_counts_as_closed(tmp_path, sample):
    doc = sample_dxf.build_doc()
    e = _layer(doc, "ROOF_OUTLINE")[0]
    pts = [(p[0], p[1]) for p in e.get_points("xy")]
    e.set_points(pts + [pts[0]], format="xy")
    e.closed = False
    assert read_dxf(_save(doc, tmp_path)).spec == sample.spec


def test_error_missing_layer(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "PARAPET"):
            m.delete_entity(e)
    _bad(tmp_path, f, "layer PARAPET .*nothing on it")


def test_error_parapet_with_only_text(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "PARAPET"):
            if e.dxftype() != "TEXT":
                m.delete_entity(e)
    _bad(tmp_path, f, "PARAPET needs exactly one closed polyline, found 0")


def test_error_parapet_missing_height(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "PARAPET"):
            if e.dxftype() == "TEXT":
                m.delete_entity(e)
    _bad(tmp_path, f, "PARAPET: shape 1 has no H=")


def test_error_two_outlines(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=True,
                                                     dxfattribs={"layer": "ROOF_OUTLINE"}), "exactly one")


def test_error_two_parapets(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_lwpolyline([(500, 500), (600, 500), (600, 600)], close=True,
                                                     dxfattribs={"layer": "PARAPET"}), "PARAPET needs exactly one")


def test_error_arc_in_outline(tmp_path):
    def f(doc, m):
        e = _layer(doc, "ROOF_OUTLINE")[0]
        pts = list(e.get_points("xyb"))
        pts[0] = (pts[0][0], pts[0][1], 0.3)
        e.set_points(pts, format="xyb")
    _bad(tmp_path, f, "arcs")


def test_error_self_crossing(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_lwpolyline([(1000, 2000), (3000, 3000), (3000, 2000), (1000, 2500)],
                                                     close=True, dxfattribs={"layer": "STORE"}), "crosses itself")


def test_error_two_point_polygon(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_lwpolyline([(1000, 2000), (2000, 3000)], close=True,
                                                     dxfattribs={"layer": "STORE"}), "fewer than 3")


def test_error_line_on_polygon_layer(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "ROOF_OUTLINE"):
            m.delete_entity(e)
        m.add_line((0, 0), (1, 1), dxfattribs={"layer": "ROOF_OUTLINE"})
    _bad(tmp_path, f, "exactly one closed polyline, found 0")


def test_error_missing_height(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "STAIR"):
            if e.dxftype() == "TEXT":
                m.delete_entity(e)
    _bad(tmp_path, f, "STAIR: shape 1 has no H=")


def test_error_tank_missing_stand(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "TANK"):
            if e.dxftype() == "TEXT":
                e.dxf.text = "H=1500"
    _bad(tmp_path, f, "no S=")


def test_error_pipe_missing_diameter(tmp_path):
    def f(doc, m):
        for e in _layer(doc, "PIPE"):
            if e.dxftype() == "TEXT":
                m.delete_entity(e)
    _bad(tmp_path, f, "no D=")


def test_error_conflicting_labels(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_text("H=900", dxfattribs={"layer": "STAIR", "insert": (1000, 9000)}),
         "two labels")


def test_error_room_not_rectangle(tmp_path):
    def f(doc, m):
        m.add_lwpolyline([(7500, 9000), (9500, 9000), (9500, 11000), (8500, 11000), (8500, 10000), (7500, 10000)],
                         close=True, dxfattribs={"layer": "STORE"})
        m.add_text("H=2400", dxfattribs={"layer": "STORE", "insert": (8000, 9500)})
    _bad(tmp_path, f, "STORE: shape 2 is not an axis-aligned rectangle")


def test_error_door_off_room(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_line((9000, 5000), (10000, 5000), dxfattribs={"layer": "DOOR"}),
         "does not lie on")


def test_door_within_tolerance_still_hosted(tmp_path):
    doc = sample_dxf.build_doc()
    doc.modelspace().add_line((1000, 8400.5), (1500, 8400.5), dxfattribs={"layer": "DOOR"})
    assert len(read_dxf(_save(doc, tmp_path)).doors) == 3


def test_error_feature_outside_parapet(tmp_path):
    def f(doc, m):
        c = [e for e in _layer(doc, "TANK") if e.dxftype() == "CIRCLE"][0]
        c.dxf.center = (14000, 7300)          # pokes over the inner edge at y = 7770
    _bad(tmp_path, f, "Tank1 lies outside")


def test_feature_touching_inner_edge_is_inside(tmp_path):
    doc = sample_dxf.build_doc()
    c = [e for e in _layer(doc, "TANK") if e.dxftype() == "CIRCLE"][0]
    c.dxf.center = (14000, 7170)             # top at 7770, exactly the inner edge
    assert read_dxf(_save(doc, tmp_path)).features[2].y == pytest.approx(6.57)


def test_error_two_north_lines(tmp_path):
    _bad(tmp_path, lambda doc, m: m.add_line((0, 0), (0, 1), dxfattribs={"layer": "NORTH"}), "NORTH needs one")


def test_error_zero_north(tmp_path):
    def f(doc, m):
        n = [e for e in _layer(doc, "NORTH") if e.dxftype() == "LINE"][0]
        n.dxf.end = n.dxf.start
    _bad(tmp_path, f, "zero length")


def test_pipe_as_line_and_other_dims_ignored(tmp_path):
    doc = sample_dxf.build_doc()
    m = doc.modelspace()
    m.add_line((1000, 2000), (3000, 2000), dxfattribs={"layer": "PIPE"})
    m.add_text("D=25", dxfattribs={"layer": "PIPE", "insert": (2000, 2100)})
    m.add_radius_dim(center=(5000, 5000), radius=500, angle=45, dimstyle="EZDXF",
                     dxfattribs={"layer": "DIMENSIONS"}).render()
    m.add_point((1, 1), dxfattribs={"layer": "DOOR"})          # not a line: ignored
    d = read_dxf(_save(doc, tmp_path))
    assert sorted(p.diameter for p in d.pipes) == [25, 50] and len(d.dimensions) == 5
    assert np.array(d.pipes[1].points) == pytest.approx(np.array([[1.0, 2.0], [3.0, 2.0]]))


# ---------------- phase2 uses the tank stand ----------------

def test_tank_parts_use_stand():
    f = phase2.Feature("T", "tank", 0, 0, 1.2, 1.2, 2.1, stand=0.6)
    plinth, body, lid = phase2.feature_parts(f, 0.15)
    assert plinth[0][:, 2].max() == pytest.approx(0.75)
    assert body[0][:, 2].min() == pytest.approx(0.75) and body[0][:, 2].max() == pytest.approx(2.25)
    assert phase2.Feature("U", "tank", 0, 0, 1, 1, 1.6).stand == 0.4


def test_layout_json_has_stand(tmp_path, sample):
    p = phase2.write_layout_json(tmp_path / "r.json", sample.spec, sample.features)
    f = json.loads(p.read_text())["features"]
    assert [x["stand"] for x in f] == [0.4, 0.4, 0.6, 0.6]


# ---------------- smoke: the CLIs ----------------

def test_make_sample_dxf_cli(tmp_path):
    dxf, js = _load("make_sample_dxf").main(["--out", str(tmp_path / "s")])
    assert dxf.stat().st_size > 10000 and json.loads(js.read_text())["length"] == 18000


def test_phase2_solar_from_dxf_smoke(tmp_path, sample):
    out = tmp_path / "o"
    lay = _load("phase2_solar").main(["--dxf", str(sample_dxf.write_sample(tmp_path / "roof.dxf")),
                                      "--out", str(out)])
    assert lay.count > 20
    assert all((out / f"rooftop_solar.{e}").stat().st_size > 200 for e in ("step", "glb", "json"))
    # keep-outs still hold: no panel on a structure, all inside the parapets
    floor = phase2.inner_outline(sample.spec)
    for p in lay.panels:
        assert floor.contains(p) and not any(p.intersects(f.footprint) for f in sample.features)
    # panels face south (the drawing's equator side at LAT 19.07)
    assert np.allclose(lay.facing, [0, -1], atol=1e-6)
    js = json.loads((out / "rooftop_solar.json").read_text())
    assert len(js["features"]) == 4


def test_phase2_solar_dxf_overrides_and_box(tmp_path):
    out = tmp_path / "o"
    lay = _load("phase2_solar").main(["--dxf", str(sample_dxf.write_sample(tmp_path / "roof.dxf")),
                                      "--out", str(out), "--box", "--facing-deg", "90", "--latitude", "10"])
    assert np.allclose(lay.facing, [0, 1], atol=1e-6)
    assert json.loads((out / "rooftop_solar.json").read_text())["features"] == []


# ---------------- integration: real files through ezdxf and CadQuery ----------------

def test_sample_dxf_passes_ezdxf_audit():
    import ezdxf
    doc = ezdxf.readfile(SAMPLE_DXF)
    a = doc.audit()
    assert not a.errors and doc.header["$INSUNITS"] == 4
    assert {lay.dxf.name for lay in doc.layers} >= set(sample_dxf.LAYERS)


@pytest.mark.integration
def test_step_from_dxf_reopens_with_right_size(tmp_path, sample):
    import cadquery as cq
    _, corners = phase2.design(sample.spec, features=sample.features, facing_deg=sample.facing_deg(),
                               latitude=sample.latitude)
    p = phase2.export_step(tmp_path / "r.step", sample.spec, corners, features=sample.features)
    shape = cq.importers.importStep(str(p)).val()
    bb = shape.BoundingBox()
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == pytest.approx((0, 0, 18000, 12000), abs=0.5)
    assert bb.zmax == pytest.approx(150 + 2700, abs=0.5)        # room tops are the highest parts
    slab = 18000 * 12000 - 6000 * 4000
    assert shape.Volume() > slab * 150
    plinths = [b for b in (x.BoundingBox() for x in shape.Solids())
               if b.xlen == pytest.approx(1200, abs=0.5) and b.ylen == pytest.approx(1200, abs=0.5)
               and b.zmin == pytest.approx(150, abs=0.5)]
    assert len(plinths) == 2 and all(b.zlen == pytest.approx(600, abs=0.5) for b in plinths)


# ---------------- labels, north, site, drains: finer cases ----------------

def test_each_tank_gets_its_own_label(tmp_path):
    doc = sample_dxf.build_doc()
    t = [e for e in _layer(doc, "TANK") if e.dxftype() == "TEXT"]
    t[1].dxf.text = "H=1000 S=300"
    d = read_dxf(_save(doc, tmp_path))
    tanks = [f for f in d.features if f.kind == "tank"]
    assert np.array([(f.h, f.stand) for f in tanks]) == pytest.approx(np.array([(2.1, 0.6), (1.3, 0.3)]))


def test_north_pointing_down_is_270(tmp_path):
    doc = sample_dxf.build_doc()
    n = [e for e in _layer(doc, "NORTH") if e.dxftype() == "LINE"][0]
    n.dxf.end = (n.dxf.start.x, n.dxf.start.y - 1000)
    d = read_dxf(_save(doc, tmp_path))
    assert d.north_deg == pytest.approx(270.0) and d.facing_deg() == pytest.approx(90.0)


def test_site_info_merges_several_texts(tmp_path):
    doc = sample_dxf.build_doc()
    doc.modelspace().add_text("SLAB=180", dxfattribs={"layer": "SITE_INFO", "insert": (0, -4000)})
    d = read_dxf(_save(doc, tmp_path))
    assert d.site == {"LAT": 19.07, "LON": 72.88, "TZ": 5.5, "FLOORS": 3, "FLOOR_H": 3200, "SLAB": 180}


def test_only_points_are_drains(tmp_path):
    doc = sample_dxf.build_doc()
    doc.modelspace().add_line((100, 100), (200, 200), dxfattribs={"layer": "DRAIN"})
    assert len(read_dxf(_save(doc, tmp_path)).drains) == 3


# ---------------- the sample drawing itself ----------------

def test_parapet_points_exact():
    s = dict(length=10, width=8, notch_length=4, notch_width=3, parapet_thickness=1)
    assert sample_dxf.parapet_points(s) == [(1, 1), (9, 1), (9, 4), (5, 4), (5, 7), (1, 7)]
    assert sample_dxf.outline_points(s) == [(0, 0), (10, 0), (10, 5), (6, 5), (6, 8), (0, 8)]


def test_offset_moves_raw_coordinates(tmp_path):
    import ezdxf
    p = sample_dxf.write_sample(tmp_path / "o.dxf", offset=(1000.0, -500.0))
    e = _layer(ezdxf.readfile(p), "ROOF_OUTLINE")[0]
    assert [tuple(q) for q in e.get_points("xy")][:2] == [(1000.0, -500.0), (19000.0, -500.0)]


def test_room_labels_sit_inside_their_rooms():
    doc = sample_dxf.build_doc()
    for r in sample_dxf.SAMPLE["rooms"]:
        t = [e for e in _layer(doc, r["kind"].upper()) if e.dxftype() == "TEXT"][0]
        x, y = t.dxf.insert.x, t.dxf.insert.y
        assert r["x"] < x < r["x"] + r["w"] and r["y"] < y < r["y"] + r["d"]


def test_phase2_solar_dxf_with_layers_and_units(tmp_path):
    doc = sample_dxf.build_doc(units=0)
    for e in doc.modelspace():
        if e.dxf.layer == "ROOF_OUTLINE":
            e.dxf.layer = "A-ROOF"
    p = _save(doc, tmp_path)
    m = tmp_path / "layers.json"
    m.write_text(json.dumps({"ROOF_OUTLINE": "A-ROOF"}))
    lay = _load("phase2_solar").main(["--dxf", str(p), "--layers", str(m), "--units", "mm",
                                      "--out", str(tmp_path / "o")])
    assert lay.count > 20
