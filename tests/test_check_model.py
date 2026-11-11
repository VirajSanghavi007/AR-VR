"""M7 checker: the sample model passes, deliberately wrong models fail, the CLI and STEP reader work."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rooftop import check_model as C
from rooftop import details as D
from rooftop.dxf_read import Dimension, Pipe, read_dxf

ROOT = Path(__file__).resolve().parents[1]
DXF = ROOT / "data/sample_roof/roof.dxf"
MEAS = json.loads((ROOT / "data/sample_roof/measurements.json").read_text())


@pytest.fixture(scope="module")
def dwg():
    return read_dxf(DXF)


def parts_of(dwg, features=None, drains=None, pipes=None, floors=None, spec=None, doors=None):
    f, fh = D.site_floors(dwg.site)
    return D.build(spec or dwg.spec, dwg.features if features is None else features, (),
                   dwg.doors if doors is None else doors, dwg.drains if drains is None else drains,
                   dwg.pipes if pipes is None else pipes, floors or f, fh)


def rows_of(dwg, model):
    return C.dimension_rows(model, dwg.dimensions) + C.measurement_rows(model, MEAS)


def failing(rows):
    return sorted(r.name for r in rows if not r.ok)


@pytest.fixture(scope="module")
def good(dwg):
    return C.groups_from_parts(parts_of(dwg))


# ---------------------------------------------------------------- rows and table

def test_row_passes_up_to_the_tolerance_and_fails_when_missing():
    assert C.Row("a", 100, 105).ok and C.Row("a", 100, 95).ok
    assert not C.Row("a", 100, 105.1).ok and not C.Row("a", 100, 94.9).ok
    assert C.Row("a", 100, 102, tol=1).ok is False and C.Row("a", 100, 102, tol=2).ok
    assert C.Row("a", 100, 107).error == 7
    assert not C.Row("a", 100, float("nan")).ok


def test_table_marks_failures_missing_values_and_counts():
    t = C.table([C.Row("ok one", 10, 10.04), C.Row("bad", 10, 20), C.Row("gone", 5, float("nan"))])
    lines = t.splitlines()
    assert lines[0].startswith("| check |") and len(lines) == 2 + 3 + 2
    assert "| ok one | 10.0 | 10.0 | +0.0 | pass |" in t                # never "-0.0"
    assert "| bad | 10.0 | 20.0 | +10.0 | FAIL |" in t
    assert "| gone | 5.0 | missing | - | FAIL |" in t
    assert lines[-1] == "1/3 checks pass (tolerance 5 mm)"
    assert C.table([C.Row("n", 1, 0.96)]).splitlines()[2] == "| n | 1.0 | 1.0 | +0.0 | pass |"
    assert C.table([]).splitlines()[-1] == "no checks"


# ---------------------------------------------------------------- reading

def test_groups_from_parts_boxes_in_mm_and_skips_glb_only_parts():
    parts = [D.box("A", "concrete", (0, 0, 0), (1, 2, 0.5)), D.box("A", "concrete", (2, 0, 0), (3, 1, 1)),
             D.box("Ground", "ground", (-1, -1, -1), (5, 5, 0), step=False),
             D.cyl("T", "tank", (1, 1, 0), (0, 0, 1), 0.5, 2)]
    g = C.groups_from_parts(parts)
    assert set(g) == {"A_concrete", "T_tank"}
    assert len(g["A_concrete"]) == 2
    lo, hi = g["A_concrete"][0]
    assert lo.tolist() == [0, 0, 0] and hi.tolist() == [1000, 2000, 500]
    lo, hi = g["T_tank"][0]
    assert lo == pytest.approx([500, 500, 0]) and hi == pytest.approx([1500, 1500, 2000])


def test_extent_and_plan_cover_all_named_groups():
    g = {"A": [(np.array([0, 0, 0.]), np.array([1, 1, 1.]))], "B": [(np.array([2, -1, -3.]), np.array([3, 0, 0.]))]}
    lo, hi = C.extent(g, "A", "B")
    assert lo.tolist() == [0, -1, -3] and hi.tolist() == [3, 1, 1]
    assert C.extent(g, "A")[1].tolist() == [1, 1, 1]
    assert C.extent(g, "C") is None
    assert C.plan(g, "A", "B").area == pytest.approx(2)
    assert C.plan(g, "C").is_empty


# ---------------------------------------------------------------- dimensions

def test_station_snaps_along_the_extension_line():
    corners = np.array([[0, 0], [18000, 0], [18000, 8000], [12000, 12000]], float)
    u = np.array([1.0, 0])
    # (18000, 12000) is not a corner, but the slab's east edge is on its extension line
    assert C._station(corners, np.array([18000, 12000.]), u, 50).tolist() == [18000, 8000]
    assert C._station(corners, np.array([12010, 12000.]), u, 50).tolist() == [12000, 12000]
    assert C._station(corners, np.array([15000, 12000.]), u, 50) is None
    assert C._station(np.zeros((0, 2)), np.array([0, 0.]), u, 50) is None
    assert C._station(corners, np.array([18040, 0.]), u, 50).tolist() == [18000, 0]
    assert C._station(corners, np.array([18060, 0.]), u, 50) is None


def test_dimension_rows_measure_drawing_and_dict_dimensions(good):
    dims = [Dimension((0, 0), (18, 0), 18000), {"name": "w", "p1": [0, 0], "p2": [0, 12000], "value": 12000},
            {"p1": [800, 8400], "p2": [3800, 8400], "value": 3010}, Dimension((0, 0), (0, 30), 30000)]
    rows = C.dimension_rows(good, dims, tol=5)
    assert [r.name for r in rows] == ["DIMENSIONS dim1", "DIMENSIONS w", "DIMENSIONS dim3", "DIMENSIONS dim4"]
    assert [r.model for r in rows[:3]] == pytest.approx([18000, 12000, 3000])
    assert [r.ok for r in rows] == [True, True, False, False]           # stair is 3000, dim says 3010
    assert np.isnan(rows[3].model)                                       # nothing at y = 30 m
    assert C.dimension_rows(good, dims[:1], tol=1)[0].tol == 1


def test_snap_groups_are_slab_and_room_walls_not_the_building(good):
    s = C.snap_groups(good)
    assert s[0] == "Slab_concrete" and "Stair1_plaster" in s and "Parapet_plaster" in s
    assert "Building_plaster" not in s and "Stair1_concrete" not in s


def test_dimension_rows_use_room_wall_corners():
    g = {"Slab_concrete": [(np.array([0, 0, 0.]), np.array([10000, 10000, 150.]))],
         "R_plaster": [(np.array([2000, 3000, 150.]), np.array([4500, 3200, 2000.]))],
         "Building_plaster": [(np.array([-500, 0, -9000.]), np.array([20000, 10000, 0.]))]}
    r = C.dimension_rows(g, [{"p1": [2000, 3000], "p2": [4500, 3000], "value": 2500},
                             {"p1": [-500, 0], "p2": [10000, 0], "value": 10500}])
    assert r[0].model == pytest.approx(2500) and r[0].ok
    assert np.isnan(r[1].model)                                          # building corners are not used


# ---------------------------------------------------------------- the sample passes

def test_sample_model_passes_every_check(dwg, good):
    rows = rows_of(dwg, good)
    assert len(rows) == 48
    assert failing(rows) == []
    assert max(abs(r.error) for r in rows) < 0.01


def test_measurement_rows_names_cover_the_sample(good):
    names = [r.name for r in C.measurement_rows(good, MEAS)]
    for n in ("length", "width", "diagonal", "notch_length", "notch_width", "parapet_thickness", "parapet_height",
              "Stair1 door start", "Store1 door end", "Tank2 stand", "Drain3 y", "Pipe1 d", "building height"):
        assert n in names
    assert C.measurement_rows(good, {}) == []


# ---------------------------------------------------------------- wrong models fail

def wrong(dwg, name, **kw):
    feats = [replace(f, **kw) if f.name == name else f for f in dwg.features]
    return C.groups_from_parts(parts_of(dwg, features=feats))


def test_room_moved_10_mm_fails_its_position_only(dwg):
    f = next(f for f in dwg.features if f.name == "Store1")
    rows = rows_of(dwg, wrong(dwg, "Store1", x=f.x + 0.01))
    assert failing(rows) == ["Store1 x"]                                  # the drawn door stays put


def test_room_wrong_size_and_height_fail(dwg):
    f = next(f for f in dwg.features if f.name == "Stair1")
    assert "Stair1 w" in failing(rows_of(dwg, wrong(dwg, "Stair1", w=f.w + 0.006)))
    assert failing(rows_of(dwg, wrong(dwg, "Stair1", d=f.d - 0.006))) == ["Stair1 d"]
    assert failing(rows_of(dwg, wrong(dwg, "Stair1", h=f.h + 0.006))) == ["Stair1 h"]
    assert failing(rows_of(dwg, wrong(dwg, "Stair1", y=f.y + 0.004))) == []        # inside 5 mm


def test_door_in_the_wrong_place_fails(dwg):
    doors = [replace(d, x0=d.x0 + 0.02, x1=d.x1 + 0.02) if d.room == "Stair1" else d for d in dwg.doors]
    rows = rows_of(dwg, C.groups_from_parts(parts_of(dwg, doors=doors)))
    assert failing(rows) == ["Stair1 door end", "Stair1 door start"]
    r = {r.name: r for r in rows}
    assert r["Stair1 door start"].error == pytest.approx(20) and r["Stair1 door end"].error == pytest.approx(20)


def test_tank_wrong_place_size_or_stand_fails(dwg):
    f = next(f for f in dwg.features if f.name == "Tank1")
    assert failing(rows_of(dwg, wrong(dwg, "Tank1", x=f.x + 0.01))) == ["Tank1 cx"]
    assert failing(rows_of(dwg, wrong(dwg, "Tank1", y=f.y - 0.01))) == ["Tank1 cy"]
    assert failing(rows_of(dwg, wrong(dwg, "Tank1", stand=f.stand + 0.01))) == ["Tank1 h", "Tank1 stand"]
    assert failing(rows_of(dwg, wrong(dwg, "Tank1", h=f.h + 0.01))) == ["Tank1 h"]
    assert "Tank1 r" in failing(rows_of(dwg, wrong(dwg, "Tank1", w=f.w + 0.02, d=f.d + 0.02)))


def test_wrong_roof_drains_pipes_and_floors_fail(dwg):
    spec = dwg.spec
    assert {"length", "diagonal", "DIMENSIONS dim1"} <= set(
        failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, spec=replace(spec, length=spec.length + 10))))))
    assert {"width", "diagonal"} <= set(
        failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, spec=replace(spec, width=spec.width + 10))))))
    assert "notch_length" in failing(rows_of(dwg, C.groups_from_parts(
        parts_of(dwg, spec=replace(spec, notch_length=spec.notch_length + 10)))))
    assert "notch_width" in failing(rows_of(dwg, C.groups_from_parts(
        parts_of(dwg, spec=replace(spec, notch_width=spec.notch_width + 10)))))
    assert failing(rows_of(dwg, C.groups_from_parts(
        parts_of(dwg, spec=replace(spec, parapet_thickness=spec.parapet_thickness + 10))))) == ["parapet_thickness"]
    assert failing(rows_of(dwg, C.groups_from_parts(
        parts_of(dwg, spec=replace(spec, parapet_height=spec.parapet_height + 10))))) == ["parapet_height"]
    drains = [tuple(d) for d in dwg.drains]
    drains[1] = (drains[1][0], drains[1][1] + 0.01)
    assert failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, drains=drains)))) == ["Drain2 y"]
    p = dwg.pipes[0]
    pipes = [Pipe(p.points, p.diameter + 20)]                           # +10 mm each side
    assert set(failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, pipes=pipes))))) == {
        "Pipe1 d", "Pipe1 x max", "Pipe1 x min", "Pipe1 y max", "Pipe1 y min"}
    pipes = [Pipe(tuple((x + 0.01, y) for x, y in p.points), p.diameter)]
    assert failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, pipes=pipes)))) == ["Pipe1 x max", "Pipe1 x min"]
    assert failing(rows_of(dwg, C.groups_from_parts(parts_of(dwg, floors=2)))) == ["building height"]


def test_one_thick_parapet_side_fails(dwg, good):
    """Only the east parapet 10 mm too thick: the check must find the worst side, not the first."""
    (a0, b0, _), (a1, b1, _) = C.extent(good, "Slab_concrete")
    t = MEAS["parapet_thickness"]
    lo, hi = C.extent(good, "Parapet_plaster")
    model = dict(good, Parapet_plaster=good["Parapet_plaster"] + [
        (np.array([a1 - t - 10, b0, lo[2]]), np.array([a1, b1, hi[2]]))])
    r = [x for x in C.measurement_rows(model, MEAS) if x.name == "parapet_thickness"][0]
    assert not r.ok and r.model == pytest.approx(t + 10, abs=0.01)


def test_missing_groups_fail_as_missing(dwg, good):
    gone = {k: v for k, v in good.items() if not k.startswith(("Store1", "Tank2", "Drain1", "Pipe1", "Coping",
                                                                "Building"))}
    rows = rows_of(dwg, gone)
    miss = {r.name for r in rows if np.isnan(r.model)}
    assert {"Store1 x", "Store1 h", "Store1 door start", "Tank2 cx", "Tank2 stand", "Drain1 x", "Pipe1 d",
            "building height"} <= miss
    assert "parapet_height" not in miss                                   # parapet walls still there
    empty = C.measurement_rows({}, MEAS)
    assert all(np.isnan(r.model) for r in empty) and len(empty) == 41    # no notch rows without a slab


def test_opening_finds_the_gap_on_either_side_or_none():
    from shapely.geometry import box
    walls = box(0, 0, 1000, 200).union(box(2000, 0, 3000, 200))
    assert C._opening(walls, [1000, 0, 2000, 0], np.array([1500, 1000.])) == pytest.approx([0, 1000])
    assert C._opening(walls, [2000, 0, 1000, 0], np.array([1500, 1000.])) == pytest.approx([0, 1000])
    top = box(0, 1800, 1000, 2000).union(box(1500, 1800, 3000, 2000))
    assert C._opening(top, [1000, 2000, 1500, 2000], np.array([1500, 1000.])) == pytest.approx([0, 500])
    assert C._opening(box(0, 0, 3000, 200), [1000, 0, 2000, 0], np.array([1500, 1000.])) is None


# ---------------------------------------------------------------- STEP and CLI

@pytest.fixture(scope="module")
def sample_step(dwg, tmp_path_factory):
    return D.export_step(tmp_path_factory.mktemp("m7") / "rooftop_detailed.step", parts_of(dwg), [])


def test_groups_from_step_match_the_parts(good, sample_step):
    """Integration: the STEP file read back with OpenCascade gives the same boxes as the parts."""
    g = C.groups_from_step(sample_step)
    assert set(good) <= set(g)
    for k in ("Slab_concrete", "Stair1_plaster", "Tank1_tank", "Pipe1_pvc", "Building_plaster", "Drain2_drain"):
        assert len(g[k]) == len(good[k])
        for i in (0, 1):
            assert C.extent(g, k)[i] == pytest.approx(C.extent(good, k)[i], abs=0.01)


def _cli():
    spec = importlib.util.spec_from_file_location("check_dxf_model", ROOT / "scripts/check_dxf_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_passes_the_sample_and_fails_a_wrong_measurement(sample_step, tmp_path, capsys):
    """Smoke + integration: real drawing, real STEP; a tape value 10 mm off makes it exit 1."""
    cli = _cli()
    rep = tmp_path / "r" / "check.md"
    assert cli.main(["--dxf", str(DXF), "--model", str(sample_step), "--report", str(rep)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Model check: rooftop_detailed.step vs roof.dxf") and "48/48 checks pass" in out
    assert rep.read_text(encoding="utf-8") == out
    bad = dict(MEAS, rooms=[dict(MEAS["rooms"][0], w=MEAS["rooms"][0]["w"] + 10)] + MEAS["rooms"][1:])
    mp = tmp_path / "measurements.json"
    mp.write_text(json.dumps(bad))
    assert cli.main(["--dxf", str(DXF), "--model", str(sample_step), "--measurements", str(mp)]) == 1
    out = capsys.readouterr().out
    assert "| Stair1 w | 3010.0 | 3000.0 | -10.0 | FAIL |" in out and "47/48" in out
    assert cli.main(["--dxf", str(DXF), "--model", str(sample_step), "--measurements", str(mp), "--tol", "10"]) == 0
    capsys.readouterr()
    assert cli.main(["--dxf", str(DXF), "--model", str(sample_step), "--measurements", str(tmp_path / "no.json")]) == 0
    assert "5/5 checks pass" in capsys.readouterr().out                  # DIMENSIONS only
    assert cli.main(["--dxf", str(DXF), "--model", str(sample_step), "--measurements", str(tmp_path / "no.json"),
                     "--tol", "-1"]) == 1                                # --tol reaches the DIMENSIONS rows too
    assert "0/5 checks pass (tolerance -1" in capsys.readouterr().out


def test_cli_without_a_model_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit):
        _cli().main(["--dxf", str(DXF), "--model", str(tmp_path / "none.step")])
