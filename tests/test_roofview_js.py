"""preview/roofview.js (M6): the pure logic behind roof.html (sun per month and hour, yearly energy,
HUD text, proposal camera shots, the neighbour block), run under node and checked against
rooftop/sun.py and hand values."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from rooftop import sun as S

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
JS = (PREVIEW / "roofview.js").as_uri()
INFO = json.loads((PREVIEW / "models" / "rooftop_detailed.json").read_text())


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, info=INFO):
    code = (f'import * as R from "{JS}"; const info = {json.dumps(info)};'
            f" console.log(JSON.stringify({expr}));")
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def plan(w):
    """world (x, y up, z) back to plan (x, y, z up)."""
    return [w[0], -w[2], w[1]]


def test_to_world_turns_plan_z_up_into_y_up(node_available):
    assert run("R.toWorld([1, 2, 3])") == [1, 3, -2]


@pytest.mark.parametrize("month,hour", [(12, 9.0), (12, 12.5), (6, 16.25), (3, 7.0), (12, 22.0)])
def test_sun_for_matches_sun_py(node_available, month, hour):
    got = run(f"R.sunFor(info, {month}, {hour})")
    alt, az = S.sun_at(INFO["latitude"], INFO["longitude"], INFO["tz"], month, hour)
    assert got["alt"] == pytest.approx(alt, abs=1e-6) and got["az"] == pytest.approx(az, abs=1e-6)
    assert got["up"] is (alt > 0)
    assert plan(got["dir"]) == pytest.approx(S.sun_dir(alt, az, INFO["north_deg"]), abs=1e-9)


def test_sun_for_uses_plan_north(node_available):
    turned = dict(INFO, north_deg=0.0)
    a = run("R.sunFor(info, 12, 12.5)", turned)
    alt, az = S.sun_at(INFO["latitude"], INFO["longitude"], INFO["tz"], 12, 12.5)
    assert plan(a["dir"]) == pytest.approx(S.sun_dir(alt, az, 0.0), abs=1e-9)
    no_north = {k: v for k, v in INFO.items() if k != "north_deg"}
    assert run("R.sunFor(info, 12, 12.5).dir", no_north) == pytest.approx(run("R.sunFor(info, 12, 12.5).dir"))


def test_sun_strength_ramps_smoothly_to_full_at_25_degrees(node_available):
    got = run("[-10, 0, 5, 12.5, 25, 60, 90].map((a) => R.sunStrength(a))")
    assert got[0] == 0 and got[1] == 0
    assert got[4] == pytest.approx(3.2) and got[5] == pytest.approx(3.2) and got[6] == pytest.approx(3.2)
    k = math.sin(math.radians(5)) / math.sin(math.radians(25))
    assert got[2] == pytest.approx(3.2 * k * k * (3 - 2 * k))
    k = math.sin(math.radians(12.5)) / math.sin(math.radians(25))
    assert got[3] == pytest.approx(3.2 * k * k * (3 - 2 * k))
    assert 0 < got[2] < got[3] < got[4]
    assert run("R.sunStrength(40, 2)") == pytest.approx(2)


def test_yearly_energy_from_sun_hours_and_derate(node_available):
    y = run("R.yearly(41, 550)")
    assert y["peakKw"] == pytest.approx(22.55)
    assert y["kwhDay"] == pytest.approx(41 * 550 * 5 * 0.8 / 1000)
    assert y["kwhYear"] == pytest.approx(y["kwhDay"] * 365)
    y = run("R.yearly(10, 400, 4, 0.75)")
    assert y["kwhDay"] == pytest.approx(10 * 400 * 4 * 0.75 / 1000) and y["peakKw"] == pytest.approx(4)


def test_clock_rounds_minutes_and_wraps(node_available):
    assert run('[7.0833, 12.5, 0, 9.999, 23.9999, 24.25, 13.75].map((h) => R.clock(h))') == \
        ["07:05", "12:30", "00:00", "10:00", "00:00", "00:15", "13:45"]


def test_compass_points(node_available):
    assert run("[0, 22, 23, 45, 90, 135, 180, 225, 270, 315, 337.6, 360, -45, 720].map((a) => R.compass(a))") == \
        ["N", "N", "NE", "NE", "E", "SE", "S", "SW", "W", "NW", "N", "N", "NW", "N"]


def test_hud_lines(node_available):
    up = run('R.hud(41, 550, {up: true, alt: 47.4, az: 181}, 12, 12.5)')
    kwh = round(41 * 550 * 5 * 0.8 / 1000 * 365)
    assert up[0] == f"41 panels · 22.55 kWp · about {kwh:,} kWh a year"
    assert up[1] == "21 Dec 12:30 · sun 47° up, from S"
    down = run('R.hud(3, 400, {up: false, alt: -5, az: 300}, 6, 21.25, 6)')
    assert down[0] == f"3 panels · 1.20 kWp · about {round(3 * 400 * 6 * 0.8 / 1000 * 365):,} kWh a year"
    assert down[1] == "21 Jun 21:15 · sun below the horizon"
    assert run("R.MONTHS.length") == 12 and run("R.MONTHS[0]") == "Jan"


def test_noon_for_is_solar_noon(node_available):
    for m in (3, 12):
        assert run(f"R.noonFor(info, {m})") == pytest.approx(S.solar_noon(INFO["longitude"], INFO["tz"], m), abs=1e-6)


def test_plan_bounds_of_the_l_roof(node_available):
    assert run("R.planBounds(info.cells)") == {"x0": 0, "y0": 0, "x1": 18, "y1": 12}
    assert run("R.planBounds([[2, -1, 5, 3], [-4, 0, 1, 9]])") == {"x0": -4, "y0": -1, "x1": 5, "y1": 9}


def test_shots_frame_the_roof_from_the_panel_side(node_available):
    panel = [9, 0.5, -3]
    s = run(f"R.shots(info, {panel})")
    assert set(s) == {"aerial", "eye", "front", "closeup"} == set(run("R.SHOT_NAMES"))
    span = math.hypot(18, 12)
    centre = [9, 0, -6]                                              # world of plan (9, 6, 0)
    assert s["aerial"]["target"] == pytest.approx(centre)
    # facing_deg 270: the panels face plan -y, i.e. world +z; every overview camera stands on that side
    for k in ("aerial", "eye", "front"):
        assert s[k]["pos"][2] > centre[2] + span * 0.7, k
    assert s["aerial"]["pos"] == pytest.approx([9 + span * 0.45, span * 0.95, -6 + span * 0.75])
    assert s["aerial"]["fov"] == 40
    assert s["eye"]["pos"][1] == pytest.approx(-9.6 + 1.6)                # eye height above the ground
    assert s["eye"]["pos"] == pytest.approx([9 - span * 0.35, -8.0, -6 + span * 1.25])
    assert s["eye"]["target"] == pytest.approx([9, 0.8, -6]) and s["eye"]["fov"] == 50
    assert s["front"]["pos"] == pytest.approx([9, -9.6 * 0.35 + 2, -6 + span * 1.6])
    assert s["front"]["target"] == pytest.approx([9, -9.6 * 0.45, -6]) and s["front"]["fov"] == 38
    c = s["closeup"]
    assert c["target"] == panel and c["fov"] == 45
    assert c["pos"] == pytest.approx([9 + 1.2, 2.0, -3 + 2.6])
    assert 2 < math.dist(c["pos"], panel) < 4                           # close, but the whole panel in view


def test_shots_follow_facing_and_default_base(node_available):
    east = dict(INFO, facing_deg=0.0)
    s = run("R.shots(info, [0, 0, 0])", east)
    span = math.hypot(18, 12)
    assert s["front"]["pos"] == pytest.approx([9 + span * 1.6, -9.6 * 0.35 + 2, -6])
    assert s["closeup"]["pos"] == pytest.approx([2.6, 1.5, -1.2])
    bare = {k: v for k, v in INFO.items() if k not in ("facing_deg", "building_base")}
    d = run("R.shots(info, [0, 0, 0])", bare)
    assert d["eye"]["pos"][1] == pytest.approx(-9 + 1.6)                 # base defaults to -9 m
    assert d["front"]["pos"] == pytest.approx([9, -9 * 0.35 + 2, -6 + span * 1.6])  # facing defaults to 270


def test_neighbour_sits_across_the_street_on_the_sunny_side(node_available):
    n = run("R.neighbour(info)")
    # facing 270 = plan -y: the block is south of the roof, 18 + 4 m wide, 10 m deep, 8 m of street away
    assert n == pytest.approx({"x": 9, "y": 6 - (6 + 8 + 5), "sx": 22, "sy": 10, "h": 18})
    e = run("R.neighbour(info, 5, 30, 6)", dict(INFO, facing_deg=10.0))   # rounds to +x
    assert e == pytest.approx({"x": 9 + 9 + 5 + 3, "y": 6, "sx": 6, "sy": 16, "h": 30})
    w = run("R.neighbour(info)", dict(INFO, facing_deg=170.0))            # rounds to -x
    assert w["x"] == pytest.approx(9 - 9 - 8 - 5) and w["y"] == pytest.approx(6)
    no = run("R.neighbour(info)", {k: v for k, v in INFO.items() if k != "facing_deg"})
    assert no == pytest.approx(n)
    # the block clears the roof by the street gap
    assert (6 - n["y"]) - n["sy"] / 2 - 6 == pytest.approx(8)


def test_sun_for_is_down_just_below_the_horizon(node_available):
    # the last minutes before sunrise: the sun is under 1 degree below the horizon, still "down"
    h = next(h / 100 for h in range(500, 800) if -1 < S.sun_at(INFO["latitude"], INFO["longitude"], INFO["tz"], 12, h / 100)[0] < 0)
    got = run(f"R.sunFor(info, 12, {h})")
    assert -1 < got["alt"] < 0 and got["up"] is False
