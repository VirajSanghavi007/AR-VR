"""rooftop/sun.py and preview/sun.js (M5): sun position against hand calculations, shadows,
and the JS copy agreeing with Python."""
import json
import math
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rooftop import sun as S

LAT, LON, TZ = 19.07, 72.88, 5.5                     # Mumbai, the sample drawing's site
JS = (Path(__file__).resolve().parents[1] / "preview" / "sun.js").as_uri()


def hand(lat, decl, ha):
    """Textbook spherical trig: altitude and azimuth (clockwise from north) for a declination
    and hour angle (negative = morning)."""
    p, d, h = (math.radians(v) for v in (lat, decl, ha))
    alt = math.asin(math.sin(p) * math.sin(d) + math.cos(p) * math.cos(d) * math.cos(h))
    ca = (math.sin(d) - math.sin(alt) * math.sin(p)) / (math.cos(alt) * math.cos(p))
    az = math.degrees(math.acos(max(-1, min(1, ca))))
    return math.degrees(alt), (az if ha <= 0 else 360 - az)


def test_julian_day_epoch_and_timezone():
    assert S.julian_day(datetime(2000, 1, 1, 12)) == 2451545.0
    aware = datetime(2000, 1, 1, 17, 30, tzinfo=timezone(timedelta(hours=5.5)))
    assert S.julian_day(aware) == 2451545.0
    assert S.sun_position(LAT, LON, aware) == S.sun_position(LAT, LON, datetime(2000, 1, 1, 12))


def test_declination_and_equation_of_time_known_values():
    d, e = S._solar(S.julian_day(datetime(2026, 6, 21, 12)))
    assert abs(d - 23.44) < 0.02 and abs(e - (-1.7)) < 0.3          # June solstice, EoT about -1.7 min
    d, e = S._solar(S.julian_day(datetime(2026, 12, 21, 12)))
    assert abs(d + 23.44) < 0.02 and abs(e - 1.8) < 0.4
    d, e = S._solar(S.julian_day(datetime(2026, 11, 3, 12)))
    assert abs(e - 16.4) < 0.2                                       # the yearly EoT peak
    d, e = S._solar(S.julian_day(datetime(2026, 2, 11, 12)))
    assert abs(e + 14.2) < 0.2


# almanac declination at Mumbai's solar noon (about 07:08 UTC): the 2026 March equinox is
# 20 March 14:46 UTC, so 21 March is 16 h later at about +0.4 deg/day
@pytest.mark.parametrize("month, day, decl", [(3, 21, 0.27), (6, 21, 23.44), (12, 21, -23.44)])
def test_noon_altitude_matches_hand_values(month, day, decl):
    noon = S.solar_noon(LON, TZ, month, day)
    alt, az = S.sun_at(LAT, LON, TZ, month, noon, day)
    assert abs(alt - (90 - abs(LAT - decl))) < 0.03                 # 71.20, 85.63, 47.49
    assert min(abs(az - 180), abs(az), abs(az - 360)) < 0.5        # due south or due north
    if month == 12:
        assert abs(az - 180) < 0.5
    if month == 6:
        assert min(abs(az), abs(az - 360)) < 0.5                      # Mumbai: sun north in June
    for dh in (-0.25, 0.25):
        assert S.sun_at(LAT, LON, TZ, month, noon + dh, day)[0] < alt


def test_solar_noon_in_mumbai_is_about_12_38():
    h = S.solar_noon(LON, TZ, 12, 21)
    assert abs(h - (12 - 4 * LON / 60 + TZ - 1.8 / 60)) < 0.02 and 12.4 < h < 12.7


@pytest.mark.parametrize("ha", [-45.0, 0.0, 45.0])
def test_december_shadow_direction_and_length_match_hand_calculation(ha):
    """The M5 acceptance check: a 1 m pole at 9:00, noon and 15:00 solar time on 21 December."""
    noon = S.solar_noon(LON, TZ, 12, 21)
    alt, az = S.sun_at(LAT, LON, TZ, 12, noon + ha / 15, 21)
    decl = S._solar(S.julian_day(S.local_time(2026, 12, 21, noon + ha / 15, TZ)))[0]
    halt, haz = hand(LAT, decl, ha)
    assert abs(alt - halt) < 0.05 and abs((az - haz + 180) % 360 - 180) < 0.05
    dx, dy = S.shadow(1.0, alt, az, north_deg=90.0)                  # plan north = +y
    assert abs(math.hypot(dx, dy) - 1 / math.tan(math.radians(halt))) < 0.005
    bearing = math.degrees(math.atan2(dx, dy)) % 360                  # clockwise from north
    assert abs((bearing - (haz + 180) + 180) % 360 - 180) < 0.05
    if ha < 0:
        assert dx < 0 and dy > 0                                      # morning: shadow to north-west
    elif ha > 0:
        assert dx > 0 and dy > 0
    else:
        assert abs(dx) < 1e-3 and dy > 0 and abs(dy - 1 / math.tan(math.radians(47.49))) < 0.02


def test_sun_dir_respects_plan_north():
    x, y, z = S.sun_dir(0.0, 90.0, 90.0)                             # east on the horizon, north +y
    assert abs(x - 1) < 1e-12 and abs(y) < 1e-12 and abs(z) < 1e-12
    x, y, z = S.sun_dir(0.0, 90.0, 0.0)                              # north +x: east is -y
    assert abs(x) < 1e-12 and abs(y + 1) < 1e-12
    x, y, z = S.sun_dir(30.0, 180.0, 90.0)
    assert abs(z - 0.5) < 1e-12 and abs(y + math.cos(math.radians(30))) < 1e-12 and abs(x) < 1e-12
    assert abs(math.hypot(*S.sun_dir(37, 211, 33)) - 1) < 1e-12


def test_shadow_none_at_night_and_scales_with_height():
    assert S.shadow(1.0, 0.0, 180) is None and S.shadow(1.0, -5.0, 180) is None
    a = S.shadow(1.0, 45.0, 180.0)
    b = S.shadow(2.5, 45.0, 180.0)
    assert abs(a[1] - 1.0) < 1e-12 and abs(a[0]) < 1e-12 and abs(b[1] - 2.5) < 1e-12


def test_night_and_sunrise():
    alt, _ = S.sun_at(LAT, LON, TZ, 6, 0.0)
    assert alt < -30
    rise = [h / 60 for h in range(4 * 60, 8 * 60) if S.sun_at(LAT, LON, TZ, 6, h / 60)[0] > 0][0]
    assert 5.95 < rise < 6.15                                          # Mumbai June sunrise about 6:00 (centre)


def test_hour_angle_wraps_across_midnight():
    a = S.sun_position(LAT, LON, datetime(2026, 3, 21, 23, 59))
    b = S.sun_position(LAT, LON, datetime(2026, 3, 22, 0, 1))
    assert abs(a[0] - b[0]) < 1 and abs((a[1] - b[1] + 180) % 360 - 180) < 2


# ---------------------------------------------------------------- the JS copy

@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as S from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_js_matches_python_over_a_year(node_available):
    cases = [(m, h) for m in range(1, 13) for h in (5.0, 7.5, 9.25, 10.1234, 12.6, 15.0, 16.98765, 18.4, 23.9)]  # odd seconds too
    js = run("[" + ",".join(f"S.sunAt({LAT},{LON},{TZ},{m},{h})" for m, h in cases) + "]")
    for (m, h), r in zip(cases, js):
        alt, az = S.sun_at(LAT, LON, TZ, m, h)
        assert abs(r["alt"] - alt) < 1e-6 and abs((r["az"] - az + 180) % 360 - 180) < 1e-6
    south = run("S.sunAt(-33.87, 151.21, 10, 6, 12)")                  # Sydney winter
    assert abs(south["alt"] - S.sun_at(-33.87, 151.21, 10, 6, 12)[0]) < 1e-6


def test_js_noon_dir_shadow_and_julian(node_available):
    assert abs(run(f"S.solarNoon({LON},{TZ},12)") - S.solar_noon(LON, TZ, 12)) < 1e-9
    assert abs(run("S.julianDay(Date.UTC(2000,0,1,12))") - 2451545.0) < 1e-9
    assert run("S.localTime(2026,3,21,12,5.5)") == int((datetime(2026, 3, 21, 6, 30) - datetime(1970, 1, 1)).total_seconds() * 1000)
    for args in ((37, 211, 33), (0, 90, 0), (60, 10, 90)):
        assert all(abs(a - b) < 1e-12 for a, b in zip(run(f"S.sunDir({args[0]},{args[1]},{args[2]})"), S.sun_dir(*args)))
    assert run("S.shadow(1, -1, 180)") is None
    assert all(abs(a - b) < 1e-12 for a, b in zip(run("S.shadow(2, 40, 135, 70)"), S.shadow(2, 40, 135, 70)))
    assert run("S.sunAt(19.07, 72.88, 5.5, 3, 12, 21, 2026)") == run("S.sunAt(19.07, 72.88, 5.5, 3, 12)")
