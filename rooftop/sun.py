"""M5: where the sun is, from latitude, longitude, date and time (NOAA solar position
equations, about 0.01 degree from 1901 to 2099), and the shadow a height casts on the roof.
preview/sun.js is the same maths for the browser; tests check the two agree.

Angles in degrees. Azimuth is clockwise from true north (90 = east). Plan directions use
the drawing's north_deg: the direction of north in the plan, counter-clockwise from +x.
"""
import math
from datetime import datetime, timedelta, timezone


def julian_day(when):
    """Julian day of a datetime (naive = UTC)."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return (when - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0


def _solar(jd):
    """(declination deg, equation of time minutes) for a Julian day."""
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    om = 125.04 - 1934.136 * t
    lam = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(om))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(om))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam))))
    y = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eot = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                           - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    return decl, eot


def sun_position(lat, lon, when):
    """(altitude, azimuth) of the sun's centre, no refraction. when: datetime (naive = UTC)."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    decl, eot = _solar(julian_day(when))
    minutes = when.hour * 60 + when.minute + when.second / 60 + when.microsecond / 6e7
    ha = (minutes + eot + 4 * lon) / 4 - 180                        # hour angle, deg (only used via sin/cos)
    la, de, h = (math.radians(v) for v in (lat, decl, ha))
    cz = math.sin(la) * math.sin(de) + math.cos(la) * math.cos(de) * math.cos(h)
    zen = math.acos(max(-1.0, min(1.0, cz)))
    az = math.degrees(math.atan2(math.sin(h), math.cos(h) * math.sin(la) - math.tan(de) * math.cos(la))) + 180
    return 90 - math.degrees(zen), az % 360


def local_time(year, month, day, hour, tz_hours):
    """UTC datetime of a local clock time (hour may be fractional) in a fixed UTC offset."""
    return datetime(year, month, day) + timedelta(hours=hour - tz_hours)


def sun_at(lat, lon, tz_hours, month, hour, day=21, year=2026):
    """(altitude, azimuth) at a local clock time: what the page sliders ask for."""
    return sun_position(lat, lon, local_time(year, month, day, hour, tz_hours))


def solar_noon(lon, tz_hours, month, day=21, year=2026):
    """Local clock hour of solar noon (the sun due north or south, highest)."""
    h = 12.0
    for _ in range(3):
        _, eot = _solar(julian_day(local_time(year, month, day, h, tz_hours)))
        h = 12 - 4 * lon / 60 - eot / 60 + tz_hours
    return h


def sun_dir(alt, az, north_deg=90.0):
    """Unit vector towards the sun in the plan frame (x, y, z up)."""
    a = math.radians(north_deg - az)                                 # clockwise az from plan north
    c = math.cos(math.radians(alt))
    return (c * math.cos(a), c * math.sin(a), math.sin(math.radians(alt)))


def shadow(height, alt, az, north_deg=90.0):
    """Plan offset (dx, dy) of the shadow tip of a point height above the ground, or None
    when the sun is down."""
    if alt <= 0:
        return None
    x, y, z = sun_dir(alt, az, north_deg)
    k = height / z
    return (-x * k, -y * k)
