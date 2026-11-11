// M5: sun position (NOAA equations, same maths as rooftop/sun.py) and roof shadows.
// Angles in degrees; azimuth clockwise from true north; plan north_deg counter-clockwise from +x.
const R = Math.PI / 180;

export function julianDay(ms) { return ms / 86400000 + 2440587.5; }

function solar(jd) {
  const t = (jd - 2451545.0) / 36525.0;
  const l0 = (((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360) + 360) % 360;
  const m = 357.52911 + t * (35999.05029 - 0.0001537 * t);
  const e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
  const mr = m * R;
  const c = Math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
    + Math.sin(2 * mr) * (0.019993 - 0.000101 * t) + Math.sin(3 * mr) * 0.000289;
  const om = 125.04 - 1934.136 * t;
  const lam = l0 + c - 0.00569 - 0.00478 * Math.sin(om * R);
  const eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60;
  const eps = eps0 + 0.00256 * Math.cos(om * R);
  const decl = Math.asin(Math.sin(eps * R) * Math.sin(lam * R)) / R;
  const y = Math.tan(eps * R / 2) ** 2;
  const l0r = l0 * R;
  const eot = 4 / R * (y * Math.sin(2 * l0r) - 2 * e * Math.sin(mr) + 4 * e * y * Math.sin(mr) * Math.cos(2 * l0r)
    - 0.5 * y * y * Math.sin(4 * l0r) - 1.25 * e * e * Math.sin(2 * mr));
  return { decl, eot };
}

// ms: UTC milliseconds (Date.UTC / Date.getTime). Returns { alt, az }.
export function sunPosition(lat, lon, ms) {
  const { decl, eot } = solar(julianDay(ms));
  const d = new Date(ms);
  const minutes = d.getUTCHours() * 60 + d.getUTCMinutes() + d.getUTCSeconds() / 60 + d.getUTCMilliseconds() / 60000;
  const ha = (minutes + eot + 4 * lon) / 4 - 180;   // hour angle, only used via sin/cos
  const la = lat * R, de = decl * R, h = ha * R;
  const cz = Math.sin(la) * Math.sin(de) + Math.cos(la) * Math.cos(de) * Math.cos(h);
  const zen = Math.acos(Math.max(-1, Math.min(1, cz)));
  const az = Math.atan2(Math.sin(h), Math.cos(h) * Math.sin(la) - Math.tan(de) * Math.cos(la)) / R + 180;
  return { alt: 90 - zen / R, az: ((az % 360) + 360) % 360 };
}

export function localTime(year, month, day, hour, tz) {
  return Date.UTC(year, month - 1, day) + (hour - tz) * 3600000;
}

// What the page sliders ask for: month 1-12, local clock hour (fractional), day of month 21.
export function sunAt(lat, lon, tz, month, hour, day = 21, year = 2026) {
  return sunPosition(lat, lon, localTime(year, month, day, hour, tz));
}

export function solarNoon(lon, tz, month, day = 21, year = 2026) {
  let h = 12;
  for (let i = 0; i < 3; i++) {
    const { eot } = solar(julianDay(localTime(year, month, day, h, tz)));
    h = 12 - 4 * lon / 60 - eot / 60 + tz;
  }
  return h;
}

// Unit vector towards the sun in the plan frame [x, y, z up].
export function sunDir(alt, az, northDeg = 90) {
  const a = (northDeg - az) * R, c = Math.cos(alt * R);
  return [c * Math.cos(a), c * Math.sin(a), Math.sin(alt * R)];
}

// Plan offset [dx, dy] of the shadow tip of a point `height` up, or null at night.
export function shadow(height, alt, az, northDeg = 90) {
  if (alt <= 0) return null;
  const [x, y, z] = sunDir(alt, az, northDeg);
  const k = height / z;
  return [-x * k, -y * k];
}
