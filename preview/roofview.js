// Pure logic behind roof.html (the Phase 2 customer roof viewer), kept free of three.js so node tests
// can check it: the sun for a month and time, yearly energy, the HUD lines, and the four proposal
// camera shots. Plan coordinates (rooftop_detailed.json): x east-ish, y along the drawing, z up, metres.
// three.js / glTF world: (x, y, z)plan -> (x, z, -y), the same turn as rooftop/ar_export.py Z_TO_Y.
import { sunAt, sunDir, solarNoon } from "./sun.js?v=1";
import { energy } from "./panel.js?v=9";

export const toWorld = ([x, y, z]) => [x, z, -y];

// Sun for the site at hour (local clock time, decimal) on the 21st of month.
// dir = unit vector towards the sun in world coordinates; up = above the horizon.
export function sunFor(info, month, hour) {
  const { alt, az } = sunAt(info.latitude, info.longitude, info.tz, month, hour);
  return { alt, az, up: alt > 0, dir: toWorld(sunDir(alt, az, info.north_deg ?? 90)) };
}

// Directional light strength for a sun altitude: none below the horizon, rising smoothly to full
// by about 25 degrees (the atmosphere dims a low sun), plus a dim floor so night is not pitch black.
export function sunStrength(alt, full = 3.2) {
  const k = Math.min(1, Math.max(0, Math.sin((alt * Math.PI) / 180) / Math.sin((25 * Math.PI) / 180)));
  return full * k * k * (3 - 2 * k);
}

// Peak kW, kWh per year and per day for n panels of `watts` (panel.js energy: sun hours x derate).
export function yearly(n, watts, sunHours = 5, derate = 0.8) {
  const e = energy(n, watts, sunHours, derate);
  return { peakKw: e.peakW / 1000, kwhDay: e.kwhDay, kwhYear: e.kwhDay * 365 };
}

// "07:05" from 7.0833 h; minutes rounded, 24 wraps to 00.
export function clock(hour) {
  const m = Math.round(hour * 60) % (24 * 60);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// The HUD text: system size and output, then the sun.
export function hud(n, watts, sun, month, hour, sunHours = 5) {
  const y = yearly(n, watts, sunHours);
  const grp = (v) => Math.round(v).toLocaleString("en-US");
  const s = sun.up ? `sun ${sun.alt.toFixed(0)}° up, from ${compass(sun.az)}` : "sun below the horizon";
  return [`${n} panels · ${y.peakKw.toFixed(2)} kWp · about ${grp(y.kwhYear)} kWh a year`,
          `21 ${MONTHS[month - 1]} ${clock(hour)} · ${s}`];
}

export function compass(az) {
  const pts = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return pts[Math.round((((az % 360) + 360) % 360) / 45) % 8];
}

// Solar noon for the site, the default time on the slider.
export const noonFor = (info, month) => solarNoon(info.longitude, info.tz, month);

// Plan bounds of the roof footprint cells [[x0, y0, x1, y1], ...].
export function planBounds(cells) {
  const xs = cells.flatMap((c) => [c[0], c[2]]), ys = cells.flatMap((c) => [c[1], c[3]]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

// The four proposal stills, as world-space {pos, target, fov}. Front = the side the panels face
// (facing_deg, plan degrees counter-clockwise from +x, as in phase2), so the glass is seen.
// panel = world centre of one panel for the close-up; base = building base height (negative).
export function shots(info, panel) {
  const b = planBounds(info.cells), cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
  const span = Math.hypot(b.x1 - b.x0, b.y1 - b.y0), base = info.building_base ?? -9;
  const f = ((info.facing_deg ?? 270) * Math.PI) / 180, fx = Math.cos(f), fy = Math.sin(f);
  const at = (x, y, z) => toWorld([x, y, z]);
  const centre = at(cx, cy, 0);
  const side = [-fy, fx];                                    // to the right of the facing direction
  return {
    aerial: { pos: at(cx + fx * span * 0.75 + side[0] * span * 0.45, cy + fy * span * 0.75 + side[1] * span * 0.45, span * 0.95),
              target: centre, fov: 40 },
    eye: { pos: at(cx + fx * span * 1.25 - side[0] * span * 0.35, cy + fy * span * 1.25 - side[1] * span * 0.35, base + 1.6),
           target: at(cx, cy, 0.8), fov: 50 },
    front: { pos: at(cx + fx * span * 1.6, cy + fy * span * 1.6, base * 0.35 + 2), target: at(cx, cy, base * 0.45), fov: 38 },
    closeup: { pos: [panel[0] + fx * 2.6 + side[0] * 1.2, panel[1] + 1.5, panel[2] - (fy * 2.6 + side[1] * 1.2)],
               target: panel, fov: 45 },
  };
}

export const SHOT_NAMES = ["aerial", "eye", "front", "closeup"];

// The optional neighbour block: a taller building across the street on the panels' sunny side,
// 4 m wider than the roof edge it faces, `gap` m of street away, d deep, h tall above the building base.
// Returns {x, y, sx, sy, h}: plan centre and plan size (sx along x, sy along y). Facing is rounded to
// the nearer plan axis, so the block stays square to the street.
export function neighbour(info, gap = 8, h = 18, d = 10) {
  const b = planBounds(info.cells), cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
  const f = ((info.facing_deg ?? 270) * Math.PI) / 180;
  const alongX = Math.abs(Math.cos(f)) > Math.abs(Math.sin(f));
  const fx = alongX ? Math.sign(Math.cos(f)) : 0, fy = alongX ? 0 : Math.sign(Math.sin(f));
  const reach = alongX ? (b.x1 - b.x0) / 2 : (b.y1 - b.y0) / 2;
  const edge = alongX ? b.y1 - b.y0 : b.x1 - b.x0;
  const dist = reach + gap + d / 2;
  return { x: cx + fx * dist, y: cy + fy * dist, sx: alongX ? d : edge + 4, sy: alongX ? edge + 4 : d, h };
}
