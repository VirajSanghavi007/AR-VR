// Movable solar panels for tagar.html: looks, cell layout, touch math, array patterns and an energy
// estimate (no three.js, so node can test it).
// Positions are design (roof) metres on the roof plane as [x, z]; yaw is three.js rotation.y in radians.

// Real module sizes (metres): w across the short side, d along the long side (portrait).
// watts: typical rated power for a module of that size and kind (a guide for the pitch, not a datasheet).
export const PANEL_LOOKS = [
  // Standard module: dark mono cells, silver frame.
  { key: "standard", name: "Standard", w: 1.038, d: 1.755, cols: 6, rows: 20, halfCut: true,
    cell: "#1a2232", busbar: "#5c6878", gap: "#e9edf1", frame: "#b9c0c8", frameMm: 30, watts: 400 },
  // BIPV: frameless, the face is a shingle-roof texture so it blends into the surrounding architecture.
  // Size matches the picture (800 x 1120 px, 0.714) so the bricks are not stretched. A patterned face
  // gives up output: ~120 W/m2 against ~220 W/m2 for a plain mono module.
  { key: "bipv", name: "BIPV shingle", w: 1.1, d: 1.54, cols: 1, rows: 1, halfCut: false,
    cell: "#8a3f2b", busbar: "#8a3f2b", gap: null, frame: "#6b3a2a", frameMm: 0, texture: "bipv_shingle.jpg", watts: 200,
    // glass-glass laminate (real facade modules are ~10-12 mm), a dark fired-ceramic border round the
    // printed face, and a teal glass edge; the mortar joints read as shallow relief under the glass
    // laid flat it is the roof covering itself, so it sits flush with no rack under it
    thickMm: 12, borderMm: 14, edge: "#35504c", relief: 1.6, flush: true },
];

export const lookByKey = (key) => PANEL_LOOKS.find((l) => l.key === key) || PANEL_LOOKS[0];

// Next look in the list, wrapping round: dir +1 or -1.
export function nextLook(key, dir = 1) {
  const i = PANEL_LOOKS.findIndex((l) => l.key === key);
  return PANEL_LOOKS[(((i < 0 ? 0 : i) + dir) % PANEL_LOOKS.length + PANEL_LOOKS.length) % PANEL_LOOKS.length].key;
}

// Cell rectangles in 0..1 panel-face coordinates (x across, y along), inside the frame, with a
// 2 mm gap between cells and, for half-cut modules, a wider gap across the middle.
export function cellLayout(look) {
  const fx = look.frameMm / 1000 / look.w, fy = look.frameMm / 1000 / look.d;
  const gx = 0.002 / look.w, gy = 0.002 / look.d, mid = look.halfCut ? 0.012 / look.d : 0;
  const cw = (1 - 2 * fx - (look.cols - 1) * gx) / look.cols;
  const ch = (1 - 2 * fy - mid - (look.rows - 1) * gy) / look.rows;
  const cells = [];
  for (let r = 0; r < look.rows; r++) {
    const y = fy + r * (ch + gy) + (look.halfCut && r >= look.rows / 2 ? mid : 0);
    for (let c = 0; c < look.cols; c++) cells.push({ x: fx + c * (cw + gx), y, w: cw, h: ch });
  }
  return cells;
}

// Point p = [x, z] in the panel's own frame [u across, v along], panel = {x, z, yaw}.
export function toPanelFrame(p, panel) {
  const dx = p[0] - panel.x, dz = p[1] - panel.z, c = Math.cos(panel.yaw), s = Math.sin(panel.yaw);
  return [dx * c - dz * s, dx * s + dz * c];
}

// Is a touch at p on the panel footprint? slop (metres) widens it so a small model is still easy to hit.
export function panelHit(p, panel, look, slop = 0) {
  const [u, v] = toPanelFrame(p, panel);
  return Math.abs(u) <= look.w / 2 + slop && Math.abs(v) <= look.d / 2 + slop;
}

// Yaw change (for rotation.y) when two fingers move from a0, b0 to a1, b1 on the roof plane.
export function twistDelta(a0, b0, a1, b1) {
  const ang = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);
  let d = ang(a1, b1) - ang(a0, b0);
  d = Math.atan2(Math.sin(d), Math.cos(d));          // wrap to (-pi, pi]
  return -d;                                          // rotation.y turns +x toward -z
}

// A long press: held at least holdMs without the finger drifting more than slopPx.
export const isLongPress = (downAt, now, movedPx, holdMs = 450, slopPx = 12) => now - downAt >= holdMs && movedPx <= slopPx;

// Keep the panel centre inside box = {minX, maxX, minZ, maxZ}.
export const clampToBox = (p, box) => [Math.min(box.maxX, Math.max(box.minX, p[0])), Math.min(box.maxZ, Math.max(box.minZ, p[1]))];

// Yaw in whole degrees 0..359 for the HUD.
export const yawDeg = (yaw) => ((Math.round((yaw * 180) / Math.PI) % 360) + 360) % 360;

// Height map for the relief under a printed face: luminance of an RGBA image (dark mortar = low),
// blurred by `blur` pixels (box, so the joints are bevelled, not razor edges) and stretched to the
// full 0..255 range. Returns grey RGBA of the same size. `rim` px of the outer edge are forced low
// so the border sits down where the glass edge is.
export function heightMap(rgba, w, h, blur = 1, rim = 0) {
  if (rgba.length !== w * h * 4) throw new Error("image must be w x h RGBA");
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) lum[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2];
  let src = lum;
  if (blur > 0) {
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h), n = 2 * blur + 1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let t = 0;
      for (let k = -blur; k <= blur; k++) t += lum[y * w + Math.min(w - 1, Math.max(0, x + k))];
      tmp[y * w + x] = t / n;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let t = 0;
      for (let k = -blur; k <= blur; k++) t += tmp[Math.min(h - 1, Math.max(0, y + k)) * w + x];
      out[y * w + x] = t / n;
    }
    src = out;
  }
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < src.length; i++) { lo = Math.min(lo, src[i]); hi = Math.max(hi, src[i]); }
  const span = hi - lo || 1, res = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inRim = x < rim || y < rim || x >= w - rim || y >= h - rim;
    const v = inRim ? 0 : ((src[y * w + x] - lo) / span) * 255, o = (y * w + x) * 4;
    res[o] = res[o + 1] = res[o + 2] = v; res[o + 3] = 255;
  }
  return res;
}

// Border width in image pixels for a face `faceM` metres wide drawn `px` pixels wide.
export const borderPx = (borderMm, faceM, px) => Math.round((borderMm / 1000 / faceM) * px);

// Where a w x d panel's centre may go so it stays inside the tag rectangle (rectW along the long
// side, rectH across, real metres) shown at `scale` after `turns` quarter turns. Panel metres, centred.
export function moveBox(rectW, rectH, turns, scale, w, d) {
  const [rx, rz] = turns % 2 ? [rectH, rectW] : [rectW, rectH];
  const hx = Math.max(0, (rx / scale - w) / 2), hz = Math.max(0, (rz / scale - d) / 2);
  return { minX: -hx, maxX: hx, minZ: -hz, maxZ: hz };
}

// ---------- several panels: an array of {x, z, yaw}, all of one look ----------
export const GAP_M = 0.02;                           // clamp gap between neighbouring modules

// Does panel a sit on panel b (both w x d)? Exact when they share a yaw, which the patterns keep.
export function overlaps(a, b, w, d) {
  const [u, v] = toPanelFrame([a.x, a.z], b);
  return Math.abs(u) < w - 1e-6 && Math.abs(v) < d - 1e-6;
}

// Where one more panel goes: beside panel `from` (right, left, behind, in front, then further out),
// turned like it, on the first spot that is free.
export function addSlot(panels, w, d, from = panels.length - 1, gap = GAP_M) {
  if (!panels.length) return { x: 0, z: 0, yaw: 0 };
  const p = panels[Math.min(panels.length - 1, Math.max(0, from))], c = Math.cos(p.yaw), s = Math.sin(p.yaw);
  for (let k = 1; ; k++) {
    for (const [u, v] of [[k * (w + gap), 0], [-k * (w + gap), 0], [0, k * (d + gap)], [0, -k * (d + gap)]]) {
      const q = { x: p.x + u * c + v * s, z: p.z - u * s + v * c, yaw: p.yaw };   // panel frame back to roof
      if (!panels.some((o) => overlaps(q, o, w, d))) return q;
    }
  }
}

// Indices of the back row: panels within half a panel of the largest z.
export function lastRow(panels, d) {
  const zmax = Math.max(...panels.map((p) => p.z));
  return panels.flatMap((p, i) => (p.z > zmax - d / 2 ? [i] : []));
}

// Pattern: copy the back row one row further back. Returns a new array.
export const addRow = (panels, d, gap = GAP_M) =>
  panels.concat(lastRow(panels, d).map((i) => ({ ...panels[i], z: panels[i].z + d + gap })));

// Drop the back row, unless it is the only one. Returns a new array.
export function removeRow(panels, d) {
  const back = new Set(lastRow(panels, d));
  return back.size === panels.length ? panels.slice() : panels.filter((_, i) => !back.has(i));
}

// Footprint of the whole array (turned panels count by their corners): centre and size, metres.
export function arrayBounds(panels, w, d) {
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (const p of panels) {
    const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
    for (const [u, v] of [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]) {
      const x = p.x + u * c + v * s, z = p.z - u * s + v * c;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
    }
  }
  return { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
}

// Another look has another size: stretch the spacing so neighbours stay one gap apart.
export const relayout = (panels, from, to, gap = GAP_M) =>
  panels.map((p) => ({ ...p, x: (p.x * (to.w + gap)) / (from.w + gap), z: (p.z * (to.d + gap)) / (from.d + gap) }));

// Rough output of n panels of `watts` each: peak power, energy per day from the site's peak sun hours
// (kWh/m2/day, ~4-6 in India) after typical losses (derate: heat, inverter, wiring, dust), and that
// energy as an average per hour over the whole day and over the daylight hours.
export function energy(n, watts, sunHours = 5, derate = 0.8, daylightHours = 12) {
  const whDay = n * watts * sunHours * derate;
  return { peakW: n * watts, kwhDay: whDay / 1000, avgW: whDay / 24, dayAvgW: whDay / daylightHours };
}

// The mounting under one panel, in the panel object's frame (roof at y = 0, panel tilted by tiltDeg
// about x with its back edge, -z, up). Returns y = height of the tilt pivot (the panel's underside
// centre) and parts = boxes {x, y, z, sx, sy, sz, rotX} (centre, size, turn about x), all metres.
// Tilted: per side a rail under the panel, a short front and a tall rear leg down to a foot rail on
// the roof, and a brace between the rear legs; lift = gap under the low front edge. Flat: two rails
// on the roof, or nothing for a flush (BIPV) look.
export function mountParts(look, tiltDeg, { lift = 0.15, bar = 0.04, inset = 0.2, legInset = 0.1 } = {}) {
  const a = (tiltDeg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), { w, d } = look;
  const xs = [-1, 1].map((k) => k * (w / 2 - inset * w));
  if (a === 0) {
    if (look.flush) return { y: 0.002, parts: [] };
    return { y: bar, parts: xs.map((x) => ({ x, y: bar / 2, z: 0, sx: bar, sy: bar, sz: d * 0.95, rotX: 0 })) };
  }
  const zf = d / 2 - legInset * d, zr = -zf;                   // leg positions along the panel
  const y = lift + bar * c + zf * s;                           // front leg top lands exactly at lift
  const under = (zt) => ({ Y: y - bar * c - zt * s, Z: -bar * s + zt * c });   // rail underside, panel z = zt
  const F = under(zf), R = under(zr), parts = [];
  for (const x of xs) {
    parts.push({ x, y: y - (bar / 2) * c, z: -(bar / 2) * s, sx: bar, sy: bar, sz: d * 0.95, rotX: a });  // rail
    parts.push({ x, y: F.Y / 2, z: F.Z, sx: bar, sy: F.Y, sz: bar, rotX: 0 });                           // front leg
    parts.push({ x, y: R.Y / 2, z: R.Z, sx: bar, sy: R.Y, sz: bar, rotX: 0 });                           // rear leg
    parts.push({ x, y: bar / 4, z: (F.Z + R.Z) / 2, sx: bar, sy: bar / 2, sz: F.Z - R.Z + bar, rotX: 0 }); // foot rail
  }
  parts.push({ x: 0, y: R.Y * 0.55, z: R.Z, sx: xs[1] - xs[0], sy: bar * 0.6, sz: bar * 0.6, rotX: 0 });  // rear brace
  return { y, parts };
}
