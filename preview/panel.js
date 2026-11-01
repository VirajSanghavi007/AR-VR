// One movable solar panel for tagar.html: looks, cell layout and touch math (no three.js, so node can test it).
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
    thickMm: 12, borderMm: 14, edge: "#35504c", relief: 1.6 },
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
