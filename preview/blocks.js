// Architect's design on site (site.html): parametric blocks, drawn outlines and wall lines, materials,
// snapping, size labels, undo, design options A/B/C, the design file, and model-to-site alignment.
// Pure, no three.js: build3d.js turns the parts into meshes.
//
// Everything is in room metres (tagmap.js room frame: y up, floor at y = 0 once the floor is known).
// An object stands on the floor at (x, z), turned by yaw (three.js rotation.y), base raised by y.
// Its parts are in the object's own frame: x along its width, y up, z across (depth), origin at the
// middle of its base.

export const MATERIALS = {
  wood:     { name: "Wood",     color: "#a8743f", roughness: 0.72, metalness: 0 },
  glass:    { name: "Glass",    color: "#b9d6e2", roughness: 0.05, metalness: 0, opacity: 0.32 },
  concrete: { name: "Concrete", color: "#9c9b96", roughness: 0.92, metalness: 0 },
  plaster:  { name: "Plaster",  color: "#ebe6dc", roughness: 0.95, metalness: 0 },
  metal:    { name: "Metal",    color: "#a2abb5", roughness: 0.32, metalness: 0.85 },
  // fixed finishes some blocks use for parts that are not the picked material
  soil:  { name: "Soil",  color: "#4b3a2a", roughness: 1, metalness: 0, fixed: true },
  plant: { name: "Plant", color: "#4f7f3a", roughness: 0.85, metalness: 0, fixed: true },
  pv:    { name: "PV",    color: "#1a2232", roughness: 0.4, metalness: 0.1, fixed: true },
};
export const PICKABLE = ["wood", "glass", "concrete", "plaster", "metal"];
export const nextMaterial = (key, dir = 1) => PICKABLE[(((PICKABLE.indexOf(key) + dir) % PICKABLE.length) + PICKABLE.length) % PICKABLE.length];

// The library: default size (w along, h up, d across, metres), base height y, material, what to call it.
export const BLOCKS = {
  wall:    { name: "Wall",        w: 2.0,   h: 2.4,  d: 0.12,  y: 0,   mat: "plaster" },
  door:    { name: "Door",        w: 0.9,   h: 2.1,  d: 0.12,  y: 0,   mat: "wood" },
  window:  { name: "Window",      w: 1.2,   h: 1.2,  d: 0.1,   y: 0.9, mat: "metal" },
  pergola: { name: "Pergola",     w: 3.0,   h: 2.4,  d: 2.7,   y: 0,   mat: "wood" },
  planter: { name: "Planter",     w: 1.2,   h: 0.45, d: 0.4,   y: 0,   mat: "concrete" },
  bench:   { name: "Bench",       w: 1.6,   h: 0.45, d: 0.42,  y: 0,   mat: "wood" },
  panel:   { name: "Solar panel", w: 1.038, h: 0.6,  d: 1.755, y: 0,   mat: "metal", tiltDeg: 15 },
};
// drawn shapes: a closed floor outline pulled up into a solid, and a wall line pulled up into walls
export const DRAWN = { slab: { name: "Outline", mat: "concrete", h: 0.15 }, walls: { name: "Wall line", mat: "plaster", h: 2.4, d: 0.12 } };
export const LIMITS = { min: 0.02, max: 30 };
const clampLen = (v, lo = LIMITS.min, hi = LIMITS.max) => Math.min(hi, Math.max(lo, v));

let nextId = 1;
export const newId = () => `o${Date.now().toString(36)}${(nextId++).toString(36)}`;

// A new block of `type` at (x, z) turned by yaw, at its default size.
export function newBlock(type, x = 0, z = 0, yaw = 0) {
  const b = BLOCKS[type];
  if (!b) throw new Error(`no block called ${type}`);
  return { id: newId(), type, x, z, y: b.y, yaw, w: b.w, h: b.h, d: b.d, mat: b.mat };
}

// A drawn shape from room points pts [[x, z], ...]: kind "slab" (closed outline) or "walls" (open line).
// The points are kept relative to their centre, which becomes the object's (x, z).
export function newDrawn(kind, pts, h = DRAWN[kind].h) {
  if (kind === "slab" && pts.length < 3) throw new Error("an outline needs 3 or more corners");
  if (kind === "walls" && pts.length < 2) throw new Error("a wall line needs 2 or more points");
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cz = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const rel = pts.map(([x, z]) => [x - cx, z - cz]);
  const b = bounds2(rel);
  return { id: newId(), type: kind, x: cx, z: cz, y: 0, yaw: 0, w: b.w, h, d: kind === "walls" ? DRAWN.walls.d : b.d,
           mat: DRAWN[kind].mat, pts: rel, base: { w: b.w, d: b.d } };
}

export function bounds2(pts) {
  const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minZ: Math.min(...zs), maxZ: Math.max(...zs),
           w: Math.max(...xs) - Math.min(...xs), d: Math.max(...zs) - Math.min(...zs) };
}

// A drawn shape's points at its current size: resizing stretches the outline from its centre.
export function scaledPts(o) {
  if (!o.pts) return null;
  const sx = o.base && o.base.w > 1e-6 ? o.w / o.base.w : 1, sz = o.type === "slab" && o.base && o.base.d > 1e-6 ? o.d / o.base.d : sx;
  return o.pts.map(([x, z]) => [x * sx, z * (o.type === "walls" ? sx : sz)]);
}

// Box part: centre c, size s, optional turn about x (rx) and y (ry), material key ("main" = the object's).
const box = (c, s, mat = "main", extra = {}) => ({ kind: "box", c, s, mat, ...extra });

// The parts of one object, in its own frame (see the top of the file).
export function blockParts(o) {
  const { w, h, d } = o, P = [];
  switch (o.type) {
    case "wall":
      P.push(box([0, h / 2, 0], [w, h, d]));
      break;
    case "door": {
      const f = 0.06, lw = w - 2 * f, lh = h - f;
      P.push(box([-(w - f) / 2, h / 2, 0], [f, h, d]), box([(w - f) / 2, h / 2, 0], [f, h, d]), box([0, h - f / 2, 0], [w, f, d]));
      P.push(box([0, lh / 2, 0], [lw - 0.006, lh - 0.006, 0.04]));                         // leaf
      P.push(box([lw / 2 - 0.08, Math.min(1.05, lh * 0.5), 0.035], [0.12, 0.02, 0.03], "metal"));   // handle
      break;
    }
    case "window": {
      const f = 0.05;
      P.push(box([0, f / 2, 0], [w, f, d]), box([0, h - f / 2, 0], [w, f, d]),
             box([-(w - f) / 2, h / 2, 0], [f, h, d]), box([(w - f) / 2, h / 2, 0], [f, h, d]),
             box([0, h / 2, 0], [0.04, h - 2 * f, d * 0.8]));                              // mullion
      P.push(box([0, h / 2, 0], [w - 2 * f, h - 2 * f, 0.012], "glass"));
      break;
    }
    case "pergola": {
      const p = 0.12, beam = 0.2, raft = 0.05;
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) P.push(box([sx * (w / 2 - p / 2), (h - beam) / 2, sz * (d / 2 - p / 2)], [p, h - beam, p]));
      for (const sz of [-1, 1]) P.push(box([0, h - beam / 2, sz * (d / 2 - p / 2)], [w + 0.2, beam, 0.08]));
      const n = Math.max(2, Math.round(w / 0.4) + 1);
      for (let i = 0; i < n; i++) P.push(box([-w / 2 + (i * w) / (n - 1), h + raft / 2, 0], [raft, raft * 3, d + 0.3]));
      break;
    }
    case "planter": {
      const t = 0.05;
      P.push(box([0, h / 2, 0], [w, h, d]));
      P.push(box([0, h - t / 2, 0], [w - 2 * t, 0.02, d - 2 * t], "soil"));
      const n = Math.max(1, Math.round(w / 0.3));
      for (let i = 0; i < n; i++) P.push({ kind: "ball", c: [-w / 2 + ((i + 0.5) * w) / n, h + 0.1, 0], r: Math.min(0.14, d / 2.4), mat: "plant" });
      break;
    }
    case "bench": {
      const slat = 0.04, n = Math.max(2, Math.round(d / 0.1));
      for (let i = 0; i < n; i++) P.push(box([0, h - slat / 2, -d / 2 + ((i + 0.5) * d) / n], [w, slat, d / n - 0.012]));
      for (const sx of [-1, 1]) P.push(box([sx * (w / 2 - 0.15), (h - slat) / 2, 0], [0.06, h - slat, d * 0.9], "metal"));
      break;
    }
    case "panel": {
      const a = ((BLOCKS.panel.tiltDeg) * Math.PI) / 180, lift = 0.15, front = d / 2 - 0.1 * d;
      const pivot = lift + front * Math.sin(a) + 0.035;           // front edge clears the floor by lift
      P.push(box([0, pivot, 0], [w, 0.035, d], "pv", { rx: a }));
      for (const sx of [-1, 1]) {
        const x = sx * (w / 2 - 0.2 * w);
        const yf = pivot - front * Math.sin(a), yr = pivot + front * Math.sin(a);
        P.push(box([x, yf / 2, front * Math.cos(a)], [0.04, yf, 0.04]), box([x, yr / 2, -front * Math.cos(a)], [0.04, yr, 0.04]));
      }
      break;
    }
    case "slab":
      P.push({ kind: "prism", pts: scaledPts(o), y0: 0, h, mat: "main" });
      break;
    case "walls": {
      const pts = scaledPts(o);
      for (let i = 0; i + 1 < pts.length; i++) {
        const [a, b] = [pts[i], pts[i + 1]], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (len < 1e-4) continue;
        // walls overlap by their thickness at the joints, so the corners close
        P.push(box([(a[0] + b[0]) / 2, h / 2, (a[1] + b[1]) / 2], [len + o.d, h, o.d], "main", { ry: -Math.atan2(b[1] - a[1], b[0] - a[0]) }));
      }
      break;
    }
    default: throw new Error(`no parts for ${o.type}`);
  }
  return P;
}

// Room point -> the object's own frame (x along, z across), and back.
export function toLocal(o, [px, pz]) {
  const dx = px - o.x, dz = pz - o.z, c = Math.cos(o.yaw), s = Math.sin(o.yaw);
  return [dx * c - dz * s, dx * s + dz * c];
}
export function toRoom(o, [lx, lz]) {
  const c = Math.cos(o.yaw), s = Math.sin(o.yaw);
  return [o.x + lx * c + lz * s, o.z - lx * s + lz * c];
}

// Does the floor point p fall on the object's footprint (plus slop metres)?
export function hitsObject(o, p, slop = 0.05) {
  const [lx, lz] = toLocal(o, p);
  if (o.type === "slab") return pointInPoly([lx, lz], scaledPts(o)) || polyDist([lx, lz], scaledPts(o), true) < slop;
  if (o.type === "walls") return polyDist([lx, lz], scaledPts(o), false) < o.d / 2 + slop;
  return Math.abs(lx) <= o.w / 2 + slop && Math.abs(lz) <= Math.max(o.d, 0.1) / 2 + slop;
}
// the object under p: the last one drawn wins (it is on top), -1 for none
export function objectAt(objs, p, slop = 0.05) {
  for (let i = objs.length - 1; i >= 0; i--) if (hitsObject(objs[i], p, slop)) return i;
  return -1;
}

export function pointInPoly([x, z], pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, zi] = pts[i], [xj, zj] = pts[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}
export function polyDist(p, pts, closed) {
  let best = Infinity;
  const n = closed ? pts.length : pts.length - 1;
  for (let i = 0; i < n; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length], ab = [b[0] - a[0], b[1] - a[1]], L = ab[0] ** 2 + ab[1] ** 2;
    const t = L ? Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L)) : 0;
    best = Math.min(best, Math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1]));
  }
  return best;
}
export const polyArea = (pts) => Math.abs(pts.reduce((s, [x, z], i) => { const [x2, z2] = pts[(i + 1) % pts.length]; return s + x * z2 - x2 * z; }, 0)) / 2;
export const polyLength = (pts, closed = false) => pts.reduce((s, p, i) => (i + 1 < pts.length || closed ? s + Math.hypot(pts[(i + 1) % pts.length][0] - p[0], pts[(i + 1) % pts.length][1] - p[1]) : s), 0);

// Snapping while drawing: the next corner after prev lands on a whole step (5 cm) from it, and its
// direction on a whole angle step (15°) from the line before (or from the room axes for the 1st side),
// so walls come out square and at round lengths. Closing: within closeM of the first corner = closed.
export function snapPoint(p, prev = null, before = null, { step = 0.05, angleDeg = 15 } = {}) {
  if (!prev) return [Math.round(p[0] / step) * step, Math.round(p[1] / step) * step];
  const dx = p[0] - prev[0], dz = p[1] - prev[1], len = Math.round(Math.hypot(dx, dz) / step) * step;
  const ref = before ? Math.atan2(prev[1] - before[1], prev[0] - before[0]) : 0, a = (angleDeg * Math.PI) / 180;
  const ang = ref + Math.round((Math.atan2(dz, dx) - ref) / a) * a;
  return [prev[0] + len * Math.cos(ang), prev[1] + len * Math.sin(ang)];
}
export const closesOutline = (pts, p, closeM = 0.2) => pts.length >= 3 && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) < closeM;

// Height from a vertical drag: dy screen pixels up (negative = down) at pxPerM, rounded to 5 cm.
export const dragHeight = (h0, dyPx, pxPerM = 200, step = 0.05) => clampLen(Math.round((h0 + dyPx / pxPerM) / step) * step, step);

// Resize / move / turn: new objects (the old ones stay as they were, for undo).
export const resized = (o, { w, h, d, y } = {}) => ({ ...o, ...(w !== undefined && { w: clampLen(w) }), ...(h !== undefined && { h: clampLen(h) }),
  ...(d !== undefined && { d: clampLen(d) }), ...(y !== undefined && { y: Math.max(0, Math.min(10, y)) }) });
export const moved = (o, x, z) => ({ ...o, x, z });
export const turned = (o, da) => ({ ...o, yaw: o.yaw + da });

// Labels: "2.40 m", and per object what a customer reads off it.
export const fmtM = (v) => `${v.toFixed(2)} m`;
export function sizeLabel(o) {
  if (o.type === "walls") return `${fmtM(polyLength(scaledPts(o)))} of wall, ${fmtM(o.h)} high`;
  if (o.type === "slab") return `${fmtM(o.w)} × ${fmtM(o.d)}, ${polyArea(scaledPts(o)).toFixed(1)} m², ${fmtM(o.h)} high`;
  if (o.type === "wall") return `${fmtM(o.w)} long, ${fmtM(o.h)} high`;
  if (o.type === "door" || o.type === "window") return `${fmtM(o.w)} × ${fmtM(o.h)}`;
  return `${fmtM(o.w)} × ${fmtM(o.d)}, ${fmtM(o.h)} high`;
}
export const objectName = (o) => (BLOCKS[o.type] || DRAWN[o.type] || { name: o.type }).name;

// Undo: a stack of design snapshots (JSON strings), at most `max`.
export class History {
  constructor(max = 60) { this.max = max; this.stack = []; }
  push(state) { const s = JSON.stringify(state); if (this.stack[this.stack.length - 1] !== s) this.stack.push(s); if (this.stack.length > this.max) this.stack.shift(); }
  // the state before the last push, or null when there is nothing to undo
  undo() { if (this.stack.length < 2) return null; this.stack.pop(); return JSON.parse(this.stack[this.stack.length - 1]); }
  get canUndo() { return this.stack.length > 1; }
}

// ---------- the design: options A/B/C on one tag map ----------
export const OPTIONS = ["A", "B", "C"];
export const DESIGN_KIND = "arvr-site-design", DESIGN_VERSION = 1;
export const emptyDesign = () => ({ options: { A: [], B: [], C: [] }, active: "A", names: { A: "Option A", B: "Option B", C: "Option C" } });
// copy option `from` over option `to` (fresh ids, so the copies are separate objects)
export function copyOption(design, from, to) {
  return { ...design, options: { ...design.options, [to]: design.options[from].map((o) => ({ ...o, id: newId(), pts: o.pts && o.pts.map((p) => [...p]) })) } };
}

const OBJ_KEYS = ["id", "type", "x", "z", "y", "yaw", "w", "h", "d", "mat", "pts", "base"];
const round = (v) => (typeof v === "number" ? Math.round(v * 1e4) / 1e4 : Array.isArray(v) ? v.map(round) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, round(x)])) : v);
export function designRecord(design, map, extra = {}) {
  const options = {};
  for (const k of OPTIONS) options[k] = (design.options[k] || []).map((o) => round(Object.fromEntries(OBJ_KEYS.filter((key) => o[key] !== undefined).map((key) => [key, o[key]]))));
  return { kind: DESIGN_KIND, version: DESIGN_VERSION, units: "m", frame: "tag map room frame: y up, floor at y = 0",
           saved: new Date().toISOString(), active: design.active, names: design.names, options, map, ...extra };
}
export function parseDesign(src) {
  const r = typeof src === "string" ? JSON.parse(src) : src;
  if (!r || r.kind !== DESIGN_KIND) throw new Error("not a site design file");
  if (r.version !== DESIGN_VERSION) throw new Error(`design version ${r.version}, this page reads ${DESIGN_VERSION}`);
  const options = {};
  for (const k of OPTIONS) {
    options[k] = (r.options && r.options[k] || []).map((o, i) => {
      if (!BLOCKS[o.type] && !DRAWN[o.type]) throw new Error(`option ${k}, object ${i + 1}: unknown type ${o.type}`);
      for (const key of ["x", "z", "w", "h", "d"]) if (!Number.isFinite(o[key])) throw new Error(`option ${k}, object ${i + 1}: ${key} is not a number`);
      return { y: 0, yaw: 0, mat: (BLOCKS[o.type] || DRAWN[o.type]).mat, ...o, id: o.id || newId() };
    });
  }
  return { design: { options, active: OPTIONS.includes(r.active) ? r.active : "A", names: { ...emptyDesign().names, ...(r.names || {}) } }, map: r.map || null, record: r };
}
export const designFileName = (d = new Date()) => `site-design-${d.toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`;

// ---------- lining an imported model up with the site ----------
// Model points m1, m2 and the real points r1, r2 they belong on (each [x, y, z]; m in model metres, r in
// room metres). The model is turned about y so m1 -> m2 points along r1 -> r2 and shifted so m1 sits on
// r1; scale stays 1 (a designer's model is already to size), but the length ratio is returned as a check.
export function alignTwo(m1, m2, r1, r2) {
  const dm = [m2[0] - m1[0], m2[2] - m1[2]], dr = [r2[0] - r1[0], r2[2] - r1[2]];
  const lm = Math.hypot(...dm), lr = Math.hypot(...dr);
  if (lm < 0.01 || lr < 0.01) throw new Error("pick two points at least 1 cm apart for the direction");
  // three.js rotation.y by yaw takes (x, z) to (x c + z s, -x s + z c): angle of a vector (x, z) is atan2(-z, x)
  const yaw = Math.atan2(-dr[1], dr[0]) - Math.atan2(-dm[1], dm[0]);
  const c = Math.cos(yaw), s = Math.sin(yaw);
  // height stays as it is: the model keeps standing on the floor, only its plan place and turn change
  const t = [r1[0] - (m1[0] * c + m1[2] * s), 0, r1[2] - (-m1[0] * s + m1[2] * c)];
  return { yaw, t, ratio: lr / lm, lengthModel: lm, lengthReal: lr };
}
// a model point after alignTwo's {yaw, t}
export function applyAlign({ yaw, t }, p) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [p[0] * c + p[2] * s + t[0], p[1] + t[1], -p[0] * s + p[2] * c + t[2]];
}
// The model corner nearest p among its bounding box's 8 corners (min/max [x, y, z]): taps snap to corners.
export function nearestCorner(min, max, p) {
  // in plan: a tap anywhere up a corner post (or on the roof above it) picks that corner, at floor height
  let best = null, bd = Infinity;
  for (const x of [min[0], max[0]]) for (const z of [min[2], max[2]]) {
    const dd = Math.hypot(x - p[0], z - p[2]);
    if (dd < bd) { bd = dd; best = [x, min[1], z]; }
  }
  return { corner: best, dist: bd };
}
