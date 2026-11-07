// A designer's CAD model (.glb) placed in the room for tagar.html: units, seating it on the surface,
// the 1 m reference bar, and the saved pose file. No three.js or DOM here, so node can test it.
//
// Room frame = the locked tag rectangle (group in tagar.html), metres: origin at the rectangle's
// centre, x along its long side (u), y out of the surface (n), z across it (v; on a wall, down).
// The model is shown at its real size times the scale slider, never at the panel array's scale.

// Metres per model unit.
export const UNIT_M = { m: 1, cm: 0.01, mm: 0.001, in: 0.0254 };

// Unit guess from the model's largest raw dimension: real things in a room are well under 100 m,
// so a model over 100 units across was drawn in millimetres (the brief's "wrong by 1000x" case).
export function guessUnits(maxRaw) {
  if (!(maxRaw > 0)) throw new Error("the model has no size: is the file empty?");
  return maxRaw > 100 ? "mm" : "m";
}

// Raw size [x, y, z] in model units -> metres.
export function realSize(sizeRaw, units) {
  const k = UNIT_M[units];
  if (!k) throw new Error(`unknown units "${units}"`);
  return sizeRaw.map((v) => v * k);
}

// "1.20 × 0.80 × 0.45 m" (width × height × depth).
export const fmtSize = (s) => `${s.map((v) => v.toFixed(2)).join(" × ")} m`;

// Translation that seats a box {min, max} (already mounted: y out of the surface) on the surface:
// centred over the origin in x and z, its lowest point (on a wall: its back) on y = 0.
export const seat = (min, max) => [-(min[0] + max[0]) / 2, -min[1], -(min[2] + max[2]) / 2];

// The mount turn about x: on a wall the model's up (+y) goes up the wall (-z, since +z is down)
// and its front (+z) faces out of the wall; on a table or floor it stands as drawn.
export const mountTurnX = (wall) => (wall ? -Math.PI / 2 : 0);

// Scale slider in percent, whole numbers 10..500 (100 = the model's real size).
export function clampPct(v) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(500, Math.max(10, n)) : 100;
}

// Turn slider in whole degrees 0..359.
export const clampDeg = (v) => { const n = Math.round(Number(v)); return Number.isFinite(n) ? ((n % 360) + 360) % 360 : 0; };

// Stripes of the 1 m reference bar: `step` metres each, alternating dark and light, from x = 0.
export function rulerTicks(length = 1, step = 0.1) {
  const n = Math.round(length / step);
  if (n < 1 || Math.abs(n * step - length) > 1e-9) throw new Error("length must be a whole number of steps");
  return Array.from({ length: n }, (_, i) => ({ from: +(i * step).toFixed(6), to: +((i + 1) * step).toFixed(6), dark: i % 2 === 0 }));
}

// Where the bar starts: 10 cm past the model's +x side, at its centre across, so it lies beside it.
export const rulerStart = (boxMin, boxMax, gap = 0.1) => [boxMax[0] + gap, (boxMin[2] + boxMax[2]) / 2];

// Pose file name: the model name, its real size as shown (mm, width x height x depth) and units,
// e.g. "cabinet-1200x800x450mm-pose.json".
export function poseFileName(name, sizeM) {
  const base = String(name || "model").replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "model";
  return `${base}-${sizeM.map((v) => Math.round(v * 1000)).join("x")}mm-pose.json`;
}

export const POSE_VERSION = 1;

// The saved pose: everything needed to put the model back in the same place on the same tags.
// p = { name, units, sizeRaw, pct, deg, x, z, wall, tags: {ids, width, height} }
export function poseRecord(p, when = new Date()) {
  const size = realSize(p.sizeRaw, p.units), k = clampPct(p.pct) / 100;
  return {
    version: POSE_VERSION,
    model: p.name,
    units: p.units,
    size_as_drawn_m: size.map((v) => +v.toFixed(4)),
    scale_pct: clampPct(p.pct),
    size_shown_m: size.map((v) => +(v * k).toFixed(4)),
    position_m: [+p.x.toFixed(4), 0, +p.z.toFixed(4)],
    turn_deg: clampDeg(p.deg),
    mount: p.wall ? "wall" : "floor",
    frame: "tag rectangle: origin at its centre, x along the long side, y out of the surface, z across (on a wall, down)",
    tags: { ids: p.tags.ids, width_m: +p.tags.width.toFixed(4), height_m: +p.tags.height.toFixed(4) },
    saved: when.toISOString(),
  };
}

// Read a pose file back (object or JSON text). Returns { name, units, pct, deg, x, z, wall, ids }
// or throws a message a person can act on.
export function parsePose(src) {
  let o = src;
  if (typeof src === "string") { try { o = JSON.parse(src); } catch (e) { throw new Error("not a pose file: it is not JSON"); } }
  if (!o || o.version !== POSE_VERSION) throw new Error(`not a pose file from this page (version ${o && o.version})`);
  if (!UNIT_M[o.units]) throw new Error(`unknown units "${o.units}"`);
  const pos = o.position_m;
  if (!Array.isArray(pos) || pos.length !== 3 || !pos.every(Number.isFinite)) throw new Error("the pose file has no position");
  return { name: o.model, units: o.units, pct: clampPct(o.scale_pct), deg: clampDeg(o.turn_deg), x: pos[0], z: pos[2],
           wall: o.mount === "wall", ids: o.tags ? o.tags.ids : "" };
}

// A loaded pose belongs to these tags only if the ids match; warn otherwise (the frame differs).
export const sameTags = (poseIds, rectIds) => String(poseIds) === String(rectIds);
