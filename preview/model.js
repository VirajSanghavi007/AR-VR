// A designer's CAD model (.glb) placed in the room for tagar.html: units, seating it on the surface,
// the 1 m reference bar, and the saved pose file. No three.js or DOM here, so node can test it.
//
// Room frame = the locked tag rectangle (group in tagar.html), metres: origin at the rectangle's
// centre, x along its long side (u), y out of the surface (n), z across it (v; on a wall, down).
// The model is shown at its real size times the scale slider, never at the panel array's scale.

// Metres per model unit.
export const UNIT_M = { m: 1, cm: 0.01, mm: 0.001, in: 0.0254, ft: 0.3048 };

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

// ---------- a room model aligned to the real room (seen from inside) ----------
// The tags are taped on one wall of the real room. The room model's matching wall is put on the tag
// plane, its inside facing out of the wall (toward the viewer), its floor at the measured height
// below the tags and its left corner at the measured distance to their left (facing the wall).

// Which face of the model's box goes on the tag wall: the turn about the model's up axis that brings
// that face to the back (-z, which the wall mount lays on the wall). Seen from inside, facing it.
export const ROOM_WALLS = ["back", "right", "front", "left"];
const WALL_FACE = { back: "-Z", right: "+X", front: "+Z", left: "-X" };
export const wallFace = (w) => WALL_FACE[w];
export function wallYaw(w) {
  const i = ROOM_WALLS.indexOf(w);
  if (i < 0) throw new Error(`unknown wall "${w}"`);
  return (i * Math.PI) / 2;
}
export const nextWall = (w) => ROOM_WALLS[(ROOM_WALLS.indexOf(w) + 1) % 4];

// A typed length in metres ("1.2", "1,2", " 1.2 m"); NaN if there is none.
export function parseLen(v) {
  const s = String(v ?? "").trim().replace(",", ".").replace(/\s*m$/i, "");
  return s === "" ? NaN : Number(s);
}

// Where the seated room (centred on the tags, wall width W, wall height H, metres) must move, in the
// tag frame [x right along the wall, z down], so the tag centre is `along` from the left corner and
// `height` above the floor. nudge = [right, up] in metres for fine alignment by eye.
export function roomOffset(W, H, along, height, nudge = [0, 0]) {
  return [W / 2 - along + nudge[0], height - H / 2 - nudge[1]];
}

// Problems with the measured tag position on a W x H wall, as sentences (empty = fine).
export function roomCheck(W, H, along, height) {
  const out = [];
  if (!Number.isFinite(along)) out.push("type the distance from the left corner to the tags' centre");
  else if (along < 0 || along > W) out.push(`the tags' centre is ${along.toFixed(2)} m from the left corner but this wall is ${W.toFixed(2)} m wide: wrong wall?`);
  if (!Number.isFinite(height)) out.push("type the height of the tags' centre above the floor");
  else if (height < 0 || height > H) out.push(`the tags' centre is ${height.toFixed(2)} m up but the room is ${H.toFixed(2)} m high`);
  return out;
}

// ---------- which file is the model (SketchUp exports, glTF) ----------
// SketchUp saves .skp, which no browser can read; its exports can be: File > Export > 3D Model >
// COLLADA (.dae, any SketchUp) or OBJ (.obj + .mtl colours, SketchUp Pro). glTF (.glb/.gltf) from Blender etc.
const KINDS = { glb: "gltf", gltf: "gltf", dae: "dae", obj: "obj" };
const PRIORITY = ["gltf", "dae", "obj"];
export const extOf = (name) => { const m = /\.([^./\\]+)$/.exec(String(name || "")); return m ? m[1].toLowerCase() : ""; };

// The loader a file needs: "gltf", "dae", "obj", or null for a side file (.mtl, textures).
export function modelKind(name) {
  const e = extOf(name);
  if (e === "skp") throw new Error("a .skp file cannot be opened in a browser: in SketchUp use File > Export > 3D Model and pick COLLADA (.dae) or OBJ");
  return KINDS[e] || null;
}

// From the files picked together, the model itself (glTF first, then .dae, then .obj) and the rest
// (its .mtl and textures). Throws a sentence if none of them is a model.
export function pickModelFiles(names) {
  const kinds = names.map(modelKind);
  for (const k of PRIORITY) {
    const i = kinds.indexOf(k);
    if (i >= 0) return { main: i, kind: k, extras: names.map((_, j) => j).filter((j) => j !== i) };
  }
  throw new Error("no model among these files: pick a .glb, .gltf, .dae or .obj (with its .mtl and textures)");
}

// File name a loader asks for, without folders or ?query, lower case, so "Textures\Roof.JPG" finds "roof.jpg".
export const baseName = (url) => {
  const s = String(url).split(/[?#]/)[0].split(/[\\/]/).pop();
  try { return decodeURIComponent(s).toLowerCase(); } catch (e) { return s.toLowerCase(); }
};

// Units an OBJ says it was drawn in. SketchUp writes "# File units = inches" (or meters, centimeters,
// millimeters, feet) in the header; OBJ itself has no units. null when the file does not say.
const OBJ_UNITS = { inches: "in", inch: "in", meters: "m", meter: "m", metres: "m", centimeters: "cm", centimetres: "cm",
                    millimeters: "mm", millimetres: "mm", feet: "ft", foot: "ft" };
export function objFileUnits(text) {
  const m = /^#\s*File units\s*=\s*([A-Za-z]+)/im.exec(String(text).slice(0, 2000));
  return m ? OBJ_UNITS[m[1].toLowerCase()] || null : null;
}

// Units a freshly loaded file is in: COLLADA carries its own <unit> (three.js already converts it to
// metres), an OBJ may say in its header, glTF is metres by spec but often exported wrong, so guess.
export function fileUnits(kind, objUnits = null) {
  if (kind === "dae") return "m";
  if (kind === "obj") return objUnits;
  return null;
}

// ---------- sample files, so the page works with nothing picked (while testing) ----------
// Loaded on page open unless the address has ?samples=0 (or off / false / no). A file the user picks
// replaces them. along / height: where the tags sit on the room's back wall, filled only if empty.
export const SAMPLES = {
  model: { url: "models/sample_sketchup_pergola.dae?v=1", name: "sample_sketchup_pergola.dae", units: "auto" },
  room: { url: "models/office_scan_room.dae?v=1", name: "office_scan_room.dae", units: "auto", wall: "back",
          along: "1.20", height: "1.10" },
};
export function samplesOn(search) {
  const v = new URLSearchParams(search || "").get("samples");
  return v === null || !["0", "off", "false", "no"].includes(v.trim().toLowerCase());
}
