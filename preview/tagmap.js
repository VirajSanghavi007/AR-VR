// Tag map for 360° anchoring (site.html): every AprilTag taped round the site gets a fixed pose in one
// room frame, so any of them in view (one is enough) puts the design back where it belongs.
// Pure math, no three.js. Matrices are WebXR style: 16 numbers, column-major. Points are [x, y, z] metres.
//
// Room frame: y is up (ARCore's world is levelled by gravity, so only a turn about y and a shift are ever
// fitted), the origin is the centre of the tags the setup started on, and on a wall z points out of that
// wall into the room with x to the right as you face it. Started on the floor: z = the tags' printed "down".
import { invert4, transform, cross, dot, unit, norm } from "./tagrect.js?v=7";

export const MAP_VERSION = 1;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const mean = (pts) => mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);
const flat = (v) => [v[0], 0, v[2]];

// A matrix from axes x, y, z and origin o.
export const basis = (x, y, z, o) => [...x, 0, ...y, 0, ...z, 0, ...o, 1];

// Rotation about y by a (three.js rotation.y) with translation t.
export function yawMatrix(a, t = [0, 0, 0]) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, t[0], t[1], t[2], 1];
}
export function mul4(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
// The turn about y a levelled pose matrix holds (radians, three.js rotation.y).
export const yawOf = (m) => Math.atan2(m[8], m[0]);

// One tag from its 4 world corners in printed order (top-left, top-right, bottom-right, bottom-left):
// centre, outward normal (towards the camera that read it), printed right and down directions, and its
// surface: a wall, the floor, or a ceiling, by how its normal leans.
export function tagPose(corners) {
  const [tl, tr, br, bl] = corners, right = unit(add(sub(tr, tl), sub(br, bl))), down = unit(add(sub(bl, tl), sub(br, tr)));
  const normal = unit(cross(down, right));
  const surface = normal[1] > Math.SQRT1_2 ? "floor" : normal[1] < -Math.SQRT1_2 ? "ceiling" : "wall";
  return { center: mean(corners), normal, right, down, surface };
}

// worldFromRoom for a new map, from the tags the setup locked on (each [4 world corners]).
export function roomFrame(tagCorners) {
  if (!tagCorners.length) throw new Error("no tags to start the map on");
  const poses = tagCorners.map(tagPose), o = mean(poses.map((p) => p.center)), y = [0, 1, 0];
  const wall = poses.filter((p) => p.surface === "wall");
  let z = wall.length ? flat(mean(wall.map((p) => p.normal))) : flat(mean(poses.map((p) => p.down)));
  if (norm(z) < 1e-6) z = [0, 0, 1];
  z = unit(z);
  return basis(cross(y, z), y, z, o);
}

// An empty map started on the tags in tagCorners {id: [4 world corners]}; returns {map, worldFromRoom}.
export function startMap(tagCorners, size) {
  const ids = Object.keys(tagCorners);
  const worldFromRoom = roomFrame(ids.map((id) => tagCorners[id]));
  const map = { version: MAP_VERSION, size, origin: ids.map(Number).sort((a, b) => a - b), floorY: null, tags: {} };
  for (const id of ids) registerTag(map, id, tagCorners[id], worldFromRoom, 30);
  return { map, worldFromRoom };
}

// Add tag id seen at world corners (or average it into what the map has: a running mean that stops
// growing at `cap` sightings, so a tag keeps settling but an old bad one can still be outvoted).
export function registerTag(map, id, corners, worldFromRoom, weight = 1, cap = 30) {
  const inv = invert4(worldFromRoom), room = corners.map((p) => transform(inv, p)), key = String(id);
  const t = map.tags[key];
  if (!t) { map.tags[key] = { surface: tagPose(room).surface, corners: room, seen: weight }; return map.tags[key]; }
  const k = weight / (Math.min(t.seen, cap) + weight);
  t.corners = t.corners.map((p, i) => add(p, mul(sub(room[i], p), k)));
  t.seen += weight;
  t.surface = tagPose(t.corners).surface;
  return t;
}

// Best levelled pose (turn about y + shift) taking room points P onto world points Q: worldFromRoom and
// the RMS of what is left over (metres). The turn is closed-form: the angle of sum(b x a) against sum(b . a)
// over the centred points, in the floor plane.
export function fitYaw(P, Q) {
  if (P.length !== Q.length || P.length < 2) throw new Error("need 2 or more matching points");
  const p = mean(P), q = mean(Q);
  let C = 0, S = 0;
  P.forEach((pt, i) => {
    const a = sub(pt, p), b = sub(Q[i], q);
    C += b[0] * a[0] + b[2] * a[2];
    S += b[0] * a[2] - b[2] * a[0];
  });
  const ang = Math.atan2(S, C), R = yawMatrix(ang), t = sub(q, transform(R, p));
  const M = yawMatrix(ang, t);
  const rms = Math.sqrt(P.reduce((s, pt, i) => s + norm(sub(transform(M, pt), Q[i])) ** 2, 0) / P.length);
  return { worldFromRoom: M, rms, yaw: ang };
}

// Where the room is now, from the mapped tags in view: seen = {id: [4 world corners]} (unmapped ids are
// skipped). A tag that disagrees with the others by more than tol (moved, or misread) is dropped and the
// rest refitted. Returns {worldFromRoom, rms, used: [ids], dropped: [ids]} or null when no mapped tag is seen.
export function locate(map, seen, tol = 0.04) {
  let ids = Object.keys(seen).filter((id) => map.tags[id]);
  const dropped = [];
  while (ids.length) {
    const P = ids.flatMap((id) => map.tags[id].corners), Q = ids.flatMap((id) => seen[id]);
    const f = fitYaw(P, Q);
    if (f.rms <= tol || ids.length === 1) {
      // span: how far apart the used tags are (level metres). The turn a fit gives is only as good as this:
      // from one 16 cm tag it is off by a degree or more, from tags on two walls a metre or more apart it is good.
      const cs = ids.map((id) => mean(map.tags[id].corners));
      const span = Math.max(0, ...cs.flatMap((a) => cs.map((b) => Math.hypot(a[0] - b[0], a[2] - b[2]))));
      return { ...f, used: ids.map(Number), dropped, span, pc: mean(P), qc: mean(Q) };
    }
    const err = ids.map((id) => { const c = map.tags[id].corners; return c.reduce((s, pt, i) => s + norm(sub(transform(f.worldFromRoom, pt), seen[id][i])), 0); });
    const worst = err.indexOf(Math.max(...err));
    dropped.push(Number(ids[worst]));
    ids = ids.filter((_, i) => i !== worst);
  }
  return null;
}

// Ease pose a towards b by k (0..1): turn and shift each, so a noisy single-tag fix does not shake the design.
// A jump bigger than snapM (metres) is taken at once (the tags were re-taped, or tracking relocalised).
export function blendPose(a, b, k = 0.35, snapM = 0.5) {
  if (!a) return b.slice();
  const ta = [a[12], a[13], a[14]], tb = [b[12], b[13], b[14]];
  if (norm(sub(ta, tb)) > snapM) return b.slice();
  let d = yawOf(b) - yawOf(a);
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return yawMatrix(yawOf(a) + k * d, add(ta, mul(sub(tb, ta), k)));
}

// Fold a tag fix (from locate) into the pose in use, prev. The tags in view fix where the room is (their
// centre lands where they are seen, eased by k); the turn moves towards the fix's only by
// span² / (span² + spanRef²) of the way (at least minTurn), since ARCore's own heading drifts far less than
// a 1-2 tag fit's turn is off. A jump bigger than snapM is taken whole (tags re-taped, tracking relocalised).
export function fusePose(prev, fix, { k = 0.35, spanRef = 1.5, minTurn = 0.03, snapM = 0.5 } = {}) {
  if (!prev) return fix.worldFromRoom.slice();
  const shown = transform(prev, fix.pc);
  if (norm(sub(shown, fix.qc)) > snapM) return fix.worldFromRoom.slice();
  const s2 = (fix.span || 0) ** 2, wTurn = Math.max(minTurn, s2 / (s2 + spanRef * spanRef));
  let d = yawOf(fix.worldFromRoom) - yawOf(prev);
  d = Math.atan2(Math.sin(d), Math.cos(d));
  const yaw = yawOf(prev) + k * wTurn * d, R = yawMatrix(yaw);
  // shift so the tags' centre lands on where it is seen (eased by k from where prev shows it)
  const t0 = sub(shown, transform(R, fix.pc)), t1 = sub(fix.qc, transform(R, fix.pc));
  return yawMatrix(yaw, add(t0, mul(sub(t1, t0), k)));
}

// How far two room poses put the design apart: the largest shift of the sample points (room metres, by
// default the corners of a 6 x 6 m area 0-2.5 m up round the origin) and the turn between them (degrees).
export function poseDelta(a, b, pts = null) {
  pts = pts || [-3, 3].flatMap((x) => [0, 2.5].flatMap((y) => [-3, 3].map((z) => [x, y, z])));
  const mm = Math.max(...pts.map((p) => norm(sub(transform(a, p), transform(b, p))))) * 1000;
  let d = yawOf(b) - yawOf(a);
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return { mm, deg: Math.abs((d * 180) / Math.PI) };
}

// The map as a file, and back (checked). Corners rounded to 0.1 mm.
export function mapRecord(map) {
  const tags = {};
  for (const [id, t] of Object.entries(map.tags))
    tags[id] = { surface: t.surface, seen: Math.round(t.seen), corners: t.corners.map((p) => p.map((v) => Math.round(v * 1e4) / 1e4)) };
  return { version: MAP_VERSION, size: map.size, origin: map.origin, floorY: map.floorY, tags };
}
export function parseMap(src) {
  const m = typeof src === "string" ? JSON.parse(src) : src;
  if (!m || typeof m !== "object" || !m.tags) throw new Error("not a tag map (no tags)");
  if (m.version !== MAP_VERSION) throw new Error(`tag map version ${m.version}, this page reads ${MAP_VERSION}`);
  for (const [id, t] of Object.entries(m.tags))
    if (!/^\d+$/.test(id) || !Array.isArray(t.corners) || t.corners.length !== 4 || t.corners.some((p) => p.length !== 3 || p.some((v) => !Number.isFinite(v))))
      throw new Error(`tag ${id}: needs 4 corners of 3 numbers`);
  return { version: MAP_VERSION, size: +m.size || 0.16, origin: m.origin || [], floorY: Number.isFinite(m.floorY) ? m.floorY : null,
           tags: Object.fromEntries(Object.entries(m.tags).map(([id, t]) => [id, { surface: t.surface || tagPose(t.corners).surface, corners: t.corners, seen: t.seen || 1 }])) };
}

// One line per tag for the HUD/list: "3 wall 1.20 m right, 1.35 m up".
export function describeMap(map) {
  return Object.entries(map.tags).sort((a, b) => a[0] - b[0]).map(([id, t]) => {
    const c = mean(t.corners);
    return `${id}: ${t.surface} at ${c.map((v) => v.toFixed(2)).join(", ")} m (${Math.round(t.seen)} sightings)`;
  });
}

// A walk round a w x d room (centred on 0, walls at x = ±w/2, z = ±d/2) `inset` metres from the walls,
// looking square at the nearest wall: n stops [{eye, look}] once round. For checking 360° anchoring.
export function wallWalk(w, d, inset = 1.0, n = 48, eyeY = 1.4, lookY = 1.3) {
  const hx = w / 2 - inset, hz = d / 2 - inset, per = 4 * (hx + hz), out = [];
  for (let i = 0; i < n; i++) {
    let s = (i / n) * per, eye, look;
    // back wall (z = -d/2) first, walking right, then the right wall, the front wall, the left wall
    if (s < 2 * hx) { eye = [-hx + s, eyeY, -hz]; look = [eye[0], lookY, -d / 2]; }
    else if ((s -= 2 * hx) < 2 * hz) { eye = [hx, eyeY, -hz + s]; look = [w / 2, lookY, eye[2]]; }
    else if ((s -= 2 * hz) < 2 * hx) { eye = [hx - s, eyeY, hz]; look = [eye[0], lookY, d / 2]; }
    else { s -= 2 * hx; eye = [-hx, eyeY, hz - s]; look = [-w / 2, lookY, eye[2]]; }
    out.push({ eye, look });
  }
  return out;
}

// Path length of a walk (metres): tracking drift grows with it.
export const walkLength = (stops) => stops.reduce((s, st, i) => (i ? s + norm(sub(st.eye, stops[i - 1].eye)) : 0), 0);
