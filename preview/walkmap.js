// Walk-round lock for panels.html: up to 8 AprilTags round the panel area. Before the lock, the 4 tags that
// make the biggest well-squared rectangle set the area; after it, every tag seen is kept in a tag map
// (tagmap.js) so whichever tags are in view (one is enough) keep the panels where they belong as you walk
// a full circle. Pure math, no three.js. Matrices are WebXR style: 16 numbers, column-major.
import { fitRectangle, invert4, transform, cross, unit, norm } from "./tagrect.js?v=9";
import * as TM from "./tagmap.js?v=3";

export const MAX_TAGS = 8;

// All k-element subsets of 0..n-1 (k <= n), in lexicographic order.
export function combos(n, k) {
  const out = [], pick = [];
  const go = (start) => {
    if (pick.length === k) { out.push(pick.slice()); return; }
    for (let i = start; i <= n - (k - pick.length); i++) { pick.push(i); go(i + 1); pick.pop(); }
  };
  if (k >= 0 && k <= n) go(0);
  return out;
}

// From tags [{id, center: [x, y, z]}] on a surface with normal n, the 4 at the corners of the biggest
// rectangle whose corners are all within maxErrDeg of square. Returns {ids (sorted), rect} or null.
// With exactly 4 tags that is just those 4 (when square enough). Two rectangles within `tie` (2%) of the
// same area: the squarer one wins (extra tags round the area can by chance make a second, tilted one).
export function bestRectangle(tags, n, maxErrDeg = 12, tie = 0.02) {
  let best = null;
  for (const c of combos(tags.length, 4)) {
    const pick = c.map((i) => tags[i]);
    let r;
    try { r = fitRectangle(pick.map((t) => t.center), n); } catch (e) { continue; }
    if (r.angleErrDeg > maxErrDeg) continue;
    const area = r.width * r.height, near = best && Math.abs(area - best.area) <= tie * Math.max(area, best.area);
    if (!best || (near ? r.angleErrDeg < best.rect.angleErrDeg : area > best.area))
      best = { area, ids: pick.map((t) => t.id).sort((a, b) => a - b), rect: r };
  }
  return best && { ids: best.ids, rect: best.rect };
}

// Levelled room frame (worldFromRoom) for a locked rectangle {center, u, n, v}: y straight up, x the
// rectangle's long side laid level (on a wall u is already level), origin the rectangle's centre.
export function lockFrame(r) {
  const y = [0, 1, 0];
  let x = [r.u[0], 0, r.u[2]];
  if (norm(x) < 1e-6) x = [r.v[0], 0, r.v[2]];
  x = unit(x);
  return TM.basis(x, y, cross(x, y), r.center);
}

// A new walk map from the lock: rect (with u, n, v, center), corners {id: [4 world corners]} of every tag
// in view, rectIds = the 4 corner tags (counted as 30 sightings each, the rest as `extra`).
// Returns {map, wfr, roomFromLock}: lockPose = wfr * roomFromLock at any later time.
export function startWalk(rect, corners, rectIds, size, extra = 3) {
  const wfr = lockFrame(rect);
  const lockPose = TM.basis(rect.u, rect.n, rect.v, rect.center);
  const map = { version: TM.MAP_VERSION, size, origin: rectIds.slice().sort((a, b) => a - b), floorY: null, tags: {} };
  for (const [id, c] of Object.entries(corners))
    TM.registerTag(map, id, c, wfr, rectIds.includes(Number(id)) ? 30 : extra);
  return { map, wfr, roomFromLock: TM.mul4(invert4(wfr), lockPose) };
}

// How much a fix's turn is trusted (tagmap fusePose): site.html walks rooms, where tags metres apart give the
// turn; here the tags sit round a table-size area, seen from about a metre, so a span of 0.6 m already counts
// as wide and even one tag's own turn is given some weight.
export const FUSE = { spanRef: 0.6, minTurn: 0.15 };

// One tracking tick. st = {map, wfr, roomFromLock, pending}, seen = {id: [4 world corners]}. Fixes wfr
// from the mapped tags in view (tagmap locate + fusePose), then adds each unmapped tag once it was seen
// `need` times (its mean room corners). A sighting only counts when at least 2 mapped tags at least
// minSpan metres apart give this tick's fit: it is placed by that fit, not by the eased pose (which lags a
// moment behind), and one tag's own turn is too rough to place a tag half a metre away.
// Returns {fix (or null), added: [ids]}.
export function walkStep(st, seen, { need = 3, tol = 0.04, fuse = FUSE, minSpan = 0.25 } = {}) {
  st.pending = st.pending || {};
  const fix = Object.keys(seen).length ? TM.locate(st.map, seen, tol) : null;
  const added = [];
  if (!fix) return { fix: null, added };
  st.wfr = TM.fusePose(st.wfr, fix, fuse);
  if (fix.used.length < 2 || fix.span < minSpan) return { fix, added };
  const inv = invert4(fix.worldFromRoom);
  for (const [id, c] of Object.entries(seen)) {
    if (st.map.tags[id]) continue;              // mapped tags stay put: refitting them from a drifted pose would drift the map
    (st.pending[id] = st.pending[id] || []).push(c.map((p) => transform(inv, p)));
    if (st.pending[id].length >= need) {
      const ps = st.pending[id], k = ps.length;
      const mean = [0, 1, 2, 3].map((i) => [0, 1, 2].map((j) => ps.reduce((s, cs) => s + cs[i][j], 0) / k));
      st.map.tags[id] = { surface: TM.tagPose(mean).surface, corners: mean, seen: k };
      delete st.pending[id];
      added.push(Number(id));
    }
  }
  return { fix, added };
}

// Where the panels' lock frame is now: worldFromRoom * roomFromLock.
export const lockPoseNow = (st) => TM.mul4(st.wfr, st.roomFromLock);

// HUD line for the walk: which tags hold the lock now and how many are mapped.
export function walkLine(st, fix) {
  const mapped = Object.keys(st.map.tags).length;
  if (!fix) return `No mapped tag in view: holding by phone tracking · ${mapped} tag${mapped === 1 ? "" : "s"} mapped`;
  const ids = fix.used.slice().sort((a, b) => a - b);
  return `Held by tag${ids.length === 1 ? "" : "s"} ${ids.join(", ")} · ${mapped} tag${mapped === 1 ? "" : "s"} mapped`;
}
