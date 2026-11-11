// AprilTag capture checks for tags.html. No dependencies (tested from Python through node).
// Markers are { id, corners: [[x, y] x4] } in original-image pixels. tag_scale.py needs every
// tag in >= 2 registered photos; more views and bigger tags give a steadier scale.

export const MIN_VIEWS = 2;      // tag_scale.py drops a tag seen in fewer photos
export const GOOD_VIEWS = 3;
export const MIN_EDGE_PX = 40;   // below this, corner error is a large share of the edge

// "0-3, 7" -> [0, 1, 2, 3, 7]
export function parseIds(text) {
  const ids = new Set();
  for (const part of String(text).split(/[\s,]+/).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) throw new Error(`not an id or range: "${part}"`);
    const a = Number(m[1]), b = m[2] === undefined ? a : Number(m[2]);
    if (b < a || b > 586) throw new Error(`bad range "${part}" (36h11 ids are 0-586)`);
    for (let i = a; i <= b; i++) ids.add(i);
  }
  return [...ids].sort((x, y) => x - y);
}

// js-aruco2 markers (corners {x, y} in the detected, possibly downscaled image) -> original pixels
export function toMarkers(arMarkers, scale = 1) {
  return arMarkers.map((m) => ({ id: m.id, corners: m.corners.map((c) => [c.x * scale, c.y * scale]) }));
}

export function edgeLengths(corners) {
  return corners.map((p, i) => {
    const q = corners[(i + 1) % 4];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
}

// js-aruco2 can find one sharp tag twice: the outline of its black square and a contour just inside
// it, same id, same centre. Keep the bigger of such a pair; the same id somewhere else stays (a real
// duplicate). Order of the rest is kept.
const centre = (c) => [c.reduce((s, p) => s + p[0], 0) / c.length, c.reduce((s, p) => s + p[1], 0) / c.length];
export function mergeNested(markers) {
  const keep = [];
  for (const m of [...markers].sort((a, b) => Math.min(...edgeLengths(b.corners)) - Math.min(...edgeLengths(a.corners)))) {
    const [x, y] = centre(m.corners);
    const inside = keep.some((k) => {
      const [kx, ky] = centre(k.corners);
      return k.id === m.id && Math.hypot(kx - x, ky - y) < 0.5 * Math.min(...edgeLengths(k.corners));
    });
    if (!inside) keep.push(m);
  }
  return markers.filter((m) => keep.includes(m));
}

// One frame or photo: per-tag size and problems
export function assessFrame(markers, expected, minEdgePx = MIN_EDGE_PX) {
  const want = new Set(expected);
  const count = {};
  for (const m of markers) count[m.id] = (count[m.id] || 0) + 1;
  return markers.map((m) => {
    const minEdge = Math.min(...edgeLengths(m.corners));
    const problems = [];
    if (!want.has(m.id)) problems.push("not in your id list");
    if (count[m.id] > 1) problems.push("same id twice in one photo");
    if (minEdge < minEdgePx) problems.push(`small (${Math.round(minEdge)} px edge): move closer`);
    return { id: m.id, corners: m.corners, minEdge, problems };
  });
}

// photos: [{ name, tags: assessFrame(...) output }] -> readiness for tag_scale.py
export function summarize(photos, expected) {
  const views = Object.fromEntries(expected.map((id) => [id, 0]));
  const unknown = new Set(), duplicates = [], noTags = [], small = [];
  for (const p of photos) {
    if (!p.tags.length) noTags.push(p.name);
    const seen = new Set();
    for (const t of p.tags) {
      if (!(t.id in views)) { unknown.add(t.id); continue; }
      if (seen.has(t.id)) { if (!duplicates.includes(p.name)) duplicates.push(p.name); continue; }
      seen.add(t.id);
      views[t.id]++;
      if (t.minEdge < MIN_EDGE_PX) small.push(`${p.name} (id ${t.id})`);
    }
  }
  const missing = expected.filter((id) => views[id] === 0);
  const tooFew = expected.filter((id) => views[id] > 0 && views[id] < MIN_VIEWS);
  const weak = expected.filter((id) => views[id] >= MIN_VIEWS && views[id] < GOOD_VIEWS);
  const usable = expected.filter((id) => views[id] >= MIN_VIEWS);

  const messages = [];
  if (missing.length) messages.push(`Never seen: id ${missing.join(", ")}.`);
  if (tooFew.length) messages.push(`Only 1 photo: id ${tooFew.join(", ")} (needs ${MIN_VIEWS}+, else tag_scale.py drops it).`);
  if (weak.length) messages.push(`Only ${MIN_VIEWS} photos: id ${weak.join(", ")} (works, ${GOOD_VIEWS}+ is steadier).`);
  if (duplicates.length) messages.push(`Same id twice in: ${duplicates.join(", ")}. Each id must be printed once.`);
  if (unknown.size) messages.push(`Unexpected ids: ${[...unknown].sort((a, b) => a - b).join(", ")}. Check the id list or remove stray tags.`);
  if (small.length) messages.push(`Small tags in ${small.length} view(s): ${small.slice(0, 5).join(", ")}${small.length > 5 ? ", ..." : ""}.`);
  if (noTags.length) messages.push(`${noTags.length} photo(s) with no tag (fine for COLMAP, no help for scale).`);

  // Ready = scale can be computed AND cross-checked: >= 2 usable tags, nothing that corrupts it
  const ready = usable.length >= 2 && !duplicates.length;
  return { views, missing, tooFew, weak, usable, unknown: [...unknown], duplicates, noTags, small, ready, messages };
}
