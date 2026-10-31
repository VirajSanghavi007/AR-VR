// Tabletop AR on 4 AprilTags: pure math for tagar.html (no three.js, so node can test it).
// Matrices are WebXR style: 16 numbers, column-major. Points are [x, y, z] in metres.

export const NICE_SCALES = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000];
export const FILL = 0.95;              // content covers 95 % of the tag rectangle, so the tags stay visible

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const unit = (a) => mul(a, 1 / norm(a));
const mean = (pts) => mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);

export function invert4(m) {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) throw new Error("matrix is not invertible");
  const d = 1 / det;
  return [
    (a11 * b11 - a12 * b10 + a13 * b09) * d, (a02 * b10 - a01 * b11 - a03 * b09) * d,
    (a31 * b05 - a32 * b04 + a33 * b03) * d, (a22 * b04 - a21 * b05 - a23 * b03) * d,
    (a12 * b08 - a10 * b11 - a13 * b07) * d, (a00 * b11 - a02 * b08 + a03 * b07) * d,
    (a32 * b02 - a30 * b05 - a33 * b01) * d, (a20 * b05 - a22 * b02 + a23 * b01) * d,
    (a10 * b10 - a11 * b08 + a13 * b06) * d, (a01 * b08 - a00 * b10 - a03 * b06) * d,
    (a30 * b04 - a31 * b02 + a33 * b00) * d, (a21 * b02 - a20 * b04 - a23 * b00) * d,
    (a11 * b07 - a10 * b09 - a12 * b06) * d, (a00 * b09 - a01 * b07 + a02 * b06) * d,
    (a31 * b01 - a30 * b03 - a32 * b00) * d, (a20 * b03 - a21 * b01 + a22 * b00) * d,
  ];
}

export function transform(m, p) {
  const [x, y, z] = p;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
          (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
          (m[2] * x + m[6] * y + m[10] * z + m[14]) / w];
}

// World ray through camera-image pixel (px, py), y measured down from the top of the image.
// proj = XRView.projectionMatrix, camToWorld = XRView.transform.matrix.
export function pixelRay(px, py, width, height, proj, camToWorld) {
  const inv = invert4(proj);
  const x = (2 * px) / width - 1, y = 1 - (2 * py) / height;
  const near = transform(camToWorld, transform(inv, [x, y, -1]));
  const far = transform(camToWorld, transform(inv, [x, y, 1]));
  return { origin: near, dir: unit(sub(far, near)) };
}

// Where a ray meets the plane (point p0, normal n); null if parallel or behind the camera.
export function rayPlane(ray, p0, n) {
  const den = dot(ray.dir, n);
  if (Math.abs(den) < 1e-6) return null;
  const t = dot(sub(p0, ray.origin), n) / den;
  return t > 0 ? add(ray.origin, mul(ray.dir, t)) : null;
}

// Rectangle through 4 points on a plane (normal n points up, away from the table).
// Returns centre, axes u (long side), n, v = u x n (so u, n, v is a right-handed x, y, z),
// side lengths and how far the 4 points are from a true rectangle.
export function fitRectangle(points, n) {
  if (points.length !== 4) throw new Error(`need 4 tag points, got ${points.length}`);
  n = unit(n);
  const c = mean(points);
  const flat = points.map((p) => sub(p, mul(n, dot(sub(p, c), n))));
  const e1 = unit(sub(flat[0], c)), e2 = cross(n, e1);
  const ang = (p) => Math.atan2(dot(sub(p, c), e2), dot(sub(p, c), e1));
  let P = flat.slice().sort((a, b) => ang(a) - ang(b));
  const side = (i) => norm(sub(P[(i + 1) % 4], P[i]));
  if (side(0) + side(2) < side(1) + side(3)) P = [P[1], P[2], P[3], P[0]];   // P0 -> P1 is a long side
  const width = (side(0) + side(2)) / 2, height = (side(1) + side(3)) / 2;
  let u = add(sub(P[1], P[0]), sub(P[2], P[3]));
  u = unit(sub(u, mul(n, dot(u, n))));
  const v = cross(u, n);
  let angleErrDeg = 0;
  for (let i = 0; i < 4; i++) {
    const a = unit(sub(P[(i + 1) % 4], P[i])), b = unit(sub(P[(i + 3) % 4], P[i]));
    angleErrDeg = Math.max(angleErrDeg, Math.abs(90 - (Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * 180) / Math.PI));
  }
  if (norm(cross(sub(P[2], P[0]), sub(P[3], P[1]))) / 2 < 1e-4)          // under 1 cm2
    throw new Error("tags are in a line, not a rectangle");
  const d1 = norm(sub(P[2], P[0])), d2 = norm(sub(P[3], P[1]));
  return { center: c, u, n, v, width, height, corners: P, angleErrDeg, diagErrPct: (200 * Math.abs(d1 - d2)) / (d1 + d2) };
}

// Mean edge of one tag from its 4 world corners (metres): a check on the surface plane.
export function tagEdge(corners) {
  let s = 0;
  for (let i = 0; i < 4; i++) s += norm(sub(corners[(i + 1) % 4], corners[i]));
  return s / 4;
}

// Scale that fits a design (extent w along its x, d along its z) into the rectangle after
// `turns` quarter turns; odd turns put the design's z along the rectangle's long side.
export function fitScale(w, d, rectW, rectH, turns = 0, fill = FILL) {
  const [ew, ed] = turns % 2 ? [d, w] : [w, d];
  return (fill * Math.min(rectW / ew, rectH / ed));
}

// Quarter turns (0 or 1) that make the design as big as possible in the rectangle.
export const bestTurns = (w, d, rectW, rectH) => (fitScale(w, d, rectW, rectH, 1) > fitScale(w, d, rectW, rectH, 0) ? 1 : 0);

// 1:N label for a model scale s (model metres per real metre).
export function scaleLabel(s) {
  const N = 1 / s;
  return "1:" + (N >= 100 ? Math.round(N) : Math.round(N * 10) / 10);
}

// Next round 1:N from the current scale: dir +1 = bigger model (smaller N), -1 = smaller model.
export function nextNiceScale(s, dir) {
  const N = 1 / s, eps = 1e-6;
  const n = dir > 0 ? [...NICE_SCALES].reverse().find((k) => k < N * (1 - eps))
                    : NICE_SCALES.find((k) => k > N * (1 + eps));
  return 1 / (n ?? (dir > 0 ? NICE_SCALES[0] : NICE_SCALES[NICE_SCALES.length - 1]));
}

// True when the last `need` rectangles agree within tol metres (centre and both sides):
// the lock waits for this so one bad frame cannot place the model.
export function isStable(history, need = 3, tol = 0.01) {
  if (history.length < need) return false;
  const last = history.slice(-need), c = mean(last.map((r) => r.center));
  return last.every((r) => norm(sub(r.center, c)) < tol &&
    Math.abs(r.width - last[0].width) < tol && Math.abs(r.height - last[0].height) < tol);
}

// Camera frame for the detector: RGBA from gl.readPixels (rows bottom-up when flipY), shrunk by a
// whole step to at most maxPixels (the default keeps most phone frames whole: at half size the
// thin white margin round a distant tag merges with its black square and it stops decoding), each output pixel the mean of its step x step block (plain
// skipping aliases thin tag borders away). Returns the factor to map detector pixels back.
export function downsample(rgba, width, height, flipY, maxPixels = 2.6e6) {
  const step = Math.max(1, Math.ceil(Math.sqrt((width * height) / maxPixels)));
  const w = Math.floor(width / step), h = Math.floor(height / step), k = step * step;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = flipY ? height - 1 - (y * step + dy) : y * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const si = (sy * width + x * step + dx) * 4;
          r += rgba[si]; g += rgba[si + 1]; b += rgba[si + 2];
        }
      }
      const di = (y * w + x) * 4;
      out[di] = r / k; out[di + 1] = g / k; out[di + 2] = b / k; out[di + 3] = 255;
    }
  }
  return { data: out, width: w, height: h, factor: step };
}

// The 4 expected tags from one detection, or why not. markers: [{id, corners: [[x, y] x4]}].
export function pickTags(markers, ids) {
  const seen = new Map();
  for (const m of markers) if (ids.includes(m.id)) seen.set(m.id, (seen.get(m.id) || []).concat([m]));
  const dup = [...seen].filter(([, l]) => l.length > 1).map(([id]) => id);
  if (dup.length) return { ok: false, why: `id ${dup.join(", ")} seen twice: every tag needs its own id` };
  // 2-tag mode: any 2 tags will do (e.g. ids 0 and 2 of a 4-tag sheet), when exactly 2 are in view
  if (ids.length === 2 && ids.some((id) => !seen.has(id)) && markers.length === 2 && markers[0].id !== markers[1].id)
    return { ok: true, tags: markers.slice().sort((a, b) => a.id - b.id) };
  const missing = ids.filter((id) => !seen.has(id));
  if (missing.length) return { ok: false, found: ids.length - missing.length, why: `${ids.length - missing.length} of ${ids.length} tags in view, missing id ${missing.join(", ")}` };
  return { ok: true, tags: ids.map((id) => seen.get(id)[0]) };
}

// Direction of a tag's top edge on the table, from its 4 world corners in detector order
// (top-left, top-right, bottom-right, bottom-left as printed), averaged over top and bottom edges.
export const tagAxis = (corners) => unit(add(sub(corners[1], corners[0]), sub(corners[2], corners[3])));

// Rectangle from 2 tags at opposite corners, both laid the same way up: the tags' own edges give
// the rectangle's directions, the diagonal between their centres gives its size. Same result shape
// as fitRectangle; angleErrDeg is how far apart the two tags' directions are.
export function fitFromTwo(centers, axes, n) {
  if (centers.length !== 2 || axes.length !== 2) throw new Error(`need 2 tag points, got ${centers.length}`);
  n = unit(n);
  const flat = (a) => sub(a, mul(n, dot(a, n)));
  const a0 = unit(flat(axes[0])), a1 = unit(flat(axes[1]));
  const angleErrDeg = (Math.acos(Math.max(-1, Math.min(1, dot(a0, a1)))) * 180) / Math.PI;
  const s = add(a0, a1);
  let u = norm(s) > 1e-6 ? unit(s) : a0, v = cross(u, n);
  const d = flat(sub(centers[1], centers[0]));
  let a = dot(d, u), b = dot(d, v);
  if (Math.min(Math.abs(a), Math.abs(b)) < 0.02)
    throw new Error("the 2 tags are along one side: put them at opposite corners of the rectangle");
  if (Math.abs(b) > Math.abs(a)) { u = v; v = cross(u, n); [a, b] = [dot(d, u), dot(d, v)]; }   // u = long side
  const c0 = centers[0];
  const center = mul(add(c0, centers[1]), 0.5);
  const corners = [c0, add(c0, mul(u, a)), add(c0, d), add(c0, mul(v, b))];
  return { center, u, n, v, width: Math.abs(a), height: Math.abs(b), corners, angleErrDeg, diagErrPct: 0 };
}

// Flip u and v of a new fit when they point against the old one, so a re-fit of the same tags
// never spins the model 180 degrees (u, n, v stays right-handed).
export function alignAxes(r, prev) {
  if (!prev || dot(r.u, prev.u) >= 0) return r;
  return { ...r, u: mul(r.u, -1), v: mul(r.v, -1) };
}

// Did the tags move: centre or a side by more than tol metres, or the long side by more than tolDeg.
export function rectMoved(a, b, tol = 0.02, tolDeg = 3) {
  const turn = (Math.acos(Math.min(1, Math.abs(dot(a.u, b.u)))) * 180) / Math.PI;
  return norm(sub(a.center, b.center)) > tol || Math.abs(a.width - b.width) > tol ||
    Math.abs(a.height - b.height) > tol || turn > tolDeg;
}

// 2-tag mode, per frame obs = {centers: [c0, c1], axes: [a0, a1]}. A single small tag's edge direction
// jitters by a few degrees, which swings the fitted size by centimetres, so the fit uses the mean of
// the last `need` frames, once every centre is within tol metres and every axis within tolDeg of it.
export function averageTwo(obs, need = 4, tol = 0.01, tolDeg = 6) {
  if (obs.length < need) return null;
  const last = obs.slice(-need), cosTol = Math.cos((tolDeg * Math.PI) / 180);
  const centers = [0, 1].map((k) => mean(last.map((o) => o.centers[k])));
  const axes = [0, 1].map((k) => unit(last.map((o) => unit(o.axes[k])).reduce(add, [0, 0, 0])));
  const ok = last.every((o) => [0, 1].every((k) =>
    norm(sub(o.centers[k], centers[k])) < tol && dot(unit(o.axes[k]), axes[k]) > cosTol));
  return ok ? { centers, axes } : null;
}

// Did any tag centre move more than tol metres (same order in both lists)?
export const tagsMoved = (a, b, tol = 0.02) => a.some((p, i) => norm(sub(p, b[i])) > tol);
