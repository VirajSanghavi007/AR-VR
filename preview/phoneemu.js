// Desktop stand-in for the phone in tagar.html: renders printed AprilTags lying on a table from a
// virtual phone camera, so the page's own detection, 2/4-tag fit, lock and follow code can run
// without ARCore. Only the camera image and the poses are faked; everything after them is real.

// 36h11 code (9 hex digits, from tag36h11.js) -> 6 x 6 data bits, row-major from the top-left, 1 = white.
export function tagBits(hex) {
  const bits = [...hex].map((d) => parseInt(d, 16).toString(2).padStart(4, "0")).join("");
  if (bits.length !== 36 || /[^01]/.test(bits)) throw new Error(`not a 36-bit tag code: ${hex}`);
  return Array.from({ length: 6 }, (_, r) => [...bits.slice(r * 6, r * 6 + 6)].map(Number));
}

// The printed tag as 10 x 10 cells: 1 white margin, 1 black border, 6 x 6 data. 1 = white.
export function tagCells(hex) {
  const bits = tagBits(hex);
  return Array.from({ length: 10 }, (_, r) => Array.from({ length: 10 }, (_, c) =>
    r === 0 || r === 9 || c === 0 || c === 9 ? 1 : r === 1 || r === 8 || c === 1 || c === 8 ? 0 : bits[r - 2][c - 2]));
}

// A seeded random source (Park-Miller) and a roughly normal sample from it, so runs repeat exactly.
export function rng(seed = 1) {
  let s = Math.max(1, Math.floor(seed) % 2147483647);
  const uni = () => (s = (s * 16807) % 2147483647) / 2147483647;
  // sum of 6 uniforms has variance 1/2, so scale by sqrt 2 for a unit spread
  return { uni, gauss: () => { let t = 0; for (let i = 0; i < 6; i++) t += uni(); return (t - 3) * Math.SQRT2; } };
}

export class PhoneEmulator {
  // codes = AR.DICTIONARIES.APRILTAG_36h11.codeList; w x h = camera image pixels; vfov in degrees.
  constructor(THREE, renderer, codes, { w = 720, h = 1280, vfov = 62 } = {}) {
    Object.assign(this, { THREE, renderer, codes, w, h });
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x6b6f75);
    const table = new THREE.Mesh(new THREE.PlaneGeometry(4, 3).rotateX(-Math.PI / 2),
                                 new THREE.MeshBasicMaterial({ color: 0xa08c74 }));
    this.scene.add(table);
    this.camera = new THREE.PerspectiveCamera(vfov, w / h, 0.02, 20);
    this.target = new THREE.WebGLRenderTarget(w, h, { samples: 4 });
    this.buf = new Uint8Array(w * h * 4);
    this.tags = new THREE.Group();
    this.scene.add(this.tags);
  }

  texture(id) {
    const cells = tagCells(this.codes[id]), px = 16, c = document.createElement("canvas");
    c.width = c.height = 10 * px;
    const g = c.getContext("2d");
    cells.forEach((row, r) => row.forEach((v, k) => { g.fillStyle = v ? "#f4f4f2" : "#111"; g.fillRect(k * px, r * px, px, px); }));
    const t = new this.THREE.CanvasTexture(c);
    t.magFilter = this.THREE.NearestFilter;
    return t;
  }

  // tags = [{id, at: [x, z], yawDeg = 0}] flat on the table (y = 0); size = black square edge in metres.
  // Yaw 0 = printed upright with its top edge along +x and its top toward -z.
  // wall: the tags hang on the wall z = 0 facing +z instead, at = [x, y], yaw 0 = upright.
  setTags(tags, size = 0.16, wall = false) {
    const T = this.THREE;
    this.tags.clear();
    for (const t of tags) {
      const g = new T.PlaneGeometry(size * 1.25, size * 1.25);
      const m = new T.Mesh(wall ? g : g.rotateX(-Math.PI / 2), new T.MeshBasicMaterial({ map: this.texture(t.id) }));
      if (wall) m.position.set(t.at[0], t.at[1], 0.0005);
      else m.position.set(t.at[0], 0.0005, t.at[1]);
      m.rotation[wall ? "z" : "y"] = ((t.yawDeg || 0) * Math.PI) / 180;
      this.tags.add(m);
    }
  }

  // Tags anywhere (several walls, floor): tags = [{id, at: [x, y, z], face: [nx, ny, nz], spinDeg = 0}];
  // face = the way the printed side looks (out of its wall), spin = turned in its own plane from upright
  // (on a wall: top edge level; on the floor: top towards -z). size as in setTags.
  setPosedTags(tags, size = 0.16) {
    const T = this.THREE;
    this.tags.clear();
    for (const t of tags) {
      const m = new T.Mesh(new T.PlaneGeometry(size * 1.25, size * 1.25), new T.MeshBasicMaterial({ map: this.texture(t.id) }));
      const n = new T.Vector3(...t.face).normalize(), level = Math.abs(n.y) < 0.9;
      const up = level ? new T.Vector3(0, 1, 0) : new T.Vector3(0, 0, -Math.sign(n.y));
      const x = new T.Vector3().crossVectors(up, n).normalize(), y = new T.Vector3().crossVectors(n, x);
      m.matrixAutoUpdate = false;
      m.matrix.makeBasis(x, y, n).multiply(new T.Matrix4().makeRotationZ(((t.spinDeg || 0) * Math.PI) / 180))
        .setPosition(t.at[0] + n.x * 0.0005, t.at[1] + n.y * 0.0005, t.at[2] + n.z * 0.0005);
      this.tags.add(m);
    }
  }

  // Grey walls (and a ceiling) round a w x d x h room centred on the origin, so the camera image is a room.
  setRoom(w = 4, d = 4, h = 2.6) {
    const T = this.THREE;
    if (this.room) this.scene.remove(this.room);
    this.room = new T.Group();
    const mat = (c) => new T.MeshBasicMaterial({ color: c, side: T.DoubleSide });
    const walls = [[[0, h / 2, -d / 2], 0, w, 0xc9c4ba], [[0, h / 2, d / 2], Math.PI, w, 0xc4bfb5],
                   [[-w / 2, h / 2, 0], Math.PI / 2, d, 0xbfbab0], [[w / 2, h / 2, 0], -Math.PI / 2, d, 0xcdc8be]];
    for (const [p, ry, len, c] of walls) {
      const m = new T.Mesh(new T.PlaneGeometry(len, h), mat(c));
      m.position.set(...p); m.rotation.y = ry;
      this.room.add(m);
    }
    const ceil = new T.Mesh(new T.PlaneGeometry(w, d).rotateX(Math.PI / 2), mat(0xe2e0db));
    ceil.position.y = h;
    const floor = new T.Mesh(new T.PlaneGeometry(w, d).rotateX(-Math.PI / 2), mat(0x8d8478));
    floor.position.y = -0.001;                                   // under the table, so tags on it still show
    this.room.add(ceil, floor);
    this.scene.add(this.room);
  }

  // Tracking drift: the page sees every pose (camera, hit tests) through this matrix (a three.js Matrix4,
  // identity = none). driftStep turns it by deg about y and shifts it by mm in a random level direction.
  resetDrift() { this.drift = new this.THREE.Matrix4(); }
  driftStep(r, mm, deg) {
    const T = this.THREE, a = r.uni() * 2 * Math.PI;
    const step = new T.Matrix4().makeRotationY((r.gauss() * deg * Math.PI) / 180)
      .setPosition((Math.cos(a) * mm) / 1000, (r.gauss() * mm * 0.2) / 1000, (Math.sin(a) * mm) / 1000);
    this.drift = step.multiply(this.drift || new T.Matrix4());
  }

  // Point the phone from eye at look (world metres), shaken by shake = {mm, deg} with random r.
  aim(eye, look, r, shake = { mm: 4, deg: 0.4 }) {
    const T = this.THREE, j = (k) => r.gauss() * k;
    this.camera.position.set(eye[0] + j(shake.mm / 1000), eye[1] + j(shake.mm / 1000), eye[2] + j(shake.mm / 1000));
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(new T.Vector3(...look));
    this.camera.rotateZ(j((shake.deg * Math.PI) / 180));
    this.camera.rotateX(j((shake.deg * Math.PI) / 180));
    this.camera.updateMatrixWorld(true);
  }

  // The camera image as the phone's camera texture reads back: RGBA, rows bottom-up.
  capture() {
    const rr = this.renderer, prev = rr.getRenderTarget(), xr = rr.xr.enabled;
    rr.xr.enabled = false;
    rr.setRenderTarget(this.target);
    rr.render(this.scene, this.camera);
    rr.readRenderTargetPixels(this.target, 0, 0, this.w, this.h, this.buf);
    rr.setRenderTarget(prev);
    rr.xr.enabled = xr;
    return this.buf;
  }

  // The XRView the page sees: the true pose plus tracking error of trackMm, with a camera.
  view(r, trackMm = 1) {
    const m = this.camera.matrixWorld.clone(), e = m.elements;
    for (const i of [12, 13, 14]) e[i] += (r.gauss() * trackMm) / 1000;
    if (this.drift) m.premultiply(this.drift);
    return { projectionMatrix: Float32Array.from(this.camera.projectionMatrix.elements),
             transform: { matrix: Float32Array.from(e) }, camera: { width: this.w, height: this.h } };
  }

  // A hit-test pose on the table at p whose up axis is off by about normalDeg (ARCore's plane estimate).
  hit(p, r, normalDeg = 1) {
    const T = this.THREE, a = (normalDeg * Math.PI) / 180;
    const q = new T.Quaternion().setFromEuler(new T.Euler(r.gauss() * a, 0, r.gauss() * a));
    const m = new T.Matrix4().makeRotationFromQuaternion(q).setPosition(p[0], 0, p[1]);
    if (this.drift) m.premultiply(this.drift);
    return Array.from(m.elements);
  }
}
