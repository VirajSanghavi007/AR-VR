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
  setTags(tags, size = 0.16) {
    const T = this.THREE;
    this.tags.clear();
    for (const t of tags) {
      const m = new T.Mesh(new T.PlaneGeometry(size * 1.25, size * 1.25).rotateX(-Math.PI / 2),
                           new T.MeshBasicMaterial({ map: this.texture(t.id) }));
      m.position.set(t.at[0], 0.0005, t.at[1]);
      m.rotation.y = ((t.yawDeg || 0) * Math.PI) / 180;
      this.tags.add(m);
    }
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
    return { projectionMatrix: Float32Array.from(this.camera.projectionMatrix.elements),
             transform: { matrix: Float32Array.from(e) }, camera: { width: this.w, height: this.h } };
  }

  // A hit-test pose on the table at p whose up axis is off by about normalDeg (ARCore's plane estimate).
  hit(p, r, normalDeg = 1) {
    const T = this.THREE, a = (normalDeg * Math.PI) / 180;
    const q = new T.Quaternion().setFromEuler(new T.Euler(r.gauss() * a, 0, r.gauss() * a));
    return Array.from(new T.Matrix4().makeRotationFromQuaternion(q).setPosition(p[0], 0, p[1]).elements);
  }
}
