// Meshes for a site design (blocks.js objects), size labels, and model files (glTF, COLLADA, OBJ + MTL),
// shared by site.html (AR on site) and viewer.html (the customer's link). Uses the page's three.js.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { MATERIALS, blockParts, sizeLabel, objectName } from "./blocks.js?v=3";
import { modelKind, baseName, objFileUnits, fileUnits, guessUnits, UNIT_M, seat } from "./model.js?v=4";

// One three.js material per finish, made once; wrap(m) adds a page's shader tweaks (the camera look).
export function materialSet(wrap = (m) => m, envIntensity = 1) {
  const cache = new Map();
  return (key) => {
    if (!cache.has(key)) {
      const s = MATERIALS[key] || MATERIALS.plaster;
      const opts = { color: new THREE.Color(s.color), roughness: s.roughness, metalness: s.metalness, envMapIntensity: envIntensity };
      const m = s.opacity ? new THREE.MeshPhysicalMaterial({ ...opts, transparent: true, opacity: s.opacity, transmission: 0, clearcoat: 1, clearcoatRoughness: 0.05, depthWrite: false })
                          : new THREE.MeshStandardMaterial(opts);
      m.name = s.name;
      cache.set(key, wrap(m));
    }
    return cache.get(key);
  };
}

const boxGeo = new THREE.BoxGeometry(1, 1, 1), ballGeo = new THREE.SphereGeometry(1, 16, 12);

// The object as a group in room metres: posed at (x, y, z), turned by yaw; userData.id = its id.
export function objectMesh(o, mat) {
  const g = new THREE.Group();
  g.name = `${objectName(o)} ${o.id}`;
  g.userData.id = o.id;
  for (const p of blockParts(o)) {
    const material = mat(p.mat === "main" ? o.mat : p.mat);
    let m;
    if (p.kind === "box") {
      m = new THREE.Mesh(boxGeo, material);
      m.scale.set(...p.s);
      m.position.set(...p.c);
      m.rotation.set(p.rx || 0, p.ry || 0, 0, "YXZ");
    } else if (p.kind === "ball") {
      m = new THREE.Mesh(ballGeo, material);
      m.scale.setScalar(p.r);
      m.position.set(...p.c);
    } else if (p.kind === "prism") {
      // shape in (x, -z) so that after turning the extrusion up, shape point (x, -z) lands on room (x, z)
      const shape = new THREE.Shape(p.pts.map(([x, z]) => new THREE.Vector2(x, -z)));
      const geo = new THREE.ExtrudeGeometry(shape, { depth: p.h, bevelEnabled: false }).rotateX(-Math.PI / 2);
      geo.translate(0, p.y0, 0);
      m = new THREE.Mesh(geo, material);
    }
    m.castShadow = !material.transparent;
    m.receiveShadow = true;
    g.add(m);
  }
  g.position.set(o.x, o.y || 0, o.z);
  g.rotation.y = o.yaw || 0;
  return g;
}

// A label that always faces the camera: white text on a dark pill, heightM metres tall in the room.
export function labelSprite(text, heightM = 0.09) {
  const fs = 44, pad = 18, c = document.createElement("canvas"), g = c.getContext("2d");
  g.font = `600 ${fs}px system-ui, sans-serif`;
  c.width = Math.ceil(g.measureText(text).width + 2 * pad); c.height = fs + 2 * pad;
  g.font = `600 ${fs}px system-ui, sans-serif`;
  g.fillStyle = "rgba(12,18,26,.82)";
  g.beginPath(); g.roundRect(0, 0, c.width, c.height, c.height / 2); g.fill();
  g.fillStyle = "#fff"; g.textBaseline = "middle"; g.fillText(text, pad, c.height / 2 + 2);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, toneMapped: false }));
  s.scale.set((heightM * c.width) / c.height, heightM, 1);
  s.renderOrder = 10;
  s.userData.text = text;
  return s;
}

// Every object of one option (a list), plus a size label over each when labels is on.
export function designGroup(objs, mat, { labels = false, labelM = 0.09 } = {}) {
  const root = new THREE.Group();
  root.name = "design";
  for (const o of objs) {
    const g = objectMesh(o, mat);
    root.add(g);
    if (labels) {
      const s = labelSprite(sizeLabel(o), labelM);
      s.position.set(o.x, (o.y || 0) + o.h + 0.18, o.z);
      s.userData.label = o.id;
      root.add(s);
    }
  }
  return root;
}

// Read a model file (url, its file name, side files by lower-case base name for .mtl and textures)
// in metres, seated on y = 0 and centred: { obj (a group), kind, units, size [x, y, z] metres }.
export async function readModel(url, name, side = {}, units = "auto") {
  const kind = modelKind(name);
  if (!kind) throw new Error("not a model file: pick a .glb, .gltf, .dae or .obj");
  const mgr = new THREE.LoadingManager();
  mgr.setURLModifier((u) => side[baseName(u)] || u);
  let obj, said = fileUnits(kind);
  if (kind === "gltf") obj = (await new GLTFLoader(mgr).loadAsync(url)).scene;
  else if (kind === "dae") obj = (await new ColladaLoader(mgr).loadAsync(url)).scene;
  else {
    const text = await (await fetch(url)).text(), loader = new OBJLoader(mgr);
    said = fileUnits(kind, objFileUnits(text));
    const lib = /^mtllib\s+(.+?)\s*$/m.exec(text);
    if (lib) {
      const mtlUrl = side[baseName(lib[1])] || (url.startsWith("blob:") ? null : new URL(lib[1], new URL(url, location.href)).href);
      if (mtlUrl) {
        try { const mats = await new MTLLoader(mgr).loadAsync(mtlUrl); mats.preload(); loader.setMaterials(mats); }
        catch (e) { console.warn(`${lib[1]}: ${e.message || e}; grey instead`); }
      }
    }
    obj = loader.parse(text);
  }
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; if (kind !== "gltf") for (const m of [o.material].flat()) m.side = THREE.DoubleSide; } });
  const raw = new THREE.Box3().setFromObject(obj), rawSize = raw.getSize(new THREE.Vector3()).toArray();
  const u = units === "auto" ? (said || guessUnits(Math.max(...rawSize))) : units;
  const k = UNIT_M[u], inner = new THREE.Group();
  inner.add(obj);
  obj.scale.multiplyScalar(k);
  const box = new THREE.Box3().setFromObject(inner);
  obj.position.add(new THREE.Vector3(...seat(box.min.toArray(), box.max.toArray())));
  const out = new THREE.Group();
  out.name = name;
  out.add(inner);
  return { obj: out, kind, units: u, size: rawSize.map((v) => v * k) };
}

// The bounding box of root's meshes in root's own frame (whatever root is posed at): { min, max } arrays.
export function localBox(root) {
  root.updateMatrixWorld(true);
  const inv = root.matrixWorld.clone().invert(), box = new THREE.Box3(), b = new THREE.Box3();
  root.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    b.copy(o.geometry.boundingBox).applyMatrix4(new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld));
    box.union(b);
  });
  return { min: box.min.toArray(), max: box.max.toArray() };
}

// Sun and sky for a design scene (the sun's shadow box covers about span metres round the origin).
export function sunAndSky(span = 8, intensity = 1.8) {
  const hemi = new THREE.HemisphereLight(0xdfe9f5, 0x6b6257, intensity * 0.83);
  const sun = new THREE.DirectionalLight(0xffffff, intensity);
  sun.position.set(3, 6, 2.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 0.5, far: 30 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
  return { hemi, sun };
}
