"""Export the roof + panel design as glTF binary (.glb) for 3D and AR viewers.

glTF is metres, Y-up. The model origin is put at the roof-plane origin so the
AR app can anchor it to a real point on the roof (corner tap, marker, geo pose).

Panels are built realistically: an aluminium frame body with a glass laminate on
top that carries a generated texture of 120 half-cut mono cells with busbars.
For AR, export with include_roof=False: the real roof is the surface, and a grey
slab drawn over it is what makes the overlay look pasted on.
"""
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

# z-up (our levelled frame) -> y-up (glTF)
Z_TO_Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)

ROOF_RGB = (0.55, 0.55, 0.55)
FRAME_WIDTH = 0.03       # visible aluminium rim, metres
GLASS_LIFT = 0.002       # laminate sits just above the frame top (avoids z-fighting)


@dataclass(frozen=True)
class Material:
    rgb: tuple
    metallic: float = 0.0
    roughness: float = 0.6
    texture: bytes = None    # PNG or JPEG bytes for baseColorTexture
    tile: bool = False       # texture repeats (world-metre UVs) instead of clamping
    normal: bytes = None     # PNG or JPEG tangent-space normalTexture (same UVs, same sampler)


def image_mime(data):
    """glTF mimeType of embedded image bytes (PNG or JPEG, the two glTF allows)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    raise ValueError("embedded textures must be PNG or JPEG")


def cell_texture_png(width=1536, cols=20, rows=6, busbars=9):
    """PNG of a 120 half-cut cell module face (long side along image width).

    rows x cols cells: 6 across the short side, 20 half-cells along the long side,
    with the half-cut split gap in the middle and thin silver busbars along the
    string direction.
    """
    from PIL import Image

    h = int(round(width * 1.134 / 1.722))
    img = np.empty((h, width, 3), np.float32)
    img[:] = (0.86, 0.87, 0.89)                     # white backsheet visible in gaps
    margin, gap, mid = 0.012, 0.0025, 0.006         # fractions of width
    mx, gx, midx = int(width * margin), max(1, int(width * gap)), int(width * mid)
    usable_w = width - 2 * mx - midx - (cols - 2) * gx
    cw = usable_w / cols
    my = mx
    ch = (h - 2 * my - (rows - 1) * gx) / rows
    yy = np.linspace(0, 1, int(ch))[:, None]
    for r in range(rows):
        y0 = int(round(my + r * (ch + gx)))
        for c in range(cols):
            x0 = mx + c * (cw + gx) + (midx - gx if c >= cols // 2 else 0)
            x0 = int(round(x0))
            cell = np.empty((int(ch), int(cw), 3), np.float32)
            shade = 0.9 + 0.1 * yy                      # subtle sheen across each cell
            cell[:] = np.array([0.05, 0.07, 0.13]) * shade[..., None][:, :1]
            for b in range(busbars):
                by = int((b + 0.5) * cell.shape[0] / busbars)
                cell[by:by + max(1, int(ch * 0.008)), :] = (0.36, 0.38, 0.43)
            img[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell
    out = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out, "PNG", optimize=True)
    return out.getvalue()


def _materials():
    return {
        "roof": Material(ROOF_RGB, 0.0, 0.9),
        "frame": Material((0.78, 0.80, 0.83), 0.9, 0.35),
        "glass": Material((1.0, 1.0, 1.0), 0.0, 0.12, cell_texture_png()),
    }


def box_from_quad(corners, thickness):
    """Vertices (8,3) and triangles (12,3) of a thin slab under a planar quad."""
    c = np.asarray(corners, float)
    n = _normal(c)
    v = np.vstack([c - n * thickness, c])
    t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, t


def _normal(c):
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n = n / np.linalg.norm(n)
    return n if n[2] >= 0 else -n


def laminate_quad(corners, frame_width=FRAME_WIDTH, lift=GLASS_LIFT):
    """Glass quad inset by the frame rim, with UVs: u runs along the panel's long edge."""
    c = np.asarray(corners, float)
    e1, e3 = c[1] - c[0], c[3] - c[0]
    a, b = np.linalg.norm(e1), np.linalg.norm(e3)
    d1, d3 = e1 / a, e3 / b
    fw = min(frame_width, 0.25 * min(a, b))
    inset = np.array([c[0] + fw * d1 + fw * d3, c[1] - fw * d1 + fw * d3,
                      c[2] - fw * d1 - fw * d3, c[3] + fw * d1 - fw * d3])
    v = inset + _normal(c) * lift
    uv = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) if a >= b
          else np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    return v, np.array([[0, 1, 2], [0, 2, 3]]), uv.astype(float)


def polygon_mesh(poly, frame, h=0.0):
    """Triangulate a (possibly concave, holed) roof polygon onto its plane."""
    tris = shapely.constrained_delaunay_triangles(poly)
    verts, faces = [], []
    for tri in tris.geoms:
        ring = np.asarray(tri.exterior.coords)[:3]
        base = len(verts)
        verts.extend(frame.to_3d(ring, h))
        faces.append([base, base + 1, base + 2])
    return np.asarray(verts), np.asarray(faces)


def build_scene(frame, outline, panel_corners, panel_thickness=0.035, include_roof=True):
    """List of (vertices, triangles, uv or None, material key) in the levelled z-up frame."""
    parts = []
    if include_roof and outline is not None and not outline.is_empty:
        v, t = polygon_mesh(outline, frame, h=-0.005)
        parts.append((v, t, None, "roof"))
    for c in panel_corners:
        v, t = box_from_quad(c, panel_thickness)
        parts.append((v, t, None, "frame"))
        parts.append(laminate_quad(c) + ("glass",))
    return parts


def _pad4(b, fill=b"\x00"):
    return b + fill * (-len(b) % 4)


def glb_bytes(parts, materials):
    """Minimal glTF 2.0 binary: one primitive per material, uint32 indices,
    optional TEXCOORD_0 + embedded PNG/JPEG baseColorTexture and normalTexture, optional NORMAL.

    parts: [(vertices (N,3) y-up metres, triangles (M,3), uv (N,2) or None, material key
    [, normals (N,3) y-up])]. NORMAL is written for a material only when every one of its
    parts has normals.
    Open3D's own .glb writer produces out-of-range buffer views, hence this.
    """
    groups = {}
    for part in parts:
        v, t, uv, key = part[:4]
        nrm = part[4] if len(part) > 4 else None
        g = groups.setdefault(key, {"v": [], "t": [], "uv": [], "nrm": [], "n": 0})
        g["nrm"].append(None if nrm is None else np.asarray(nrm, np.float32))
        g["v"].append(np.asarray(v, np.float32))
        g["t"].append(np.asarray(t, np.uint32) + g["n"])
        g["uv"].append(np.asarray(uv, np.float32) if uv is not None else np.zeros((len(v), 2), np.float32))
        g["n"] += len(v)

    blob, views, accessors, prims, mats = b"", [], [], [], []
    images, textures = [], []

    def add_view(data, target=None):
        nonlocal blob
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        blob = _pad4(blob + data)
        return len(views) - 1

    for key, g in groups.items():
        m = materials[key]
        pos, idx = np.vstack(g["v"]), np.vstack(g["t"]).ravel()
        attrs = {"POSITION": len(accessors)}
        accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126,
                          "count": len(pos), "type": "VEC3",
                          "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        if all(n is not None for n in g["nrm"]):
            nrm = np.vstack(g["nrm"])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            attrs["NORMAL"] = len(accessors)
            accessors.append({"bufferView": add_view(nrm.astype(np.float32).tobytes(), 34962),
                              "componentType": 5126, "count": len(nrm), "type": "VEC3"})
        pbr = {"baseColorFactor": [*m.rgb, 1.0], "metallicFactor": m.metallic,
               "roughnessFactor": m.roughness}
        mat = {"name": key, "pbrMetallicRoughness": pbr, "doubleSided": True}

        def add_tex(data):
            images.append({"bufferView": add_view(data), "mimeType": image_mime(data)})
            textures.append({"source": len(images) - 1, "sampler": 1 if m.tile else 0})
            return {"index": len(textures) - 1}

        if m.texture is not None or m.normal is not None:
            uv = np.vstack(g["uv"])
            attrs["TEXCOORD_0"] = len(accessors)
            accessors.append({"bufferView": add_view(uv.tobytes(), 34962), "componentType": 5126,
                              "count": len(uv), "type": "VEC2"})
        if m.texture is not None:
            pbr["baseColorTexture"] = add_tex(m.texture)
        if m.normal is not None:
            mat["normalTexture"] = add_tex(m.normal)
        idx_acc = len(accessors)
        accessors.append({"bufferView": add_view(idx.tobytes(), 34963), "componentType": 5125,
                          "count": int(idx.size), "type": "SCALAR"})
        mats.append(mat)
        prims.append({"attributes": attrs, "indices": idx_acc, "material": len(mats) - 1, "mode": 4})

    gltf = {"asset": {"version": "2.0", "generator": "rooftop.ar_export"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "RoofSolar"}],
            "meshes": [{"primitives": prims}], "materials": mats,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    if images:
        gltf.update(images=images, textures=textures,
                    samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071},
                              {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\x00") + blob)


def write_glb(path, frame, outline, panel_corners, panel_thickness=0.035, include_roof=True):
    """Single .glb, y-up metres, origin at the roof-plane origin."""
    parts = [((v - frame.origin) @ Z_TO_Y.T, t, uv, key)
             for v, t, uv, key in build_scene(frame, outline, panel_corners,
                                              panel_thickness, include_roof)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(parts, _materials()))
    return path


def _gltf_ring(frame, ring2d):
    """2D roof-plane ring -> list of [x, y, z] in the glTF frame (y-up, origin at roof origin)."""
    p = (frame.to_3d(np.asarray(ring2d, float)) - frame.origin) @ Z_TO_Y.T
    return np.round(p, 4).tolist()


def anchor_points(outline):
    """Two reference corners for tap placement: the long edge of the outline's
    minimum rotated rectangle, snapped to the nearest real outline vertex
    (a corner the user can actually stand on), as 2D roof-plane points (A, B)."""
    rect = np.asarray(outline.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
    verts = np.asarray(outline.exterior.coords)
    snap = lambda p: verts[np.argmin(np.linalg.norm(verts - p, axis=1))]
    longest = max(np.linalg.norm(b - a) for a, b in edges)
    # of the two long sides, keep the one whose rectangle corners lie closest to real vertices
    long_sides = [(a, b) for a, b in edges if np.linalg.norm(b - a) > longest * (1 - 1e-9)]
    a, b = min(long_sides, key=lambda e: sum(np.linalg.norm(snap(p) - p) for p in e))
    return snap(a), snap(b)


def anchor_data(frame, outline, obstacles=None):
    """Data the AR page needs to place the design from two tapped corners."""
    a, b = anchor_points(outline)
    rings = lambda g: [] if g is None or g.is_empty else [
        _gltf_ring(frame, np.asarray(p.exterior.coords))
        for p in getattr(g, "geoms", [g])]
    pa, pb = _gltf_ring(frame, [a, b])
    return {"units": "m", "frame": "glTF y-up, origin = roof-plane origin",
            "anchor_a": pa, "anchor_b": pb,
            "anchor_length_m": round(float(np.linalg.norm(b - a)), 3),
            "outline": rings(outline), "keepouts": rings(obstacles)}


def write_anchor(path, frame, outline, obstacles=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor_data(frame, outline, obstacles)))
    return path
