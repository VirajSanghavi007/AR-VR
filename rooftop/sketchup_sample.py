"""A SketchUp-style test model for the AR page's model picker: a small solar pergola (posts, beams,
rails, six panels on a tilted roof), written the way SketchUp exports it.

- COLLADA (.dae): SketchUp's own 3D export. Inches, Z up (as SketchUp writes it); three.js's
  ColladaLoader reads the <unit> and <up_axis> and turns it into metres, Y up.
- OBJ + MTL: SketchUp's other export (Pro). Inches, Y up ("swap YZ" on, SketchUp's default), colours in
  the .mtl. OBJ carries no units, so the page asks for them (or guesses).

Geometry is in metres, z up, x across the front, y back from the front posts, until written out.
Faces are wound outward (counter-clockwise seen from outside) with flat normals.
"""
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

INCH = 0.0254


@dataclass(frozen=True)
class Part:
    material: str
    vertices: np.ndarray        # (N, 3) metres, z up
    faces: np.ndarray           # (M, 3) vertex indices, outward winding


MATERIALS = {                   # name -> diffuse RGB 0..1
    "Wood_Cedar": (0.55, 0.36, 0.2),
    "Aluminium": (0.78, 0.8, 0.83),
    "Solar_Glass": (0.06, 0.1, 0.22),
}

# The 8 corners of a slab: bottom quad 0..3 (counter-clockwise from above), top quad 4..7 above them.
SLAB_FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                       [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                       [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def slab(quad, thickness, material):
    """A prism: the quad (4 corners, counter-clockwise seen from above its normal) pushed out along its normal."""
    q = np.asarray(quad, float)
    n = np.cross(q[1] - q[0], q[3] - q[0])
    n /= np.linalg.norm(n)
    return Part(material, np.vstack([q, q + n * thickness]), SLAB_FACES.copy())


def box(lo, hi, material):
    """Axis-aligned box from corner lo to corner hi (metres)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return slab([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], z1 - z0, material)


def pergola(width=3.0, depth=2.4, front=2.5, back=2.2, cols=3, rows=2, gap=0.05, post=0.1):
    """Parts of the pergola: 4 posts, 2 beams, 2 rails along the slope, cols x rows panels on top."""
    parts = []
    for x in (0.0, width - post):
        parts.append(box([x, 0, 0], [x + post, post, front], "Wood_Cedar"))
        parts.append(box([x, depth - post, 0], [x + post, depth, back], "Wood_Cedar"))
    parts.append(box([0, 0, front], [width, post, front + 0.15], "Wood_Cedar"))
    parts.append(box([0, depth - post, back], [width, depth, back + 0.15], "Wood_Cedar"))
    top_f, top_b = front + 0.15, back + 0.15
    z_at = lambda y: top_f + (top_b - top_f) * y / depth                      # roof plane height at y
    for x in (width * 0.25, width * 0.75):
        parts.append(slab([[x - 0.03, 0, top_f], [x + 0.03, 0, top_f], [x + 0.03, depth, top_b], [x - 0.03, depth, top_b]],
                          0.04, "Aluminium"))
    pw = (width - (cols - 1) * gap) / cols
    pd = (depth - (rows - 1) * gap) / rows                                      # panel extent in plan y
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * (pw + gap), r * (pd + gap)
            x1, y1 = x0 + pw, y0 + pd
            quad = [[x0, y0, z_at(y0) + 0.04], [x1, y0, z_at(y0) + 0.04], [x1, y1, z_at(y1) + 0.04], [x0, y1, z_at(y1) + 0.04]]
            parts.append(slab(quad, 0.035, "Solar_Glass"))
    return parts


def bounds(parts):
    v = np.vstack([p.vertices for p in parts])
    return v.min(0), v.max(0)


def face_normals(vertices, faces):
    v = np.asarray(vertices, float)
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _floats(a):
    return " ".join(f"{x:.6g}" for x in np.asarray(a, float).ravel())


def dae_text(parts, unit=INCH, unit_name="inch"):
    """COLLADA 1.4.1 like SketchUp writes: one geometry per material, Z_UP, coordinates in `unit`s."""
    mats = [m for m in MATERIALS if any(p.material == m for p in parts)]
    eff = "".join(f'<effect id="{m}-fx"><profile_COMMON><technique sid="common"><lambert><diffuse>'
                  f'<color>{_floats(MATERIALS[m])} 1</color></diffuse></lambert></technique></profile_COMMON></effect>'
                  for m in mats)
    lib_mat = "".join(f'<material id="{m}-mat" name="{m}"><instance_effect url="#{m}-fx"/></material>' for m in mats)
    geos, nodes = [], []
    for m in mats:
        ps = [p for p in parts if p.material == m]
        v, f, k = [], [], 0
        for p in ps:
            v.append(p.vertices / unit)
            f.append(p.faces + k)
            k += len(p.vertices)
        v, f = np.vstack(v), np.vstack(f)
        n = face_normals(v, f)
        idx = np.column_stack([f[:, 0], np.arange(len(f)), f[:, 1], np.arange(len(f)), f[:, 2], np.arange(len(f))])
        g = f"{m}-geo"
        geos.append(
            f'<geometry id="{g}" name="{m}"><mesh>'
            f'<source id="{g}-pos"><float_array id="{g}-pos-a" count="{v.size}">{_floats(v)}</float_array>'
            f'<technique_common><accessor source="#{g}-pos-a" count="{len(v)}" stride="3"><param name="X" type="float"/>'
            f'<param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>'
            f'<source id="{g}-nrm"><float_array id="{g}-nrm-a" count="{n.size}">{_floats(n)}</float_array>'
            f'<technique_common><accessor source="#{g}-nrm-a" count="{len(n)}" stride="3"><param name="X" type="float"/>'
            f'<param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>'
            f'<vertices id="{g}-vtx"><input semantic="POSITION" source="#{g}-pos"/></vertices>'
            f'<triangles count="{len(f)}" material="{m}"><input semantic="VERTEX" source="#{g}-vtx" offset="0"/>'
            f'<input semantic="NORMAL" source="#{g}-nrm" offset="1"/><p>{" ".join(map(str, idx.ravel()))}</p></triangles>'
            f'</mesh></geometry>')
        nodes.append(f'<node id="{m}-node" name="{escape(m)}"><instance_geometry url="#{g}"><bind_material><technique_common>'
                     f'<instance_material symbol="{m}" target="#{m}-mat"/></technique_common></bind_material>'
                     f'</instance_geometry></node>')
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
            '<asset><contributor><authoring_tool>SketchUp-style sample (rooftop/sketchup_sample.py)</authoring_tool>'
            f'</contributor><unit meter="{unit}" name="{unit_name}"/><up_axis>Z_UP</up_axis></asset>'
            f'<library_effects>{eff}</library_effects><library_materials>{lib_mat}</library_materials>'
            f'<library_geometries>{"".join(geos)}</library_geometries>'
            f'<library_visual_scenes><visual_scene id="scene" name="SketchUp">{"".join(nodes)}</visual_scene></library_visual_scenes>'
            '<scene><instance_visual_scene url="#scene"/></scene></COLLADA>\n')


def obj_text(parts, mtl_name, unit=INCH, unit_name="inches"):
    """OBJ as SketchUp exports with "swap YZ" on: Y up (x, z, -y), in `unit`s (named in the header the way SketchUp
    writes it, "# File units = inches"), one group per part, flat normals."""
    out = ["# SketchUp-style sample (rooftop/sketchup_sample.py)", f"# File units = {unit_name}", f"mtllib {mtl_name}"]
    k = kn = 0                                                                  # vertices / normals written so far
    for i, p in enumerate(parts):
        v = p.vertices[:, [0, 2, 1]] * [1, 1, -1] / unit                         # z up -> y up, keeps handedness
        n = face_normals(v, p.faces)
        out.append(f"g part_{i}")
        out.append(f"usemtl {p.material}")
        out += [f"v {_floats(x)}" for x in v]
        out += [f"vn {_floats(x)}" for x in n]
        out += [f"f {a + k + 1}//{j + kn + 1} {b + k + 1}//{j + kn + 1} {c + k + 1}//{j + kn + 1}"
                for j, (a, b, c) in enumerate(p.faces)]
        k += len(v)
        kn += len(n)
    return "\n".join(out) + "\n"


def mtl_text(parts):
    mats = [m for m in MATERIALS if any(p.material == m for p in parts)]
    return "".join(f"newmtl {m}\nKd {_floats(MATERIALS[m])}\nKa 0 0 0\nd 1\n\n" for m in mats)


def write_sample(out_dir, name="sample_sketchup_pergola"):
    """Write {name}.dae, {name}.obj and {name}.mtl; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = pergola()
    files = {"dae": out_dir / f"{name}.dae", "obj": out_dir / f"{name}.obj", "mtl": out_dir / f"{name}.mtl"}
    files["dae"].write_text(dae_text(parts), encoding="utf-8")
    files["obj"].write_text(obj_text(parts, files["mtl"].name), encoding="utf-8")
    files["mtl"].write_text(mtl_text(parts), encoding="utf-8")
    return files
