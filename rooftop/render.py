"""Render a STEP file (the CAD output) as a shaded 3D image, no panels."""
from pathlib import Path

import numpy as np


def tessellate_step(path, tol=5.0):
    """Return (vertices Nx3, triangles Mx3) in the STEP's units (mm)."""
    import cadquery as cq

    shape = cq.importers.importStep(str(path)).val()
    verts, tris = shape.tessellate(tol)
    v = np.array([(p.x, p.y, p.z) for p in verts], float)
    return v, np.array(tris, int)


def shade(v, tris, light=(-0.5, -0.7, 0.6), base=(0.72, 0.72, 0.70)):
    """Per-triangle RGB from a two-sided Lambert term plus ambient (STEP winding varies)."""
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    l = np.array(light, float)
    l /= np.linalg.norm(l)
    k = 0.35 + 0.65 * np.abs(n @ l)
    return np.clip(k[:, None] * np.array(base), 0, 1)


def render_png(step_path, out_path, elev=28, azim=-55, size=(1600, 1100), extra=()):
    """extra: (vertices mm, triangles, base rgb) meshes drawn on top, e.g. panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, tris = tessellate_step(step_path)
    n0, t0, cols = len(v), len(tris), []
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for ev, et, rgb in extra:
        off = len(v)
        v = np.vstack([v, ev])
        tris = np.vstack([tris, et + off])
        cols.append(shade(ev, et, base=rgb))
    cols = [shade(v[:n0], tris[:t0])] + cols
    col = Poly3DCollection(v[tris], facecolors=np.vstack(cols), edgecolor=(0.25, 0.25, 0.25, 0.35), linewidth=0.3)
    ax.add_collection3d(col)
    lo, hi = v.min(0), v.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("mm"); ax.set_ylabel("mm"); ax.set_zlabel("mm")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path
