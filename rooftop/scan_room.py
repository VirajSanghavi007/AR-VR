"""Phase 3 interior: a laser-scanned room (point cloud) -> a simple triangle mesh -> SketchUp exports.

Steps (scan_to_room): merge the scans with normals facing their scanner, level the floor, turn the walls
onto the x/y axes, crop to the room's box, put the floor's corner at the origin, Poisson-mesh, decimate,
then sort the faces into Floor / Wall / Ceiling / Furniture so SketchUp gets one material (and one
geometry) per kind. Written with rooftop.sketchup_sample: .dae (inches, Z_UP, SketchUp's own export)
and .obj + .mtl. A .skp itself needs SketchUp (File > Import the .dae, then Save).

Units: metres, z up, until written out. Open3D is imported inside the functions that need it.
"""
import json
from pathlib import Path

import numpy as np

from .sketchup_sample import INCH, Part, dae_text, mtl_text, obj_text

ROOM_MATERIALS = {                       # name -> diffuse RGB 0..1
    "Floor": (0.55, 0.47, 0.38),
    "Wall": (0.86, 0.85, 0.81),
    "Ceiling": (0.95, 0.95, 0.94),
    "Furniture": (0.36, 0.42, 0.5),
}
TOOL = "SketchUp-style export of a laser-scanned room (rooftop/scan_room.py)"


# ---------- frame: level, square, crop ----------

def rotation_to_z(n):
    """3x3 rotation taking unit vector n onto +z (Rodrigues); n pointing down turns about x."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def wall_yaw(normals, max_tilt=0.2):
    """Angle (radians, in [-pi/4, pi/4)) of the room's wall directions about z, from the normals of
    near-vertical surfaces. Walls at right angles share one angle modulo 90 degrees, so it averages
    4 x angle on the circle."""
    n = np.asarray(normals, float)
    n = n[np.abs(n[:, 2]) < max_tilt]
    if len(n) == 0:
        raise ValueError("No wall-like (vertical) surfaces")
    a = 4 * np.arctan2(n[:, 1], n[:, 0])
    w = np.hypot(n[:, 0], n[:, 1])
    yaw = np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()) / 4
    return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def floor_ceiling(z, nz, band=0.15):
    """Floor and ceiling heights: up-facing points near the lowest up-facing ones, down-facing points
    near the highest down-facing ones (normals facing into the room). Medians, so stray hits don't count."""
    z, nz = np.asarray(z, float), np.asarray(nz, float)
    up, down = z[nz > 0.9], z[nz < -0.9]
    if len(up) == 0 or len(down) == 0:
        raise ValueError("No floor or no ceiling (need up- and down-facing surfaces)")
    lo, hi = np.percentile(up, 1), np.percentile(down, 99)
    floor = float(np.median(up[(up >= lo - band / 3) & (up <= lo + band)]))
    ceil = float(np.median(down[(down >= hi - band) & (down <= hi + band / 3)]))
    if ceil <= floor:
        raise ValueError(f"Ceiling ({ceil:.2f}) not above floor ({floor:.2f})")
    return floor, ceil


def plan_box(xy, q=0.5):
    """Room footprint (x0, y0, x1, y1): the q-th and (100-q)-th percentiles, so a few points seen
    through a window or door don't stretch it."""
    xy = np.asarray(xy, float)
    lo, hi = np.percentile(xy, q, axis=0), np.percentile(xy, 100 - q, axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def room_frame(points, normals):
    """(R, t, size): p_room = R @ p + t puts the floor at z = 0, walls along x and y and the room's
    plan corner at the origin; size = (length x, width y, height z) in metres."""
    p, n = np.asarray(points, float), np.asarray(normals, float)
    up = np.median(n[n[:, 2] > 0.9], axis=0) if (n[:, 2] > 0.9).any() else np.array([0, 0, 1.0])
    R = rotation_to_z(up)
    n1 = n @ R.T
    R = rot_z(-wall_yaw(n1)) @ R
    p2, n2 = p @ R.T, n @ R.T
    floor, ceil = floor_ceiling(p2[:, 2], n2[:, 2])
    inside = (p2[:, 2] > floor + 0.1) & (p2[:, 2] < ceil - 0.1)
    x0, y0, x1, y1 = plan_box(p2[inside, :2])
    t = -np.array([x0, y0, floor])
    return R, t, (x1 - x0, y1 - y0, ceil - floor)


def crop_box(points, size, margin=0.05):
    """Mask of points inside [0, size] grown by margin on every side."""
    p = np.asarray(points, float)
    lo, hi = -margin, np.asarray(size, float) + margin
    return np.all((p >= lo) & (p <= hi), axis=1)


# ---------- faces ----------

def classify_faces(centers, normals, size, tol=0.12, wall_band=0.2):
    """Label per face: Floor (faces up, at the floor), Ceiling (faces down, at the ceiling), Wall
    (near-vertical, within wall_band of a side of the room box), else Furniture."""
    c, n = np.asarray(centers, float), np.asarray(normals, float)
    L, W, H = size
    lab = np.full(len(c), "Furniture", dtype=object)
    side = np.min(np.c_[c[:, 0], L - c[:, 0], c[:, 1], W - c[:, 1]], axis=1)
    lab[(np.abs(n[:, 2]) < 0.4) & (side < wall_band)] = "Wall"
    lab[(n[:, 2] > 0.7) & (c[:, 2] < tol)] = "Floor"
    lab[(n[:, 2] < -0.7) & (c[:, 2] > H - tol)] = "Ceiling"
    return lab


def mesh_parts(vertices, faces, labels):
    """One Part per label (ROOM_MATERIALS order), each with only its own vertices, re-indexed."""
    v, f, lab = np.asarray(vertices, float), np.asarray(faces, int), np.asarray(labels)
    parts = []
    for m in ROOM_MATERIALS:
        ff = f[lab == m]
        if len(ff) == 0:
            continue
        used, inv = np.unique(ff, return_inverse=True)
        parts.append(Part(m, v[used], inv.reshape(-1, 3)))
    return parts


def plane_parts(size):
    """Floor (z = 0, facing up) and Ceiling (z = height, facing down) as one rectangle each over the room
    box: the levelled planes the scan measured, without the scanner's blind spot under its tripod and the
    shadows of furniture. Faces wind toward the inside of the room."""
    L, W, H = (float(s) for s in size)
    quad = lambda z: np.array([[0, 0, z], [L, 0, z], [L, W, z], [0, W, z]], float)
    return [Part("Floor", quad(0.0), np.array([[0, 1, 2], [0, 2, 3]])),
            Part("Ceiling", quad(H), np.array([[0, 2, 1], [0, 3, 2]]))]


def room_parts(vertices, faces, labels, size, planes=True):
    """Parts for the export: with planes, the scanned Floor / Ceiling faces swapped for plane_parts."""
    if not planes:
        return mesh_parts(vertices, faces, labels)
    lab = np.asarray(labels)
    keep = (lab != "Floor") & (lab != "Ceiling")
    scanned = {p.material: p for p in mesh_parts(vertices, np.asarray(faces)[keep], lab[keep])}
    flat = {p.material: p for p in plane_parts(size)}
    return [scanned.get(m) or flat[m] for m in ROOM_MATERIALS if m in scanned or m in flat]


def drop_degenerate(vertices, faces, min_area=1e-10):
    """Faces with (near) zero area or repeated vertices removed (they have no normal)."""
    v, f = np.asarray(vertices, float), np.asarray(faces, int)
    a = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    return f[a > min_area]                     # a repeated vertex also gives zero area


# ---------- Open3D steps ----------

def load_scans(scan_dir, voxel=0.02, log=print):
    """ETH3D scans (scan_alignment.mlp) merged into the image frame, each downsampled with normals turned
    toward its own scanner (the scan's origin). Returns an Open3D cloud."""
    import open3d as o3d
    from .validate import parse_mlp

    scan_dir = Path(scan_dir)
    scans = parse_mlp(scan_dir / "scan_alignment.mlp")
    out = o3d.geometry.PointCloud()
    for i, (name, m) in enumerate(scans.items(), 1):
        pc = o3d.io.read_point_cloud(str(scan_dir / name)).voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        pc.orient_normals_towards_camera_location(np.zeros(3))
        out += pc.transform(m)
        log(f"      scan {i}/{len(scans)} {name}: {len(pc.points)} points")
    return out


def to_room(cloud):
    """Cloud moved into the room frame and cropped to the room box; returns (cloud, size)."""
    import open3d as o3d

    p, n = np.asarray(cloud.points), np.asarray(cloud.normals)
    R, t, size = room_frame(p, n)
    p2, n2 = p @ R.T + t, n @ R.T
    keep = crop_box(p2, size)
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p2[keep]))
    out.normals = o3d.utility.Vector3dVector(n2[keep])
    return out, size


def mesh_cloud(cloud, size, depth=9, trim=0.04, triangles=60_000):
    """Screened Poisson surface, low-density skin trimmed, cropped to the room box, decimated.
    Returns (vertices, faces) numpy arrays, degenerate faces removed."""
    import open3d as o3d

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth, n_threads=-1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    bb = o3d.geometry.AxisAlignedBoundingBox(np.full(3, -0.03), np.asarray(size, float) + 0.03)
    mesh = mesh.crop(bb)
    if len(mesh.triangles) > triangles:
        mesh = mesh.simplify_quadric_decimation(triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return v, drop_degenerate(v, f)


def face_normals_centers(vertices, faces):
    v, f = np.asarray(vertices, float), np.asarray(faces, int)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True), v[f].mean(1)


def write_room(out_dir, name, parts, size, source="", planes=False):
    """{name}.dae / .obj / .mtl (SketchUp-style, inches) and {name}.json (size, face counts, source)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: out_dir / f"{name}.{k}" for k in ("dae", "obj", "mtl", "json")}
    files["dae"].write_text(dae_text(parts, materials=ROOM_MATERIALS, tool=TOOL), encoding="utf-8")
    files["obj"].write_text(obj_text(parts, files["mtl"].name, tool=TOOL), encoding="utf-8")
    files["mtl"].write_text(mtl_text(parts, materials=ROOM_MATERIALS), encoding="utf-8")
    info = {"size_m": [round(float(s), 3) for s in size], "units_in_files": "inch", "up": "z",
            "faces": {p.material: int(len(p.faces)) for p in parts},
            "floor_ceiling": "fitted planes" if planes else "scanned mesh", "source": source}
    files["json"].write_text(json.dumps(info, indent=1), encoding="utf-8")
    return files, info


def scan_to_room(scan_dir, out_dir, name, voxel=0.02, depth=9, triangles=60_000, source="", planes=True, log=print):
    """The whole chain; prints [k/5] stage lines with elapsed time. Returns (files, info)."""
    import time

    import open3d as o3d

    t0 = time.time()
    step = lambda k, msg: log(f"[{k}/5] {msg} ({time.time() - t0:.0f}s elapsed)")
    step(1, f"loading scans from {scan_dir}, voxel {voxel} m")
    cloud = load_scans(scan_dir, voxel, log)
    step(2, f"levelling and squaring {len(cloud.points)} points")
    cloud, size = to_room(cloud)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_dir / f"{name}_cloud.ply"), cloud)
    step(3, f"room {size[0]:.2f} x {size[1]:.2f} m, {size[2]:.2f} m high; Poisson mesh (depth {depth})")
    v, f = mesh_cloud(cloud, size, depth, triangles=triangles)
    step(4, f"{len(f)} triangles; sorting floor / walls / ceiling / furniture")
    n, c = face_normals_centers(v, f)
    parts = room_parts(v, f, classify_faces(c, n, size), size, planes)
    step(5, "writing SketchUp .dae / .obj")
    files, info = write_room(out_dir, name, parts, size, source, planes)
    log(f"done in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v}" for k, v in info["faces"].items()))
    return files, info
