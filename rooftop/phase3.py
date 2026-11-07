"""Phase 3 viewer data: the roof CAD made from a scan (scripts/design_panels.py output) and
the laser scan itself, put in the glTF frame of that CAD so preview/vr.html?phase=3 can show
both in Orbit, VR and AR.

Phase 3 is "no AutoCAD drawing: a drone LiDAR scan becomes the CAD with a solar layer".
Until drone data exists, the ETH3D terrace laser scan stands in for the LiDAR.
"""
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from rooftop.ar_export import Z_TO_Y


def cloud_to_gltf(points, transform):
    """Scan points (cloud frame) -> the roof GLB's frame: scale, level, roof origin at 0, y-up."""
    p = np.asarray(points, float) * float(transform["unit_scale"])
    levelled = p @ np.asarray(transform["R_level"], float).T
    return (levelled - np.asarray(transform["roof_origin"], float)) @ Z_TO_Y.T


def plan_ring(ring_gltf):
    """glTF ring [[x, y, z], ...] -> plan [[x, y], ...] (plan y = -glTF z, as in rooftop_solar.json)."""
    r = np.asarray(ring_gltf, float)
    return np.round(np.c_[r[:, 0], -r[:, 2]], 3).tolist()


def crop_to_outline(points, colors, rings_plan, margin=3.0):
    """Points whose plan position lies in the outline's bounding box grown by `margin` metres."""
    pts = np.asarray(points, float)
    allp = np.vstack([np.asarray(r, float) for r in rings_plan])
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    x, y = pts[:, 0], -pts[:, 2]
    keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
    return pts[keep], (None if colors is None else np.asarray(colors)[keep])


def thin(points, colors, n, seed=0):
    """At most n points, a repeatable random subset (order kept)."""
    pts = np.asarray(points)
    if len(pts) <= n:
        return pts, colors
    idx = np.sort(np.random.default_rng(seed).choice(len(pts), n, replace=False))
    return pts[idx], (None if colors is None else np.asarray(colors)[idx])


def height_colors(points):
    """Grey ramp by height for scans without colour: low dark, high light (0..1 floats)."""
    y = np.asarray(points, float)[:, 1]
    span = max(float(np.ptp(y)), 1e-9)
    g = 0.35 + 0.55 * (y - y.min()) / span
    return np.c_[g, g, g * 1.04].clip(0, 1)


def write_points_ply(path, points, colors):
    """Binary little-endian PLY: float32 x y z + uchar red green blue (three.js PLYLoader reads it)."""
    pts = np.asarray(points, np.float32)
    rgb = (np.asarray(colors, float).clip(0, 1) * 255).round().astype(np.uint8)
    if len(pts) != len(rgb):
        raise ValueError("points and colors differ in length")
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = pts, rgb
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + rec.tobytes())
    return path


def read_points_ply(path):
    """Inverse of write_points_ply: (points float32 (N,3), colors uint8 (N,3))."""
    b = Path(path).read_bytes()
    end = b.index(b"end_header\n") + len(b"end_header\n")
    n = int(next(l for l in b[:end].decode().splitlines() if l.startswith("element vertex")).split()[2])
    rec = np.frombuffer(b[end:], dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=n)
    return rec["p"].copy(), rec["c"].copy()


def sidecar(summary, anchor, scan_points=0):
    """What vr.html needs besides the GLB: numbers for the caption, outline/keep-outs in plan metres."""
    outline = [plan_ring(r) for r in anchor["outline"]]
    ys = [p[1] for r in anchor["outline"] for p in r]
    return {
        "phase": 3, "site": summary["site"],
        "panels": int(summary["panels"]), "kwp": float(summary["kwp"]),
        "outline_area_m2": round(float(summary["outline_area_m2"]), 1),
        "usable_area_m2": round(float(summary["usable_area_m2"]), 1),
        "roof_y": round(float(np.mean(ys)), 3) if ys else 0.0,
        "outline": outline,
        "keepouts": [plan_ring(r) for r in anchor.get("keepouts", [])],
        "scan_points": int(scan_points),
    }


def build(out_dir, models_dir, name, scan=None, max_points=120_000, margin=3.0):
    """Copy the design GLB, write the sidecar JSON and (if scan=(points, colors|None)) the scan PLY."""
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    summary = json.loads((out_dir / "panels_summary.json").read_text())
    anchor = json.loads((out_dir / "panels_anchor.json").read_text())
    models_dir.mkdir(parents=True, exist_ok=True)
    files = {"glb": shutil.copyfile(out_dir / "roof_panels.glb", models_dir / f"{name}.glb")}
    n = 0
    if scan is not None:
        pts = cloud_to_gltf(scan[0], summary["transform"])
        rings = [plan_ring(r) for r in anchor["outline"]]
        pts, col = crop_to_outline(pts, scan[1], rings, margin)
        pts, col = thin(pts, col, max_points)
        if col is None or len(col) == 0 or np.ptp(np.asarray(col, float)) < 1e-6:
            col = height_colors(pts) if len(pts) else np.zeros((0, 3))
        files["scan"] = write_points_ply(models_dir / f"{name}_scan.ply", pts, col)
        n = len(pts)
    data = sidecar(summary, anchor, n)
    if n:
        data["scan"] = f"{name}_scan.ply"
    files["json"] = models_dir / f"{name}.json"
    files["json"].write_text(json.dumps(data, separators=(",", ":")))
    return files, data
