"""Dense cloud -> level -> roof patch -> outline/keep-outs -> panel layout -> STEP/GLB/PLY/PNG.

Usage:
  python scripts/design_panels.py eth3d/terrace
  python scripts/design_panels.py eth3d/terrace --tilt 15 --latitude 47.4 --facing-deg -90
  python scripts/design_panels.py south_building --unit-scale 2.1 --select highest
Cloud is assumed metric unless --unit-scale (metres per cloud unit) is given;
--unit-scale tags reads the AprilTag scale from scripts/tag_scale.py.
--facing-deg is the direction the panels face (towards the equator), measured in the
roof's 2D frame (0 = +u axis). Without a north reference in the scan, set it by eye
from panels_plan.png or from a compass bearing taken on site.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from inspect_cloud import find_fused  # noqa: E402
from rooftop import layout as L  # noqa: E402
from rooftop import panels as P  # noqa: E402
from rooftop.ar_export import write_anchor, write_glb  # noqa: E402
from rooftop.cloud import clean_cloud, extract_planes, split_clusters  # noqa: E402
from rooftop.geometry import PlaneFrame, estimate_up, level_transform, read_camera_centres  # noqa: E402


def find_cameras(site_dir: Path):
    hits = sorted(site_dir.rglob("images.txt"))
    return read_camera_centres(hits[0]) if hits else None


def roof_candidates(planes, max_tilt, eps, min_points):
    """Near-horizontal plane patches, largest first: (frame, points) pairs."""
    out = []
    for pl in planes:
        frame = PlaneFrame.from_points(pl.points, pl.normal)
        if frame.tilt_deg() > max_tilt:
            continue
        for patch in split_clusters(pl.points, eps=eps, min_points=min_points):
            out.append((PlaneFrame.from_points(patch, pl.normal), patch))
    return sorted(out, key=lambda fp: len(fp[1]), reverse=True)


def choose_roof(cands, select="largest", roof_index=None, min_share=0.1):
    """Index into cands. 'highest' = highest patch with >= min_share of the largest's points
    (roofs sit above the ground, which is often the biggest flat patch)."""
    if roof_index is not None:
        if not 0 <= roof_index < len(cands):
            raise IndexError(f"--roof-index {roof_index} out of range (0..{len(cands) - 1})")
        return roof_index
    if select == "largest":
        return 0
    if select == "highest":
        big = len(cands[0][1])
        ok = [i for i, (_, p) in enumerate(cands) if len(p) >= min_share * big]
        return max(ok, key=lambda i: cands[i][0].origin[2])
    raise ValueError(f"Unknown select mode {select!r}")


def run(site, unit_scale=1.0, voxel=0.05, roof_index=None, select="largest", max_tilt=15.0,
        panel_w=1.134, panel_h=1.722, gap=0.02, setback=0.5, clearance=0.3,
        watt_per_panel=400, hull_ratio=0.15, tilt=0.0, facing_deg=None, latitude=None,
        standoff=0.1, return_objects=False):
    site_dir = ROOT / "data" / site
    out = ROOT / "outputs" / site
    out.mkdir(parents=True, exist_ok=True)
    o3d.utility.random.seed(0)  # RANSAC repeatable run to run

    pcd = o3d.io.read_point_cloud(str(find_fused(site_dir)))
    pcd.scale(unit_scale, center=np.zeros(3))
    pcd = clean_cloud(pcd, voxel_size=voxel)
    pts = np.asarray(pcd.points)
    print(f"[1/6] cloud: {len(pts)} pts after cleaning")

    cams = find_cameras(site_dir)
    ref = cams * unit_scale if cams is not None else pts
    up = estimate_up(extract_planes(pcd, distance_threshold=voxel * 1.5, max_planes=3), ref)
    R = level_transform(up)
    pcd.rotate(R, center=np.zeros(3))
    pts = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"[2/6] levelled: up was {np.round(up, 3)} ({'cameras' if cams is not None else 'cloud'} as reference)")

    planes = extract_planes(pcd, distance_threshold=voxel * 1.5, max_planes=8)
    cands = roof_candidates(planes, max_tilt, eps=voxel * 3, min_points=10)
    if not cands:
        raise RuntimeError(f"No plane within {max_tilt} deg of horizontal")
    print("      candidates: idx  points  height_m  tilt_deg")
    for i, (f, p) in enumerate(cands[:8]):
        print(f"                  {i:3d} {len(p):7d} {f.origin[2]:9.2f} {f.tilt_deg():9.1f}")
    idx = choose_roof(cands, select, roof_index)
    frame, patch = cands[idx]
    print(f"[3/6] roof patch #{idx} ({select if roof_index is None else 'manual'}): "
          f"{len(patch)} pts, tilt {frame.tilt_deg():.1f} deg")

    uv_roof, _ = frame.to_2d(patch)
    outline = L.roof_outline(uv_roof, ratio=hull_ratio)
    uv_all, h_all = frame.to_2d(pts)
    obstacles = L.keepouts(uv_all, h_all, outline, clearance=clearance,
                           cell=max(voxel * 2, 0.1), min_points=5)
    usable = L.usable_area(outline, setback=setback, obstacles=obstacles)
    sun_alt = L.winter_sun_altitude(latitude) if tilt > 0 else None
    if tilt > 0 and facing_deg is None:
        raise ValueError("--tilt needs --facing-deg (direction panels face, towards the equator)")
    lay = L.layout_panels(usable, panel_w=panel_w, panel_h=panel_h, gap=gap,
                          tilt_deg=tilt, facing_deg=facing_deg, sun_alt_deg=sun_alt)
    print(f"[4/6] outline {outline.area:.1f} m2, usable {usable.area:.1f} m2, "
          f"{lay.count} panels ({lay.orientation}, tilt {tilt:g} deg"
          + (f", row gap {lay.row_gap:.2f} m @ sun {sun_alt:.1f} deg" if tilt > 0 else "") + ")")

    corners = P.panel_corners_3d(frame, lay.panels, standoff=standoff,
                                 facing=lay.facing if tilt > 0 else None, rise=lay.rise)
    outline3d = frame.to_3d(P.polygon_ring(outline))
    kwp = lay.count * watt_per_panel / 1000
    print("[5/6] exporting")
    files = {
        "overlay": P.write_overlay(out / "panels_overlay.ply", pts, rgb, corners, outline3d),
        "plan": P.plot_plan(out / "panels_plan.png", uv_roof, outline, usable, obstacles, lay.panels,
                            title=f"{site}: {lay.count} x {watt_per_panel} W = {kwp:.1f} kWp"
                                  + (f", {tilt:g} deg tilt" if tilt > 0 else "")),
        "step": P.export_step(out / "roof_panels.step", frame, outline, corners, obstacles),
        "glb": write_glb(out / "roof_panels.glb", frame, outline, corners),
        "glb_ar": write_glb(out / "panels_ar.glb", frame, outline, corners, include_roof=False),
        "anchor": write_anchor(out / "panels_anchor.json", frame, outline, obstacles),
    }
    summary = {
        "site": site, "roof_index": idx, "roof_tilt_deg": round(frame.tilt_deg(), 2),
        "outline_area_m2": round(outline.area, 2), "usable_area_m2": round(usable.area, 2),
        "panels": lay.count, "orientation": lay.orientation, "panel_tilt_deg": tilt,
        "row_gap_m": round(lay.row_gap, 3), "panel_size_m": [panel_w, panel_h], "kwp": round(kwp, 2),
        "transform": {  # cloud (after unit_scale) -> levelled frame; roof frame within it
            "unit_scale": unit_scale, "R_level": R.round(9).tolist(),
            "roof_origin": frame.origin.tolist(), "roof_u": frame.u.tolist(),
            "roof_v": frame.v.tolist(), "roof_n": frame.n.tolist(),
        },
        "files": {k: str(v) for k, v in files.items()},
    }
    (out / "panels_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[6/6] wrote {', '.join(p.name for p in files.values())}")
    if return_objects:
        return summary, {"R": R, "frame": frame, "outline": outline, "usable": usable,
                         "layout": lay, "corners": corners, "obstacles": obstacles}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--unit-scale", default="1.0",
                    help="metres per cloud unit, or 'tags' to use outputs/<site>/tag_scale.json")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--select", choices=["largest", "highest"], default="largest")
    ap.add_argument("--roof-index", type=int, default=None, help="override --select")
    ap.add_argument("--max-tilt", type=float, default=15.0)
    ap.add_argument("--setback", type=float, default=0.5)
    ap.add_argument("--watt", type=int, default=400)
    ap.add_argument("--tilt", type=float, default=0.0, help="panel tilt on the roof, deg")
    ap.add_argument("--facing-deg", type=float, default=None)
    ap.add_argument("--latitude", type=float, default=None)
    a = ap.parse_args(argv)
    if a.tilt > 0 and a.latitude is None:
        ap.error("--tilt needs --latitude (for winter-noon row spacing)")
    if a.unit_scale == "tags":
        from tag_scale import load_scale
        unit_scale = load_scale(a.site, ROOT)
        print(f"unit scale from AprilTags: {unit_scale:.6f} m per unit")
    else:
        unit_scale = float(a.unit_scale)
    s = run(a.site, unit_scale=unit_scale, voxel=a.voxel, roof_index=a.roof_index,
            select=a.select, max_tilt=a.max_tilt, setback=a.setback, watt_per_panel=a.watt,
            tilt=a.tilt, facing_deg=a.facing_deg, latitude=a.latitude)
    print(json.dumps({k: v for k, v in s.items() if k not in ("files", "transform")}, indent=2))


if __name__ == "__main__":
    main()
