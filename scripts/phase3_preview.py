"""Phase 3 model for the phone viewer: scan-made roof CAD + panels + the laser scan as a LiDAR layer.

Usage:
  python scripts/design_panels.py eth3d/terrace          (once: makes outputs/eth3d/terrace/*)
  python scripts/phase3_preview.py                       (writes preview/models/phase3_terrace.*)
  python scripts/phase3_preview.py eth3d/terrace --name phase3_terrace --no-scan
Then open preview/vr.html?phase=3 (Orbit, Cardboard VR, AR on a table or at full size).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import phase3  # noqa: E402


def load_scan(scan_dir, voxel):
    """ETH3D laser scans merged into the image frame (rooftop.validate.load_ground_truth)."""
    from rooftop.validate import load_ground_truth
    gt = load_ground_truth(scan_dir, voxel=voxel)
    cols = np.asarray(gt.colors) if gt.has_colors() else None
    return np.asarray(gt.points), cols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site", nargs="?", default="eth3d/terrace")
    ap.add_argument("--name", default="phase3_terrace")
    ap.add_argument("--models", default=str(ROOT / "preview" / "models"))
    ap.add_argument("--no-scan", action="store_true", help="skip the laser-scan layer")
    ap.add_argument("--voxel", type=float, default=0.04, help="scan downsample before thinning, metres")
    ap.add_argument("--max-points", type=int, default=120_000)
    a = ap.parse_args(argv)
    t0 = time.time()
    scan = None
    scan_dir = ROOT / "data" / a.site / "dslr_scan_eval"
    if not a.no_scan and scan_dir.is_dir():
        print(f"[1/2] loading laser scan {scan_dir} (voxel {a.voxel} m)", flush=True)
        scan = load_scan(scan_dir, a.voxel)
        print(f"      {len(scan[0])} points, {time.time() - t0:.0f}s elapsed", flush=True)
    files, data = phase3.build(ROOT / "outputs" / a.site, a.models, a.name, scan, a.max_points)
    print(f"[2/2] wrote {', '.join(str(Path(f).name) for f in files.values())}: "
          f"{data['panels']} panels, {data['kwp']} kWp, {data['scan_points']} scan points, {time.time() - t0:.0f}s")
    return files


if __name__ == "__main__":
    main()
