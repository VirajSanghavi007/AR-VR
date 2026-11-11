"""Turn a laser-scanned room (ETH3D dslr_scan_eval) into a simple mesh and write it as SketchUp exports.

    python scripts/scan_room_to_sketchup.py [eth3d/office] [--name office_scan_room] [--publish]

Writes outputs/<site>/<name>.dae (SketchUp COLLADA, inches, Z up), .obj + .mtl, .json (size, faces)
and <name>_cloud.ply (the levelled, cropped cloud). Floor and ceiling are planes fitted from the scan
(the scanner barely sees the floor); --scanned-floor keeps the scanned faces instead. --publish copies the .dae/.obj/.mtl to
preview/models for tagar.html's room card. In SketchUp: File > Import > the .dae, then save as .skp.
Data: ETH3D "office" (https://www.eth3d.net/data/office_dslr_scan_eval.7z, CC BY-NC-SA 4.0).
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.scan_room import scan_to_room  # noqa: E402

SOURCE = ("ETH3D high-res multi-view benchmark, scene 'office', laser scan (Schoeps et al., CVPR 2017), "
          "https://www.eth3d.net, CC BY-NC-SA 4.0")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site", nargs="?", default="eth3d/office")
    ap.add_argument("--name", default="office_scan_room")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="default outputs/<site>")
    ap.add_argument("--voxel", type=float, default=0.02, help="scan downsample, metres")
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    ap.add_argument("--triangles", type=int, default=60_000)
    ap.add_argument("--scanned-floor", action="store_true",
                    help="keep the scanned floor/ceiling faces instead of the fitted planes")
    ap.add_argument("--publish", action="store_true", help="also copy .dae/.obj/.mtl to preview/models")
    a = ap.parse_args(argv)
    scan_dir = Path(a.data) / a.site / "dslr_scan_eval"
    if not (scan_dir / "scan_alignment.mlp").is_file():
        sys.exit(f"No scans at {scan_dir}: download {a.site.split('/')[-1]}_dslr_scan_eval.7z from "
                 "https://www.eth3d.net/datasets and extract it there")
    out = Path(a.out) if a.out else ROOT / "outputs" / a.site
    files, info = scan_to_room(scan_dir, out, a.name, a.voxel, a.depth, a.triangles, SOURCE,
                               planes=not a.scanned_floor)
    for f in files.values():
        print(f"wrote {f} ({f.stat().st_size // 1024} KB)")
    if a.publish:
        for k in ("dae", "obj", "mtl"):
            shutil.copy(files[k], ROOT / "preview" / "models" / files[k].name)
        print("copied to preview/models")
    return files, info


if __name__ == "__main__":
    main()
