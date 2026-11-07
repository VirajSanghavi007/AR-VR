"""Build the sample room CAD (STEP + GLB) for the AR page's Room model card.

  python scripts/make_room_cad.py [--width 4000 --depth 3500 --height 2800] [--out outputs/sample_room] [--publish]

Sizes are the room's inside measurements in mm. --publish also copies the GLB to preview/models/sample_room.glb.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import room  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    d = room.RoomSpec()
    ap.add_argument("--width", type=float, default=d.width)
    ap.add_argument("--depth", type=float, default=d.depth)
    ap.add_argument("--height", type=float, default=d.height)
    ap.add_argument("--out", default=str(ROOT / "outputs/sample_room"))
    ap.add_argument("--no-step", action="store_true", help="GLB only (skips CadQuery)")
    ap.add_argument("--publish", action="store_true", help="also copy the GLB to preview/models")
    a = ap.parse_args(argv)
    spec = room.RoomSpec(width=a.width, depth=a.depth, height=a.height, openings=d.openings)
    bad = room.check(spec)
    if bad:
        ap.error("; ".join(bad))
    out = Path(a.out)
    glb = room.write_glb(out / "sample_room.glb", spec)
    print(f"GLB  {glb}")
    if not a.no_step:
        print(f"STEP {room.write_step(out / 'sample_room.step', spec)}")
    if a.publish:
        dst = ROOT / "preview/models/sample_room.glb"
        shutil.copyfile(glb, dst)
        print(f"published {dst}")
    print(f"room inside {spec.width / 1000:.2f} x {spec.depth / 1000:.2f} m, {spec.height / 1000:.2f} m high; "
          f"back wall = the model's -Z side (where the tags go by default)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
