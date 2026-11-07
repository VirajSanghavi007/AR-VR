"""Cut one room out of an IFC building model and write it as a GLB for the AR page's Room model card.

  python scripts/ifc_room_to_glb.py data/duplex/Duplex_A_20110907.ifc --room A102 [--out outputs/duplex] [--publish]

--room is the room number or name (as the IFC's IfcSpace Name or LongName). --publish also copies the
GLB to preview/models/<stem>.glb so the AR page can load it.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import ifc_room  # noqa: E402

FACE = {"back": "-Z", "right": "+X", "front": "+Z", "left": "-X"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc")
    ap.add_argument("--room", required=True)
    ap.add_argument("--margin", type=float, default=0.02, help="wall/floor/ceiling skin kept around the room, m")
    ap.add_argument("--out", default=str(ROOT / "outputs/ifc_room"))
    ap.add_argument("--name", help="GLB file stem (default: <ifc stem>_<room>)")
    ap.add_argument("--publish", action="store_true", help="also copy the GLB to preview/models")
    a = ap.parse_args(argv)
    if not Path(a.ifc).is_file():
        ap.error(f"no file {a.ifc}")
    try:
        cut = ifc_room.load_room(a.ifc, a.room, a.margin)
    except ValueError as e:
        ap.error(str(e))
    stem = a.name or f"{Path(a.ifc).stem}_{a.room}".replace(" ", "_")
    glb = ifc_room.write_glb(Path(a.out) / f"{stem}.glb", cut)
    print(f"GLB  {glb}")
    if a.publish:
        dst = ROOT / "preview/models" / glb.name
        shutil.copyfile(glb, dst)
        print(f"published {dst}")
    w, d, h = cut.size
    print(f"{cut.name}: inside {w:.2f} x {d:.2f} m, {h:.2f} m high, {len(cut.parts)} parts")
    for kind, side, width in cut.openings:
        print(f"  {kind} {width:.2f} m wide on the {side} wall (the model's {FACE[side]} side)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
