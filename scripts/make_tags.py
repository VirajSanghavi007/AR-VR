"""Printable AprilTag sheet (36h11), one tag per A4 page at an exact size.

Usage:
  python scripts/make_tags.py                      # ids 0-3, 160 mm -> outputs/apriltags_36h11_160mm.pdf
  python scripts/make_tags.py --size-mm 120 --ids 0 1 2 3 4 5
Print at 100 % ("Actual size", not "Fit to page") and check the 100 mm line with a ruler.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.tags import write_tag_sheet  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mm", type=float, default=160.0, help="edge of the black square")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or ROOT / "outputs" / f"apriltags_36h11_{a.size_mm:g}mm.pdf"
    path, _ = write_tag_sheet(out, a.ids, a.size_mm)
    print(f"wrote {path} ({len(a.ids)} pages, black square {a.size_mm:g} mm)")
    return path


if __name__ == "__main__":
    main()
