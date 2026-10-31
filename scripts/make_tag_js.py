"""Write preview/tag36h11.js: the AprilTag 36h11 code list for the browser tag checker.

Usage:
  python scripts/make_tag_js.py                     # -> preview/tag36h11.js
  python scripts/make_tag_js.py --max-errors 3 --out some/other.js
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.tags import write_js_dictionary  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "preview" / "tag36h11.js")
    ap.add_argument("--max-errors", type=int, default=4, help="bit errors corrected per tag (0-5)")
    a = ap.parse_args(argv)
    path, n = write_js_dictionary(a.out, a.max_errors)
    print(f"wrote {path} ({n} tag ids)")
    return path


if __name__ == "__main__":
    main()
