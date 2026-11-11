"""Write the sample tag photos tags.html's "Use sample photos" button checks.

    python scripts/make_sample_tag_photos.py [--out preview/samples]

Four 160 mm 36h11 tags (ids 0-3) on a table, six phone views (one with id 3 cut off), plus tag_photos.json.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.sample_photos import write_sample_photos  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "preview" / "samples"))
    a = ap.parse_args(argv)
    paths, index = write_sample_photos(a.out)
    for p, ph in zip(paths, index["photos"]):
        print(f"wrote {p} ({p.stat().st_size // 1024} KB), tags in frame: {ph['ids']}")
    return paths, index


if __name__ == "__main__":
    main()
