"""Write the SketchUp-style sample model (a solar pergola) for tagar.html's model picker.

    python scripts/make_sketchup_sample.py [--out preview/models]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rooftop.sketchup_sample import bounds, pergola, write_sample  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "preview" / "models"))
    ap.add_argument("--name", default="sample_sketchup_pergola")
    a = ap.parse_args(argv)
    files = write_sample(a.out, a.name)
    lo, hi = bounds(pergola())
    for f in files.values():
        print(f"wrote {f} ({f.stat().st_size // 1024} KB)")
    print("size (m, x y z):", " x ".join(f"{v:.2f}" for v in hi - lo))
    return files


if __name__ == "__main__":
    main()
