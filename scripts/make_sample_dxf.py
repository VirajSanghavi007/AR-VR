"""Phase 2, M1: write the sample rooftop drawing and its true values.

  python scripts/make_sample_dxf.py [--out data/sample_roof]

Writes roof.dxf (mm, layer rules in docs/phase2_dxf.md) and measurements.json.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import sample_dxf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/sample_roof"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    dxf = sample_dxf.write_sample(out / "roof.dxf")
    js = sample_dxf.write_measurements(out / "measurements.json")
    print(dxf, js, sep="\n")
    return dxf, js


if __name__ == "__main__":
    main()
