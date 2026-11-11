"""M7: check the detailed 3D model against its drawing's DIMENSIONS and measurements.json.

  python scripts/check_dxf_model.py --dxf data/sample_roof/roof.dxf
      [--model outputs/sample_roof/rooftop_detailed.step] [--measurements data/sample_roof/measurements.json]
      [--tol 5] [--report outputs/sample_roof/model_check.md]

Prints a markdown table of errors in mm, pass/fail at the tolerance; exit code 1 when any check fails.
Make the model first with scripts/phase2_solar.py --dxf <drawing> --detailed.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import check_model as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--model", default=None, help="default outputs/<site>/rooftop_detailed.step")
    ap.add_argument("--measurements", default=None, help="default measurements.json next to the drawing")
    ap.add_argument("--tol", type=float, default=C.TOL, help="mm")
    ap.add_argument("--report", default=None, help="also write the table to this .md file")
    a = ap.parse_args(argv)
    import json

    from rooftop.dxf_read import read_dxf

    dxf = Path(a.dxf)
    model_path = Path(a.model or ROOT / "outputs" / dxf.resolve().parent.name / "rooftop_detailed.step")
    meas_path = Path(a.measurements or dxf.parent / "measurements.json")
    if not model_path.exists():
        ap.error(f"no model at {model_path}; run scripts/phase2_solar.py --dxf {a.dxf} --detailed")
    dwg = read_dxf(dxf, a.layers)
    model = C.groups_from_step(model_path)
    rows = C.dimension_rows(model, dwg.dimensions, tol=a.tol)
    if meas_path.exists():
        rows += C.measurement_rows(model, json.loads(meas_path.read_text()), tol=a.tol)
    text = f"# Model check: {model_path.name} vs {dxf.name}\n\n" + C.table(rows)
    print(text)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text + "\n", encoding="utf-8")
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
