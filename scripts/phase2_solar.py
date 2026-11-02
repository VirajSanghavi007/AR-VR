"""Phase 2: auto-place solar panels on the roof CAD and export STEP and GLB (view with preview/vr.html).

  python scripts/phase2_solar.py [data/rooftop_01/measurements.json] [--tilt 15] [--facing-deg 90]
"""
import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import phase2  # noqa: E402
from rooftop.cad import RooftopSpec  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default=str(ROOT / "data/rooftop_01/measurements.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/rooftop_01"))
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--latitude", type=float, default=19.0)
    ap.add_argument("--facing-deg", type=float, default=90.0)
    ap.add_argument("--setback", type=float, default=0.5)
    ap.add_argument("--notch-length", type=float, default=None, help="mm; L-shaped building (default 3500)")
    ap.add_argument("--notch-width", type=float, default=None, help="mm (default 2500)")
    ap.add_argument("--box", action="store_true", help="plain rectangle, no rooftop structures")
    ap.add_argument("--publish", action="store_true", help="also copy the GLB + layout JSON to preview/models")
    a = ap.parse_args(argv)
    spec, out = RooftopSpec.from_json(a.spec), Path(a.out)
    if not a.box:
        spec = replace(spec, notch_length=a.notch_length or spec.notch_length or 3500,
                       notch_width=a.notch_width or spec.notch_width or 2500)
    feats = [] if a.box else phase2.default_features(spec)
    lay, corners = phase2.design(spec, a.setback, a.tilt, a.latitude, a.facing_deg, features=feats)
    step = phase2.export_step(out / "rooftop_solar.step", spec, corners, features=feats)
    glb = phase2.write_glb(out / "rooftop_solar.glb", spec, corners, feats)
    js = phase2.write_layout_json(out / "rooftop_solar.json", spec, feats)
    kw = len(corners) * 0.55
    print(f"{lay.count} panels ({lay.orientation}, tilt {a.tilt:g}), ~{kw:.1f} kW at 550 W, "
          f"{len(feats)} rooftop structures")
    print(step, glb, js, sep="\n")
    if a.publish:
        for f in (glb, js):
            shutil.copy(f, ROOT / "preview" / "models" / f.name)
    return lay


if __name__ == "__main__":
    main()
