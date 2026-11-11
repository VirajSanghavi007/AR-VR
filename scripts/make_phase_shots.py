"""Take the pictures for the phase pages (preview/phase1.html, phase2.html, phase3.html).

    python scripts/make_phase_shots.py [--out preview/phases] [--only phase2_roof.jpg ...]

Opens tagar.html and vr.html in headless Chrome and saves JPEGs to preview/phases/; also copies the
Phase 3 roof plan (outputs/eth3d/terrace/panels_plan.png from design_panels.py) when it exists.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import phase_shots as ps  # noqa: E402

PLAN = ROOT / "outputs" / "eth3d" / "terrace" / "panels_plan.png"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "preview" / "phases"))
    ap.add_argument("--only", nargs="*", help="shot names to take (default all)")
    ap.add_argument("--plan", default=str(PLAN))
    a = ap.parse_args(argv)
    shots = [s for s in ps.SHOTS if not a.only or s["name"] in a.only]
    paths = []
    if shots:
        chrome = ps.find_chrome()
        if not chrome:
            sys.exit("Chrome not found")
        paths += ps.capture(shots, ROOT / "preview", a.out, chrome)
    if (not a.only or "phase3_plan.jpg" in a.only) and Path(a.plan).is_file():
        paths.append(ps.to_jpeg(a.plan, Path(a.out) / "phase3_plan.jpg"))
    for p in paths:
        print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
    return paths


if __name__ == "__main__":
    main()
