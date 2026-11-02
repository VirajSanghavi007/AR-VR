"""Render a STEP file as a 3D PNG: python scripts/render_cad.py [step] [png]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.render import render_png  # noqa: E402

if __name__ == "__main__":
    step = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/rooftop_01/rooftop.step"
    png = Path(sys.argv[2]) if len(sys.argv) > 2 else step.with_name("rooftop_3d.png")
    print(render_png(step, png))
