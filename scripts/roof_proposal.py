"""M6 one-page customer proposal: the four roof stills (scripts/roof_stills.py) and the system
figures (panel count, peak kW, yearly kWh) on one A4 page, written as Typst and compiled to PDF.

    python scripts/roof_stills.py                 # outputs/sample_roof/stills/*.png first
    python scripts/roof_proposal.py               # outputs/sample_roof/proposal.pdf
    python scripts/roof_proposal.py --sun-hours 5.5 --title "Sharma residence" --out somewhere.pdf

Energy is the same estimate as roof.html (preview/roofview.js yearly): panels x watts x sun hours
x derate per day, times 365. The .typ is written next to the stills folder (Typst only reads files
under its root) and kept, so it can be edited and recompiled by hand.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = (("aerial", "From above"), ("eye", "From the street"), ("front", "Front"), ("closeup", "Panel close-up"))


def figures(info, sun_hours=5.0, derate=0.8):
    """{panels, watts, peak_kw, kwh_day, kwh_year} from the model sidecar (rooftop_detailed.json)."""
    n = int(info["panels"])
    watts = info["peak_kw"] * 1000 / n if n else 0.0
    kwh_day = n * watts * sun_hours * derate / 1000
    return {"panels": n, "watts": watts, "peak_kw": n * watts / 1000, "kwh_day": kwh_day, "kwh_year": kwh_day * 365}


def typst_str(s):
    """A Typst string literal."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_typ(info, stills_rel, title="Rooftop solar proposal", sun_hours=5.0, derate=0.8):
    """Typst source for the page; stills_rel = folder of the PNGs relative to the .typ file."""
    f = figures(info, sun_hours, derate)
    imgs = ",\n  ".join(f"figure(image({typst_str(f'{stills_rel}/{k}.png')}, height: 6.3cm), caption: {typst_str(c)})"
                        for k, c in SHOTS)
    rows = [("Panels", f"{f['panels']} x {f['watts']:.0f} W"),
            ("System size", f"{f['peak_kw']:.2f} kWp"),
            ("Energy per year", f"about {round(f['kwh_year']):,} kWh"),
            ("Energy per day", f"about {f['kwh_day']:.0f} kWh"),
            ("Panel tilt", f"{info.get('tilt_deg', 0):g} degrees"),
            ("Site", f"{info.get('latitude', 0):.2f} N, {info.get('longitude', 0):.2f} E")]
    table = ",\n  ".join(f"[{a}], [*{b}*]" for a, b in rows)
    return f"""#set page(paper: "a4", flipped: true, margin: 1.2cm)
#set text(font: "Libertinus Serif", size: 10.5pt)
#show figure.caption: set text(size: 9pt)
#set figure(gap: 0.4em, numbering: none)

#text(size: 20pt, weight: "bold", {typst_str(title)})

#grid(columns: (1fr, 1fr), gutter: 1.2em,
  table(columns: 2, stroke: none, column-gutter: 1.2em, inset: (x: 0pt, y: 3pt),
  {table}),
  [The panels follow the roof drawing, keep clear of the stair room, store, water tanks and their
   winter shadows, and leave walkways to the drains. Energy assumes {sun_hours:g} peak sun hours a day
   and {round((1 - derate) * 100)} % losses (wiring, heat, inverter, dust); the real figure varies
   with weather and shading.]
)

#v(0.4em)
#grid(columns: (1fr, 1fr), gutter: 0.6em,
  {imgs}
)
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=str(ROOT / "preview" / "models" / "rooftop_detailed.json"))
    ap.add_argument("--stills", default=str(ROOT / "outputs" / "sample_roof" / "stills"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sample_roof" / "proposal.pdf"))
    ap.add_argument("--title", default="Rooftop solar proposal")
    ap.add_argument("--sun-hours", type=float, default=5.0)
    ap.add_argument("--derate", type=float, default=0.8)
    a = ap.parse_args(argv)
    stills = Path(a.stills).resolve()
    missing = [k for k, _ in SHOTS if not (stills / f"{k}.png").is_file()]
    if missing:
        ap.error(f"missing stills {missing} in {stills}; run scripts/roof_stills.py first")
    typst = shutil.which("typst")
    if not typst:
        ap.error("typst not found on PATH")
    info = json.loads(Path(a.json).read_text())
    typ = stills.parent / (Path(a.out).stem + ".typ")
    typ.write_text(build_typ(info, stills.name, a.title, a.sun_hours, a.derate), encoding="utf-8")
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([typst, "compile", "--root", str(stills.parent), str(typ), str(out)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 1
    f = figures(info, a.sun_hours, a.derate)
    print(f"{f['panels']} panels, {f['peak_kw']:.2f} kWp, about {round(f['kwh_year']):,} kWh a year")
    print(f"wrote {out} (source {typ})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
