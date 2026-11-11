"""Phase 2: auto-place solar panels on the roof CAD and export STEP and GLB (view with preview/vr.html).

  python scripts/phase2_solar.py [data/rooftop_01/measurements.json] [--tilt 15] [--facing-deg 90]
  python scripts/phase2_solar.py --dxf data/sample_roof/roof.dxf [--layers layers.json] [--detailed]

With --dxf the roof, parapet and structures come from the drawing (docs/phase2_dxf.md);
latitude and facing come from its SITE_INFO and NORTH layers unless given.
--detailed also writes rooftop_detailed.step/.glb/.json (rooftop/details.py): parapet with
coping, sloped screed and drains, rooms with doors, ribbed tanks on stands, pipes, the
building below, panel mounting with walkways, inverter and cable tray.
"""
import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import details, phase2  # noqa: E402
from rooftop.cad import RooftopSpec  # noqa: E402
from rooftop.dxf_read import read_dxf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default=str(ROOT / "data/rooftop_01/measurements.json"))
    ap.add_argument("--dxf", default=None, help="read the roof from this drawing instead of a spec")
    ap.add_argument("--layers", default=None, help="layers.json mapping for a drawing's own layer names")
    ap.add_argument("--units", default=None, help="drawing unit when $INSUNITS is unset: mm, cm, m, in, ft")
    ap.add_argument("--out", default=None, help="default outputs/<site>")
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--latitude", type=float, default=None, help="default 19, or the drawing's LAT")
    ap.add_argument("--facing-deg", type=float, default=None, help="default 90, or the drawing's equator side")
    ap.add_argument("--setback", type=float, default=0.5)
    ap.add_argument("--notch-length", type=float, default=None, help="mm; L-shaped building (default 3500)")
    ap.add_argument("--notch-width", type=float, default=None, help="mm (default 2500)")
    ap.add_argument("--box", action="store_true", help="plain rectangle, no rooftop structures")
    ap.add_argument("--detailed", action="store_true", help="also write the detailed model (needs --dxf)")
    ap.add_argument("--plain", action="store_true", help="detailed GLB without textures (flat colours)")
    ap.add_argument("--texture-px", type=int, default=256, help="texture pixels per metre (tile = 1 m)")
    ap.add_argument("--walkway", type=float, default=0.6, help="m between rows in the detailed model")
    ap.add_argument("--publish", action="store_true", help="also copy the GLB + layout JSON to preview/models")
    a = ap.parse_args(argv)
    if a.dxf:
        dwg = read_dxf(a.dxf, a.layers, a.units)
        spec, out = dwg.spec, Path(a.out or ROOT / "outputs" / Path(a.dxf).resolve().parent.name)
        feats = [] if a.box else dwg.features
        lat = a.latitude if a.latitude is not None else (dwg.latitude if dwg.latitude is not None else 19.0)
        facing = a.facing_deg if a.facing_deg is not None else dwg.facing_deg()
    else:
        spec, out = RooftopSpec.from_json(a.spec), Path(a.out or ROOT / "outputs/rooftop_01")
        if not a.box:
            spec = replace(spec, notch_length=a.notch_length or spec.notch_length or 3500,
                           notch_width=a.notch_width or spec.notch_width or 2500)
        feats = [] if a.box else phase2.default_features(spec)
        lat = 19.0 if a.latitude is None else a.latitude
        facing = 90.0 if a.facing_deg is None else a.facing_deg
    lay, corners = phase2.design(spec, a.setback, a.tilt, lat, facing, features=feats)
    step = phase2.export_step(out / "rooftop_solar.step", spec, corners, features=feats)
    glb = phase2.write_glb(out / "rooftop_solar.glb", spec, corners, feats)
    js = phase2.write_layout_json(out / "rooftop_solar.json", spec, feats)
    kw = len(corners) * 0.55
    print(f"{lay.count} panels ({lay.orientation}, tilt {a.tilt:g}), ~{kw:.1f} kW at 550 W, "
          f"{len(feats)} rooftop structures")
    print(step, glb, js, sep="\n")
    files = [glb, js]
    if a.detailed:
        if not a.dxf:
            ap.error("--detailed needs --dxf")
        files += write_detailed(out, dwg, feats, a, lat, facing)
    if a.publish:
        for f in files:
            shutil.copy(f, ROOT / "preview" / "models" / f.name)
    return lay


def write_detailed(out, dwg, feats, a, lat, facing):
    """Detailed model: panels on 0.3 m mounting with a walkway between rows."""
    lay, corners = phase2.design(dwg.spec, a.setback, a.tilt, lat, facing, standoff=0.3, features=feats,
                                 walkway=a.walkway)
    floors, floor_h = details.site_floors(dwg.site)
    parts = details.build(dwg.spec, feats, corners, dwg.doors, dwg.drains, dwg.pipes, floors, floor_h)
    step = details.export_step(out / "rooftop_detailed.step", parts, corners)
    from rooftop.textures import texture_set

    tex = None if a.plain else texture_set(details.COLOURS, px=a.texture_px)
    glb = details.write_glb(out / "rooftop_detailed.glb", parts, corners, tex)
    js = details.write_json(out / "rooftop_detailed.json", dwg, feats, lay, corners, lat, facing, floors, floor_h)
    print(f"detailed: {lay.count} panels, {len(parts)} parts, walkway {a.walkway:g} m")
    print(step, glb, js, sep="\n")
    return [glb, js]


if __name__ == "__main__":
    main()
