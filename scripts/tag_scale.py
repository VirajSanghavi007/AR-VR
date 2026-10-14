"""Metric scale of a COLMAP reconstruction from AprilTags of known size on the roof.

Usage:
  python scripts/tag_scale.py rooftop_01 --tag-size 0.16
  python scripts/tag_scale.py rooftop_01 --tag-size 0.16 --model data/rooftop_01/colmap/sparse/0
Writes outputs/<site>/tag_scale.json; then:
  python scripts/design_panels.py rooftop_01 --unit-scale tags
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop import tags as T  # noqa: E402


def find_model(site_dir: Path):
    """First folder under colmap/ that holds a sparse model (cameras + images, .txt or .bin)."""
    for d in sorted([site_dir / "colmap", site_dir]):
        if not d.exists():
            continue
        for f in sorted(d.rglob("images.*")):
            if f.suffix in (".txt", ".bin") and (f.parent / f"cameras{f.suffix}").exists() and "dense" not in f.parts:
                return f.parent
    raise FileNotFoundError(f"no COLMAP sparse model under {site_dir}")


def run(site, tag_size, model=None, images=None, root=ROOT):
    site_dir = root / "data" / site
    model = Path(model) if model else find_model(site_dir)
    images = Path(images) if images else site_dir / "images"
    print(f"[1/3] model {model}, images {images}")
    cams, found = T.detect_in_model(model, images)
    print(f"[2/3] triangulating {len(found)} tag ids")
    tags3d = T.tag_corners_3d(cams, found)
    res = T.scale_from_tags(tags3d, tag_size)
    res["tag_size_m"] = tag_size
    res["model"] = str(model)
    res["tag_distances_m"] = T.tag_distances(tags3d, res["metres_per_unit"])
    res["tags"] = {str(k): v for k, v in res["tags"].items()}
    out = root / "outputs" / site
    out.mkdir(parents=True, exist_ok=True)
    (out / "tag_scale.json").write_text(json.dumps(res, indent=2))
    spread = f", tags agree within {res['spread_pct']:.2f} %" if res["spread_pct"] is not None else " (only 1 tag: no cross-check)"
    print(f"[3/3] {res['metres_per_unit']:.6f} m per model unit from {res['n_tags']} tag(s){spread}")
    for pair, d in res["tag_distances_m"].items():
        print(f"      tag {pair}: {d:.3f} m apart (check with a tape)")
    return res


def load_scale(site, root=ROOT):
    p = root / "outputs" / site / "tag_scale.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/tag_scale.py {site} --tag-size <m> first")
    return json.loads(p.read_text())["metres_per_unit"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--tag-size", type=float, required=True, help="black square edge, metres (0.16 for 160 mm)")
    ap.add_argument("--model", default=None, help="COLMAP sparse model folder")
    ap.add_argument("--images", default=None, help="image root that COLMAP image names are relative to")
    a = ap.parse_args(argv)
    return run(a.site, a.tag_size, a.model, a.images)


if __name__ == "__main__":
    main()
