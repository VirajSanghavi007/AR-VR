import json
import struct

import numpy as np
import pytest

from rooftop import phase2
from rooftop.cad import RooftopSpec

SPEC = RooftopSpec(12000, 8000)


def test_usable_area_inside_parapets_and_setback():
    a = phase2.usable_area(SPEC, setback=0.5)
    assert a.bounds == pytest.approx((0.7, 0.7, 11.3, 7.3))


def test_panels_stay_inside_usable_area():
    lay, corners = phase2.design(SPEC, tilt_deg=15)
    area = phase2.usable_area(SPEC)
    assert lay.count > 10 and len(corners) == lay.count
    assert all(area.buffer(1e-6).contains(p) for p in lay.panels)


def test_flat_has_more_panels_than_tilted_and_sits_on_slab():
    flat, c = phase2.design(SPEC, tilt_deg=0)
    tilted, _ = phase2.design(SPEC, tilt_deg=15)
    assert flat.count >= tilted.count
    assert np.allclose(np.array(c)[..., 2], 0.15 + 0.05)


def test_tilted_back_edge_is_higher():
    lay, corners = phase2.design(SPEC, tilt_deg=15)
    z = np.array(corners)[..., 2]
    assert np.allclose(z.min(), 0.20) and z.max() > 0.20 + 0.3


def test_too_small_roof_fits_nothing():
    lay, corners = phase2.design(RooftopSpec(2000, 2000), tilt_deg=0)
    assert lay.count == 0 and corners == []


def test_glb_is_valid_and_has_roof_and_panels(tmp_path):
    _, corners = phase2.design(SPEC)
    p = phase2.write_glb(tmp_path / "a" / "r.glb", SPEC, corners)
    b = p.read_bytes()
    magic, ver, total = struct.unpack("<4sII", b[:12])
    assert (magic, ver, total) == (b"glTF", 2, len(b))
    n = struct.unpack("<I", b[12:16])[0]
    js = json.loads(b[20:20 + n])
    assert len(js["meshes"][0]["primitives"]) == 3  # roof, frame, glass


def test_step_smoke_has_panel_solids(tmp_path):
    _, corners = phase2.design(SPEC)
    p = phase2.export_step(tmp_path / "r.step", SPEC, corners)
    txt = p.read_text(errors="ignore")
    assert txt.count("Panel_") >= len(corners)


def test_viewer_page_points_at_existing_model_and_keeps_flat_shading():
    from pathlib import Path
    prev = Path(__file__).resolve().parents[1] / "preview"
    html = (prev / "vr.html").read_text(encoding="utf-8")
    assert (prev / "models" / "rooftop_solar.glb").exists()
    assert "models/rooftop_solar.glb" in html
    # the GLB has no normals: replacement materials must stay flat-shaded or the panels render black
    assert html.count("flatShading: true") == 2
    assert "fromScene(scene, 0, 0.1, 10000)" in html      # default PMREM far plane clips the sky


# ---- L-shaped building and rooftop structures ----
from shapely.geometry import box as _box  # noqa: E402

LSPEC = RooftopSpec(12000, 8000, notch_length=3500, notch_width=2500)


def test_notch_must_be_both_or_neither_and_leave_roof():
    with pytest.raises(ValueError):
        RooftopSpec(12000, 8000, notch_length=3000)
    with pytest.raises(ValueError):
        RooftopSpec(12000, 8000, notch_length=-1, notch_width=-1)
    with pytest.raises(ValueError):
        RooftopSpec(12000, 8000, notch_length=11700, notch_width=2000)
    with pytest.raises(ValueError):
        RooftopSpec(12000, 8000, notch_length=2000, notch_width=7700)


def test_outline_is_an_L_and_cells_tile_it_exactly():
    o = phase2.outline(LSPEC)
    assert o.area == pytest.approx(12 * 8 - 3.5 * 2.5)
    assert not o.contains(_box(11, 6, 11.5, 7))            # the cut-away corner
    cells = [_box(*c) for c in phase2.rect_cells(o)]
    assert sum(c.area for c in cells) == pytest.approx(o.area)
    assert all(a.intersection(b).area < 1e-9 for i, a in enumerate(cells) for b in cells[i + 1:])


def test_building_boxes_do_not_overlap_and_slab_matches_footprint():
    boxes = phase2.building_boxes(LSPEC)
    rects = [(_box(b.origin[0] / 1000, b.origin[1] / 1000, (b.origin[0] + b.size[0]) / 1000,
                   (b.origin[1] + b.size[1]) / 1000), b) for b in boxes]
    slab = [r for r, b in rects if b.name.startswith("RoofSlab")]
    walls = [r for r, b in rects if b.name.startswith("Parapet")]
    assert sum(r.area for r in slab) == pytest.approx(phase2.outline(LSPEC).area)
    ring = phase2.outline(LSPEC).difference(phase2.inner_outline(LSPEC))
    assert sum(r.area for r in walls) == pytest.approx(ring.area)
    assert all(a.intersection(b).area < 1e-9 for i, a in enumerate(walls) for b in walls[i + 1:])


def test_usable_area_avoids_the_notch():
    a = phase2.usable_area(LSPEC, 0.5)
    assert a.area < phase2.usable_area(SPEC, 0.5).area
    assert not a.intersects(_box(8.4, 5.0, 12, 8))


def test_default_features_fit_inside_and_do_not_touch_each_other():
    feats = phase2.default_features(LSPEC)
    assert {f.kind for f in feats} == {"stair", "store", "tank"} and len(feats) == 4
    floor = phase2.inner_outline(LSPEC)
    assert all(floor.contains(f.footprint) for f in feats)
    assert all(not a.footprint.buffer(0.3).intersects(b.footprint)
               for i, a in enumerate(feats) for b in feats[i + 1:])
    assert phase2.default_features(RooftopSpec(1500, 1500)) == []


def test_keepout_shadow_falls_away_from_the_panels_side():
    f = phase2.Feature("S", "stair", 5, 3, 2, 2, 2.4)
    k = phase2.keepout([f], facing_deg=90.0)
    assert k.bounds[1] < 3 - 0.5 - 1.5                      # shadow runs towards -y, past the clearance
    assert k.bounds[3] == pytest.approx(3 + 2 + 0.5)        # no shadow on the panels' side
    assert phase2.keepout([]).is_empty


def test_panels_keep_clear_of_structures():
    feats = phase2.default_features(LSPEC)
    with_f, corners = phase2.design(LSPEC, features=feats)
    without, _ = phase2.design(LSPEC)
    ko = phase2.keepout(feats)
    assert 0 < with_f.count < without.count
    assert all(not p.intersects(ko) for p in with_f.panels)
    assert len(corners) == with_f.count


def test_cylinder_mesh_is_closed_and_round():
    v, t = phase2._cylinder_mesh(2.0, 3.0, 1.0, 0.5, 2.0, n=12)
    assert len(v) == 26 and len(t) == 48
    assert np.allclose(np.hypot(v[:24, 0] - 2.0, v[:24, 1] - 3.0), 0.5)
    assert v[:, 2].min() == 1.0 and v[:, 2].max() == 3.0
    assert t.max() == 25


def test_feature_parts_materials_and_door_faces_the_roof():
    stair = phase2.Feature("S", "stair", 1, 1, 2.6, 2.8, 2.4)
    parts = phase2.feature_parts(stair, 0.15)
    keys = [p[3] for p in parts]
    assert keys.count("door") == 1 and keys.count("roof") == 2
    door = next(p for p in parts if p[3] == "door")[0]
    assert door[:, 1].min() == pytest.approx(1 + 2.8)       # on the +y face, away from the parapet
    assert door[:, 2].min() == pytest.approx(0.15)
    tank = phase2.feature_parts(phase2.Feature("T", "tank", 1, 1, 1.4, 1.4, 1.6), 0.15)
    assert [p[3] for p in tank].count("tank") == 2
    assert max(p[0][:, 2].max() for p in tank) == pytest.approx(0.15 + 1.6 + 0.12)


def test_glb_with_features_has_tank_and_door_primitives(tmp_path):
    feats = phase2.default_features(LSPEC)
    _, corners = phase2.design(LSPEC, features=feats)
    b = phase2.write_glb(tmp_path / "r.glb", LSPEC, corners, feats).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    js = json.loads(b[20:20 + n])
    assert {m["name"] for m in js["materials"]} == {"roof", "frame", "glass", "tank", "door"}


def test_layout_json_describes_footprint_and_structures(tmp_path):
    feats = phase2.default_features(LSPEC)
    j = json.loads(phase2.write_layout_json(tmp_path / "x" / "l.json", LSPEC, feats).read_text())
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in j["cells"]) == pytest.approx(
        phase2.outline(LSPEC).area)
    assert [f["name"] for f in j["features"]][0] == "StairHeadroom"


def test_step_with_structures_names_them(tmp_path):
    feats = phase2.default_features(LSPEC)
    _, corners = phase2.design(LSPEC, features=feats)
    txt = phase2.export_step(tmp_path / "r.step", LSPEC, corners, features=feats).read_text(errors="ignore")
    for name in ("StairHeadroom", "StoreRoom", "WaterTank1"):
        assert name in txt


def test_cli_smoke_builds_an_L_roof_with_structures_and_publishes(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase2_solar", root / "scripts" / "phase2_solar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lay = mod.main(["--out", str(tmp_path)])
    assert lay.count > 0
    assert {p.name for p in tmp_path.iterdir()} == {"rooftop_solar.step", "rooftop_solar.glb", "rooftop_solar.json"}
    j = json.loads((tmp_path / "rooftop_solar.json").read_text())
    assert len(j["cells"]) > 1 and len(j["features"]) == 4
    mod.main(["--out", str(tmp_path / "b"), "--box"])
    jb = json.loads((tmp_path / "b" / "rooftop_solar.json").read_text())
    assert len(jb["cells"]) == 1 and jb["features"] == []


def test_viewer_loads_the_layout_and_handles_tank_and_door_materials():
    from pathlib import Path
    prev = Path(__file__).resolve().parents[1] / "preview"
    html = (prev / "vr.html").read_text(encoding="utf-8")
    assert (prev / "models" / "rooftop_solar.json").exists()
    assert "models/rooftop_solar.json" in html
    assert 'm.name === "tank" || m.name === "door"' in html


def test_tank_body_is_slightly_narrower_than_its_plinth():
    parts = phase2.feature_parts(phase2.Feature("T", "tank", 1, 1, 1.4, 1.4, 1.6), 0.15)
    body = next(p for p in parts if p[3] == "tank")[0]
    assert body[:, 0].max() - 1.7 == pytest.approx(0.7 * 0.92 - 0.0, abs=1e-6) or \
        (body[:, 0].max() - body[:, 0].min()) / 2 == pytest.approx(0.7 * 0.92)


def test_default_features_keep_a_gap_of_at_least_30cm():
    feats = phase2.default_features(RooftopSpec(6000, 6000))
    gaps = [a.footprint.distance(b.footprint) for i, a in enumerate(feats) for b in feats[i + 1:]]
    assert gaps and min(gaps) >= 0.3


def test_cli_publish_copies_model_into_preview(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase2_solar2", root / "scripts" / "phase2_solar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "preview" / "models").mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.main([str(root / "data/rooftop_01/measurements.json"), "--out", str(tmp_path / "o"), "--publish"])
    assert {p.name for p in (tmp_path / "preview" / "models").iterdir()} == {"rooftop_solar.glb", "rooftop_solar.json"}


def test_design_walkway_sets_the_row_gap():
    spec = RooftopSpec(12000, 10000)
    lay0, _ = phase2.design(spec, latitude=19.0, facing_deg=270.0)
    lay1, c1 = phase2.design(spec, latitude=19.0, facing_deg=270.0, walkway=1.2)
    assert lay0.row_gap < 1.2 and lay1.row_gap == pytest.approx(1.2) and lay1.count < lay0.count
    assert len(c1) == lay1.count


def _cli(name):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / "phase2_solar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def test_cli_detailed_writes_step_glb_json_and_publishes(tmp_path, monkeypatch):
    mod, root = _cli("phase2_solar_det")
    (tmp_path / "preview" / "models").mkdir(parents=True)
    dxf = str(root / "data/sample_roof/roof.dxf")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.main(["--dxf", dxf, "--detailed", "--out", str(tmp_path / "o"), "--publish", "--walkway", "0.8"])
    assert {p.name for p in (tmp_path / "o").iterdir()} >= {"rooftop_detailed.step", "rooftop_detailed.glb",
                                                           "rooftop_detailed.json"}
    assert {p.name for p in (tmp_path / "preview" / "models").iterdir()} == {
        "rooftop_solar.glb", "rooftop_solar.json", "rooftop_detailed.glb", "rooftop_detailed.json"}
    j = json.loads((tmp_path / "o" / "rooftop_detailed.json").read_text())
    assert j["detailed"] and j["row_gap"] == pytest.approx(0.8) and j["floors"] == 3 and j["latitude"] == 19.07
    assert j["facing_deg"] == 270.0 and j["panels"] > 20


def test_cli_detailed_needs_a_drawing(tmp_path):
    mod, root = _cli("phase2_solar_det2")
    with pytest.raises(SystemExit):
        mod.main([str(root / "data/rooftop_01/measurements.json"), "--out", str(tmp_path), "--detailed"])


def test_cli_detailed_textures_by_default_and_plain_on_request(tmp_path, monkeypatch):
    mod, root = _cli("phase2_solar_tex")
    dxf = str(root / "data/sample_roof/roof.dxf")
    calls = []
    import rooftop.textures as T
    real = T.texture_set
    monkeypatch.setattr(T, "texture_set", lambda c, **k: calls.append(k) or real(c, **k))

    def mats(path):
        b = path.read_bytes()
        n = struct.unpack("<I", b[12:16])[0]
        return {m["name"]: m for m in json.loads(b[20:20 + n])["materials"]}

    mod.main(["--dxf", dxf, "--detailed", "--out", str(tmp_path / "t"), "--texture-px", "64"])
    assert calls == [{"px": 64}]
    m = mats(tmp_path / "t" / "rooftop_detailed.glb")
    assert "normalTexture" in m["concrete"] and "baseColorTexture" in m["concrete"]["pbrMetallicRoughness"]
    big = tmp_path / "t" / "rooftop_detailed.glb"
    mod.main(["--dxf", dxf, "--detailed", "--out", str(tmp_path / "p"), "--plain"])
    assert len(calls) == 1
    m = mats(tmp_path / "p" / "rooftop_detailed.glb")
    assert "normalTexture" not in m["concrete"] and "baseColorTexture" not in m["concrete"]["pbrMetallicRoughness"]
    assert (tmp_path / "p" / "rooftop_detailed.glb").stat().st_size < big.stat().st_size
    assert mod.main.__module__ and "--texture-px" in open(root / "scripts" / "phase2_solar.py").read()
    calls.clear()
    mod.main(["--dxf", dxf, "--detailed", "--out", str(tmp_path / "d")])
    assert calls == [{"px": 256}]
