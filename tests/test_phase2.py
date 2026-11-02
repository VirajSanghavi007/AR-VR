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
