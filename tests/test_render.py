import numpy as np
import pytest

from rooftop.cad import RooftopSpec, export_step, rooftop_boxes
from rooftop.render import render_png, shade, tessellate_step


def test_shade_is_two_sided_and_bounded():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    up = shade(v, np.array([[0, 1, 2]]))
    down = shade(v, np.array([[0, 2, 1]]))
    assert np.allclose(up, down)
    assert up.min() >= 0 and up.max() <= 1


def test_shade_brighter_facing_light():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    lit = shade(v, np.array([[0, 1, 2]]), light=(0, 0, 1))
    side = shade(v, np.array([[0, 1, 2]]), light=(1, 0, 0))
    assert lit.sum() > side.sum()


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    p = tmp_path_factory.mktemp("cad") / "r.step"
    return export_step(rooftop_boxes(RooftopSpec(4000, 3000)), p)


def test_tessellate_bounds_match_spec(step):
    v, t = tessellate_step(step)
    assert np.allclose(v.min(0), 0, atol=1e-3)
    assert np.allclose(v.max(0), [4000, 3000, 1050], atol=1e-3)
    assert t.max() < len(v)


def test_render_smoke(step, tmp_path):
    out = render_png(step, tmp_path / "o" / "r.png", size=(400, 300))
    assert out.stat().st_size > 2000
