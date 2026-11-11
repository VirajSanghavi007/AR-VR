"""rooftop/textures.py (M4): tiling procedural PBR textures, normal maps and JPEG packing."""
import io

import numpy as np
import pytest
from PIL import Image

from rooftop import details as D
from rooftop import textures as T


def wrap_vs_interior(a, axis):
    """Mean |difference| across the wrap edge and between interior neighbours along one axis."""
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a.mean(-1)
    first, last = np.take(a, 0, axis), np.take(a, -1, axis)
    inner = np.abs(np.diff(a, axis=axis)).mean()
    return np.abs(first - last).mean(), inner


def step_along(a, axis):
    a = np.asarray(a, float)
    return np.abs(np.diff(a, axis=axis)).mean()


# ---------------------------------------------------------------- colour

def test_to_srgb_matches_the_standard_curve_and_clips():
    got = T.to_srgb([0.0, 0.002, 0.0031308, 0.18, 0.5, 1.0, -0.3, 1.7])
    want = [0, 0.002 * 12.92, 0.0031308 * 12.92, 1.055 * 0.18 ** (1 / 2.4) - 0.055,
            1.055 * 0.5 ** (1 / 2.4) - 0.055, 1, 0, 1]
    assert got == pytest.approx(want, abs=1e-9)
    assert T.to_srgb(0.5) == pytest.approx(0.7354, abs=1e-4)


# ---------------------------------------------------------------- noise

def test_fbm_is_normalised_deterministic_and_tiles():
    a = T.fbm(64, 2.0, np.random.default_rng(5))
    b = T.fbm(64, 2.0, np.random.default_rng(5))
    assert a.shape == (64, 64) and np.array_equal(a, b) and np.isfinite(a).all()
    assert abs(a.mean()) < 1e-9 and a.std() == pytest.approx(1, abs=1e-6)
    for ax in (0, 1):
        edge, inner = wrap_vs_interior(a, ax)
        assert edge < 2 * inner                                  # no seam where the tile repeats
    assert not np.array_equal(a, T.fbm(64, 2.0, np.random.default_rng(6)))


def test_fbm_higher_beta_is_smoother():
    rough = T.fbm(128, 0.5, np.random.default_rng(1))
    smooth = T.fbm(128, 3.0, np.random.default_rng(1))
    assert step_along(smooth, 1) < 0.5 * step_along(rough, 1)


def test_fbm_streaks_run_vertically_when_ay_damps_y():
    s = T.fbm(128, 2.0, np.random.default_rng(2), ax=0.08, ay=1.0)
    assert step_along(s, 0) < 0.4 * step_along(s, 1)              # changes slowly down a column
    s = T.fbm(128, 2.0, np.random.default_rng(2), ax=1.0, ay=0.08)
    assert step_along(s, 1) < 0.4 * step_along(s, 0)


def test_spangle_is_a_periodic_voronoi_of_flat_crystals():
    n, cells = 48, 12
    v, owner = T.spangle(n, cells, np.random.default_rng(3))
    seeds = np.random.default_rng(3).random((cells, 2)) * n          # same draws as spangle
    assert owner.shape == (n, n) and set(np.unique(owner)) <= set(range(cells))
    assert np.all(np.abs(v) <= 1)
    for i in range(cells):                                         # one brightness per crystal
        assert np.unique(v[owner == i]).size <= 1
    rng = np.random.default_rng(0)
    for y, x in rng.integers(0, n, (60, 2)):                       # every pixel owned by its nearest seed on the torus
        dx = np.abs(x - seeds[:, 0]); dy = np.abs(y - seeds[:, 1])
        d = np.minimum(dx, n - dx) ** 2 + np.minimum(dy, n - dy) ** 2
        assert d[owner[y, x]] == pytest.approx(d.min())
    assert len(np.unique(owner)) > cells // 2


def test_weathering_ranges_from_clean_to_amount_and_streaks_down():
    w = T.weathering(128, np.random.default_rng(4), 0.3)
    assert w.min() >= 0 and w.max() == pytest.approx(0.3, rel=1e-6)
    assert step_along(w, 0) < step_along(w, 1)                      # run-off is long in y
    assert np.all(T.weathering(32, np.random.default_rng(4), 0.0) == 0)
    edge, inner = wrap_vs_interior(w, 0)
    assert edge < 2 * inner + 1e-9


# ---------------------------------------------------------------- surfaces

@pytest.mark.parametrize("kind", T.KINDS)
def test_surface_averages_to_the_material_colour_and_tiles(kind):
    rgb = D.COLOURS[kind][0]
    img, h = T.surface(kind, rgb, px=96, seed=1, weather=0.5)
    assert img.dtype == np.uint8 and img.shape == (96, 96, 3) and h.shape == (96, 96)
    want = T.to_srgb(np.asarray(rgb)) * 255
    assert np.abs(img.reshape(-1, 3).mean(0) - want).max() < 0.06 * 255
    for ax in (0, 1):
        edge, inner = wrap_vs_interior(img, ax)
        assert edge < 2.5 * inner + 0.5
        edge, inner = wrap_vs_interior(h, ax)
        assert edge < 2.5 * inner + 1e-9
    a, _ = T.surface(kind, rgb, px=48, seed=7)
    b, _ = T.surface(kind, rgb, px=48, seed=7)
    c, _ = T.surface(kind, rgb, px=48, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_surface_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown texture kind 'brick'"):
        T.surface("brick", (0.5, 0.5, 0.5))


def test_weather_darkens_and_grain_varies():
    rgb = (0.5, 0.5, 0.5)
    clean, _ = T.surface("concrete", rgb, px=96, seed=2, weather=0.0)
    dirty, _ = T.surface("concrete", rgb, px=96, seed=2, weather=1.0)
    assert dirty.mean() < clean.mean() - 3
    base = T.to_srgb(0.5) * 255
    assert clean.std() > 0.05 * base                                 # concrete grain is visible
    pvc, _ = T.surface("pvc", rgb, px=96, seed=2, weather=0.0)
    assert pvc.std() < clean.std() / 3                               # pvc is smooth
    assert clean.mean() == pytest.approx(base, abs=4)


def test_concrete_has_aggregate_speckles():
    rgb = (0.3, 0.3, 0.3)
    img, h = T.surface("concrete", rgb, px=128, seed=3, weather=0.0)
    base = T.to_srgb(0.3) * 255
    lum = img.mean(-1)
    bright = (lum > base * 1.22).mean()
    assert 0.002 < bright < 0.05                                     # about the speckle density
    assert h.max() > 3                                               # stones stand proud
    plaster, ph = T.surface("plaster", rgb, px=128, seed=3, weather=0.0)
    assert (plaster.mean(-1) > base * 1.22).mean() < 0.001


def test_galvanised_spangle_shows_flat_crystal_patches():
    img, h = T.surface("galv", (0.6, 0.6, 0.6), px=128, seed=4, weather=0.0)
    hi = np.abs(np.diff(h, axis=1))
    # a crystal boundary is a sharp step in height; inside a crystal the height is grain only
    assert np.percentile(hi, 99.5) > 4 * np.median(hi)
    cells = 128 * 128 // 200
    assert cells == 81
    g, _ = T.surface("pvc", (0.6, 0.6, 0.6), px=128, seed=4, weather=0.0)
    assert img.std() > g.std()


def test_plaster_height_has_broad_trowel_swirls():
    _, h = T.surface("plaster", (0.8, 0.8, 0.8), px=128, seed=5, weather=0.0)
    _, m = T.surface("membrane", (0.8, 0.8, 0.8), px=128, seed=5, weather=0.0)
    # the swirl adds 0.6 of a unit-std field on top of the unit grain
    assert h.std() > 1.1 and m.std() < 1.1


# ---------------------------------------------------------------- normal maps and JPEG

def decode(nm):
    return nm.astype(float) / 255 * 2 - 1


def test_normal_map_of_a_flat_height_points_straight_up():
    nm = T.normal_map(np.zeros((8, 8)), 2.0)
    assert nm.dtype == np.uint8 and nm.shape == (8, 8, 3)
    assert np.all(nm == [128, 128, 255])
    assert np.all(T.normal_map(np.full((8, 8), 3.0), 2.0) == [128, 128, 255])


def test_normal_map_slopes_unit_length_and_strength():
    n = 64
    x = np.arange(n)
    h = np.tile(np.sin(2 * np.pi * x / n), (n, 1))                  # rises along x (columns)
    nm = decode(T.normal_map(h, 1.0))
    assert np.allclose(np.linalg.norm(nm, axis=-1), 1, atol=0.02)
    assert nm[0, 0, 0] < -0.05 and nm[0, n // 2, 0] > 0.05           # x tilts away from the uphill side
    assert np.abs(nm[..., 1]).max() < 0.01                           # no y slope
    # the right difference: centred, wrapping round
    d = (np.sin(2 * np.pi / n) - np.sin(-2 * np.pi / n)) / 2
    assert nm[0, 0, 0] == pytest.approx(-d / np.sqrt(1 + d * d), abs=0.01)
    hy = h.T                                                         # rises down the rows
    ny = decode(T.normal_map(hy, 1.0))
    assert ny[0, 0, 1] > 0.05 and np.abs(ny[..., 0]).max() < 0.01
    assert ny[0, 0, 1] == pytest.approx(nm[0, 0, 0] * -1, abs=0.01)
    strong = decode(T.normal_map(h, 4.0))
    assert abs(strong[0, 0, 0]) > 3 * abs(nm[0, 0, 0])
    # scaled by the height's peak, so the units of height do not matter
    assert np.abs(T.normal_map(h * 10, 1.0).astype(int) - T.normal_map(h, 1.0)).max() <= 1


def test_jpeg_bytes_decode_and_quality_matters():
    img, _ = T.surface("concrete", (0.4, 0.4, 0.4), px=64, seed=1)
    hi, lo = T.jpeg(img, 95), T.jpeg(img, 30)
    assert hi[:3] == b"\xff\xd8\xff" and len(lo) < len(hi)
    back = np.asarray(Image.open(io.BytesIO(hi)))
    assert back.shape == img.shape and np.abs(back.astype(int) - img).mean() < 4
    assert T.jpeg(img) == T.jpeg(img, 82)


def test_texture_set_packs_albedo_and_normal_per_kind():
    s = T.texture_set(D.COLOURS, kinds=("concrete", "galv"), px=32, seed=3, weather=0.2, quality=70)
    assert set(s) == {"concrete", "galv"}
    for k, (alb, nrm) in s.items():
        assert alb[:3] == b"\xff\xd8\xff" and nrm[:3] == b"\xff\xd8\xff"
        img = np.asarray(Image.open(io.BytesIO(alb)))
        want, h = T.surface(k, D.COLOURS[k][0], 32, 3 + ("concrete", "galv").index(k), 0.2)
        assert alb == T.jpeg(want, 70) and nrm == T.jpeg(T.normal_map(h, T.STYLE[k]["bump"]), 70)
        assert img.shape == (32, 32, 3)
    full = T.texture_set(D.COLOURS, px=16)
    assert set(full) == set(T.KINDS)
    assert full["concrete"][0] == T.jpeg(T.surface("concrete", D.COLOURS["concrete"][0], 16, 0, 0.5)[0], 82)


# ---------------------------------------------------------------- exact construction

def test_freqs_are_integer_cycles_per_tile():
    kx, ky = T._freqs(4)
    assert kx.tolist() == [[0, 1, -2, -1]] * 4
    assert ky.tolist() == [[0] * 4, [1] * 4, [-2] * 4, [-1] * 4]


def test_fbm_spectrum_is_white_noise_over_f_to_half_beta():
    n, beta, ax, ay = 32, 1.6, 0.5, 1.0
    out = T.fbm(n, beta, np.random.default_rng(9), ax, ay)
    white = np.fft.fft2(np.random.default_rng(9).standard_normal((n, n)))
    kx, ky = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="xy")
    f = np.hypot(kx * ax, ky * ay)
    ratio = (np.fft.fft2(out) * f ** (beta / 2) / white)[f > 0]
    assert np.allclose(ratio, ratio[0], rtol=1e-6)                      # one real gain for every frequency
    assert abs(ratio[0].imag) < 1e-9 and ratio[0].real > 0
    assert abs(np.fft.fft2(out)[0, 0]) < 1e-9                           # no DC


def test_fbm_warns_about_nothing():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T.fbm(16, 2.0, np.random.default_rng(0))


def test_weathering_is_blotches_plus_streaks_exactly():
    n, amount = 64, 0.3
    w = T.weathering(n, np.random.default_rng(4), amount)
    rng = np.random.default_rng(4)
    blotch = T.fbm(n, 3.0, rng)
    streak = T.fbm(n, 2.0, rng, ax=0.08, ay=1.0)
    d = 0.5 * np.clip(blotch, 0, None) + 0.7 * np.clip(streak, 0, None)
    assert np.allclose(w, amount * d / d.max(), atol=1e-12)


def _rebuild(kind, rgb, px, seed):
    """surface() at weather 0 rebuilt from its parts: (shade, height)."""
    s = T.STYLE[kind]
    rng = np.random.default_rng(seed)
    grain = T.fbm(px, s["beta"], rng)
    shade, height = 1 + s["grain"] * grain, grain.copy()
    if s["speckle"]:
        dots = rng.random((px, px))
        shade = shade + 0.25 * (dots < s["speckle"]) - 0.2 * (dots > 1 - s["speckle"])
        height = height + 1.5 * (dots < s["speckle"])
    if kind == "galv":
        sp, _ = T.spangle(px, max(8, px * px // 200), rng)
        shade, height = shade + 0.035 * sp, height + 0.3 * sp
    return shade, height


@pytest.mark.parametrize("kind", ["concrete", "membrane", "galv", "pvc"])
def test_surface_is_grain_speckle_and_spangle_exactly(kind):
    rgb, px, seed = (0.35, 0.4, 0.45), 48, 11
    img, h = T.surface(kind, rgb, px=px, seed=seed, weather=0.0)
    shade, height = _rebuild(kind, rgb, px, seed)
    assert np.allclose(h, height, atol=1e-12)
    want = np.clip(T.to_srgb(np.asarray(rgb))[None, None, :] * shade[..., None], 0, 1)
    assert np.array_equal(img, np.floor(want * 255 + 0.5).astype(np.uint8))   # rounded, not truncated


def test_surface_weather_subtracts_the_weathering_field():
    rgb, px, seed, weather = (0.5, 0.5, 0.5), 48, 2, 0.7
    img, _ = T.surface("pvc", rgb, px=px, seed=seed, weather=weather)
    rng = np.random.default_rng(seed)
    grain = T.fbm(px, T.STYLE["pvc"]["beta"], rng)
    shade = 1 + T.STYLE["pvc"]["grain"] * grain - T.weathering(px, rng, T.STYLE["pvc"]["streak"] * weather * 2)
    want = np.clip(T.to_srgb(np.asarray(rgb))[None, None, :] * shade[..., None], 0, 1)
    assert np.array_equal(img, np.floor(want * 255 + 0.5).astype(np.uint8))


def test_normal_map_of_a_ridge_uses_centred_differences():
    h = np.zeros((8, 8))
    h[:, 5] = 1.0                                                   # a one-pixel ridge down column 5
    nm = decode(T.normal_map(h, 1.0))
    # centred: column 4 sees +0.5, column 5 sees 0 (both neighbours flat), column 6 sees -0.5
    want = {4: [-0.5, 0, 1], 5: [0, 0, 1], 6: [0.5, 0, 1], 0: [0, 0, 1]}
    for col, v in want.items():
        v = np.array(v) / np.linalg.norm(v)
        assert nm[3, col] == pytest.approx(v, abs=0.01), col
    r = h.T                                                          # the same ridge along row 5
    ny = decode(T.normal_map(r, 1.0))
    assert ny[4, 3] == pytest.approx(np.array([0, 0.5, 1]) / np.hypot(0.5, 1), abs=0.01)
