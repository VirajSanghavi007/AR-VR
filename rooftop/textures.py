"""M4: procedural tiling PBR textures for the detailed roof model (concrete, plaster, membrane,
galvanised steel, PVC), with weathering, compressed to JPEG for the GLB.

Every map tiles seamlessly: the noise is white noise filtered in the frequency domain (an FFT of a
periodic signal is periodic), the galvanised spangle is a Voronoi pattern on a torus, and the
normal map takes its slopes with wrap-around differences. The UVs in rooftop/details.py are world
metres, so one tile covers 1 x 1 m; textures are drawn at `px` pixels per metre.

Colours: the material's linear rgb (details.COLOURS, what glTF's baseColorFactor means) is turned
into sRGB for the image, so a textured surface averages to the same colour as the untextured one.
"""
import io

import numpy as np

KINDS = ("concrete", "plaster", "membrane", "galv", "pvc")

# per kind: grain amplitude, grain spectrum slope (higher = smoother), speckle density, streak
# (dirt run-off) strength, bump strength for the normal map
STYLE = {
    "concrete": dict(grain=0.10, beta=1.6, speckle=0.004, streak=0.10, bump=2.0),
    "plaster": dict(grain=0.05, beta=2.2, speckle=0.0, streak=0.14, bump=1.2),
    "membrane": dict(grain=0.07, beta=1.2, speckle=0.02, streak=0.06, bump=1.5),
    "galv": dict(grain=0.03, beta=2.6, speckle=0.0, streak=0.08, bump=0.4),
    "pvc": dict(grain=0.015, beta=3.0, speckle=0.0, streak=0.05, bump=0.2),
}


def to_srgb(lin):
    """Linear 0..1 to sRGB 0..1 (the standard curve)."""
    lin = np.clip(np.asarray(lin, float), 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def _freqs(n):
    k = np.fft.fftfreq(n) * n
    return np.meshgrid(k, k, indexing="xy")             # kx varies along x (columns), ky along y


def fbm(n, beta, rng, ax=1.0, ay=1.0):
    """n x n periodic noise, zero mean, unit std, power ~ 1/f^beta. ax, ay weight the frequencies:
    ay > ax damps change along y, giving features long in y (vertical streaks)."""
    kx, ky = _freqs(n)
    f = np.hypot(kx * ax, ky * ay)
    f[0, 0] = 1.0
    spec = np.fft.fft2(rng.standard_normal((n, n))) / f ** (beta / 2)
    out = np.real(np.fft.ifft2(spec))
    return (out - out.mean()) / (out.std() + 1e-12)


def spangle(n, cells, rng):
    """Galvanised spangle: a periodic Voronoi pattern, one random brightness (-1..1) per crystal."""
    seeds = rng.random((cells, 2)) * n
    y, x = np.mgrid[0:n, 0:n].astype(float)
    best = np.full((n, n), np.inf)
    owner = np.zeros((n, n), int)
    for i, (sx, sy) in enumerate(seeds):
        dx = np.abs(x - sx)
        dy = np.abs(y - sy)
        d = np.minimum(dx, n - dx) ** 2 + np.minimum(dy, n - dy) ** 2      # distance on the torus
        m = d < best
        best[m] = d[m]
        owner[m] = i
    return rng.uniform(-1, 1, cells)[owner], owner


def weathering(n, rng, amount):
    """Dirt: soft blotches plus run-off streaks (long in y), 0 = clean, darkens by up to ~amount."""
    blotch = fbm(n, 3.0, rng)
    streak = fbm(n, 2.0, rng, ax=0.08, ay=1.0)
    d = 0.5 * np.clip(blotch, 0, None) + 0.7 * np.clip(streak, 0, None)
    return amount * d / (d.max() + 1e-12)


def surface(kind, rgb, px=256, seed=0, weather=0.5):
    """(albedo uint8 (px, px, 3) sRGB, height float (px, px)) of one tiling material."""
    if kind not in STYLE:
        raise ValueError(f"unknown texture kind {kind!r}")
    s = STYLE[kind]
    rng = np.random.default_rng(seed)
    grain = fbm(px, s["beta"], rng)
    height = grain.copy()
    shade = 1 + s["grain"] * grain
    if s["speckle"]:                                    # aggregate / granules: small light and dark dots
        dots = rng.random((px, px))
        shade += np.where(dots < s["speckle"], 0.25, 0) - np.where(dots > 1 - s["speckle"], 0.2, 0)
        height += np.where(dots < s["speckle"], 1.5, 0)
    if kind == "galv":
        sp, _ = spangle(px, max(8, px * px // 200), rng)          # crystals ~1.5 cm across
        shade += 0.035 * sp
        height += 0.3 * sp
    if kind == "plaster":                               # trowel marks: broad low swirls
        height += 0.6 * fbm(px, 3.5, rng)
    shade -= weathering(px, rng, s["streak"] * weather * 2)
    base = to_srgb(np.asarray(rgb, float))
    img = np.clip(base[None, None, :] * shade[..., None], 0, 1)
    return (img * 255 + 0.5).astype(np.uint8), height


def normal_map(height, strength=1.0):
    """Tangent-space normal map (uint8 RGB, +y up in the image = green) from a periodic height field."""
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / 2
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) / 2
    k = strength / max(np.abs(height).max(), 1e-12)
    n = np.stack([-dx * k, dy * k, np.ones_like(height)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)


def jpeg(img, quality=82):
    """JPEG bytes of a uint8 RGB image."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def texture_set(colours, kinds=KINDS, px=256, seed=0, weather=0.5, quality=82):
    """{kind: (albedo JPEG, normal JPEG)} for details.materials(textures=...); colours is
    details.COLOURS ({kind: (rgb, metallic, roughness)})."""
    out = {}
    for i, k in enumerate(kinds):
        img, h = surface(k, colours[k][0], px, seed + i, weather)
        out[k] = (jpeg(img, quality), jpeg(normal_map(h, STYLE[k]["bump"]), quality))
    return out
