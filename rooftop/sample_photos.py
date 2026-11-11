"""Sample photos for tags.html's "Check photos" mode, so it can be tried without taking any.

Four 36h11 tags (ids 0-3, 160 mm) lie at the corners of a 1.0 x 0.6 m rectangle on a light table;
each photo is that table seen from a different phone position (a perspective warp), with a little blur,
sensor noise and JPEG. One view leaves id 3 out of frame, as real sets often do.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from rooftop.tags import CELLS, tag_image

PHOTO_WH = (1600, 1200)

# Where the table's four corners (top-left, top-right, bottom-right, bottom-left) land in each photo,
# as fractions of its width and height. Values past 0..1 put that corner out of frame.
VIEWS = {
    "tag_photo_01_overhead.jpg": [(0.08, 0.14), (0.92, 0.14), (0.92, 0.86), (0.08, 0.86)],
    "tag_photo_02_from_front.jpg": [(0.2, 0.18), (0.8, 0.18), (0.97, 0.9), (0.03, 0.9)],
    "tag_photo_03_from_left.jpg": [(0.06, 0.08), (0.84, 0.24), (0.84, 0.78), (0.06, 0.94)],
    "tag_photo_04_from_right.jpg": [(0.16, 0.24), (0.94, 0.06), (0.94, 0.94), (0.16, 0.78)],
    "tag_photo_05_turned.jpg": [(0.22, 0.04), (0.97, 0.3), (0.78, 0.97), (0.03, 0.7)],
    "tag_photo_06_id3_cut_off.jpg": [(0.02, 0.0), (0.98, 0.04), (0.98, 1.0), (-0.1, 1.15)],
}


def table_scene(ids=(0, 1, 2, 3), tag_mm=160, span_mm=(1000, 600), border_mm=110, px_per_mm=1.0):
    """Top-down picture of the table: tags centred on the corners of the span. Returns (BGR image,
    {id: 4x2 corners of the black square in image px, clockwise from top-left})."""
    w = int(round((span_mm[0] + 2 * border_mm) * px_per_mm))
    h = int(round((span_mm[1] + 2 * border_mm) * px_per_mm))
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
    img = np.clip(205 + 18 * grain / grain.std(), 0, 255).astype(np.uint8)
    cell = max(1, int(round(tag_mm * px_per_mm / CELLS)))                   # whole pixels per tag cell
    centres = [(border_mm, border_mm), (border_mm + span_mm[0], border_mm),
               (border_mm + span_mm[0], border_mm + span_mm[1]), (border_mm, border_mm + span_mm[1])]
    corners = {}
    for tid, (cx, cy) in zip(ids, centres):
        t = tag_image(tid, cell_px=cell, margin_cells=1)
        x0, y0 = int(round(cx * px_per_mm)) - t.shape[1] // 2, int(round(cy * px_per_mm)) - t.shape[0] // 2
        img[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        m, e = cell, cell * CELLS                                            # white margin, black square (px)
        corners[tid] = np.array([[x0 + m, y0 + m], [x0 + m + e, y0 + m], [x0 + m + e, y0 + m + e], [x0 + m, y0 + m + e]], float)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), corners


def view_homography(scene_wh, quad, photo_wh=PHOTO_WH):
    """Homography taking the scene's corners to the view's corner fractions of the photo."""
    sw, sh = scene_wh
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = np.float32([[x * photo_wh[0], y * photo_wh[1]] for x, y in quad])
    return cv2.getPerspectiveTransform(src, dst)


def take_photo(scene, quad, photo_wh=PHOTO_WH, seed=0, noise=4.0, blur=0.8, floor=(70, 74, 80)):
    """The scene as a phone would see it from `quad`: warped onto a darker floor, blurred, noisy."""
    H = view_homography((scene.shape[1], scene.shape[0]), quad, photo_wh)
    out = cv2.warpPerspective(scene, H, photo_wh, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=floor)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise:
        n = np.random.default_rng(seed).normal(0, noise, out.shape)
        out = np.clip(out + n, 0, 255).astype(np.uint8)
    return out, H


def project(H, pts):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def in_frame(corners, photo_wh=PHOTO_WH, pad=2):
    """True if every corner is inside the photo (pad px from the edge)."""
    c = np.asarray(corners)
    return bool(((c[:, 0] >= pad) & (c[:, 0] < photo_wh[0] - pad) & (c[:, 1] >= pad) & (c[:, 1] < photo_wh[1] - pad)).all())


def write_sample_photos(out_dir, views=VIEWS, quality=90):
    """Write each view as a JPEG plus tag_photos.json ({"tag_mm", "ids", "photos": [{name, ids}]}, ids = tags fully
    in frame). Returns (paths, index)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene, corners = table_scene()
    paths, photos = [], []
    for i, (name, quad) in enumerate(views.items()):
        img, H = take_photo(scene, quad, seed=i)
        p = out_dir / name
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(p)
        photos.append({"name": name, "ids": sorted(t for t, c in corners.items() if in_frame(project(H, c)))})
    index = {"tag_mm": 160, "ids": sorted(corners), "photos": photos}
    (out_dir / "tag_photos.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return paths, index
