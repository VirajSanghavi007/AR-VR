"""AprilTags (family 36h11) for metric scale of a photo reconstruction.

Print tags of a known size, lay them flat on the roof, photograph as usual.
COLMAP poses are only defined up to scale; each tag's four corners are
triangulated from every photo that sees it, and scale = real edge / model edge.

Tag size = edge of the outer black square (what OpenCV and the AprilTag papers call
the tag size), not the white margin around it.
"""
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
CELLS = 8                      # 36h11: 6x6 data + 1-cell black border each side

# COLMAP camera models: id -> (name, number of params)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
                 8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}
SUPPORTED = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}


# ---------- printing ----------

def tag_image(tag_id, cell_px=40, margin_cells=1):
    """Tag as a uint8 image, black square = CELLS*cell_px, plus a white quiet zone."""
    img = cv2.aruco.generateImageMarker(DICT, int(tag_id), CELLS * cell_px)
    m = margin_cells * cell_px
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)


def write_tag_sheet(path, ids=(0, 1, 2, 3), size_mm=160.0):
    """A4 PDF, one tag per page at exactly size_mm, with a 100 mm check line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not 20 <= size_mm <= 180:
        raise ValueError("size_mm must be 20-180 to fit A4 with margins")
    A4 = (210.0, 297.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = []
    with PdfPages(path) as pdf:
        for tid in ids:
            fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, A4[0]); ax.set_ylim(0, A4[1]); ax.axis("off")   # axes units = mm
            x0, y0 = (A4[0] - size_mm) / 2, A4[1] - 40 - size_mm
            marker = cv2.aruco.generateImageMarker(DICT, int(tid), CELLS * 10)
            ax.imshow(marker, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                      extent=(x0, x0 + size_mm, y0, y0 + size_mm))
            ax.text(A4[0] / 2, A4[1] - 22, f"AprilTag 36h11  ·  id {tid}", ha="center", fontsize=16, weight="bold")
            ax.text(A4[0] / 2, y0 - 12, f"Black square must measure {size_mm:.1f} mm. Print at 100 % / Actual size.",
                    ha="center", fontsize=10)
            ly = y0 - 30
            ax.plot([55, 155], [ly, ly], color="black", lw=1)
            for x in (55, 155):
                ax.plot([x, x], [ly - 3, ly + 3], color="black", lw=1)
            ax.text(105, ly - 8, "this line must measure 100 mm", ha="center", fontsize=9)
            ax.text(A4[0] / 2, 20, "Lay flat on the roof, fully visible, before taking the photos.\n"
                    "Use each id only once per site.", ha="center", fontsize=9, color="#444")
            pdf.savefig(fig)
            plt.close(fig)
            layout.append({"id": int(tid), "x_mm": x0, "y_mm": y0, "size_mm": size_mm})
    return path, layout


# ---------- detection ----------

def detect_tags(gray):
    """{tag_id: (4, 2) pixel corners} in OpenCV order (top-left, top-right, bottom-right, bottom-left)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, params).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


# ---------- COLMAP model ----------

@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    name: str
    R: np.ndarray        # world -> camera rotation
    t: np.ndarray
    camera_id: int


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_model(model_dir):
    """cameras + images from a COLMAP sparse model folder (text or binary)."""
    d = Path(model_dir)
    if (d / "cameras.txt").exists() and (d / "images.txt").exists():
        return _read_text(d)
    if (d / "cameras.bin").exists() and (d / "images.bin").exists():
        return _read_bin(d)
    raise FileNotFoundError(f"no COLMAP cameras/images (.txt or .bin) in {d}")


def _read_text(d):
    cams = {}
    for line in (d / "cameras.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            cams[int(f[0])] = Camera(f[1], int(f[2]), int(f[3]), np.array(f[4:], float))
    imgs = []
    lines = [l for l in (d / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for line in lines[::2]:
        f = line.split(maxsplit=9)              # image names may contain spaces
        if len(f) < 10:
            continue
        imgs.append(Image(f[9], quat_to_R(*map(float, f[1:5])), np.array(f[5:8], float), int(f[8])))
    return cams, imgs


def _read_bin(d):
    cams = {}
    with open(d / "cameras.bin", "rb") as fh:
        for _ in range(struct.unpack("<Q", fh.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", fh.read(24))
            name, n = CAMERA_MODELS[mid]
            cams[cid] = Camera(name, w, h, np.array(struct.unpack(f"<{n}d", fh.read(8 * n))))
    imgs = []
    with open(d / "images.bin", "rb") as fh:
        for _ in range(struct.unpack("<Q", fh.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", fh.read(64))
            name = b""
            while (ch := fh.read(1)) != b"\x00":
                name += ch
            npts = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(24 * npts, 1)                       # skip 2D points (x, y, point3D id)
            imgs.append(Image(name.decode(), quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz]), cid))
    return cams, imgs


def normalize_points(px, cam):
    """Pixel coords -> undistorted normalized camera coords (x/z, y/z)."""
    p, m = cam.params, cam.model
    if m not in SUPPORTED:
        raise ValueError(f"camera model {m} not supported; undistort the images first (colmap image_undistorter)")
    if m == "SIMPLE_PINHOLE":
        K, dist = [p[0], p[0], p[1], p[2]], []
    elif m == "PINHOLE":
        K, dist = p[:4], []
    elif m == "SIMPLE_RADIAL":
        K, dist = [p[0], p[0], p[1], p[2]], [p[3], 0, 0, 0]
    elif m == "RADIAL":
        K, dist = [p[0], p[0], p[1], p[2]], [p[3], p[4], 0, 0]
    elif m == "OPENCV":
        K, dist = p[:4], p[4:8]
    elif m == "FULL_OPENCV":
        K, dist = p[:4], p[4:12]
    else:                                                   # OPENCV_FISHEYE
        K, dist = p[:4], p[4:8]
    Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], float)
    pts = np.asarray(px, float).reshape(-1, 1, 2)
    if m == "OPENCV_FISHEYE":
        return cv2.fisheye.undistortPoints(pts, Km, np.asarray(dist, float)).reshape(-1, 2)
    if len(dist) == 0:
        return ((pts.reshape(-1, 2) - Km[:2, 2]) / [Km[0, 0], Km[1, 1]])
    return cv2.undistortPoints(pts, Km, np.asarray(dist, float),
                               criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)


def triangulate(obs):
    """Linear multi-view triangulation. obs: [(R, t, (x, y) normalized)] -> 3D point."""
    A = []
    for R, t, (x, y) in obs:
        P = np.hstack([R, np.reshape(t, (3, 1))])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def reprojection_error(X, obs):
    """Mean distance in normalized image units between projected X and observations."""
    errs = []
    for R, t, xy in obs:
        c = R @ X + t
        errs.append(np.linalg.norm(c[:2] / c[2] - xy))
    return float(np.mean(errs))


# ---------- scale ----------

def detect_in_model(model_dir, image_root, workers=None, log=print):
    """Detect tags in every registered image; {tag_id: [(image, corners_px)]}.

    Threads, since OpenCV releases the GIL; prints progress + ETA.
    """
    cams, imgs = read_model(model_dir)
    root = Path(image_root)
    found = {}

    def one(im):
        g = cv2.imread(str(root / im.name), cv2.IMREAD_GRAYSCALE)
        return im, ({} if g is None else detect_tags(g))

    t0, total = time.time(), len(imgs)
    step = max(1, total // 20)
    with ThreadPoolExecutor(max_workers=workers or min(22, total or 1)) as ex:
        for done, (im, tags) in enumerate(ex.map(one, imgs), 1):
            for tid, c in tags.items():
                found.setdefault(tid, []).append((im, c))
            if done % step == 0 or done == total:
                el = time.time() - t0
                log(f"  tags: {done}/{total} images ({100 * done // total}%) - {el:.0f}s elapsed, "
                    f"eta {el / done * (total - done):.0f}s, {len(found)} tag ids seen")
    return cams, found


def tag_corners_3d(cams, found, min_views=2):
    """{tag_id: {"corners": (4, 3), "views": n, "reproj": mean normalized error}}."""
    out = {}
    for tid, views in found.items():
        if len(views) < min_views:
            continue
        corners, errs = [], []
        for k in range(4):
            obs = [(im.R, im.t, normalize_points(c[k:k + 1], cams[im.camera_id])[0]) for im, c in views]
            X = triangulate(obs)
            corners.append(X)
            errs.append(reprojection_error(X, obs))
        out[tid] = {"corners": np.array(corners), "views": len(views), "reproj": float(np.mean(errs))}
    return out


def scale_from_tags(tags3d, size_m):
    """Metres per model unit from known tag size, robust over several tags.

    Uses the 4 edges of each tag (not diagonals); spread_pct = how much tags disagree,
    a direct accuracy check of the reconstruction scale.
    """
    if not tags3d:
        raise ValueError("no tag was seen in at least 2 photos")
    per_tag = {}
    for tid, t in tags3d.items():
        c = t["corners"]
        edges = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
        diag = (np.linalg.norm(c[2] - c[0]) + np.linalg.norm(c[3] - c[1])) / 2
        per_tag[tid] = {"scale": size_m / float(np.mean(edges)),
                        "edge_spread_pct": 100 * float(np.ptp(edges) / np.mean(edges)),
                        "squareness": float(diag / (np.mean(edges) * np.sqrt(2))),
                        "views": t["views"], "reproj": t["reproj"]}
    s = np.array([v["scale"] for v in per_tag.values()])
    scale = float(np.median(s))
    return {"metres_per_unit": scale, "tags": per_tag, "n_tags": len(per_tag),
            "spread_pct": 100 * float(np.ptp(s) / scale) if len(s) > 1 else None}


def tag_distances(tags3d, scale):
    """Centre-to-centre distances between tags in metres (to check against a tape)."""
    cen = {tid: t["corners"].mean(0) * scale for tid, t in tags3d.items()}
    ids = sorted(cen)
    return {f"{a}-{b}": float(np.linalg.norm(cen[a] - cen[b])) for i, a in enumerate(ids) for b in ids[i + 1:]}
