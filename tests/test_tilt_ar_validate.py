"""Tilted rows, roof choice, AR (GLB) export and ground-truth validation helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from rooftop import ar_export as A
from rooftop import layout as L
from rooftop import panels as P
from rooftop import validate as V
from rooftop.geometry import PlaneFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import design_panels  # noqa: E402

FLAT = PlaneFrame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


# ---------- tilted layout ----------

def test_winter_sun_altitude():
    assert L.winter_sun_altitude(47.4) == pytest.approx(19.16)
    assert L.winter_sun_altitude(-11.0) == pytest.approx(55.56)
    with pytest.raises(ValueError):
        L.winter_sun_altitude(70)


def test_row_shadow_gap():
    # 1 m panel at 30 deg rises 0.5 m; sun at 45 deg -> 0.5 m shadow
    assert L.row_shadow_gap(1.0, 30, 45) == pytest.approx(0.5)
    assert L.row_shadow_gap(1.0, 0, 45) == 0


def test_tilted_layout_rows_spacing_and_facing():
    lay = L.layout_panels(box(0, 0, 20, 20), panel_w=1, panel_h=2, gap=0,
                          tilt_deg=30, facing_deg=-90, sun_alt_deg=45)
    assert lay.count > 0
    assert np.allclose(lay.facing, [0, -1], atol=1e-9)        # facing -y
    slope = 2 if lay.orientation == "portrait" else 1
    assert lay.rise == pytest.approx(slope * 0.5)
    assert lay.row_gap == pytest.approx(slope * 0.5)          # tan 45 = 1
    ys = sorted({round(p.bounds[1], 6) for p in lay.panels})
    pitch = np.diff(ys)
    assert np.allclose(pitch, slope * np.cos(np.radians(30)) + lay.row_gap, atol=1e-6)
    depth = {round(p.bounds[3] - p.bounds[1], 6) for p in lay.panels}
    assert depth == {round(slope * np.cos(np.radians(30)), 6)}  # footprint shrinks by cos(tilt)


def test_tilt_reduces_count_vs_flat():
    area = box(0, 0, 12, 8)
    flat = L.layout_panels(area, gap=0.02, angle=0)
    tilt = L.layout_panels(area, gap=0.02, tilt_deg=15, facing_deg=-90, sun_alt_deg=20)
    assert 0 < tilt.count < flat.count


def test_tilted_layout_requires_inputs():
    with pytest.raises(ValueError):
        L.layout_panels(box(0, 0, 5, 5), tilt_deg=10)


def test_tilted_corners_front_low_back_high():
    lay = L.layout_panels(box(0, 0, 5, 5), panel_w=1, panel_h=2, gap=0,
                          tilt_deg=30, facing_deg=-90, sun_alt_deg=45)
    c = P.panel_corners_3d(FLAT, lay.panels[:1], standoff=0.1, facing=lay.facing, rise=lay.rise)[0]
    low = c[c[:, 1] <= c[:, 1].min() + 1e-9]   # front edge (lowest y, facing -y)
    high = c[c[:, 1] >= c[:, 1].max() - 1e-9]
    assert np.allclose(low[:, 2], 0.1) and np.allclose(high[:, 2], 0.1 + lay.rise)
    # real panel length along the slope is preserved
    slope = np.linalg.norm(high[0] - low[np.argmin(np.abs(low[:, 0] - high[0, 0]))])
    assert slope == pytest.approx(2 if lay.orientation == "portrait" else 1)


def test_quad_solid_volume_tilted():
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1.0]])  # 1 x sqrt2 tilted 45 deg
    s = P.quad_solid(c, 0.04)
    assert s.Volume() == pytest.approx(1 * np.sqrt(2) * 0.04 * 1e9, rel=1e-6)


# ---------- roof choice ----------

def _cand(z, n):
    return (PlaneFrame(np.array([0, 0, z]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                       np.array([0, 0, 1.0])), np.zeros((n, 3)))


def test_choose_roof_modes():
    cands = [_cand(0, 1000), _cand(6, 300), _cand(9, 20)]  # ground, roof, tiny chimney top
    assert design_panels.choose_roof(cands, "largest") == 0
    assert design_panels.choose_roof(cands, "highest") == 1   # chimney too small to count
    assert design_panels.choose_roof(cands, "largest", roof_index=2) == 2
    with pytest.raises(IndexError):
        design_panels.choose_roof(cands, roof_index=5)
    with pytest.raises(ValueError):
        design_panels.choose_roof(cands, "biggest")


def test_cli_tilt_needs_latitude():
    with pytest.raises(SystemExit):
        design_panels.main(["x", "--tilt", "10"])


# ---------- AR export ----------

def test_box_from_quad_closed_and_oriented():
    v, t = A.box_from_quad(np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0.0]]), 0.05)
    assert v.shape == (8, 3) and t.shape == (12, 3)
    assert np.isclose(v[:, 2].min(), -0.05) and np.isclose(v[:, 2].max(), 0)
    edges = {}
    for tri in t:  # watertight: every edge shared by exactly two triangles
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = tuple(sorted((a, b)))
            edges[k] = edges.get(k, 0) + 1
    assert set(edges.values()) == {2}


def test_polygon_mesh_area_matches_concave_outline():
    poly = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    v, f = A.polygon_mesh(poly, FLAT)
    a = sum(0.5 * np.linalg.norm(np.cross(v[i1] - v[i0], v[i2] - v[i0])) for i0, i1, i2 in f)
    assert a == pytest.approx(poly.area)


def test_write_glb_is_y_up_and_metric(tmp_path):
    import open3d as o3d
    frame = PlaneFrame(np.array([5.0, 5, 2]), FLAT.u, FLAT.v, FLAT.n)
    corners = P.panel_corners_3d(frame, [box(0, 0, 1, 2)], standoff=0.1)
    path = A.write_glb(tmp_path / "m.glb", frame, box(-1, -1, 3, 3), corners)
    b = path.read_bytes()
    assert b[:4] == b"glTF" and int.from_bytes(b[8:12], "little") == len(b)
    m = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)  # independent reader (Assimp)
    v = np.asarray(m.vertices)
    assert len(m.triangles) == 12 + 2 + len(A.polygon_mesh(box(-1, -1, 3, 3), frame)[1])
    assert v[:, 1].max() == pytest.approx(0.1 + A.GLASS_LIFT, abs=1e-5)  # glass on top, 0.1 m up (glTF Y)
    assert np.ptp(v[:, 0]) == pytest.approx(4, abs=1e-5)   # 4 m roof, origin-centred
    assert abs(v[:, 0].min() + 1) < 1e-5


# ---------- validation ----------

def test_parse_mlp(tmp_path):
    p = tmp_path / "a.mlp"
    p.write_text('<MeshLabProject><MeshGroup><MLMesh label="s1.ply" filename="s1.ply"><MLMatrix44>\n'
                 '1 0 0 2 \n0 1 0 3 \n0 0 1 4 \n0 0 0 1 \n</MLMatrix44></MLMesh></MeshGroup></MeshLabProject>')
    m = V.parse_mlp(p)["s1.ply"]
    assert np.allclose(m[:3, 3], [2, 3, 4])
    (tmp_path / "b.mlp").write_text("<MeshLabProject/>")
    with pytest.raises(ValueError):
        V.parse_mlp(tmp_path / "b.mlp")


def test_distance_stats():
    s = V.distance_stats([0.01, 0.03, 0.08, 0.2])
    assert s["within_2cm"] == 0.25 and s["within_5cm"] == 0.5 and s["within_10cm"] == 0.75
    assert s["median_m"] == pytest.approx(0.055)
    with pytest.raises(ValueError):
        V.distance_stats([])


def test_fit_plane_lsq_recovers_tilt():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (500, 2))
    pts = np.c_[xy, 0.1 * xy[:, 0] + 2]  # z = 0.1 x + 2
    n, c = V.fit_plane_lsq(pts)
    assert np.allclose(n, np.array([-0.1, 0, 1]) / np.sqrt(1.01), atol=1e-9)


def test_outline_metrics():
    m = V.outline_metrics(box(0, 0, 10, 10), box(0, 0, 10, 11))
    assert m["area_error_pct"] == pytest.approx(100 * (100 - 110) / 110)
    assert m["iou"] == pytest.approx(100 / 110)
    assert m["hausdorff_m"] == pytest.approx(1)
    assert V.outline_metrics(box(0, 0, 1, 1), box(0, 0, 1, 1))["iou"] == pytest.approx(1)


def test_corner_height_errors_and_inside():
    uv = np.c_[np.random.default_rng(1).uniform(0, 4, (2000, 2))]
    h = np.where(uv[:, 0] > 2, 0.05, 0.0)
    e = V.corner_height_errors([[1, 1], [3, 3], [50, 50]], uv, h)
    assert e[0] == pytest.approx(0) and e[1] == pytest.approx(0.05) and np.isnan(e[2])
    assert V.corners_inside([[1, 1], [5, 5]], box(0, 0, 4, 4)).tolist() == [True, False]


# ---------- integration: real ETH3D terrace vs laser scan ----------

ETH = Path(__file__).resolve().parents[1] / "data" / "eth3d" / "terrace"


@pytest.fixture
def eth3d_scan_available():
    if not (ETH / "mvs" / "scene_dense.ply").is_file() or not (ETH / "dslr_scan_eval" / "scan1.ply").is_file():
        pytest.skip("ETH3D terrace dense cloud / laser scans not present")


@pytest.mark.integration
def test_validate_eth3d_terrace_meets_target(eth3d_scan_available):
    import validate_eth3d
    r = validate_eth3d.validate("eth3d/terrace", voxel=0.05)
    assert r["cloud_accuracy"]["median_m"] < 0.05
    assert r["plane"]["normal_angle_err_deg"] < 1
    assert abs(r["plane"]["height_offset_m"]) < 0.05
    assert r["outline"]["iou"] > 0.85
    assert r["panels"]["corners_outside_true_roof"] == 0
    assert r["panels"]["corner_height_abs_median_m"] < 0.10


# ---------- realistic panels + AR-only model ----------

def _gltf_json(path):
    import json
    b = Path(path).read_bytes()
    n = int.from_bytes(b[12:16], "little")
    return json.loads(b[20:20 + n]), b


def test_cell_texture_png_dimensions_and_cells():
    from io import BytesIO
    from PIL import Image
    img = np.asarray(Image.open(BytesIO(A.cell_texture_png(width=800))).convert("RGB"), float) / 255
    h, w, _ = img.shape
    assert w == 800 and h == round(800 * 1.134 / 1.722)
    dark = img.mean(axis=2) < 0.2
    assert 0.6 < dark.mean() < 0.95                       # mostly cell area
    # count cells along the middle row: dark runs separated by light gaps
    row = dark[h // 12]
    runs = np.count_nonzero(np.diff(row.astype(int)) == 1) + int(row[0])
    assert runs == 20


def test_laminate_quad_inset_lift_and_uv_along_long_edge():
    c = np.array([[0, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0.0]])  # long edge is c0->c1
    v, t, uv = A.laminate_quad(c, frame_width=0.05, lift=0.01)
    assert np.allclose(v[:, 2], 0.01)
    assert v[:, 0].min() == pytest.approx(0.05) and v[:, 1].max() == pytest.approx(1.95)
    assert np.allclose(uv[1], [1, 0])                      # u runs along the 2 m edge
    v2, _, uv2 = A.laminate_quad(c[[0, 3, 2, 1]])           # same panel, other winding
    assert np.allclose(uv2[3], [1, 0])


def test_glb_has_textured_glass_and_metal_frame(tmp_path):
    corners = P.panel_corners_3d(FLAT, [box(0, 0, 1.7, 1.1)])
    j, b = _gltf_json(A.write_glb(tmp_path / "p.glb", FLAT, box(-1, -1, 3, 3), corners))
    mats = {m["name"]: m["pbrMetallicRoughness"] for m in j["materials"]}
    assert set(mats) == {"roof", "frame", "glass"}
    assert mats["frame"]["metallicFactor"] > 0.5
    assert "baseColorTexture" in mats["glass"]
    img_view = j["bufferViews"][j["images"][0]["bufferView"]]
    start = 20 + int.from_bytes(b[12:16], "little") + 8 + img_view["byteOffset"]
    assert b[start:start + 8] == b"\x89PNG\r\n\x1a\n"
    glass = next(p for p in j["meshes"][0]["primitives"] if j["materials"][p["material"]]["name"] == "glass")
    assert "TEXCOORD_0" in glass["attributes"]


def test_ar_glb_excludes_roof(tmp_path):
    corners = P.panel_corners_3d(FLAT, [box(0, 0, 1.7, 1.1)])
    j, _ = _gltf_json(A.write_glb(tmp_path / "a.glb", FLAT, box(-1, -1, 3, 3), corners, include_roof=False))
    assert {m["name"] for m in j["materials"]} == {"frame", "glass"}


# ---------- corner-tap anchor ----------

def test_anchor_points_long_edge_of_rotated_rect():
    from shapely import affinity
    poly = affinity.rotate(box(0, 0, 12, 5), 30, origin=(0, 0))
    a, b = A.anchor_points(poly)
    assert np.linalg.norm(b - a) == pytest.approx(12, abs=1e-6)
    d = (b - a) / 12
    assert abs(abs(d @ [np.cos(np.radians(30)), np.sin(np.radians(30))]) - 1) < 1e-6


def test_anchor_points_snap_to_outline_vertex():
    # chamfered corners: no rectangle corner is a real roof vertex
    poly = Polygon([(0.5, 0), (9.5, 0), (10, 0.5), (10, 5.5), (9.5, 6), (0.5, 6), (0, 5.5), (0, 0.5)])
    verts = {tuple(v) for v in np.asarray(poly.exterior.coords)}
    a, b = A.anchor_points(poly)
    assert tuple(a) in verts and tuple(b) in verts
    assert abs(a[0] - b[0]) >= 9                              # still spans the long side


def test_anchor_data_in_glb_frame(tmp_path):
    frame = PlaneFrame(np.array([5.0, 5, 2]), FLAT.u, FLAT.v, FLAT.n)
    outline, keep = box(0, 0, 10, 4), box(2, 1, 3, 2)
    d = A.anchor_data(frame, outline, keep)
    assert d["anchor_length_m"] == pytest.approx(10)
    for p in (d["anchor_a"], d["anchor_b"]):
        assert p[1] == pytest.approx(0)                  # on the roof plane (glTF y up)
    # glTF frame: x = u, z = -v ; outline v in [0,4] -> z in [-4,0]
    ring = np.array(d["outline"][0])
    assert ring[:, 0].min() == pytest.approx(0) and ring[:, 0].max() == pytest.approx(10)
    assert ring[:, 2].min() == pytest.approx(-4) and ring[:, 2].max() == pytest.approx(0)
    assert len(d["keepouts"]) == 1
    import json as _json
    path = A.write_anchor(tmp_path / "a.json", frame, outline, None)
    assert _json.loads(path.read_text())["keepouts"] == []


def _glb_json(b):
    import json
    import struct
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


def test_glb_bytes_writes_unit_normals_only_when_every_part_has_them():
    import struct
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    t = np.array([[0, 1, 2]])
    mats = {"a": A.Material((1, 0, 0)), "b": A.Material((0, 1, 0))}
    b = A.glb_bytes([(v, t, None, "a", np.tile([0, 0, 2.0], (3, 1))),
                     (v + 1, t, None, "b", np.tile([0, 0, 1.0], (3, 1))), (v, t, None, "b")], mats)
    g = _glb_json(b)
    prims = {g["materials"][p["material"]]["name"]: p for p in g["meshes"][0]["primitives"]}
    assert "NORMAL" in prims["a"]["attributes"] and "NORMAL" not in prims["b"]["attributes"]
    acc = g["accessors"][prims["a"]["attributes"]["NORMAL"]]
    assert acc["count"] == 3 and acc["type"] == "VEC3"
    view = g["bufferViews"][acc["bufferView"]]
    n = struct.unpack("<I", b[12:16])[0]
    start = 20 + n + 8 + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(b[start:start + 36], "<f4").reshape(3, 3)
    assert np.allclose(arr, [0, 0, 1])


def test_glb_bytes_tiled_textures_use_a_repeat_sampler():
    png = A.cell_texture_png(width=64, cols=2, rows=1)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    t = np.array([[0, 1, 2]])
    uv = np.array([[0, 0], [3, 0], [0, 3]], float)
    mats = {"a": A.Material((1, 1, 1), texture=png, tile=True), "b": A.Material((1, 1, 1), texture=png)}
    g = _glb_json(A.glb_bytes([(v, t, uv, "a"), (v, t, uv, "b")], mats))
    assert g["samplers"][1]["wrapS"] == 10497 and g["samplers"][1]["wrapT"] == 10497
    assert g["samplers"][0]["wrapS"] != 10497
    by = {m["name"]: g["textures"][m["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["sampler"]
          for m in g["materials"]}
    assert by == {"a": 1, "b": 0}


def test_image_mime_knows_png_and_jpeg_only():
    from rooftop.textures import jpeg
    assert A.image_mime(A.cell_texture_png(width=64, cols=2, rows=1)) == "image/png"
    assert A.image_mime(jpeg(np.zeros((4, 4, 3), np.uint8))) == "image/jpeg"
    for bad in (b"GIF89a....", b"\x89PNG\r\n\x1a", b"\xff\xd8", b""):
        with pytest.raises(ValueError, match="PNG or JPEG"):
            A.image_mime(bad)


def test_glb_bytes_embeds_jpeg_albedo_and_normal_maps():
    import struct
    from rooftop.textures import jpeg
    alb, nrm = jpeg(np.full((8, 8, 3), 200, np.uint8)), jpeg(np.full((8, 8, 3), [128, 128, 255], np.uint8))
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    t = np.array([[0, 1, 2]])
    uv = np.array([[0, 0], [2, 0], [0, 2]], float)
    mats = {"both": A.Material((1, 1, 1), texture=alb, tile=True, normal=nrm),
            "bump": A.Material((0.5, 0.5, 0.5), normal=nrm, tile=True),
            "flat": A.Material((0.2, 0.2, 0.2))}
    b = A.glb_bytes([(v, t, uv, "both"), (v, t, uv, "bump"), (v, t, uv, "flat")], mats)
    g = _glb_json(b)
    m = {x["name"]: x for x in g["materials"]}
    assert "baseColorTexture" in m["both"]["pbrMetallicRoughness"] and "normalTexture" in m["both"]
    assert "baseColorTexture" not in m["bump"]["pbrMetallicRoughness"] and "normalTexture" in m["bump"]
    assert "normalTexture" not in m["flat"] and "baseColorTexture" not in m["flat"]["pbrMetallicRoughness"]
    assert m["bump"]["pbrMetallicRoughness"]["baseColorFactor"] == [0.5, 0.5, 0.5, 1.0]
    assert [i["mimeType"] for i in g["images"]] == ["image/jpeg"] * 3
    prims = {g["materials"][p["material"]]["name"]: p for p in g["meshes"][0]["primitives"]}
    assert "TEXCOORD_0" in prims["both"]["attributes"] and "TEXCOORD_0" in prims["bump"]["attributes"]
    assert "TEXCOORD_0" not in prims["flat"]["attributes"]
    # the normal map's bytes are what was given, in its own texture on the repeat sampler
    tex = g["textures"][m["bump"]["normalTexture"]["index"]]
    assert tex["sampler"] == 1
    view = g["bufferViews"][g["images"][tex["source"]]["bufferView"]]
    n = struct.unpack("<I", b[12:16])[0]
    start = 20 + n + 8 + view["byteOffset"]
    assert b[start:start + view["byteLength"]] == nrm

    def tex_bytes(ref):
        view = g["bufferViews"][g["images"][g["textures"][ref["index"]]["source"]]["bufferView"]]
        return b[20 + n + 8 + view["byteOffset"]:][:view["byteLength"]]
    # with both maps, each slot holds its own image
    assert tex_bytes(m["both"]["normalTexture"]) == nrm
    assert tex_bytes(m["both"]["pbrMetallicRoughness"]["baseColorTexture"]) == alb
    uvacc = g["accessors"][prims["bump"]["attributes"]["TEXCOORD_0"]]
    uview = g["bufferViews"][uvacc["bufferView"]]
    got = np.frombuffer(b[20 + n + 8 + uview["byteOffset"]:][:24], "<f4").reshape(3, 2)
    assert np.allclose(got, uv) and uvacc["type"] == "VEC2" and uvacc["count"] == 3
    assert len(b) % 4 == 0 and struct.unpack("<I", b[8:12])[0] == len(b)
