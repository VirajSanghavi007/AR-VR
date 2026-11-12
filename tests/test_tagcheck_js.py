"""Browser AprilTag checker: unit (preview/tagcheck.js through node, the 36h11 code list),
smoke (make_tag_js CLI, tags.html wiring), integration (vendored js-aruco2 + generated
dictionary decode synthetic photos exactly like OpenCV)."""
import json
import re
import shutil
import struct
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pytest

from rooftop import tags as T

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
JS = (PREVIEW / "tagcheck.js").as_uri()
sys.path.insert(0, str(ROOT / "scripts"))
import make_tag_js  # noqa: E402


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as C from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


SQ = [[0, 0], [100, 0], [100, 100], [0, 100]]


# ---------- unit: tagcheck.js ----------

def test_parse_ids(node_available):
    assert run('C.parseIds("0-3, 7 2")') == [0, 1, 2, 3, 7]
    assert run('C.parseIds("  ")') == []
    for bad in ["a", "3-1", "0-587"]:
        with pytest.raises(RuntimeError):
            run(f'C.parseIds("{bad}")')


def test_edges_and_to_markers_scale(node_available):
    assert run(f"C.edgeLengths({SQ})") == [100, 100, 100, 100]
    r = run("C.toMarkers([{id: 5, corners: [{x:1,y:2},{x:3,y:2},{x:3,y:4},{x:1,y:4}]}], 2)")
    assert r == [{"id": 5, "corners": [[2, 4], [6, 4], [6, 8], [2, 8]]}]


def test_assess_frame_flags(node_available):
    small = [[0, 0], [20, 0], [20, 20], [0, 20]]
    r = run(f"C.assessFrame([{{id:0,corners:{SQ}}},{{id:9,corners:{SQ}}},{{id:1,corners:{small}}},"
            f"{{id:2,corners:{SQ}}},{{id:2,corners:{SQ}}}], [0,1,2,3])")
    by = [(t["id"], t["problems"]) for t in r]
    assert by[0] == (0, [])
    assert by[1] == (9, ["not in your id list"])
    assert by[2][0] == 1 and by[2][1][0].startswith("small (20 px")
    assert by[3][1] == by[4][1] == ["same id twice in one photo"]
    assert r[0]["minEdge"] == 100
    assert run(f"C.assessFrame([{{id:0,corners:{SQ}}}], [0], 101)")[0]["problems"][0].startswith("small")


def sq(x, y, e):
    return [[x, y], [x + e, y], [x + e, y + e], [x, y + e]]


def test_merge_nested_keeps_the_outer_outline(node_available):
    # one tag found twice (outline + a contour 3 px inside), next to a different tag: the bigger copy stays, order kept
    ms = [{"id": 2, "corners": sq(3, 3, 94)}, {"id": 1, "corners": sq(300, 0, 50)}, {"id": 2, "corners": sq(0, 0, 100)}]
    r = run(f"C.mergeNested({json.dumps(ms)})")
    assert r == [ms[1], ms[2]]
    assert run(f"C.mergeNested({json.dumps(ms[::-1])})") == [ms[2], ms[1]]


def test_merge_nested_leaves_real_duplicates_and_other_ids(node_available):
    far = [{"id": 2, "corners": sq(0, 0, 100)}, {"id": 2, "corners": sq(400, 0, 100)}]
    assert run(f"C.mergeNested({json.dumps(far)})") == far                  # same id at two places: a real duplicate
    edge = [{"id": 2, "corners": sq(0, 0, 100)}, {"id": 2, "corners": sq(49, 0, 100)}, {"id": 2, "corners": sq(0, 51, 100)}]
    assert run(f"C.mergeNested({json.dumps(edge)})") == [edge[0], edge[2]]  # under half an edge apart: same tag
    other = [{"id": 2, "corners": sq(0, 0, 100)}, {"id": 3, "corners": sq(3, 3, 94)}]
    assert run(f"C.mergeNested({json.dumps(other)})") == other              # a different id is never merged
    assert run("C.mergeNested([])") == []
    small_first = [{"id": 4, "corners": sq(10, 10, 80)}, {"id": 4, "corners": sq(0, 0, 100)}]
    assert run(f"C.mergeNested({json.dumps(small_first)})") == [small_first[1]]


def test_pages_merge_nested_detections():
    tags = (PREVIEW / "tags.html").read_text(encoding="utf-8")
    assert 'mergeNested, assessFrame, summarize, MIN_VIEWS, GOOD_VIEWS } from "./tagcheck.js?v=2";' in tags
    assert "return mergeNested(toMarkers(detector.detect(img), 1 / k));" in tags
    tagar = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert 'import { mergeNested } from "./tagcheck.js?v=2";' in tagar
    assert "chooseTags(mergeNested(found));" in tagar


def photo(name, *ids, edge=100):
    tags = ",".join(f"{{id:{i},minEdge:{edge},problems:[]}}" for i in ids)
    return f'{{name:"{name}",tags:[{tags}]}}'


def test_summarize_ready_needs_two_tags_in_two_photos(node_available):
    s = run(f"C.summarize([{photo('a', 0, 1)},{photo('b', 0, 1, 2)},{photo('c', 0)}], [0,1,2,3])")
    assert s["views"] == {"0": 3, "1": 2, "2": 1, "3": 0}
    assert s["usable"] == [0, 1] and s["ready"] is True
    assert s["missing"] == [3] and s["tooFew"] == [2] and s["weak"] == [1]
    assert any("Never seen: id 3" in m for m in s["messages"])
    assert any("Only 1 photo: id 2" in m for m in s["messages"])
    # one usable tag gives a scale but no cross-check: not ready
    assert run(f"C.summarize([{photo('a', 0)},{photo('b', 0)}], [0,1])")["ready"] is False
    assert run(f"C.summarize([{photo('a', 0, 1)}], [0,1])")["ready"] is False


def test_summarize_duplicates_unknown_small_and_empty(node_available):
    s = run(f"C.summarize([{photo('a', 0, 0, 1)},{photo('b', 0, 1, 7)},{photo('c')},"
            f"{photo('d', 0, 1, edge=10)}], [0,1])")
    assert s["duplicates"] == ["a"] and s["ready"] is False
    assert s["views"] == {"0": 3, "1": 3}                   # a duplicate counts once per photo
    assert s["unknown"] == [7] and s["noTags"] == ["c"]
    assert s["small"] == ["d (id 0)", "d (id 1)"]
    assert run(f"C.summarize([{photo('a', 0, 1)},{photo('b', 0, 1)}], [0,1])")["messages"] == \
        ["Only 2 photos: id 0, 1 (works, 3+ is steadier)."]


# ---------- unit: code list ----------

def test_dictionary_codes_match_opencv_and_stay_apart():
    codes = T.dictionary_codes()
    assert len(codes) == 587 and len(set(codes)) == 587
    assert all(0 < c < 2 ** 36 for c in codes)
    # OpenCV's own byte list for id 0 encodes the same 36 bits (rotation 0)
    bits = cv2.aruco.Dictionary.getBitsFromByteList(T.DICT.bytesList[0:1], 6).ravel()
    assert int("".join(map(str, bits)), 2) == codes[0]
    d = min(bin(a ^ b).count("1") for a, b in combinations(codes[:120], 2))
    assert d >= 11                                          # so tau 5 can never confuse two ids


def test_write_js_dictionary(tmp_path):
    p, n = T.write_js_dictionary(tmp_path / "d.js", max_errors=3)
    txt = p.read_text()
    assert n == 587 and "tau: 4" in txt and txt.count('"') == 2 * 587
    for bad in (-1, 6):
        with pytest.raises(ValueError):
            T.write_js_dictionary(tmp_path / "x.js", bad)


# ---------- smoke ----------

def test_make_tag_js_cli_and_committed_file_is_current(tmp_path, capsys):
    out = make_tag_js.main(["--out", str(tmp_path / "t.js")])
    assert "587 tag ids" in capsys.readouterr().out
    assert out.read_text() == (PREVIEW / "tag36h11.js").read_text(), "rerun scripts/make_tag_js.py"


def test_tags_page_references_existing_files():
    html = (PREVIEW / "tags.html").read_text(encoding="utf-8")
    for src in re.findall(r'(?:src|from)="\.?/?([\w./-]+?\.js)(?:\?[^"]*)?"', html):
        assert (PREVIEW / src).exists(), src
    assert 'dictionaryName: "APRILTAG_36h11"' in html
    assert 'href="tags.html"' in (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    assert 'href="tagar.html"' in html


# ---------- integration: real decode ----------

DECODE = """
require(process.argv[1] + "/vendor/js-aruco2/cv.js");
const AR = require(process.argv[1] + "/tag36h11.js");
const buf = require("fs").readFileSync(process.argv[2]);
const w = buf.readUInt32LE(0), h = buf.readUInt32LE(4);
const m = new AR.Detector({ dictionaryName: "APRILTAG_36h11" })
  .detectImage(w, h, new Uint8ClampedArray(buf.buffer, buf.byteOffset + 8, w * h * 4));
console.log(JSON.stringify(m.map((x) => ({ id: x.id, corners: x.corners.map((c) => [c.x, c.y]) }))));
"""


def synthetic(path, ids, flip_bits=0):
    """Gray table with perspective-warped tags; returns OpenCV's own detection for comparison."""
    W, H = 1600, 1200
    img = np.full((H, W), 200, np.uint8)
    for k, tid in enumerate(ids):
        t = T.tag_image(tid, cell_px=20)
        if flip_bits:                                       # damage data cells (print smudge)
            for j in range(flip_bits):
                r, c = 2 + j, 2 + (j * 3) % 6
                t[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20] ^= 255
        s = t.shape[0]
        a = np.radians(30 * k + 10)
        src = np.float32([[0, 0], [s, 0], [s, s], [0, s]]) - s / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        dst = (src @ R.T) * [1.0, 0.6] * (1 + 0.2 * (src[:, 1:2] / s))
        M = cv2.getPerspectiveTransform(src + s / 2, np.float32(dst + [400 + 800 * (k % 2), 300 + 600 * (k // 2)]))
        mask = cv2.warpPerspective(np.full_like(t, 255), M, (W, H)) > 0
        img[mask] = cv2.warpPerspective(t, M, (W, H))[mask]
    rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    path.write_bytes(struct.pack("<II", W, H) + rgba.tobytes())
    return T.detect_tags(img)


def decode(path):
    out = subprocess.run(["node", "-e", DECODE, str(PREVIEW), str(path)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return {m["id"]: np.array(m["corners"]) for m in json.loads(out.stdout)}


def test_browser_decoder_matches_opencv(tmp_path, node_available):
    ids = [0, 1, 2, 586]
    cv_tags = synthetic(tmp_path / "a.bin", ids)
    js_tags = decode(tmp_path / "a.bin")
    assert sorted(cv_tags) == sorted(js_tags) == ids
    for tid in ids:                                         # same corner order as OpenCV / tag_scale.py
        assert np.abs(js_tags[tid] - cv_tags[tid]).max() < 3


def test_browser_decoder_corrects_a_few_bad_bits(tmp_path, node_available):
    synthetic(tmp_path / "b.bin", [3, 4], flip_bits=2)
    assert sorted(decode(tmp_path / "b.bin")) == [3, 4]
