"""scripts/size_check.py: unit (resize rule, lengths, tape rows, report text, exit codes), smoke (the CLI on the
sample design), integration (its length rule agrees with preview/blocks.js through node, on real design files)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import size_check as S  # noqa: E402

SAMPLE = ROOT / "preview" / "samples" / "site_design_sample.json"
BLOCKS = (ROOT / "preview" / "blocks.js").as_uri()


def wall(pts, w, base_w, oid="w1"):
    return {"id": oid, "type": "walls", "x": 0, "z": 0, "w": w, "h": 2.4, "d": 0.12, "pts": pts, "base": {"w": base_w, "d": 0}}


def design(objs=(), checks=()):
    return {"kind": "arvr-site-design", "version": 1, "options": {"A": list(objs), "B": [], "C": []}, "checks": list(checks)}


# ---------- unit ----------

def test_scaled_pts_wall_line_stretches_both_ways_by_width():
    o = wall([[-1, 0], [1, 0.5]], w=4, base_w=2)
    assert S.scaled_pts(o) == [[-2, 0], [2, 1]]


def test_scaled_pts_slab_uses_its_own_depth_ratio():
    o = {"type": "slab", "w": 4, "d": 1, "pts": [[-1, -1], [1, 1]], "base": {"w": 2, "d": 2}}
    assert S.scaled_pts(o) == [[-2, -0.5], [2, 0.5]]


def test_scaled_pts_without_base_or_tiny_base_is_unscaled():
    assert S.scaled_pts({"type": "walls", "w": 5, "pts": [[1, 2]]}) == [[1, 2]]
    assert S.scaled_pts({"type": "walls", "w": 5, "pts": [[1, 2]], "base": {"w": 1e-9}}) == [[1, 2]]
    slab = {"type": "slab", "w": 2, "d": 9, "pts": [[1, 1]], "base": {"w": 1, "d": 0}}
    assert S.scaled_pts(slab) == [[2, 2]]                     # no depth to scale by: falls back to the width ratio


def test_main_length_sums_wall_segments_and_uses_width_otherwise():
    o = wall([[0, 0], [3, 0], [3, 4]], w=3, base_w=3)
    assert S.main_length(o) == pytest.approx(7)
    assert S.main_length({"type": "bench", "w": 1.6}) == 1.6
    assert S.main_length({"type": "walls", "w": 2.5, "pts": []}) == 2.5


def test_load_rejects_other_files(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"kind": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a site design file"):
        S.load(p)
    p.write_text(json.dumps(design()), encoding="utf-8")
    assert S.load(p)["kind"] == "arvr-site-design"


def test_find_object_searches_every_option():
    rec = design()
    rec["options"]["C"] = [{"id": "c9", "type": "bench"}]
    assert S.find_object(rec, "c9")["type"] == "bench"
    assert S.find_object(rec, "nope") is None


def test_add_tape_adds_and_replaces_a_check():
    rec = design([wall([[-1.5, 0], [1.5, 0]], w=3, base_w=3)], checks=[{"id": "w1", "errMm": 999}, {"id": "other", "errMm": 1}])
    c = S.add_tape(rec, "w1", 2.98)
    assert c["drawn"] == 3.0 and c["tape"] == 2.98 and c["errMm"] == 20 and c["what"] == "Wall line" and c["tags"] == []
    assert [x["id"] for x in rec["checks"]] == ["other", "w1"]          # the old w1 row is replaced, not kept
    assert S.add_tape(rec, "w1", 3.01)["errMm"] == -10
    assert len(rec["checks"]) == 2


def test_add_tape_names_blocks_and_rejects_unknown_ids():
    rec = design([{"id": "b1", "type": "bench", "w": 1.6}, {"id": "s1", "type": "slab", "w": 4}])
    assert S.add_tape(rec, "b1", 1.6)["what"] == "Bench"
    assert S.add_tape(rec, "s1", 4)["what"] == "Outline"
    with pytest.raises(ValueError, match="no object 'zz'"):
        S.add_tape(rec, "zz", 1)


def row(oid, err, drawn=3.0, tags=(0, 1)):
    return {"id": oid, "what": "Wall line", "drawn": drawn, "tape": drawn - err / 1000, "errMm": err, "tags": list(tags), "at": "2026-11-11T10:00:00"}


def test_report_without_checks_says_how_to_make_one():
    text = S.report([("a.json", design())], 50)
    assert "No checks in these files yet" in text and "Tolerance: 50 mm" in text and "|" not in text


def test_report_rows_summary_and_tolerance_edge():
    files = [("a.json", design(checks=[row("w1", 50), row("w2", -51, tags=())])), ("b.json", design(checks=[row("w3", 10)]))]
    text = S.report(files, 50)
    lines = [ln for ln in text.splitlines() if ln.startswith("| a.json") or ln.startswith("| b.json")]
    assert len(lines) == 3
    assert lines[0].endswith("| +50 | 0, 1 | 2026-11-11T10:00 | pass |")       # exactly on the tolerance passes
    assert lines[1].endswith("| -51 | - | 2026-11-11T10:00 | FAIL |")
    assert "| b.json | Wall line w3 | 3.000 | 2.990 |" in lines[2]
    assert "**2 of 3 within 50 mm.** Mean error 37 mm, worst 51 mm, signed mean +3 mm" in text


def test_run_exit_codes_and_out_file(tmp_path, capsys):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(design(checks=[row("w1", 20)])), encoding="utf-8")
    bad.write_text(json.dumps(design(checks=[row("w1", 80)])), encoding="utf-8")
    assert S.run([str(good)]) == 0
    assert "| good.json |" in capsys.readouterr().out
    assert S.run([str(bad)]) == 1
    assert S.run([str(bad), "--tol-mm", "100"]) == 0
    out = tmp_path / "r.md"
    assert S.run([str(good), str(bad), "-o", str(out)]) == 1
    assert "wrote" in capsys.readouterr().out and out.read_text(encoding="utf-8").count("| Wall line w1 |") == 2


def test_run_tape_goes_into_the_first_file_only(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(design([wall([[-1.5, 0], [1.5, 0]], w=3, base_w=3)])), encoding="utf-8")
    b.write_text(json.dumps(design([wall([[-1.5, 0], [1.5, 0]], w=3, base_w=3)])), encoding="utf-8")
    assert S.run([str(a), str(b), "--tape", "w1=2.9"]) == 1                     # 100 mm off
    text = capsys.readouterr().out
    assert "| a.json | Wall line w1 | 3.000 | 2.900 | +100 |" in text and "| b.json |" not in text
    assert json.loads(a.read_text(encoding="utf-8"))["checks"] == []        # the file on disk is not rewritten


# ---------- smoke ----------

def test_cli_on_the_sample_design(tmp_path):
    out = tmp_path / "report.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "size_check.py"), str(SAMPLE), "--tape", "b1=10.0",
                        "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert "| site_design_sample.json | Wall line b1 | 10.000 | 10.000 | +0 |" in text and "**1 of 1 within 50 mm.**" in text
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "size_check.py"), str(SAMPLE)], capture_output=True, text=True)
    assert r.returncode == 0 and "No checks in these files yet" in r.stdout


# ---------- integration: the same length rule as the page ----------

@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def js_length(o):
    code = (f'import * as B from "{BLOCKS}"; const o = {json.dumps(o)};'
            'console.log(JSON.stringify(o.type === "walls" ? B.polyLength(B.scaledPts(o)) : o.w));')
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


@pytest.mark.parametrize("o", [
    wall([[-2, 1.7], [-2, -1.3], [2, -1.3], [2, 1.7]], w=5.2, base_w=4),
    wall([[0, 0], [1, 1]], w=1, base_w=1),
    {"id": "s", "type": "slab", "w": 3, "d": 2, "pts": [[-1, -1], [1, -1], [1, 1]], "base": {"w": 2, "d": 2}},
])
def test_length_matches_blocks_js(node_available, o):
    assert S.main_length(o) == pytest.approx(js_length(o), abs=1e-9)


def test_sample_wall_line_length_matches_blocks_js(node_available):
    rec = S.load(SAMPLE)
    o = S.find_object(rec, "b1")
    assert S.main_length(o) == pytest.approx(js_length(o)) == pytest.approx(10.0)
