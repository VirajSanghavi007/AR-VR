"""Unit, smoke and integration tests for scripts/run_mutants.py (no data/ duplication)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mutants as RM  # noqa: E402


def fake_project(tmp_path, data_mb=1):
    root = tmp_path / "proj"
    (root / "rooftop").mkdir(parents=True)
    (root / "rooftop" / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_mod.py").write_text(
        "from rooftop.mod import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\npythonpath = .\n")
    (root / "data").mkdir()
    (root / "data" / "big.bin").write_bytes(b"x" * (data_mb * 2**20))
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")
    return root


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "copies"
    b.mkdir()
    return b


# ---- unit ---------------------------------------------------------------

def test_copy_links_data_and_skips_git(tmp_path, base):
    root = fake_project(tmp_path, data_mb=3)
    copy = RM.make_copy(root, base)
    assert (copy / "rooftop" / "mod.py").is_file() and (copy / "pytest.ini").is_file()
    assert not (copy / ".git").exists()
    assert RM._is_link(copy / "data")                       # junction, not a folder copy
    assert (copy / "data" / "big.bin").stat().st_size == 3 * 2**20
    assert RM.tree_bytes(copy) < 2**20                      # the 3 MB file is not counted/copied
    RM.cleanup_copy(copy)


def test_cleanup_keeps_real_data(tmp_path, base):
    root = fake_project(tmp_path)
    copy = RM.make_copy(root, base)
    RM.cleanup_copy(copy)
    assert not copy.exists()
    assert (root / "data" / "big.bin").stat().st_size == 2**20


def test_cleanup_unlinks_data_before_deleting(tmp_path, base, monkeypatch):
    root = fake_project(tmp_path)
    copy = RM.make_copy(root, base)
    seen = []
    real_rmtree = RM.shutil.rmtree
    monkeypatch.setattr(RM.shutil, "rmtree",
                        lambda p, **kw: (seen.append((Path(p) / "data").exists()), real_rmtree(p, **kw)))
    RM.cleanup_copy(copy)
    assert seen == [False]                                  # the junction was gone before rmtree ran


def test_remove_link_refuses_real_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "keep.txt").write_text("k")
    with pytest.raises(OSError):
        RM.remove_link(real)
    assert (real / "keep.txt").exists()


def test_copy_too_big_raises_and_cleans_up(tmp_path, base, monkeypatch):
    root = fake_project(tmp_path)
    (root / "rooftop" / "blob.bin").write_bytes(b"y" * 2**20)   # big file in a copied folder
    monkeypatch.setattr(RM, "MAX_COPY_BYTES", 2**19)
    with pytest.raises(RuntimeError, match="copied instead of linked"):
        RM.make_copy(root, base)
    assert list(base.iterdir()) == []
    assert (root / "data" / "big.bin").exists()


def test_apply_mutant_replaces_once_and_errors_on_missing(tmp_path):
    (tmp_path / "f.py").write_bytes(b"a + b\r\na + b\r\n")
    RM.apply_mutant(tmp_path, "f.py", "a + b", "a - b")
    assert (tmp_path / "f.py").read_bytes() == b"a - b\r\na + b\r\n"
    with pytest.raises(ValueError, match="not found"):
        RM.apply_mutant(tmp_path, "f.py", "a * b", "a / b")


def test_clean_stale_removes_only_old_copies(tmp_path, base):
    root = fake_project(tmp_path)
    old, new = RM.make_copy(root, base), RM.make_copy(root, base)
    stray = base / (RM.PREFIX + "note.txt")                 # an old file, not a copy: left alone
    stray.write_text("s")
    past = time.time() - RM.STALE_AFTER_S - 60
    os.utime(old, (past, past))
    os.utime(stray, (past, past))
    assert RM.clean_stale(base) == 1
    assert stray.exists()
    assert not old.exists() and new.exists()
    assert (root / "data" / "big.bin").exists()
    RM.cleanup_copy(new)


def test_run_one_reports_missing_string_as_error(tmp_path, base):
    root = fake_project(tmp_path)
    m = {"file": "rooftop/mod.py", "old": "nope", "new": "x", "tests": ["tests"]}
    assert RM.run_one(m, True, base, root).startswith("error:")
    assert list(base.iterdir()) == []


def test_run_all_fails_fast_on_broken_baseline(tmp_path, base, monkeypatch):
    root = fake_project(tmp_path)
    monkeypatch.setattr(RM, "run_tests", lambda copy, tests: False)
    m = {"file": "rooftop/mod.py", "old": "a + b", "new": "a - b", "tests": ["tests"]}
    with pytest.raises(RuntimeError, match="fail before mutating"):
        RM.run_all([m], 2, base, root, log=lambda s: None)


def test_run_all_classifies_and_logs_progress(tmp_path, base, monkeypatch):
    root = fake_project(tmp_path)
    monkeypatch.setattr(RM, "run_tests",
                        lambda copy, tests: "a - b" not in (copy / "rooftop" / "mod.py").read_text())
    ms = [{"file": "rooftop/mod.py", "old": "a + b", "new": "a - b", "tests": ["tests"]},
          {"file": "rooftop/mod.py", "old": "def add", "new": "def  add", "tests": ["tests"]}]
    logs = []
    assert RM.run_all(ms, 2, base, root, log=logs.append) == ["killed", "survived"]
    assert logs[-1].startswith("2/2 (100%)") and "eta" in logs[-1]
    assert list(base.iterdir()) == []


# ---- smoke: real pytest subprocesses on a tiny project ------------------

def test_smoke_main_end_to_end(tmp_path, base, capsys):
    root = fake_project(tmp_path)
    spec = tmp_path / "m.json"
    spec.write_text(json.dumps([
        {"file": "rooftop/mod.py", "old": "a + b", "new": "a - b", "tests": ["tests/test_mod.py"]},
        {"file": "rooftop/mod.py", "old": "def add", "new": "def  add", "tests": ["tests/test_mod.py"]}]))
    rc = RM.main([str(spec), "--root", str(root), "--base", str(base), "--workers", "2"])
    out = capsys.readouterr().out
    assert rc == 1 and "1/2 killed" in out                  # second mutant is harmless, survives
    assert "killed     rooftop/mod.py" in out and "survived   rooftop/mod.py" in out
    assert list(base.iterdir()) == []
    assert (root / "data" / "big.bin").exists()


# ---- integration: the real arvr tree ------------------------------------

@pytest.mark.skipif(not (ROOT / "data").is_dir(), reason="no real data/ folder")
def test_real_copy_is_small_and_data_survives(tmp_path):
    before = sorted(p.name for p in (ROOT / "data").iterdir())
    copy = RM.make_copy(ROOT, tmp_path)
    try:
        assert RM.tree_bytes(copy) < RM.MAX_COPY_BYTES
        assert sorted(p.name for p in (copy / "data").iterdir()) == before
        assert (copy / "rooftop" / "geometry.py").is_file() and (copy / "tests").is_dir()
    finally:
        RM.cleanup_copy(copy)
    assert not copy.exists()
    assert sorted(p.name for p in (ROOT / "data").iterdir()) == before
