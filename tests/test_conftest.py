"""tests/conftest.py: the Chrome page file is one scheduling unit, every other test is its own."""
import re
import subprocess
import sys
from pathlib import Path

from conftest import ONE_WORKER, scope_of

ROOT = Path(__file__).resolve().parents[1]


def test_scope_of_groups_only_the_listed_files():
    browser_files = {f"tests/{f.name}" for f in Path(__file__).parent.glob("test_*.py")
                     if f.name != "test_conftest.py" and "sync_playwright" in f.read_text(encoding="utf-8")}
    assert browser_files and browser_files <= set(ONE_WORKER), browser_files - set(ONE_WORKER)
    assert scope_of("tests/test_site_page.py::test_a") == "tests/test_site_page.py"
    assert scope_of("tests/test_site_page.py::test_b[1]") == "tests/test_site_page.py"
    assert scope_of("tests/test_blocks_js.py::test_a") == "tests/test_blocks_js.py::test_a"
    assert scope_of("tests/test_x.py::Cls::test_a") == "tests/test_x.py::Cls::test_a"


def test_under_xdist_the_listed_file_runs_on_one_worker(tmp_path):
    """Smoke + integration: a real `pytest -n 3` run of a copy of the hook on throwaway test files."""
    (tmp_path / "tests").mkdir()
    conf = re.sub(r"ONE_WORKER = \([^)]*\)", 'ONE_WORKER = ("tests/test_one.py",)',
                  (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    assert 'ONE_WORKER = ("tests/test_one.py",)' in conf
    (tmp_path / "tests" / "conftest.py").write_text(conf, encoding="utf-8")
    body = "import os, time\n" + "".join(
        f"def test_{i}():\n    time.sleep(0.2)\n    print('WORKER', os.environ['PYTEST_XDIST_WORKER'])\n" for i in range(6))
    for name in ("test_one.py", "test_many.py"):
        (tmp_path / "tests" / name).write_text(body, encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "pytest", "-n", "3", "-v", "-p", "no:cacheprovider", "tests"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=600).stdout
    assert "12 passed" in out, out
    workers = lambda f: {l.split()[0] for l in out.splitlines() if f"{f}::" in l and "PASSED" in l}
    assert len(workers("test_one.py")) == 1, out
    assert len(workers("test_many.py")) > 1, out


def test_runs_without_xdist(tmp_path):
    """The mutation runner uses -p no:xdist: the xdist hook must not make pytest refuse the conftest."""
    (tmp_path / "conftest.py").write_text((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:xdist", "-p", "no:cacheprovider"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout + r.stderr


def test_test_servers_get_a_long_listen_backlog():
    import http.server
    assert http.server.ThreadingHTTPServer.request_queue_size >= 64
