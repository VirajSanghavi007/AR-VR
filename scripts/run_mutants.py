"""Mutation testing for arvr on throwaway copies that never duplicate data/.

Each mutant gets its own temp copy of the code (rooftop/, scripts/, tests/,
preview/, small root files). The heavy folders (data/, outputs/, tools/) are
directory junctions back to the real project, so a copy costs about 3 MB
instead of 5.8 GB. Copies are deleted when the mutant finishes, links first,
and copies left behind by a killed run are removed at the next start.

Spec file (JSON list):
    [{"file": "rooftop/geometry.py", "old": "a < b", "new": "a <= b",
      "tests": ["tests/test_geometry.py"]}, ...]

Usage:
    python scripts/run_mutants.py mutants.json [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "arvr_mut_"
LINKED = ("data", "outputs", "tools")          # junctioned, never copied
SKIPPED = {".git", ".venv", ".claude", ".pytest_cache", "__pycache__"} | set(LINKED)
MAX_COPY_BYTES = 100 * 2**20                   # a copy above this means data leaked in
STALE_AFTER_S = 6 * 3600
BELOW_NORMAL = 0x00004000


def make_link(target: Path, link: Path) -> None:
    """Directory junction on Windows (no admin needed), symlink elsewhere."""
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def remove_link(link: Path) -> None:
    """Remove a junction/symlink without touching what it points at.

    os.rmdir on a junction deletes only the link; on a real, non-empty folder
    it fails, so this can never empty the real data/.
    """
    if os.name != "nt" and link.is_symlink():
        link.unlink()
    else:
        os.rmdir(link)


def tree_bytes(path: Path) -> int:
    """Size of files under path, not following links or junctions."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not _is_link(Path(dirpath) / d)]
        total += sum((Path(dirpath) / f).lstat().st_size for f in filenames)
    return total


def _is_link(p: Path) -> bool:
    return p.is_symlink() or bool(getattr(p.lstat(), "st_reparse_tag", 0))


def make_copy(root: Path = ROOT, base: Path | None = None) -> Path:
    """Copy the code of root into a fresh temp folder and junction the heavy folders."""
    dest = Path(tempfile.mkdtemp(prefix=PREFIX, dir=base))
    ignore = shutil.ignore_patterns(*SKIPPED, "*.pyc")
    try:
        for item in root.iterdir():
            if item.name in SKIPPED:
                continue
            if item.is_dir():
                shutil.copytree(item, dest / item.name, ignore=ignore)
            else:
                shutil.copy2(item, dest / item.name)
        for name in LINKED:
            if (root / name).is_dir():
                make_link(root / name, dest / name)
        size = tree_bytes(dest)
        if size > MAX_COPY_BYTES:
            raise RuntimeError(f"copy is {size / 2**20:.0f} MB (> {MAX_COPY_BYTES / 2**20:.0f} MB): "
                               "a large folder is being copied instead of linked")
    except BaseException:
        cleanup_copy(dest)
        raise
    return dest


def cleanup_copy(dest: Path) -> None:
    """Unlink the junctions first, then delete the copy."""
    for name in LINKED:
        p = dest / name
        if p.exists() or p.is_symlink():
            if _is_link(p):
                remove_link(p)
    shutil.rmtree(dest, ignore_errors=True)


def clean_stale(base: Path | None = None, older_than_s: float = STALE_AFTER_S) -> int:
    """Remove copies a killed run left behind. Returns how many were removed."""
    base = Path(base or tempfile.gettempdir())
    now, n = time.time(), 0
    for d in base.glob(PREFIX + "*"):
        if d.is_dir() and now - d.stat().st_mtime > older_than_s:
            cleanup_copy(d)
            n += 1
    return n


def apply_mutant(copy: Path, file: str, old: str, new: str) -> None:
    path = copy / file
    text = path.read_bytes().decode("utf-8")
    if old not in text:
        raise ValueError(f"original string not found in {file}: {old!r}")
    path.write_bytes(text.replace(old, new, 1).encode("utf-8"))


def run_tests(copy: Path, tests: list[str]) -> bool:
    """True if the tests pass in the copy."""
    flags = BELOW_NORMAL if os.name == "nt" else 0
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:xdist", "-p", "no:cacheprovider", *tests]
    env = {**os.environ, "PYTHONPATH": str(copy), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(cmd, cwd=copy, env=env, capture_output=True, creationflags=flags)
    return r.returncode == 0


def run_one(m: dict, mutate: bool, base: Path | None = None, root: Path = ROOT) -> str:
    """'killed', 'survived', 'error: ...' for a mutant; 'pass'/'fail' for a baseline run."""
    copy = make_copy(root, base)
    try:
        if mutate:
            try:
                apply_mutant(copy, m["file"], m["old"], m["new"])
            except ValueError as e:
                return f"error: {e}"
            return "survived" if run_tests(copy, m["tests"]) else "killed"
        return "pass" if run_tests(copy, m["tests"]) else "fail"
    finally:
        cleanup_copy(copy)


def run_all(mutants: list[dict], workers: int, base: Path | None = None, root: Path = ROOT,
            log=print) -> list[str]:
    """Baseline each distinct test subset unmutated, then run every mutant."""
    removed = clean_stale(base)
    if removed:
        log(f"removed {removed} stale copies")
    subsets = {tuple(m["tests"]) for m in mutants}
    with ThreadPoolExecutor(workers) as pool:
        base_ok = dict(zip(subsets, pool.map(
            lambda s: run_one({"tests": list(s)}, False, base, root), subsets)))
    bad = [s for s, r in base_ok.items() if r != "pass"]
    if bad:
        raise RuntimeError(f"tests fail before mutating, fix these first: {bad}")

    results: list[str] = [""] * len(mutants)
    t0, done, every = time.time(), 0, max(1, len(mutants) // 20)
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(run_one, m, True, base, root): i for i, m in enumerate(mutants)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            if done % every == 0 or done == len(mutants):
                el = time.time() - t0
                log(f"{done}/{len(mutants)} ({100 * done // len(mutants)}%) - {el:.0f}s elapsed, "
                    f"eta {el / done * (len(mutants) - done):.0f}s")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--root", type=Path, default=ROOT, help="project to copy (default: arvr)")
    ap.add_argument("--base", type=Path, default=None, help="where copies go (default: temp dir)")
    a = ap.parse_args(argv)
    mutants = json.loads(a.spec.read_text(encoding="utf-8"))
    results = run_all(mutants, a.workers, a.base, a.root)
    for m, r in zip(mutants, results):
        print(f"{r:10} {m['file']}: {m['old']!r} -> {m['new']!r}")
    survived = sum(r != "killed" for r in results)
    print(f"{len(results) - survived}/{len(results)} killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
