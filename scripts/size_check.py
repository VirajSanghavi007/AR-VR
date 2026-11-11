"""Size check report for site.html: drawn sizes against a tape, as a markdown report.

site.html's Tape check (File tab) stores one row per check in the design file's "checks" list:
{id, what, drawn (m), tape (m), errMm, tags (ids the pose used), at}. This script reads one or more
saved design files (site-design-*.json), optionally adds tape readings taken later, and writes a report
with every check, the mean and worst error, and a pass/fail against the tolerance.

Usage:
    python scripts/size_check.py site-design-A.json [more.json ...] [-o report.md] [--tol-mm 50]
        [--tape <object id>=<metres> ...]

--tape adds (or replaces) a check for an object in the first file: its drawn length (a wall line's
length along its points, otherwise its width) against the tape reading you give.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

NAMES = {"walls": "Wall line", "slab": "Outline"}


def scaled_pts(o: dict) -> list[list[float]]:
    """A drawn object's points after resizing (same rule as blocks.js scaledPts)."""
    base = o.get("base") or {}
    sx = o["w"] / base["w"] if base.get("w", 0) > 1e-6 else 1.0
    sz = o["d"] / base["d"] if o["type"] == "slab" and base.get("d", 0) > 1e-6 else sx
    return [[x * sx, z * (sx if o["type"] == "walls" else sz)] for x, z in o["pts"]]


def main_length(o: dict) -> float:
    """What a tape measures on the object: a wall line's length, otherwise its width."""
    if o["type"] == "walls" and o.get("pts"):
        p = scaled_pts(o)
        return sum(math.dist(a, b) for a, b in zip(p, p[1:]))
    return float(o["w"])


def load(path: Path) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("kind") != "arvr-site-design":
        raise ValueError(f"{path.name}: not a site design file (kind is {rec.get('kind')!r})")
    return rec


def find_object(rec: dict, oid: str) -> dict | None:
    for objs in rec.get("options", {}).values():
        for o in objs:
            if o.get("id") == oid:
                return o
    return None


def add_tape(rec: dict, oid: str, tape_m: float) -> dict:
    o = find_object(rec, oid)
    if o is None:
        raise ValueError(f"no object {oid!r} in the design")
    drawn = main_length(o)
    c = {"id": oid, "what": NAMES.get(o["type"], o["type"].capitalize()), "drawn": round(drawn, 3), "tape": tape_m,
         "errMm": round((drawn - tape_m) * 1000), "tags": [], "at": datetime.now().isoformat(timespec="seconds")}
    rec["checks"] = [x for x in rec.get("checks", []) if x.get("id") != oid] + [c]
    return c


def report(files: list[tuple[str, dict]], tol_mm: float) -> str:
    rows = [(name, c) for name, rec in files for c in rec.get("checks", [])]
    out = ["# Size check report", "",
           f"Made {datetime.now():%Y-%m-%d %H:%M} from {', '.join(n for n, _ in files)}. Tolerance: {tol_mm:.0f} mm "
           "(target 50 to 100 mm).", ""]
    if not rows:
        out += ["No checks in these files yet: in site.html select the object, type the tape length in the File tab "
                "and press Tape check, then save the design.", ""]
        return "\n".join(out)
    out += ["| File | Object | Drawn (m) | Tape (m) | Error (mm) | Tags used | When | Result |",
            "|---|---|---:|---:|---:|---|---|---|"]
    for name, c in rows:
        ok = abs(c["errMm"]) <= tol_mm
        out.append(f"| {name} | {c.get('what', '')} {c['id']} | {c['drawn']:.3f} | {c['tape']:.3f} | {c['errMm']:+d} | "
                   f"{', '.join(map(str, c.get('tags', []))) or '-'} | {c.get('at', '')[:16]} | {'pass' if ok else 'FAIL'} |")
    errs = [abs(c["errMm"]) for _, c in rows]
    passed = sum(e <= tol_mm for e in errs)
    out += ["", f"**{passed} of {len(errs)} within {tol_mm:.0f} mm.** Mean error {sum(errs) / len(errs):.0f} mm, "
            f"worst {max(errs)} mm, signed mean {sum(c['errMm'] for _, c in rows) / len(rows):+.0f} mm "
            "(a steady sign points to the tag size or scale, a mixed sign to pose noise).", ""]
    return "\n".join(out)


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--tol-mm", type=float, default=50)
    ap.add_argument("--tape", action="append", default=[], metavar="ID=M")
    a = ap.parse_args(argv)
    files = [(p.name, load(p)) for p in a.files]
    for t in a.tape:
        oid, _, m = t.partition("=")
        add_tape(files[0][1], oid, float(m))
    text = report(files, a.tol_mm)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)
    rows = [c for _, r in files for c in r.get("checks", [])]
    return 0 if all(abs(c["errMm"]) <= a.tol_mm for c in rows) else 1


if __name__ == "__main__":
    sys.exit(run())
