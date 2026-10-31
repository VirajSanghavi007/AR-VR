"""phoneemu.js: the desktop phone stand-in for tagar.html. Unit (tag bits, cells, seeded noise through node),
integration (its tag drawing decodes to the right id in the page's real js-aruco2 detector), smoke (page hooks)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
JS = (PREVIEW / "phoneemu.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as E from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_tag_bits_reads_msb_first_row_major(node_available):
    bits = run('E.tagBits("800000001")')
    assert len(bits) == 6 and all(len(r) == 6 for r in bits)
    assert bits[0][0] == 1 and bits[5][5] == 1 and sum(map(sum, bits)) == 2
    assert run('E.tagBits("000000000")') == [[0] * 6] * 6
    with pytest.raises(RuntimeError, match="not a 36-bit"):
        run('E.tagBits("1234")')


def test_tag_cells_has_white_margin_black_border_and_data(node_available):
    cells = run('E.tagCells("800000001")')
    assert len(cells) == 10 and all(len(r) == 10 for r in cells)
    assert all(v == 1 for v in cells[0] + cells[9]) and all(r[0] == 1 and r[9] == 1 for r in cells)
    assert all(v == 0 for v in cells[1][1:9] + cells[8][1:9]) and all(r[1] == 0 and r[8] == 0 for r in cells[1:9])
    assert cells[2][2] == 1 and cells[7][7] == 1 and cells[2][3] == 0


def test_rng_repeats_and_is_roughly_normal(node_available):
    a = run("(() => { const r = E.rng(5); return [r.gauss(), r.gauss(), r.uni()]; })()")
    assert a == run("(() => { const r = E.rng(5); return [r.gauss(), r.gauss(), r.uni()]; })()")
    assert run("E.rng(1).uni()") == pytest.approx(16807 / 2147483647)      # Park-Miller, seed 1
    assert a != run("(() => { const r = E.rng(6); return [r.gauss(), r.gauss(), r.uni()]; })()")
    g = np.array(run("(() => { const r = E.rng(1); return Array.from({length: 4000}, () => r.gauss()); })()"))
    assert abs(g.mean()) < 0.06 and g.std() == pytest.approx(1, abs=0.06)
    u = run("(() => { const r = E.rng(2); return Array.from({length: 500}, () => r.uni()); })()")
    assert 0 < min(u) and max(u) < 1


def test_drawn_tag_decodes_to_its_id(node_available, tmp_path):
    """Rasterize tagCells like the emulator's canvas does and run the page's detector on it."""
    script = tmp_path / "decode.mjs"
    script.write_text(f"""
import {{ createRequire }} from "node:module";
import * as E from "{JS}";
const require = createRequire(import.meta.url);
const root = process.argv[2];
require(root + "/vendor/js-aruco2/cv.js");
const AR = require(root + "/tag36h11.js");
const codes = AR.DICTIONARIES.APRILTAG_36h11.codeList, px = 8, size = 10 * px, out = [];
for (const id of [0, 1, 7, 100, 586]) {{
  const cells = E.tagCells(codes[id]), data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {{
    const v = cells[Math.floor(y / px)][Math.floor(x / px)] ? 244 : 17, i = (y * size + x) * 4;
    data[i] = data[i + 1] = data[i + 2] = v; data[i + 3] = 255;
  }}
  const det = new AR.Detector({{ dictionaryName: "APRILTAG_36h11" }});
  out.push([id, det.detect({{ data, width: size, height: size }}).map((m) => m.id)]);
}}
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run(["node", str(script), str(PREVIEW)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for id_, found in json.loads(r.stdout):
        assert found == [id_], (id_, found)


def test_page_exposes_the_emulator_hooks():
    html = (PREVIEW / "tagar.html").read_text(encoding="utf-8")
    for hook in ("emuStart", "emuRun", "emuFrame", "detectStep", "PhoneEmulator", "levelNormal(["):
        assert hook in html, hook
    js = (PREVIEW / "phoneemu.js").read_text(encoding="utf-8")
    for name in ("tagBits", "tagCells", "rng", "PhoneEmulator"):
        assert f"export function {name}" in js or f"export class {name}" in js, name
