"""scripts/roof_proposal.py (M6): the one-page Typst customer proposal from the roof stills."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import roof_proposal as RP  # noqa: E402

INFO = {"panels": 41, "peak_kw": 22.55, "tilt_deg": 15, "latitude": 19.0712, "longitude": 72.8794}


@pytest.fixture(scope="module")
def typst_available():
    if not shutil.which("typst"):
        pytest.skip("typst not installed")


# ---------------------------------------------------------------- unit

def test_figures_from_the_sidecar():
    f = RP.figures(INFO)
    assert f["panels"] == 41 and f["watts"] == pytest.approx(550)
    assert f["peak_kw"] == pytest.approx(22.55)
    assert f["kwh_day"] == pytest.approx(41 * 550 * 5 * 0.8 / 1000)
    assert f["kwh_year"] == pytest.approx(f["kwh_day"] * 365)
    g = RP.figures({"panels": 10, "peak_kw": 4.0}, sun_hours=4, derate=0.75)
    assert g["watts"] == pytest.approx(400) and g["kwh_day"] == pytest.approx(10 * 400 * 4 * 0.75 / 1000)
    z = RP.figures({"panels": 0, "peak_kw": 0.0})
    assert z == {"panels": 0, "watts": 0.0, "peak_kw": 0.0, "kwh_day": 0.0, "kwh_year": 0.0}
    assert RP.figures({"panels": "3", "peak_kw": 1.5})["panels"] == 3
    assert RP.figures({"panels": 0, "peak_kw": 1.0})["watts"] == 0.0      # no panels, no watts per panel


def test_typst_str_escapes_quotes_and_backslashes():
    assert RP.typst_str("plain") == '"plain"'
    assert RP.typst_str('say "hi"') == '"say \\"hi\\""'
    assert RP.typst_str("a\\b") == '"a\\\\b"'
    assert RP.typst_str('\\"') == '"\\\\\\""'                         # backslash first, then the quote
    assert RP.typst_str(3) == '"3"'


def test_build_typ_has_the_figures_stills_and_title():
    t = RP.build_typ(INFO, "stills", title='Sharma "Sunny" house')
    assert '#text(size: 20pt, weight: "bold", "Sharma \\"Sunny\\" house")' in t
    assert 'paper: "a4", flipped: true' in t and 'font: "Libertinus Serif"' in t
    for k, cap in RP.SHOTS:
        assert f'figure(image("stills/{k}.png", height: 6.3cm), caption: "{cap}")' in t
    order = [t.index(f'"stills/{k}.png"') for k, _ in RP.SHOTS]
    assert order == sorted(order)                                   # aerial, eye, front, close-up
    for row in ("[Panels], [*41 x 550 W*]", "[System size], [*22.55 kWp*]",
                "[Energy per year], [*about 32,923 kWh*]", "[Energy per day], [*about 90 kWh*]",
                "[Panel tilt], [*15 degrees*]", "[Site], [*19.07 N, 72.88 E*]"):
        assert row in t, row
    assert "5 peak sun hours a day" in t and "20 % losses" in t
    o = RP.build_typ(INFO, "x", sun_hours=5.5, derate=0.85)
    assert "5.5 peak sun hours" in o and "15 % losses" in o and '"x/aerial.png"' in o
    assert "Rooftop solar proposal" in RP.build_typ(INFO, "s")
    bare = RP.build_typ({"panels": 2, "peak_kw": 1.0}, "s")
    assert "[*0 degrees*]" in bare and "[*0.00 N, 0.00 E*]" in bare


def test_cli_needs_the_stills(tmp_path, capsys):
    (tmp_path / "aerial.png").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        RP.main(["--stills", str(tmp_path), "--out", str(tmp_path / "p.pdf")])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "missing stills ['eye', 'front', 'closeup']" in err and "roof_stills.py" in err


def test_cli_needs_typst(tmp_path, capsys, monkeypatch):
    for k, _ in RP.SHOTS:
        (tmp_path / f"{k}.png").write_bytes(b"x")
    monkeypatch.setattr(RP.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit):
        RP.main(["--stills", str(tmp_path)])
    assert "typst not found" in capsys.readouterr().err


def test_cli_reports_a_failed_compile(tmp_path, capsys, monkeypatch):
    stills = tmp_path / "st"
    stills.mkdir()
    for k, _ in RP.SHOTS:
        (stills / f"{k}.png").write_bytes(b"x")
    js = tmp_path / "m.json"
    js.write_text(json.dumps(INFO))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "error: boom")
    monkeypatch.setattr(RP.shutil, "which", lambda name: "typst.exe")
    monkeypatch.setattr(RP.subprocess, "run", fake_run)
    assert RP.main(["--stills", str(stills), "--json", str(js), "--out", str(tmp_path / "o" / "p.pdf")]) == 1
    assert "error: boom" in capsys.readouterr().err
    typ = tmp_path / "p.typ"                                        # next to the stills folder, named after --out
    assert calls == [["typst.exe", "compile", "--root", str(tmp_path), str(typ), str((tmp_path / "o" / "p.pdf").resolve())]]
    assert '"st/aerial.png"' in typ.read_text(encoding="utf-8")
    assert (tmp_path / "o").is_dir()


# ---------------------------------------------------------------- integration (real typst)

def _pngs(folder, n=4):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i, (k, _) in enumerate(RP.SHOTS[:n]):
        a = np.zeros((90, 160, 3), np.uint8)
        a[..., i % 3] = 200
        Image.fromarray(a).save(folder / f"{k}.png")


def test_compiles_a_one_page_pdf(tmp_path, typst_available, capsys):
    stills = tmp_path / "stills"
    _pngs(stills)
    js = tmp_path / "m.json"
    js.write_text(json.dumps(INFO))
    out = tmp_path / "out" / "proposal.pdf"
    assert RP.main(["--stills", str(stills), "--json", str(js), "--out", str(out), "--title", "Test roof"]) == 0
    printed = capsys.readouterr().out
    assert "41 panels, 22.55 kWp, about 32,923 kWh a year" in printed and str(out.resolve()) in printed
    pdf = out.read_bytes()
    assert pdf[:5] == b"%PDF-"
    assert len(re.findall(rb"/Type\s*/Page[^s]", pdf)) == 1                 # fits one page
    assert (tmp_path / "proposal.typ").is_file()


def test_smoke_script_runs_from_the_command_line(tmp_path, typst_available):
    stills = tmp_path / "s"
    _pngs(stills)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "roof_proposal.py"), "--stills", str(stills),
                        "--out", str(tmp_path / "p.pdf")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "kWp" in r.stdout and (tmp_path / "p.pdf").stat().st_size > 1000
