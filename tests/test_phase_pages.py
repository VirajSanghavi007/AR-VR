"""The phase menu pages (preview/index.html, phase1/2/3.html): Phase 1 is built around the Site Designer, the
tabletop AR is a secondary table demo, and the room experiments live in Phase 3 (tagar.html?room=1)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
sys.path.insert(0, str(ROOT))
from rooftop.phase_shots import SHOTS  # noqa: E402

PAGES = ("index.html", "phase1.html", "phase2.html", "phase3.html")


def read(name):
    return (PREVIEW / name).read_text(encoding="utf-8")


def buttons(html):
    """(class, href, text) of every <a class="btn..."> in page order."""
    return [(c, h, re.sub(r"\s+", " ", t).strip())
            for c, h, t in re.findall(r'<a class="(btn[^"]*)" href="([^"]+)">(.*?)</a>', html, re.S)]


def test_phase1_leads_with_the_site_designer():
    b = buttons(read("phase1.html"))
    assert b[0] == ("btn", "site.html", "Open the Site Designer"), b
    assert [c for c, _, _ in b[1:]] == ["btn ghost"] * 3, b                  # everything else is secondary
    assert {h for _, h, _ in b[1:]} == {"tagar.html", "viewer.html", "tags.html"}, b
    assert ("btn ghost", "tagar.html", "Quick table demo") in b


def test_phase1_steps_follow_print_lock_draw_save_show():
    html = read("phase1.html")
    steps = re.findall(r"<li><b>([^<]+)</b>", html.split('<ol class="steps">', 1)[1].split("</ol>", 1)[0])
    assert steps == ["Print tags.", "Lock.", "Draw.", "Save.", "Show the customer."], steps
    for gone in ("scanned office room", "sample room", "?room=1"):            # no room experiments in Phase 1
        assert gone not in html, gone


def test_room_experiments_are_reached_from_phase3_only():
    assert 'href="tagar.html?room=1"' in read("phase3.html")
    for name in ("index.html", "phase1.html", "phase2.html"):
        assert "room=1" not in read(name), name
    by_name = {s["name"]: s for s in SHOTS}
    assert "room=1" in by_name["phase3_room.jpg"]["page"]                    # its picture needs the room card
    assert "room=1" not in by_name["phase1_ar.jpg"]["page"]


def test_index_puts_the_site_designer_first():
    html = read("index.html")
    tools = html.split("Straight to the tools:", 1)[1].split("</p>", 1)[0]
    hrefs = re.findall(r'href="([^"]+)"', tools)
    assert hrefs[:4] == ["site.html", "viewer.html", "tagar.html", "tags.html"], hrefs


def test_every_local_link_on_the_phase_pages_exists():
    for name in PAGES:
        html = read(name)
        for ref in re.findall(r'(?:href|src)="([^"#:]+?)(?:\?[^"]*)?"', html):
            if ref.startswith("//"):
                continue
            assert (PREVIEW / ref).is_file(), f"{name}: {ref}"
