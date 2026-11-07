"""model.js: the designer's CAD model in tagar.html (units, seating, 1 m bar, pose file), run through node.
Unit (each helper), integration (a pose file written to disk and read back through node's fs), smoke (page hooks)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

PREVIEW = Path(__file__).resolve().parents[1] / "preview"
JS = (PREVIEW / "model.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, pre=""):
    code = f'import * as M from "{JS}"; {pre} console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


POSE = {"name": "cabinet.glb", "units": "mm", "sizeRaw": [1200, 800, 450], "pct": 100, "deg": 30,
        "x": 0.25, "z": -0.1, "wall": True, "tags": {"ids": "3,7", "width": 0.8, "height": 0.5}}


def test_units_in_metres(node_available):
    assert run("M.UNIT_M") == {"m": 1, "cm": 0.01, "mm": 0.001, "in": 0.0254}


def test_guess_units_millimetres_only_past_100(node_available):
    assert run("M.guessUnits(1200)") == "mm"
    assert run("M.guessUnits(100.5)") == "mm"
    assert run("M.guessUnits(100)") == "m"
    assert run("M.guessUnits(4.6)") == "m"
    with pytest.raises(RuntimeError, match="no size"):
        run("M.guessUnits(0)")
    with pytest.raises(RuntimeError, match="no size"):
        run("M.guessUnits(NaN)")


def test_real_size_converts_and_rejects_unknown_units(node_available):
    assert run('M.realSize([1200, 800, 450], "mm")') == pytest.approx([1.2, 0.8, 0.45])
    assert run('M.realSize([120, 80, 45], "cm")') == pytest.approx([1.2, 0.8, 0.45])
    assert run('M.realSize([10, 1, 2], "in")') == pytest.approx([0.254, 0.0254, 0.0508])
    assert run('M.realSize([1, 2, 3], "m")') == [1, 2, 3]
    with pytest.raises(RuntimeError, match='unknown units "ft"'):
        run('M.realSize([1, 1, 1], "ft")')


def test_fmt_size(node_available):
    assert run("M.fmtSize([1.2, 0.8, 0.456])") == "1.20 × 0.80 × 0.46 m"


def test_seat_centres_and_rests_on_the_surface(node_available):
    assert run("M.seat([-1, 0.5, 2], [3, 1.5, 6])") == [-1, -0.5, -4]
    # applying the seat to the box puts its centre over the origin and its bottom on y = 0
    lo, hi = [-1, 0.5, 2], [3, 1.5, 6]
    t = run(f"M.seat({lo}, {hi})")
    assert (lo[0] + t[0] + hi[0] + t[0]) / 2 == 0 and lo[1] + t[1] == 0 and (lo[2] + t[2] + hi[2] + t[2]) / 2 == 0


def test_mount_turn_sends_up_up_the_wall(node_available):
    assert run("M.mountTurnX(false)") == 0
    a = run("M.mountTurnX(true)")
    # rotate model +y about x by a: (0, cos a, sin a) must be -z (up the wall, +z is down); +z goes out (+y)
    assert [0, math.cos(a), math.sin(a)] == pytest.approx([0, 0, -1])
    assert [0, -math.sin(a), math.cos(a)] == pytest.approx([0, 1, 0])


@pytest.mark.parametrize("v,want", [(100, 100), (9, 10), (10, 10), (500, 500), (501, 500), ("73.6", 74), ("x", 100), (None, 10)])
def test_clamp_pct(v, want, node_available):
    assert run(f"M.clampPct({json.dumps(v)})") == want


@pytest.mark.parametrize("v,want", [(0, 0), (359, 359), (360, 0), (-30, 330), (725, 5), ("12.4", 12), ("x", 0)])
def test_clamp_deg(v, want, node_available):
    assert run(f"M.clampDeg({json.dumps(v)})") == want


def test_ruler_is_ten_alternating_10cm_stripes(node_available):
    t = run("M.rulerTicks()")
    assert len(t) == 10
    assert t[0] == {"from": 0, "to": 0.1, "dark": True} and t[-1] == {"from": 0.9, "to": 1, "dark": False}
    assert all(a["to"] == b["from"] for a, b in zip(t, t[1:]))
    assert [s["dark"] for s in t] == [True, False] * 5
    assert len(run("M.rulerTicks(2, 0.25)")) == 8
    with pytest.raises(RuntimeError, match="whole number"):
        run("M.rulerTicks(1, 0.3)")
    with pytest.raises(RuntimeError, match="whole number"):
        run("M.rulerTicks(1, 0.33)")                                # 3 steps make 0.99 m: close is not a metre


def test_ruler_starts_beside_the_model(node_available):
    assert run("M.rulerStart([-0.6, 0, -0.2], [0.6, 0.8, 0.4])") == pytest.approx([0.7, 0.1])
    assert run("M.rulerStart([0, 0, 0], [1, 1, 1], 0.05)") == pytest.approx([1.05, 0.5])


def test_pose_file_name_has_size_and_units(node_available):
    assert run('M.poseFileName("cabinet.glb", [1.2, 0.8, 0.45])') == "cabinet-1200x800x450mm-pose.json"
    assert run('M.poseFileName("Wall unit (v2).GLB", [2.0004, 0.6, 0.35])') == "Wall_unit_v2-2000x600x350mm-pose.json"
    assert run('M.poseFileName("", [1, 1, 1])') == "model-1000x1000x1000mm-pose.json"
    assert run('M.poseFileName("..glb", [1, 1, 1])') == "model-1000x1000x1000mm-pose.json"


def test_pose_record_fields(node_available):
    r = run(f'M.poseRecord({json.dumps(POSE)}, new Date(Date.UTC(2026, 10, 7, 6, 0, 0)))')
    assert r["version"] == 1 and r["model"] == "cabinet.glb" and r["units"] == "mm"
    assert r["size_as_drawn_m"] == [1.2, 0.8, 0.45] and r["scale_pct"] == 100 and r["size_shown_m"] == [1.2, 0.8, 0.45]
    assert r["position_m"] == [0.25, 0, -0.1] and r["turn_deg"] == 30 and r["mount"] == "wall"
    assert r["tags"] == {"ids": "3,7", "width_m": 0.8, "height_m": 0.5}
    assert r["saved"] == "2026-11-07T06:00:00.000Z"
    assert "x along the long side" in r["frame"]


def test_pose_record_scales_shown_size_and_clamps(node_available):
    p = {**POSE, "pct": 50, "deg": -90, "wall": False}
    r = run(f"M.poseRecord({json.dumps(p)})")
    assert r["size_shown_m"] == [0.6, 0.4, 0.225] and r["turn_deg"] == 270 and r["mount"] == "floor"


def test_parse_pose_round_trip(node_available):
    back = run(f"M.parsePose(M.poseRecord({json.dumps(POSE)}))")
    assert back == {"name": "cabinet.glb", "units": "mm", "pct": 100, "deg": 30, "x": 0.25, "z": -0.1, "wall": True, "ids": "3,7"}
    assert run(f"M.parsePose(JSON.stringify(M.poseRecord({json.dumps(POSE)}))).deg") == 30


@pytest.mark.parametrize("src,msg", [
    ('"{not json"', "not JSON"),
    ('{"version": 2}', "version 2"),
    ("null", "not a pose file"),
    ('{"version": 1, "units": "ft", "position_m": [0, 0, 0]}', 'unknown units "ft"'),
    ('{"version": 1, "units": "m"}', "no position"),
    ('{"version": 1, "units": "m", "position_m": [0, "a", 0]}', "no position"),
])
def test_parse_pose_rejects_bad_files(src, msg, node_available):
    with pytest.raises(RuntimeError, match=msg):
        run(f"M.parsePose({src})")


def test_same_tags(node_available):
    assert run('M.sameTags("3,7", "3,7")') is True
    assert run('M.sameTags("3,7", "3,8")') is False


def test_pose_file_on_disk_round_trip(tmp_path, node_available):
    # integration: the page's record written as a real file under its own name, read back through node's fs
    name = run(f'M.poseFileName("{POSE["name"]}", M.realSize({POSE["sizeRaw"]}, "{POSE["units"]}"))')
    path = tmp_path / name
    path.write_text(json.dumps(run(f"M.poseRecord({json.dumps(POSE)})"), indent=1), encoding="utf8")
    back = run(f'M.parsePose(readFileSync({json.dumps(str(path))}, "utf8"))', pre='import { readFileSync } from "node:fs";')
    assert name == "cabinet-1200x800x450mm-pose.json"
    assert (back["x"], back["z"], back["deg"], back["pct"], back["wall"]) == (0.25, -0.1, 30, 100, True)


def test_page_wires_the_cad_model():
    # smoke: tagar.html loads the model, has its controls and the debug hooks the browser check drives
    html = (PREVIEW / "tagar.html").read_text(encoding="utf8")
    for hook in ['from "./model.js', "GLTFLoader", 'id="m-file"', 'id="m-units"', 'id="m-pose"', 'id="o-mctl"',
                 'id="o-mturn"', 'id="o-mscale"', 'id="o-mplace"', 'id="o-mbar"', 'id="o-msave"',
                 "rulerTicks(", "seat(", "mountTurnX(", "poseRecord(", "parsePose(", "poseFileName(",
                 "loadModel(", "placeModel(", "modelState(",
                 "if (tap && placing) placeModel(surfacePoint(t.x, t.y));", "if (placing) return;"]:
        assert hook in html, hook



def test_page_wires_the_room_model():
    # smoke: tagar.html has the room card, AR controls, the wiring into place()/report()/buttons and the debug hooks
    html = (PREVIEW / "tagar.html").read_text(encoding="utf8")
    for hook in ['id="r-file"', 'id="r-units"', 'id="r-sample"', 'id="r-wall"', 'id="r-along"', 'id="r-height"', 'id="r-info"',
                 'id="o-rctl"', 'id="o-ralong"', 'id="o-rup"', 'id="o-rwall"', 'id="o-rlook"',
                 "wallYaw, nextWall, parseLen, roomOffset, roomCheck } from \"./model.js?v=2\"",
                 "yaw.rotation.y = wallYaw(roomPos.wall);", "mount.rotation.x = mountTurnX(true);",
                 "roomOffset(room.wallW, room.wallH, along, height, roomPos.nudge)", "roomRoot.visible = ok && modelWall();",
                 "if (roomRoot) { group.add(roomRoot); poseRoom(); }", "modelLine() + roomLine()",
                 '(ROOM_BUTTONS.includes(id) && !room)', "plumbAxes(fitRectangle(centres, n))",
                 "loadRoom(url, name, units", "setRoom({", "roomState,", "roomView(eye, at)"]:
        assert hook in html, hook
    # the sample room the card's button fetches is in the published models
    urls = [u.split("?", 1)[0] for u in html.split('loadRoom("')[1:]]
    assert urls == ["models/sample_room.glb", "models/duplex_living_room.glb"], urls
    for url in urls:
        assert (PREVIEW / url).is_file(), url
    # the real room button reads it as metres and the page credits its source (CC BY 4.0)
    assert '$("r-duplex").onclick = () => { $("r-units").value = "m";' in html
    assert 'id="r-duplex"' in html and "buildingSMART International" in html and "CC BY 4.0" in html


# ---------- room model alignment ----------

def test_wall_yaw_brings_that_face_to_the_back(node_available):
    # rotating about y by a maps (x, z) -> (x cos a + z sin a, -x sin a + z cos a); the named face must end at -z
    face = {"back": (0, -1), "right": (1, 0), "front": (0, 1), "left": (-1, 0)}
    for w, (x, z) in face.items():
        a = run(f'M.wallYaw("{w}")')
        assert (x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)) == pytest.approx((0, -1), abs=1e-12), w
    with pytest.raises(RuntimeError, match='unknown wall "up"'):
        run('M.wallYaw("up")')


def test_next_wall_cycles_and_faces(node_available):
    assert run('["back", "right", "front", "left"].map(M.nextWall)') == ["right", "front", "left", "back"]
    assert run('M.ROOM_WALLS.map(M.wallFace)') == ["-Z", "+X", "+Z", "-X"]


@pytest.mark.parametrize("v,want", [("1.2", 1.2), ("1,25", 1.25), (" 0.9 m", 0.9), ("2M", 2), (3, 3), ("0", 0)])
def test_parse_len(v, want, node_available):
    assert run(f"M.parseLen({json.dumps(v)})") == want


@pytest.mark.parametrize("v", ["", "  ", None, "abc"])
def test_parse_len_none(v, node_available):
    assert run(f"Number.isNaN(M.parseLen({json.dumps(v)}))") is True


def test_room_offset_puts_corner_and_floor_where_measured(node_available):
    W, H = 4.6, 3.05
    x, z = run(f"M.roomOffset({W}, {H}, 1.2, 1.4)")
    # the seated room spans x in [-W/2, W/2], z in [-H/2, H/2] (z down); after the move, relative to the tags:
    assert -W / 2 + x == pytest.approx(-1.2)          # left corner 1.2 m to the left of the tag centre
    assert H / 2 + z == pytest.approx(1.4)            # floor 1.4 m below it
    nx, nz = run(f"M.roomOffset({W}, {H}, 1.2, 1.4, [0.05, 0.02])")
    assert (nx - x, nz - z) == pytest.approx((0.05, -0.02))   # nudge right and up (up is -z)


def test_room_check(node_available):
    assert run("M.roomCheck(4.6, 3.05, 1.2, 1.4)") == []
    assert run("M.roomCheck(4.6, 3.05, 0, 0)") == [] and run("M.roomCheck(4.6, 3.05, 4.6, 3.05)") == []
    assert "wrong wall" in run("M.roomCheck(4.6, 3.05, 5, 1.4)")[0]
    assert "wrong wall" in run("M.roomCheck(4.6, 3.05, -0.1, 1.4)")[0]
    assert "3.05 m high" in run("M.roomCheck(4.6, 3.05, 1, 3.2)")[0]
    assert "3.05 m high" in run("M.roomCheck(4.6, 3.05, 1, -0.2)")[0]
    two = run("M.roomCheck(4.6, 3.05, NaN, NaN)")
    assert len(two) == 2 and "left corner" in two[0] and "above the floor" in two[1]
