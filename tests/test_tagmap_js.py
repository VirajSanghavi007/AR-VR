"""preview/tagmap.js (360 degree anchoring for site.html) through node: unit (matrices, tag poses, the room frame,
the running-mean map, the levelled fit with outlier drop, pose easing and fusion, the map file, the walk), and an
integration check that a map started on one wall locates the room again from tags on other walls after a turn."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

JS = (Path(__file__).resolve().parents[1] / "preview" / "tagmap.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, pre=""):
    code = f'import * as T from "{JS}"; {pre}; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def M(cols):                       # column-major 16 -> numpy 4x4
    return np.array(cols, float).reshape(4, 4).T


def yaw_np(a, t=(0, 0, 0)):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = t
    return m


def cols(m):
    return np.asarray(m).T.ravel().tolist()


def wall_tag(center, face, size=0.16):
    """Printed corners TL, TR, BR, BL of a tag on a wall facing `face` (level)."""
    n = np.array(face, float)
    x = np.cross([0, 1, 0], n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    h = size / 2
    c = np.array(center, float)
    return [(c + a * x + b * y).tolist() for a, b in ((-h, h), (h, h), (h, -h), (-h, -h))]


def floor_tag(center, size=0.16):
    h, c = size / 2, np.array(center, float)
    # lying face up, printed top towards -z: right = +x, down = +z
    return [(c + d).tolist() for d in ([-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h])]


def apply(m, p):
    return (m @ np.append(p, 1))[:3]


# ---------- unit ----------

def test_yaw_matrix_matches_threejs_rotation_y_and_mul4(node_available):
    a, t = 0.7, [1, 2, 3]
    assert np.allclose(M(run(f"T.yawMatrix({a}, {t})")), yaw_np(a, t))
    assert np.allclose(M(run("T.yawMatrix(0)")), np.eye(4))
    A, B = yaw_np(0.3, [1, 0, 2]) @ np.diag([1, 2, 3, 1]), yaw_np(-1.1, [0, 5, 0])
    assert np.allclose(M(run(f"T.mul4({cols(A)}, {cols(B)})")), A @ B)
    assert np.allclose(M(run(f"T.mul4({cols(B)}, {cols(A)})")), B @ A)
    assert run(f"T.yawOf(T.yawMatrix({a}))") == pytest.approx(a)
    assert run("T.yawOf(T.yawMatrix(-2.5))") == pytest.approx(-2.5)
    assert run("T.basis([1,0,0],[0,1,0],[0,0,1],[4,5,6])") == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 5, 6, 1]


def test_tag_pose_on_wall_floor_and_ceiling(node_available):
    p = run(f"T.tagPose({wall_tag([1, 1.3, -2], [0, 0, 1])})")
    assert p["surface"] == "wall" and np.allclose(p["center"], [1, 1.3, -2])
    assert np.allclose(p["normal"], [0, 0, 1]) and np.allclose(p["right"], [1, 0, 0]) and np.allclose(p["down"], [0, -1, 0])
    f = run(f"T.tagPose({floor_tag([0, 0, 0])})")
    assert f["surface"] == "floor" and np.allclose(f["normal"], [0, 1, 0]) and np.allclose(f["down"], [0, 0, 1])
    c = run(f"T.tagPose({floor_tag([0, 2.5, 0])[::-1]})")                        # mirrored order: normal points down
    assert c["surface"] == "ceiling"
    # 40 degrees off level is still a wall, 50 degrees is a floor (the 45 degree split)
    for deg, want in ((40, "wall"), (50, "floor")):
        r = math.radians(deg)
        n = [0, math.sin(r), math.cos(r)]
        x = [1, 0, 0]
        y = np.cross(n, x)
        h = 0.08
        corners = [(a * np.array(x) + b * y).tolist() for a, b in ((-h, h), (h, h), (h, -h), (-h, -h))]
        assert run(f"T.tagPose({corners}).surface") == want
    # the same split facing down: 40 degrees is a wall, 50 degrees a ceiling
    for deg, want in ((40, "wall"), (50, "ceiling")):
        r = math.radians(deg)
        assert run(f"T.tagPose({wall_tag([0, 2, 0], [0, -math.sin(r), math.cos(r)])}).surface") == want


def test_room_frame_on_a_wall_points_z_out_of_it(node_available):
    tags = [wall_tag([2.5, 1.3, -0.2], [-1, 0, 0]), wall_tag([2.5, 1.3, 0.2], [-1, 0, 0])]
    m = M(run(f"T.roomFrame({tags})"))
    assert np.allclose(m[:3, 3], [2.5, 1.3, 0])
    assert np.allclose(m[:3, 1], [0, 1, 0]) and np.allclose(m[:3, 2], [-1, 0, 0]) and np.allclose(m[:3, 0], [0, 0, 1])
    with pytest.raises(RuntimeError, match="no tags"):
        run("T.roomFrame([])")


def test_room_frame_on_the_floor_uses_printed_down_and_a_mixed_start_prefers_walls(node_available):
    m = M(run(f"T.roomFrame({[floor_tag([1, 0, 1])]})"))
    assert np.allclose(m[:3, 2], [0, 0, 1]) and np.allclose(m[:3, 3], [1, 0, 1])
    mixed = [floor_tag([0, 0, 0]), wall_tag([0, 1, -1], [1, 0, 0])]
    assert np.allclose(M(run(f"T.roomFrame({mixed})"))[:3, 2], [1, 0, 0])
    # a level tag whose printed down is straight up/down: no level direction, falls back to +z
    up = [[-0.08, 0, 0.08], [0.08, 0, 0.08], [0.08, 0, -0.08], [-0.08, 0, -0.08]]
    straight = [[-0.08, 0.08, 0], [0.08, 0.08, 0], [0.08, -0.08, 0], [-0.08, -0.08, 0]]
    assert np.allclose(M(run(f"T.roomFrame({[up]})"))[:3, 2], [0, 0, -1])
    assert np.allclose(M(run(f"T.roomFrame([{straight}, {[c[::-1] for c in [straight]][0]}])"))[:3, 2], [0, 0, 1])


def test_start_map_registers_the_lock_tags_in_room_coordinates(node_available):
    tags = {"3": wall_tag([0.2, 1.3, -2], [0, 0, 1]), "1": wall_tag([-0.2, 1.3, -2], [0, 0, 1])}
    r = run(f"T.startMap({json.dumps(tags)}, 0.16)")
    m = r["map"]
    assert m["origin"] == [1, 3] and m["size"] == 0.16 and m["floorY"] is None and m["version"] == 1
    assert set(m["tags"]) == {"1", "3"} and all(t["seen"] == 30 and t["surface"] == "wall" for t in m["tags"].values())
    c = np.mean(m["tags"]["3"]["corners"], axis=0)
    assert np.allclose(c, [0.2, 0, 0], atol=1e-9)                              # room origin = the tags' middle


def test_register_tag_running_mean_with_cap(node_available):
    pre = "const map = {tags: {}}; const I = T.yawMatrix(0); const c0 = [[0,0,0],[1,0,0],[1,-1,0],[0,-1,0]];" \
          "const c1 = c0.map(p => [p[0] + 1, p[1], p[2]]);"
    first = run("T.registerTag(map, 7, c0, I)", pre)
    assert first["seen"] == 1 and first["surface"] == "wall"
    t = run("(T.registerTag(map, 7, c0, I), T.registerTag(map, 7, c1, I))", pre)
    assert t["seen"] == 2 and t["corners"][0][0] == pytest.approx(0.5)          # 1 old, 1 new: halfway
    t = run("(T.registerTag(map, 7, c0, I, 3), T.registerTag(map, 7, c1, I))", pre)
    assert t["corners"][0][0] == pytest.approx(0.25)                             # weight 3 against 1
    t = run("(T.registerTag(map, 7, c0, I, 100), T.registerTag(map, 7, c1, I, 1, 30))", pre)
    assert t["seen"] == 101 and t["corners"][0][0] == pytest.approx(1 / 31)       # capped at 30, not 100
    # a world pose: corners are stored in the room frame
    t = run("T.registerTag(map, 2, c0.map(p => [p[0] + 5, p[1], p[2]]), T.yawMatrix(0, [5, 0, 0]))", pre)
    assert np.allclose(t["corners"], [[0, 0, 0], [1, 0, 0], [1, -1, 0], [0, -1, 0]])


def test_fit_yaw_recovers_a_levelled_pose(node_available):
    rng = np.random.default_rng(3)
    P = rng.uniform(-2, 2, (6, 3))
    truth = yaw_np(2.6, [0.4, -1.2, 3.0])
    Q = np.array([apply(truth, p) for p in P])
    f = run(f"T.fitYaw({P.tolist()}, {Q.tolist()})")
    assert np.allclose(M(f["worldFromRoom"]), truth, atol=1e-9) and f["rms"] < 1e-9 and f["yaw"] == pytest.approx(2.6)
    Qn = Q + [[0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0], [0, 0, 0.01], [0, 0, -0.01]]
    assert run(f"T.fitYaw({P.tolist()}, {Qn.tolist()}).rms") == pytest.approx(0.01, rel=0.2)
    with pytest.raises(RuntimeError, match="2 or more"):
        run("T.fitYaw([[0,0,0]], [[0,0,0]])")
    with pytest.raises(RuntimeError, match="2 or more"):
        run("T.fitYaw([[0,0,0],[1,0,0]], [[0,0,0]])")


ROOM = {"0": wall_tag([-0.18, 0, 0], [0, 0, 1]), "1": wall_tag([0.18, 0, 0], [0, 0, 1]),
        "2": wall_tag([2.5, 0, 2.5], [-1, 0, 0]), "5": wall_tag([-2.5, 0, 2.5], [1, 0, 0])}
MAP = {"tags": {k: {"corners": v} for k, v in ROOM.items()}}


def seen_from(pose, ids, bump=None):
    out = {i: [apply(pose, p).tolist() for p in ROOM[i]] for i in ids}
    if bump:
        i, d = bump
        out[i] = [(np.array(p) + d).tolist() for p in out[i]]
    return out


def test_locate_fits_drops_a_moved_tag_and_reports_span(node_available):
    pose = yaw_np(-0.8, [1, 0.2, -3])
    r = run(f"T.locate({json.dumps(MAP)}, {json.dumps(seen_from(pose, ['0', '2', '5']))})")
    assert np.allclose(M(r["worldFromRoom"]), pose, atol=1e-9) and r["used"] == [0, 2, 5] and r["dropped"] == []
    assert r["span"] == pytest.approx(5.0) and np.allclose(r["pc"], np.mean(ROOM["0"] + ROOM["2"] + ROOM["5"], axis=0))
    s = seen_from(pose, ["0", "1", "2", "5"], bump=("2", [0.3, 0, 0]))
    r = run(f"T.locate({json.dumps(MAP)}, {json.dumps(s)})")
    assert r["dropped"] == [2] and r["used"] == [0, 1, 5] and r["rms"] < 1e-9
    assert run(f"T.locate({json.dumps(MAP)}, {json.dumps(s)}, 1.0).dropped") == []    # loose tol keeps it


def test_locate_skips_unmapped_ids_and_keeps_the_last_tag(node_available):
    pose = yaw_np(0.3, [0, 0, 1])
    s = seen_from(pose, ["1"])
    s["44"] = s["1"]
    r = run(f"T.locate({json.dumps(MAP)}, {json.dumps(s)})")
    assert r["used"] == [1] and r["span"] == 0
    assert run(f"T.locate({json.dumps(MAP)}, {json.dumps({'44': s['1']})})") is None
    assert run(f"T.locate({json.dumps(MAP)}, {{}})") is None
    two = seen_from(pose, ["0", "1"], bump=("1", [0, 0.5, 0]))
    r = run(f"T.locate({json.dumps(MAP)}, {json.dumps(two)})")
    assert len(r["used"]) == 1 and len(r["dropped"]) == 1                        # one of a disagreeing pair survives
    one = seen_from(pose, ["1"])
    c = np.mean(one["1"], axis=0)
    one["1"] = [(c + 1.5 * (np.array(p) - c)).tolist() for p in one["1"]]       # one tag that fits badly is still used
    r = run(f"T.locate({json.dumps(MAP)}, {json.dumps(one)})")
    assert r["used"] == [1] and r["dropped"] == [] and r["rms"] > 0.04


def test_blend_pose_eases_turn_and_shift_and_snaps_big_jumps(node_available):
    a, b = cols(yaw_np(0.0, [0, 0, 0])), cols(yaw_np(0.4, [0.2, 0, 0]))
    m = M(run(f"T.blendPose({a}, {b}, 0.25)"))
    assert np.allclose(m, yaw_np(0.1, [0.05, 0, 0]))
    assert run(f"T.blendPose(null, {b})") == b
    far = cols(yaw_np(0.4, [0.6, 0, 0]))
    assert run(f"T.blendPose({a}, {far})") == far
    assert np.allclose(M(run(f"T.blendPose({a}, {far}, 0.5, 1.0)"))[:3, 3], [0.3, 0, 0])
    # the short way round across +-pi
    m = M(run(f"T.blendPose({cols(yaw_np(3.0))}, {cols(yaw_np(-3.0))}, 0.5)"))
    assert run(f"T.yawOf({cols(m)})") == pytest.approx(math.pi, abs=1e-9) or run(f"T.yawOf({cols(m)})") == pytest.approx(-math.pi, abs=1e-9)


def fix(pose, span, pc=(0, 1, 0)):
    return {"worldFromRoom": cols(pose), "span": span, "pc": list(pc), "qc": apply(pose, pc).tolist()}


def test_fuse_pose_turn_weight_grows_with_span(node_available):
    prev = cols(yaw_np(0.0))
    for span, want in ((0, 0.03), (1.5, 0.5), (3.0, 0.8)):
        m = M(run(f"T.fusePose({prev}, {json.dumps(fix(yaw_np(0.2), span))}, {{k: 1}})"))
        assert run(f"T.yawOf({cols(m)})") == pytest.approx(0.2 * want, rel=1e-6), span
    m = M(run(f"T.fusePose({prev}, {json.dumps(fix(yaw_np(0.2), 1.5))})"))
    assert run(f"T.yawOf({cols(m)})") == pytest.approx(0.2 * 0.5 * 0.35)
    f = fix(yaw_np(0.2), 1.5)
    f.pop("span")
    assert run(f"T.yawOf(T.fusePose({prev}, {json.dumps(f)}, {{k: 1}}))") == pytest.approx(0.2 * 0.03)


def test_fuse_pose_lands_the_tag_centre_and_snaps(node_available):
    prev = cols(yaw_np(0.0))
    f = fix(yaw_np(0.0, [0.1, 0, 0]), 2, pc=(1, 1, 1))
    m = M(run(f"T.fusePose({prev}, {json.dumps(f)}, {{k: 1}})"))
    assert np.allclose(apply(m, [1, 1, 1]), f["qc"])
    m = M(run(f"T.fusePose({prev}, {json.dumps(f)}, {{k: 0.5}})"))
    assert np.allclose(apply(m, [1, 1, 1]), [1.05, 1, 1])
    # the fix's centre stays put under a pure turn of the fused pose (shift computed after the turn)
    f = fix(yaw_np(0.3, [0, 0, 0]), 100, pc=(2, 0, 0))
    m = M(run(f"T.fusePose({prev}, {json.dumps(f)}, {{k: 1}})"))
    assert np.allclose(apply(m, [2, 0, 0]), f["qc"], atol=1e-9)
    assert run(f"T.fusePose(null, {json.dumps(f)})") == f["worldFromRoom"]
    jump = fix(yaw_np(0.0, [0.8, 0, 0]), 1)
    assert run(f"T.fusePose({prev}, {json.dumps(jump)})") == jump["worldFromRoom"]
    assert run(f"T.fusePose({prev}, {json.dumps(jump)}, {{snapM: 1}})") != jump["worldFromRoom"]


def test_pose_delta(node_available):
    a = cols(yaw_np(0.0))
    d = run(f"T.poseDelta({a}, {cols(yaw_np(0.0, [0.01, 0, 0]))})")
    assert d["mm"] == pytest.approx(10) and d["deg"] == 0
    d = run(f"T.poseDelta({a}, {cols(yaw_np(math.radians(1)))})")
    assert d["deg"] == pytest.approx(1) and d["mm"] == pytest.approx(2 * math.hypot(3, 3) * math.sin(math.radians(0.5)) * 1000)
    d = run(f"T.poseDelta({a}, {cols(yaw_np(math.radians(1)))}, [[1, 0, 0]])")
    assert d["mm"] == pytest.approx(2 * math.sin(math.radians(0.5)) * 1000)
    assert run(f"T.poseDelta({cols(yaw_np(3.1))}, {cols(yaw_np(-3.1))}).deg") == pytest.approx(math.degrees(2 * math.pi - 6.2))
    assert run(f"T.poseDelta({a}, {cols(yaw_np(math.radians(-2)))}).deg") == pytest.approx(2)    # a turn either way is positive


def test_map_record_rounds_and_parse_map_checks(node_available):
    m = {"version": 1, "size": 0.2, "origin": [0], "floorY": -1.3,
         "tags": {"0": {"surface": "wall", "seen": 4.6, "corners": [[0.123456, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]}}}
    rec = run(f"T.mapRecord({json.dumps(m)})")
    assert rec["tags"]["0"]["corners"][0][0] == 0.1235 and rec["tags"]["0"]["seen"] == 5 and rec["floorY"] == -1.3
    back = run(f"T.parseMap({json.dumps(json.dumps(rec))})")
    assert back["size"] == 0.2 and back["floorY"] == -1.3 and back["tags"]["0"]["surface"] == "wall"
    loose = {"version": 1, "tags": {"3": {"corners": wall_tag([0, 0, 0], [0, 0, 1])}}, "floorY": "x"}
    p = run(f"T.parseMap({json.dumps(loose)})")
    assert p["size"] == 0.16 and p["floorY"] is None and p["origin"] == [] and p["tags"]["3"] == {
        "surface": "wall", "corners": loose["tags"]["3"]["corners"], "seen": 1}
    for bad, msg in (({}, "no tags"), ({"version": 2, "tags": {}}, "version 2"), ({"version": 0, "tags": {}}, "version 0"),
                     ({"version": 1, "tags": {"a": {"corners": []}}}, "tag a"),
                     ({"version": 1, "tags": {"1": {"corners": [[0, 0, 0]] * 3}}}, "tag 1"),
                     ({"version": 1, "tags": {"1": {"corners": [[0, 0, 0]] * 3 + [[0, 0]]}}}, "tag 1"),
                     ({"version": 1, "tags": {"1": {"corners": [[0, 0, 0]] * 3 + [[0, 0, None]]}}}, "tag 1"),
                     ({"version": 1, "tags": {"1": {}}}, "tag 1")):
        with pytest.raises(RuntimeError, match=msg):
            run(f"T.parseMap({json.dumps(bad)})")
    with pytest.raises(RuntimeError, match="no tags"):
        run("T.parseMap(null)")


def test_describe_map_sorted_numerically(node_available):
    m = {"tags": {"10": {"surface": "wall", "seen": 3.4, "corners": [[1, 2, 3]] * 4},
                  "2": {"surface": "floor", "seen": 1, "corners": [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]}}}
    assert run(f"T.describeMap({json.dumps(m)})") == ["2: floor at 0.50, 0.00, 0.50 m (1 sightings)",
                                                      "10: wall at 1.00, 2.00, 3.00 m (3 sightings)"]


def test_wall_walk_goes_round_looking_at_each_wall(node_available):
    stops = run("T.wallWalk(5, 4, 1, 16)")
    assert len(stops) == 16 and stops[0] == {"eye": [-1.5, 1.4, -1], "look": [-1.5, 1.3, -2]}
    eyes = np.array([s["eye"] for s in stops])
    assert np.allclose(np.abs(eyes[:, 0]).max(), 1.5) and np.allclose(np.abs(eyes[:, 2]).max(), 1.0)
    for s in stops:                                 # each look point is on a wall, square to it
        lx, _, lz = s["look"]
        ex, _, ez = s["eye"]
        assert math.isclose(abs(lx), 2.5) and math.isclose(lz, ez) or math.isclose(abs(lz), 2) and math.isclose(lx, ex)
    def wall(look):
        return "back" if look[2] == -2 else "right" if look[0] == 2.5 else "front" if look[2] == 2 else "left"
    order = [wall(s["look"]) for s in stops]
    assert order == ["back"] * 5 + ["right"] * 3 + ["front"] * 5 + ["left"] * 3       # round once, in this order
    assert run("T.walkLength(T.wallWalk(5, 4, 1, 16))") == pytest.approx(np.linalg.norm(np.diff(eyes, axis=0), axis=1).sum())
    assert run("T.walkLength([])") == 0 and run("T.walkLength([{eye: [0,0,0]}])") == 0
    assert run("T.wallWalk(4, 4, 1, 4, 1.6, 1.0)[1]") == {"eye": [1, 1.6, -1], "look": [2, 1.0, -1]}


# ---------- integration: lock on one wall, find the room again from the others ----------

def test_map_built_on_a_walk_locates_from_any_other_wall(node_available):
    world = {i: wall_tag(c, f) for i, (c, f) in {
        0: ([-0.18, 1.3, -2.5], [0, 0, 1]), 1: ([0.18, 1.3, -2.5], [0, 0, 1]),
        2: ([2.5, 1.3, -0.2], [-1, 0, 0]), 3: ([2.5, 1.3, 0.2], [-1, 0, 0]),
        4: ([0.2, 1.3, 2.5], [0, 0, -1]), 6: ([-2.5, 1.3, 0.1], [1, 0, 0])}.items()}
    rng = np.random.default_rng(0)

    def noisy(ids, pose=np.eye(4)):
        return {str(i): [(apply(pose, p) + rng.normal(0, 0.002, 3)).tolist() for p in world[i]] for i in ids}

    lock = json.dumps(noisy([0, 1]))
    later = [json.dumps(noisy([i])) for i in (2, 3, 4, 6) for _ in range(3)]
    moved = yaw_np(0.35, [0.4, 0, -0.7])                                 # tracking reset: world moved and turned
    view = json.dumps(noisy([4, 6], moved))
    pre = (f"const {{map, worldFromRoom}} = T.startMap({lock}, 0.16);"
           f"for (const s of {json.dumps(later)}) for (const [id, c] of Object.entries(JSON.parse(s))) T.registerTag(map, id, c, worldFromRoom);"
           f"const fix = T.locate(map, {view});")
    r = run("{wfr: worldFromRoom, fix, n: Object.keys(map.tags).length}", pre)
    assert r["n"] == 6 and sorted(r["fix"]["used"]) == [4, 6] and r["fix"]["rms"] < 0.005
    want = moved @ M(r["wfr"])                                     # the same room frame, seen through the moved world
    d = run(f"T.poseDelta({json.dumps(r['fix']['worldFromRoom'])}, {cols(want)})")
    assert d["mm"] < 15 and d["deg"] < 0.3
