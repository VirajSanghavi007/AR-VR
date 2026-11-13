"""preview/walkmap.js (panels.html: up to 8 AprilTags, walk all round) through node. Unit: combos, picking the 4
corner tags out of up to 8, the levelled lock frame, starting the walk map, the walk step (fix, map new tags
after 3 sightings, ignore tags seen with no mapped tag), the HUD line. Integration: a full 360 walk where only
1 to 3 tags are in view at each stop and tracking drifts keeps the lock on the truth with walkmap + tagmap
together."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

JS = (Path(__file__).resolve().parents[1] / "preview" / "walkmap.js").as_uri()
TM = (Path(__file__).resolve().parents[1] / "preview" / "tagmap.js").as_uri()

# 4 corner tags of a 1.0 x 0.6 m area and 4 more round it (the panels.html emulator layout)
EIGHT = {0: (-0.5, -0.3), 1: (0.5, -0.3), 2: (0.5, 0.3), 3: (-0.5, 0.3),
         4: (0, -0.6), 5: (0, 0.6), 6: (-0.9, 0), 7: (0.9, 0)}


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, pre=""):
    code = f'import * as W from "{JS}"; import * as TM from "{TM}"; {pre}; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module"], input=code, capture_output=True, text=True)   # stdin: no length cap
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def floor_tag(x, z, size=0.16, y=0.0):
    """Corners TL, TR, BR, BL of a tag lying face up at (x, y, z), printed top towards -z."""
    h = size / 2
    return [[x - h, y, z - h], [x + h, y, z - h], [x + h, y, z + h], [x - h, y, z + h]]


def M(cols):
    return np.array(cols, float).reshape(4, 4).T


def yaw_np(a, t=(0, 0, 0)):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = t
    return m


def moved(D, corners):
    return [(D @ np.array([*p, 1.0]))[:3].tolist() for p in corners]


# ---------- unit ----------

def test_combos(node_available):
    assert run("W.combos(4, 2)") == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert run("W.combos(4, 4)") == [[0, 1, 2, 3]]
    assert run("[W.combos(3, 4), W.combos(8, 4).length, W.combos(5, 0)]") == [[], 70, [[]]]
    assert run("W.MAX_TAGS") == 8


def test_best_rectangle_picks_the_4_corner_tags_of_8(node_available):
    tags = [{"id": i, "center": [x, 0, z]} for i, (x, z) in EIGHT.items()]
    b = run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0])")
    assert b["ids"] == [0, 1, 2, 3]
    assert sorted([b["rect"]["width"], b["rect"]["height"]]) == pytest.approx([0.6, 1.0])
    assert b["rect"]["center"] == pytest.approx([0, 0, 0], abs=1e-9)


def test_best_rectangle_breaks_an_area_tie_by_squareness(node_available):
    # in EIGHT, tags 0, 2, 4, 5 also make a 0.6 m2 rectangle (about 1 degree off square): the true corners win
    tags = [{"id": i, "center": [x, 0, z]} for i, (x, z) in EIGHT.items() if i in (0, 1, 2, 3, 4, 5)]
    assert run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0]).ids") == [0, 1, 2, 3]
    assert run(f"W.bestRectangle({json.dumps(tags[::-1])}, [0, 1, 0]).ids") == [0, 1, 2, 3]
    # outside the tie a bigger, less square rectangle still wins
    sq = [(-0.5, -0.3), (0.5, -0.3), (0.5, 0.3), (-0.5, 0.3)]
    s = math.tan(math.radians(5))
    sk = [(-0.6, -0.4), (0.6, -0.4), (0.6 + 0.8 * s, 0.4), (-0.6 + 0.8 * s, 0.4)]
    tags = [{"id": i, "center": [x, 0, z]} for i, (x, z) in enumerate(sq + sk)]
    assert run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0]).ids") == [4, 5, 6, 7]


def test_best_rectangle_prefers_the_bigger_of_two_and_sorts_ids(node_available):
    inner = [(9, -0.2, -0.1), (8, 0.2, -0.1), (7, 0.2, 0.1), (6, -0.2, 0.1)]
    outer = [(5, -0.6, -0.4), (1, 0.6, -0.4), (3, 0.6, 0.4), (2, -0.6, 0.4)]
    tags = [{"id": i, "center": [x, 0, z]} for i, x, z in inner + outer]
    assert run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0]).ids") == [1, 2, 3, 5]


def test_best_rectangle_rejects_a_skewed_set(node_available):
    s = math.tan(math.radians(25))           # a parallelogram 25 degrees off square
    pts = [(-0.5, -0.3), (0.5, -0.3), (0.5 + 0.6 * s, 0.3), (-0.5 + 0.6 * s, 0.3)]
    tags = [{"id": i, "center": [x, 0, z]} for i, (x, z) in enumerate(pts)]
    assert run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0])") is None
    assert run(f"W.bestRectangle({json.dumps(tags)}, [0, 1, 0], 30).ids") == [0, 1, 2, 3]
    assert run(f"W.bestRectangle({json.dumps(tags[:3])}, [0, 1, 0])") is None      # fewer than 4


def test_lock_frame_is_level_on_the_long_side(node_available):
    a = math.radians(30)
    u = [math.cos(a), 0.05, -math.sin(a)]                    # a slightly tilted table: x is u laid level
    f = M(run(f"W.lockFrame({{u: {json.dumps(u)}, v: [0, 0, 1], n: [0, 1, 0], center: [1, 0.7, 2]}})"))
    assert f[:3, 1] == pytest.approx([0, 1, 0])
    assert f[:3, 0] == pytest.approx([math.cos(a), 0, -math.sin(a)])
    assert f[:3, 2] == pytest.approx(np.cross(f[:3, 0], [0, 1, 0]))
    assert f[:3, 3] == pytest.approx([1, 0.7, 2])
    # on a wall with u straight up, the level axis comes from v
    g = M(run("W.lockFrame({u: [0, 1, 0], v: [0, 0, -1], n: [1, 0, 0], center: [0, 1, 0]})"))
    assert g[:3, 0] == pytest.approx([0, 0, -1]) and g[:3, 1] == pytest.approx([0, 1, 0])


def walk_pre(ids=(0, 1, 2, 3, 4)):
    corners = {i: floor_tag(*EIGHT[i]) for i in ids}
    rect = {"u": [1, 0, 0], "n": [0, 1, 0], "v": [0, 0, 1], "center": [0, 0, 0]}
    return f"const st = {{...W.startWalk({json.dumps(rect)}, {json.dumps(corners)}, [3, 1, 2, 0], 0.16), pending: {{}}}}"


def test_start_walk_maps_every_tag_and_keeps_the_lock_pose(node_available):
    s = run("{tags: Object.fromEntries(Object.entries(st.map.tags).map(([k, t]) => [k, t.seen])), origin: st.map.origin,"
            " size: st.map.size, lock: W.lockPoseNow(st)}", walk_pre())
    assert s["tags"] == {"0": 30, "1": 30, "2": 30, "3": 30, "4": 3}        # corner tags count as 30 sightings
    assert s["origin"] == [0, 1, 2, 3] and s["size"] == 0.16
    assert M(s["lock"]) == pytest.approx(np.eye(4), abs=1e-9)                # basis(u, n, v, center)
    extra = run("Object.values(W.startWalk({u: [1, 0, 0], n: [0, 1, 0], v: [0, 0, 1], center: [0, 0, 0]},"
                f" {{9: {json.dumps(floor_tag(0, 1))}}}, [], 0.16, 5).map.tags)[0].seen")
    assert extra == 5


def test_lock_pose_on_a_wall_is_the_rectangle_not_the_levelled_frame(node_available):
    # on a wall the lock frame (u, n, v) is not the levelled room frame: roomFromLock carries the difference
    s = run("{lock: W.lockPoseNow(st), wfr: st.wfr}",
            "const st = W.startWalk({u: [0, 0, -1], n: [1, 0, 0], v: [0, -1, 0], center: [2, 1.2, 0.5]}, {}, [], 0.16)")
    want = np.eye(4)
    want[:3, 0], want[:3, 1], want[:3, 2], want[:3, 3] = [0, 0, -1], [1, 0, 0], [0, -1, 0], [2, 1.2, 0.5]
    assert M(s["lock"]) == pytest.approx(want, abs=1e-9)
    assert M(s["wfr"])[:3, 1] == pytest.approx([0, 1, 0])                    # the room frame stays level


def follow(seen, n=80):
    return run(f"(() => {{ let r; for (let i = 0; i < {n}; i++) r = W.walkStep(st, seen); return {{lock: W.lockPoseNow(st), r}}; }})()",
               walk_pre() + f"; const seen = {json.dumps(seen)}")


def test_walk_step_follows_a_shift_from_one_mapped_tag(node_available):
    D = yaw_np(0, (0.03, 0, -0.02))                          # tracking drift: the world appears moved by D
    out = follow({4: moved(D, floor_tag(*EIGHT[4]))})
    assert out["r"]["fix"]["used"] == [4] and out["r"]["added"] == []
    assert M(out["lock"]) == pytest.approx(D, abs=1e-6)


def test_walk_step_follows_a_turn_from_spread_tags_and_trusts_one_tag_turn_less(node_available):
    D = yaw_np(math.radians(2), (0.03, 0, -0.02))
    many = follow({k: moved(D, floor_tag(*EIGHT[k])) for k in (0, 1, 2)})
    assert many["r"]["fix"]["used"] == [0, 1, 2]
    assert M(many["lock"]) == pytest.approx(D, abs=1e-4)
    # one small tag: its own turn is noisy, so the turn eases in slowly while the tag's place is matched at once
    one = follow({4: moved(D, floor_tag(*EIGHT[4]))}, n=20)
    yaw = math.degrees(math.atan2(one["lock"][8], one["lock"][0]))
    assert 0.5 < yaw < 1.9
    at = (M(one["lock"]) @ np.array([0, 0, -0.6, 1]))[:3]
    assert at == pytest.approx((D @ np.array([0, 0, -0.6, 1]))[:3], abs=1e-4)


def test_walk_step_maps_a_new_tag_after_3_sightings(node_available):
    seen = {0: floor_tag(*EIGHT[0]), 3: floor_tag(*EIGHT[3]), 6: floor_tag(*EIGHT[6])}
    out = run("[1, 2, 3].map(() => { const r = W.walkStep(st, seen); return {added: r.added, pending: Object.keys(st.pending),"
              " mapped: Object.keys(st.map.tags).length}; }).concat([st.map.tags[6]])",
              walk_pre() + f"; const seen = {json.dumps(seen)}")
    assert [o["added"] for o in out[:3]] == [[], [], [6]]
    assert [o["pending"] for o in out[:3]] == [["6"], ["6"], []]
    assert [o["mapped"] for o in out[:3]] == [5, 5, 6]
    assert out[3]["seen"] == 3
    assert np.array(out[3]["corners"]) == pytest.approx(np.array(floor_tag(*EIGHT[6])), abs=1e-9)     # room frame = world here


def test_walk_step_maps_new_tags_only_from_2_spread_mapped_tags(node_available):
    one = {0: floor_tag(*EIGHT[0]), 6: floor_tag(*EIGHT[6])}                     # 1 mapped tag: its turn is too rough
    close = {0: floor_tag(*EIGHT[0]), 9: floor_tag(-0.5, -0.1), 6: floor_tag(*EIGHT[6])}   # 2 mapped, 0.2 m apart
    pre = walk_pre() + f"; const one = {json.dumps(one)}, close = {json.dumps(close)}"
    out = run("(() => { for (let i = 0; i < 5; i++) W.walkStep(st, one); const a = [Object.keys(st.pending), st.map.tags[6] || null];"
              " st.map.tags[9] = {...st.map.tags[0], corners: close[9]};"
              " for (let i = 0; i < 5; i++) W.walkStep(st, close); return a.concat([Object.keys(st.pending), st.map.tags[6] || null,"
              " W.walkStep(st, close, {minSpan: 0.15}).fix.used]); })()", pre)
    assert out == [[], None, [], None, [0, 9]]
    out = run("(W.walkStep(st, close, {minSpan: 0.15}), Object.keys(st.pending))",
              pre + "; st.map.tags[9] = {...st.map.tags[0], corners: close[9]}")
    assert out == ["6"]


def test_walk_step_places_a_new_tag_by_the_fit_not_the_lagging_pose(node_available):
    D = yaw_np(math.radians(3), (0.04, 0, 0.02))      # tracking jumped: the eased pose is still far behind
    seen = {k: moved(D, floor_tag(*EIGHT[k])) for k in (1, 2, 7)}
    out = run("[1, 2, 3].map(() => W.walkStep(st, seen).added).concat([st.map.tags[7].corners])",
              walk_pre() + f"; const seen = {json.dumps(seen)}")
    assert out[:3] == [[], [], [7]]
    assert np.array(out[3]) == pytest.approx(np.array(floor_tag(*EIGHT[7])), abs=1e-9)


def test_walk_step_keeps_mapped_tags_put(node_available):
    seen = {0: [[p[0] + 0.01, p[1], p[2]] for p in floor_tag(*EIGHT[0])]}       # seen 1 cm off: the pose moves, not the map
    out = run("(W.walkStep(st, seen), W.walkStep(st, seen), st.map.tags[0].corners)", walk_pre() + f"; const seen = {json.dumps(seen)}")
    assert np.array(out) == pytest.approx(np.array(floor_tag(*EIGHT[0])), abs=1e-12)


def test_walk_step_without_a_mapped_tag_holds_and_maps_nothing(node_available):
    seen = {7: floor_tag(*EIGHT[7])}
    out = run("[W.walkStep(st, seen), W.walkStep(st, {}), st.pending, W.lockPoseNow(st)]",
              walk_pre() + f"; const seen = {json.dumps(seen)}")
    assert out[0] == {"fix": None, "added": []} and out[1] == {"fix": None, "added": []}
    assert out[2] == {}
    assert M(out[3]) == pytest.approx(np.eye(4), abs=1e-12)


def test_walk_step_pending_tag_corners_are_the_mean_of_its_sightings(node_available):
    a, b, c = (floor_tag(0.9, 0.0, y=y) for y in (0.0, 0.003, 0.006))
    out = run("(W.walkStep(st, {...c0, 7: a}), W.walkStep(st, {...c0, 7: b}), W.walkStep(st, {...c0, 7: c}), st.map.tags[7].corners)",
              walk_pre() + f"; const c0 = {json.dumps({1: floor_tag(*EIGHT[1]), 2: floor_tag(*EIGHT[2])})}, a = {json.dumps(a)}, b = {json.dumps(b)}, c = {json.dumps(c)}")
    assert np.array(out) == pytest.approx(np.array(floor_tag(0.9, 0.0, y=0.003)), abs=1e-9)


def test_walk_line(node_available):
    out = run("[W.walkLine(st, {used: [4, 1]}), W.walkLine(st, {used: [2]}), W.walkLine(st, null),"
              " W.walkLine({map: {tags: {1: {}}}}, null)]", walk_pre())
    assert out == ["Held by tags 1, 4 · 5 tags mapped", "Held by tag 2 · 5 tags mapped",
                   "No mapped tag in view: holding by phone tracking · 5 tags mapped",
                   "No mapped tag in view: holding by phone tracking · 1 tag mapped"]


# ---------- smoke ----------

def test_module_loads_in_node(node_available):
    assert run("Object.keys(W).sort()") == ["FUSE", "MAX_TAGS", "bestRectangle", "combos", "lockFrame", "lockPoseNow",
                                             "startWalk", "walkLine", "walkStep"]


# ---------- integration ----------

def test_full_circle_with_few_tags_in_view_and_drift(node_available):
    """Lock on the 4 corners (with tag 4 also seen), then walk 2 laps: at each of 24 stops per lap only the
    tags facing that stop (1 to 3) are seen, through a tracking drift that grows with distance walked. The
    lock must stay within 5 mm of the truth everywhere, every new tag gets mapped, and the raw drift grows
    well past that."""
    stops, R = [], 1.2
    drift = np.eye(4)
    walked = 0.0
    prev = None
    for lap in range(2):
        for i in range(24):
            a = 2 * math.pi * i / 24
            eye = np.array([R * math.sin(a), R * math.cos(a)])
            if prev is not None:
                d = float(np.linalg.norm(eye - prev))
                walked += d
                drift = yaw_np(math.radians(1.0 * d), (0.025 * d * math.cos(a), 0, -0.025 * d * math.sin(a))) @ drift
            prev = eye
            look = eye * 0.75                                   # looks at the near edge of the area
            near = sorted(EIGHT, key=lambda k: np.hypot(EIGHT[k][0] - look[0], EIGHT[k][1] - look[1]))[:3]
            seen = {k: moved(drift, floor_tag(*EIGHT[k])) for k in near
                    if np.hypot(EIGHT[k][0] - look[0], EIGHT[k][1] - look[1]) < 0.75}
            stops.append({"seen": seen, "D": drift.T.ravel().tolist()})
    pre = (walk_pre() + f"; const stops = {json.dumps(stops)}"
           "; const pts = [[-0.5, 0, -0.3], [0.5, 0, -0.3], [0.5, 0, 0.3], [-0.5, 0, 0.3]]")
    out = run("stops.map((s) => { let fix = null; for (let k = 0; k < 8; k++) fix = W.walkStep(st, s.seen).fix || fix;"
              " return {held: !!fix, err: TM.poseDelta(W.lockPoseNow(st), s.D, pts).mm,"
              " raw: TM.poseDelta(TM.basis([1,0,0],[0,1,0],[0,0,1],[0,0,0]), s.D, pts).mm}; })"
              ".concat([Object.keys(st.map.tags).map(Number).sort((a, b) => a - b)])", pre)
    rows, mapped = out[:-1], out[-1]
    assert mapped == list(range(8))
    assert sum(r["held"] for r in rows) >= 46                   # every stop once its tags are mapped
    assert max(r["err"] for r in rows) < 10                    # first lap: the eased pose a few mm behind
    assert max(r["err"] for r in rows[24:]) < 5
    assert max(r["raw"] for r in rows) > 40
