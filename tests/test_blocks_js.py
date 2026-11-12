"""preview/blocks.js (site.html's design model) through node: unit (materials, blocks and drawn shapes, parts,
frames, picking, polygons, snapping, sizes, undo, options, the design file, model-to-site alignment) and an
integration round trip (a design with every kind saved, parsed and drawn again gives the same parts)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "preview" / "blocks.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr, pre=""):
    code = f'import * as B from "{JS}"; {pre}; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


# ---------- materials and the library ----------

def test_next_material_cycles_both_ways_over_pickable_only(node_available):
    assert run('["wood","glass","concrete","plaster","metal"].map(k => B.nextMaterial(k))') == ["glass", "concrete", "plaster", "metal", "wood"]
    assert run('B.nextMaterial("wood", -1)') == "metal" and run('B.nextMaterial("metal", -1)') == "plaster"
    assert run('B.nextMaterial("soil")') == "wood"                      # a fixed finish goes to the first pickable
    assert run("B.PICKABLE.every(k => B.MATERIALS[k] && !B.MATERIALS[k].fixed)") is True


def test_new_block_takes_library_defaults(node_available):
    b = run('B.newBlock("window", 1, 2, 0.5)')
    assert {k: b[k] for k in ("type", "x", "z", "y", "yaw", "w", "h", "d", "mat")} == {
        "type": "window", "x": 1, "z": 2, "y": 0.9, "yaw": 0.5, "w": 1.2, "h": 1.2, "d": 0.1, "mat": "metal"}
    assert run('B.newBlock("bench")')["x"] == 0
    ids = run('[B.newBlock("wall").id, B.newBlock("wall").id, B.newId()]')
    assert len(set(ids)) == 3 and all(i.startswith("o") for i in ids)
    with pytest.raises(RuntimeError, match="no block called sofa"):
        run('B.newBlock("sofa")')


def test_new_drawn_centres_points_and_sizes(node_available):
    s = run("B.newDrawn('slab', [[1, 1], [5, 1], [5, 4], [1, 4]])")
    assert (s["x"], s["z"], s["w"], s["d"], s["h"], s["mat"]) == (3, 2.5, 4, 3, 0.15, "concrete")
    assert s["pts"] == [[-2, -1.5], [2, -1.5], [2, 1.5], [-2, 1.5]] and s["base"] == {"w": 4, "d": 3}
    w = run("B.newDrawn('walls', [[0, 0], [3, 0]], 1.1)")
    assert (w["w"], w["d"], w["h"], w["mat"], w["x"]) == (3, 0.12, 1.1, "plaster", 1.5) and w["base"] == {"w": 3, "d": 0}
    assert run("B.newDrawn('walls', [[0, 0], [3, 0]]).h") == 2.4
    with pytest.raises(RuntimeError, match="3 or more"):
        run("B.newDrawn('slab', [[0, 0], [1, 0]])")
    with pytest.raises(RuntimeError, match="2 or more"):
        run("B.newDrawn('walls', [[0, 0]])")
    assert run("B.bounds2([[-1, 2], [3, -4]])") == {"minX": -1, "maxX": 3, "minZ": -4, "maxZ": 2, "w": 4, "d": 6}


def test_scaled_pts(node_available):
    assert run("B.scaledPts({type: 'bench'})") is None
    assert run("B.scaledPts({type: 'walls', w: 6, pts: [[-1, 0], [1, 2]], base: {w: 2, d: 2}})") == [[-3, 0], [3, 6]]
    assert run("B.scaledPts({type: 'slab', w: 4, d: 1, pts: [[-1, -1], [1, 1]], base: {w: 2, d: 2}})") == [[-2, -0.5], [2, 0.5]]
    assert run("B.scaledPts({type: 'slab', w: 4, d: 1, pts: [[1, 1]], base: {w: 2, d: 0}})") == [[2, 2]]
    assert run("B.scaledPts({type: 'walls', w: 9, pts: [[1, 1]]})") == [[1, 1]]
    assert run("B.scaledPts({type: 'walls', w: 9, pts: [[1, 1]], base: {w: 0}})") == [[1, 1]]


# ---------- parts ----------

def parts(o):
    return run(f"B.blockParts({json.dumps(o)})")


def obj(type_, w, h, d, **kw):
    return {"type": type_, "x": 0, "z": 0, "y": 0, "yaw": 0, "w": w, "h": h, "d": d, **kw}


def box_extent(ps, axis):
    lo = min(p["c"][axis] - p["s"][axis] / 2 for p in ps if p["kind"] == "box")
    hi = max(p["c"][axis] + p["s"][axis] / 2 for p in ps if p["kind"] == "box")
    return lo, hi


def test_wall_and_door_parts(node_available):
    assert parts(obj("wall", 2, 2.4, 0.12)) == [{"kind": "box", "c": [0, 1.2, 0], "s": [2, 2.4, 0.12], "mat": "main"}]
    p = parts(obj("door", 0.9, 2.1, 0.12))
    assert len(p) == 5 and p[-1]["mat"] == "metal" and box_extent(p, 0) == pytest.approx((-0.45, 0.45))
    assert box_extent(p, 1) == pytest.approx((0, 2.1))
    leaf = p[3]
    assert leaf["s"] == pytest.approx([0.9 - 0.12 - 0.006, 2.1 - 0.06 - 0.006, 0.04]) and leaf["c"][1] == pytest.approx((2.1 - 0.06) / 2)
    assert p[4]["c"] == pytest.approx([(0.9 - 0.12) / 2 - 0.08, 1.02, 0.035])           # handle at half a short leaf
    assert parts(obj("door", 0.9, 2.5, 0.12))[4]["c"][1] == pytest.approx(1.05)        # capped on a tall one


def test_window_and_pergola_parts(node_available):
    p = parts(obj("window", 1.2, 1.2, 0.1))
    assert len(p) == 6 and p[-1]["mat"] == "glass" and p[-1]["s"] == pytest.approx([1.1, 1.1, 0.012])
    assert p[4]["s"] == pytest.approx([0.04, 1.1, 0.08]) and box_extent(p, 1) == pytest.approx((0, 1.2))
    p = parts(obj("pergola", 3, 2.4, 2.7))
    posts, beams, rafters = p[:4], p[4:6], p[6:]
    assert len(rafters) == round(3 / 0.4) + 1 == 9
    assert {tuple(np.round(q["c"], 6)) for q in posts} == {(sx * 1.44, 1.1, sz * 1.29) for sx in (-1, 1) for sz in (-1, 1)}
    assert all(b["s"] == pytest.approx([3.2, 0.2, 0.08]) and b["c"][1] == pytest.approx(2.3) for b in beams)
    assert rafters[0]["c"][0] == pytest.approx(-1.5) and rafters[-1]["c"][0] == pytest.approx(1.5)
    assert rafters[0]["c"][1] == pytest.approx(2.425) and rafters[0]["s"] == pytest.approx([0.05, 0.15, 3.0])
    assert len(parts(obj("pergola", 0.1, 2.4, 1))) == 4 + 2 + 2                        # at least 2 rafters


def test_planter_bench_panel_parts(node_available):
    p = parts(obj("planter", 1.2, 0.45, 0.4))
    balls = [q for q in p if q["kind"] == "ball"]
    assert p[1]["mat"] == "soil" and p[1]["s"] == pytest.approx([1.1, 0.02, 0.3]) and p[1]["c"][1] == pytest.approx(0.425)
    assert len(balls) == 4 and balls[0]["c"] == pytest.approx([-0.45, 0.55, 0]) and balls[0]["r"] == 0.14
    assert parts(obj("planter", 1.2, 0.45, 0.3))[2]["r"] == pytest.approx(0.3 / 2.4) and len([q for q in parts(obj("planter", 0.1, 0.4, 0.4)) if q["kind"] == "ball"]) == 1
    p = parts(obj("bench", 1.6, 0.45, 0.42))
    slats, legs = p[:-2], p[-2:]
    assert len(slats) == 4 and slats[0]["c"] == pytest.approx([0, 0.43, -0.42 / 2 + 0.42 / 8]) and slats[0]["s"] == pytest.approx([1.6, 0.04, 0.42 / 4 - 0.012])
    assert all(q["mat"] == "metal" for q in legs) and legs[1]["c"] == pytest.approx([0.65, 0.205, 0]) and legs[1]["s"] == pytest.approx([0.06, 0.41, 0.378])
    assert len(parts(obj("bench", 1, 0.45, 0.05))) == 2 + 2
    p = parts(obj("panel", 1.038, 0.6, 1.755))
    a = math.radians(15)
    front = 1.755 / 2 - 0.1755
    pivot = 0.15 + front * math.sin(a) + 0.035
    assert p[0]["mat"] == "pv" and p[0]["rx"] == pytest.approx(a) and p[0]["c"] == pytest.approx([0, pivot, 0])
    legs = p[1:]
    assert len(legs) == 4
    assert legs[0]["c"] == pytest.approx([-(1.038 / 2 - 0.2076), (pivot - front * math.sin(a)) / 2, front * math.cos(a)])
    assert legs[1]["s"][1] == pytest.approx(pivot + front * math.sin(a))
    assert legs[2]["c"][0] == pytest.approx(1.038 / 2 - 0.2076)


def test_drawn_parts_and_unknown_type(node_available):
    slab = run("B.newDrawn('slab', [[0, 0], [2, 0], [2, 1]])")
    p = parts(slab)
    assert p == [{"kind": "prism", "pts": slab["pts"], "y0": 0, "h": 0.15, "mat": "main"}]
    w = run("B.newDrawn('walls', [[0, 0], [3, 0], [3, 0], [3, 4]])")                # a repeated point is skipped
    p = parts(w)
    assert len(p) == 2
    assert p[0]["s"] == pytest.approx([3.12, 2.4, 0.12]) and p[0]["ry"] == pytest.approx(0)
    assert p[1]["s"] == pytest.approx([4.12, 2.4, 0.12]) and p[1]["ry"] == pytest.approx(-math.pi / 2)
    assert p[1]["c"] == pytest.approx([3 - w["x"], 1.2, 2 - w["z"]])
    with pytest.raises(RuntimeError, match="no parts for sofa"):
        parts(obj("sofa", 1, 1, 1))


# ---------- frames, picking, polygons ----------

def test_to_local_and_to_room_invert_and_follow_threejs_yaw(node_available):
    o = {"x": 1, "z": 2, "yaw": 0.7}
    lp = run(f"B.toLocal({json.dumps(o)}, [2.5, 1.2])")
    assert run(f"B.toRoom({json.dumps(o)}, {lp})") == pytest.approx([2.5, 1.2])
    # three.js rotation.y = pi/2 turns local +x to room -z
    assert run("B.toRoom({x: 0, z: 0, yaw: Math.PI / 2}, [1, 0])") == pytest.approx([0, -1])
    assert run("B.toRoom({x: 0, z: 0, yaw: Math.PI / 2}, [0, 1])") == pytest.approx([1, 0])


def test_hits_object_for_blocks_slabs_and_walls(node_available):
    bench = obj("bench", 2, 0.45, 0.4, x=1, z=1, yaw=math.pi / 2)
    assert run(f"B.hitsObject({json.dumps(bench)}, [1, 1.9])") and not run(f"B.hitsObject({json.dumps(bench)}, [1.9, 1])")
    assert run(f"B.hitsObject({json.dumps(bench)}, [1.24, 1])") and not run(f"B.hitsObject({json.dumps(bench)}, [1.26, 1])")
    assert run(f"B.hitsObject({json.dumps(bench)}, [1.26, 1], 0.1)")
    thin = obj("window", 1, 1, 0.01)
    assert run(f"B.hitsObject({json.dumps(thin)}, [0, 0.09])")                      # at least 10 cm deep to pick
    slab = run("B.newDrawn('slab', [[0, 0], [2, 0], [2, 2], [0, 2]])")
    assert run(f"B.hitsObject({json.dumps(slab)}, [1, 1])") and run(f"B.hitsObject({json.dumps(slab)}, [2.04, 1])")
    assert not run(f"B.hitsObject({json.dumps(slab)}, [2.06, 1])")
    wl = run("B.newDrawn('walls', [[0, 0], [2, 0]])")
    assert run(f"B.hitsObject({json.dumps(wl)}, [1, 0.1])") and not run(f"B.hitsObject({json.dumps(wl)}, [1, 0.12])")
    assert not run(f"B.hitsObject({json.dumps(wl)}, [-0.2, 0])")


def test_object_at_prefers_the_top_one(node_available):
    objs = [obj("bench", 2, 1, 2), obj("bench", 1, 1, 1), obj("bench", 1, 1, 1, x=5)]
    assert run(f"B.objectAt({json.dumps(objs)}, [0, 0])") == 1
    assert run(f"B.objectAt({json.dumps(objs)}, [0.8, 0])") == 0
    assert run(f"B.objectAt({json.dumps(objs)}, [5, 0])") == 2
    assert run(f"B.objectAt({json.dumps(objs)}, [9, 9])") == -1
    assert run(f"B.objectAt({json.dumps(objs)}, [0.6, 0], 0.15)") == 1


def test_polygon_helpers(node_available):
    sq = [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert run(f"B.pointInPoly([1, 1], {sq})") and not run(f"B.pointInPoly([3, 1], {sq})") and not run(f"B.pointInPoly([1, -0.1], {sq})")
    L = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    assert not run(f"B.pointInPoly([1.5, 1.5], {L})") and run(f"B.pointInPoly([0.5, 1.5], {L})")
    assert run(f"B.polyArea({L})") == 3 and run(f"B.polyArea({L[::-1]})") == 3
    assert run(f"B.polyLength({sq})") == 6 and run(f"B.polyLength({sq}, true)") == 8
    assert run("B.polyLength([[0, 0]])") == 0
    assert run(f"B.polyDist([1, -0.5], {sq}, true)") == pytest.approx(0.5)
    assert run(f"B.polyDist([-0.5, 1], {sq}, true)") == pytest.approx(0.5)
    assert run(f"B.polyDist([-0.5, 1], {sq}, false)") == pytest.approx(math.hypot(0.5, 1))   # open: no closing side
    assert run(f"B.polyDist([3, 3], {sq}, true)") == pytest.approx(math.hypot(1, 1))
    assert run("B.polyDist([1, 1], [[0, 0], [0, 0]], false)") == pytest.approx(math.sqrt(2))


# ---------- drawing helpers ----------

def test_snap_point(node_available):
    assert run("B.snapPoint([1.02, 2.98])") == pytest.approx([1.0, 3.0])
    assert run("B.snapPoint([1.12, 0.01], [0, 0])") == pytest.approx([1.1, 0])
    p = run("B.snapPoint([2.03, 0.31], [0, 0])")                                 # 8.7 deg rounds up to 15 deg
    assert p == pytest.approx([2.05 * math.cos(math.radians(15)), 2.05 * math.sin(math.radians(15))])
    p = run("B.snapPoint([2.03, 0.2], [0, 0])")                                  # 5.6 deg rounds down to 0
    assert p == pytest.approx([2.05, 0], abs=1e-9)
    p = run("B.snapPoint([1, 1.1], [0, 0])")                                      # ~48 deg -> 45 deg
    assert math.degrees(math.atan2(p[1], p[0])) == pytest.approx(45) and math.hypot(*p) == pytest.approx(1.5)
    # second side measured from the first: a near right angle snaps square to it
    p = run("B.snapPoint([2.05, 1.5], [2, 0], [0, 0])")
    assert p == pytest.approx([2, 1.5])
    p = run("B.snapPoint([2.6, 1.0], [2, 0], [0, 1])")                            # first side runs at -26.6 deg
    ref = math.atan2(-1, 2)
    ang = math.atan2(p[1], p[0] - 2) - ref
    assert (math.degrees(ang) / 15) == pytest.approx(round(math.degrees(ang) / 15))
    assert run("B.snapPoint([1.3, 0], [0, 0], null, {step: 0.25})") == pytest.approx([1.25, 0])
    p = run("B.snapPoint([1, 0.5], [0, 0], null, {angleDeg: 45})")
    assert p[1] == pytest.approx(0) or math.degrees(math.atan2(p[1], p[0])) == pytest.approx(45)


def test_closes_outline_and_drag_height(node_available):
    pts = [[0, 0], [2, 0], [2, 2]]
    assert run(f"B.closesOutline({pts}, [0.1, 0.1])") and not run(f"B.closesOutline({pts}, [0.2, 0.2])")
    assert not run(f"B.closesOutline({pts[:2]}, [0, 0])")
    assert run(f"B.closesOutline({pts}, [0.25, 0], 0.3)")
    assert run("B.dragHeight(2.4, 100)") == pytest.approx(2.9) and run("B.dragHeight(2.4, -40)") == pytest.approx(2.2)
    assert run("B.dragHeight(0.1, -1000)") == pytest.approx(0.05)                  # never below one step
    assert run("B.dragHeight(29, 1e5)") == 30
    assert run("B.dragHeight(1, 33, 100, 0.1)") == pytest.approx(1.3)


def test_resized_moved_turned(node_available):
    o = obj("wall", 2, 2.4, 0.12, y=0)
    r = run(f"B.resized({json.dumps(o)}, {{w: 50, h: 0.001, d: 0.5, y: 12}})")
    assert (r["w"], r["h"], r["d"], r["y"]) == (30, 0.02, 0.5, 10)
    assert run(f"B.resized({json.dumps(o)}, {{y: -1}}).y") == 0
    r = run(f"B.resized({json.dumps(o)}, {{h: 1.1}})")
    assert (r["w"], r["h"], r["d"]) == (2, 1.1, 0.12)
    assert run(f"B.resized({json.dumps(o)})") == o
    assert run(f"B.moved({json.dumps(o)}, 3, -1)")["x"] == 3 and run(f"B.moved({json.dumps(o)}, 3, -1)")["z"] == -1
    assert run(f"B.turned({json.dumps({**o, 'yaw': 0.5})}, 0.25).yaw") == 0.75


def test_size_labels_and_names(node_available):
    assert run("B.fmtM(2.4)") == "2.40 m"
    wl = run("B.newDrawn('walls', [[0, 0], [3, 0], [3, 2]], 1.2)")
    assert run(f"B.sizeLabel({json.dumps(wl)})") == "5.00 m of wall, 1.20 m high"
    sl = run("B.newDrawn('slab', [[0, 0], [4, 0], [4, 3], [0, 3]])")
    assert run(f"B.sizeLabel({json.dumps(sl)})") == "4.00 m × 3.00 m, 12.0 m², 0.15 m high"
    assert run(f"B.sizeLabel({json.dumps(obj('wall', 2, 2.4, 0.12))})") == "2.00 m long, 2.40 m high"
    assert run(f"B.sizeLabel({json.dumps(obj('door', 0.9, 2.1, 0.12))})") == "0.90 m × 2.10 m"
    assert run(f"B.sizeLabel({json.dumps(obj('window', 1.2, 1.0, 0.1))})") == "1.20 m × 1.00 m"
    assert run(f"B.sizeLabel({json.dumps(obj('bench', 1.6, 0.45, 0.42))})") == "1.60 m × 0.42 m, 0.45 m high"
    assert run("['panel', 'slab', 'walls', 'thing'].map(t => B.objectName({type: t}))") == ["Solar panel", "Outline", "Wall line", "thing"]


def test_history_undo(node_available):
    r = run("(() => { const h = new B.History(3); const out = [h.canUndo, h.undo()];"
            "h.push({a: 1}); h.push({a: 1}); out.push(h.stack.length, h.canUndo);"
            "h.push({a: 2}); h.push({a: 3}); h.push({a: 4}); out.push(h.stack.length, h.canUndo);"
            "out.push(h.undo(), h.undo(), h.undo(), h.canUndo); return out; })()")
    assert r == [False, None, 1, False, 3, True, {"a": 3}, {"a": 2}, None, False]
    assert run("new B.History().max") == 60


# ---------- options and the design file ----------

def test_copy_option_makes_separate_objects(node_available):
    pre = ("const d0 = B.emptyDesign(); d0.options.A = [B.newDrawn('walls', [[0, 0], [1, 0]]), B.newBlock('bench')];"
           "const d1 = B.copyOption(d0, 'A', 'C'); d1.options.C[0].pts[0][0] = 99;")
    r = run("{a: d0.options.A, c: d1.options.C, b: d1.options.B, same: d0.options.A === d1.options.A}", pre)
    assert len(r["c"]) == 2 and r["b"] == [] and r["same"] is True
    assert {o["id"] for o in r["a"]}.isdisjoint({o["id"] for o in r["c"]})
    assert r["a"][0]["pts"][0][0] != 99 and r["c"][1].get("pts") is None
    assert run("B.emptyDesign()") == {"options": {"A": [], "B": [], "C": []}, "active": "A",
                                      "names": {"A": "Option A", "B": "Option B", "C": "Option C"}}


def test_design_record_keeps_known_keys_rounded(node_available):
    pre = ("const d = B.emptyDesign(); d.active = 'B';"
           "d.options.B = [{...B.newBlock('wall', 1.234567, 2), junk: 5, mesh: {}}];")
    r = run("B.designRecord(d, {tags: {}}, {checks: [1]})", pre)
    assert r["kind"] == "arvr-site-design" and r["version"] == 1 and r["units"] == "m" and r["active"] == "B"
    o = r["options"]["B"][0]
    assert o["x"] == 1.2346 and "junk" not in o and "mesh" not in o and "pts" not in o
    assert r["options"]["A"] == [] and r["map"] == {"tags": {}} and r["checks"] == [1] and r["saved"].endswith("Z")
    w = run("B.designRecord({options: {A: [B.newDrawn('walls', [[0, 0], [1/3, 0]])]}}, null)")
    assert w["options"]["A"][0]["pts"] == [[-0.1667, 0], [0.1667, 0]] and w["options"]["B"] == [] and w["map"] is None


def test_parse_design_fills_defaults_and_rejects_bad_files(node_available):
    rec = {"kind": "arvr-site-design", "version": 1, "active": "Z", "names": {"B": "Garden"},
           "options": {"A": [{"type": "bench", "x": 0, "z": 0, "w": 1, "h": 1, "d": 1}],
                       "C": [{"id": "k", "type": "walls", "x": 0, "z": 0, "w": 1, "h": 1, "d": 1, "mat": "glass", "yaw": 2}]}}
    r = run(f"B.parseDesign({json.dumps(json.dumps(rec))})")
    a = r["design"]["options"]["A"][0]
    assert a["y"] == 0 and a["yaw"] == 0 and a["mat"] == "wood" and a["id"].startswith("o")
    c = r["design"]["options"]["C"][0]
    assert c["id"] == "k" and c["mat"] == "glass" and c["yaw"] == 2
    assert r["design"]["active"] == "A" and r["design"]["options"]["B"] == [] and r["map"] is None
    assert r["design"]["names"] == {"A": "Option A", "B": "Garden", "C": "Option C"} and r["record"]["active"] == "Z"
    assert run(f"B.parseDesign({json.dumps({**rec, 'active': 'C', 'map': {'tags': {}}})})")["design"]["active"] == "C"
    assert run(f"B.parseDesign({json.dumps({**rec, 'map': {'x': 1}})}).map") == {"x": 1}
    assert run(f"B.parseDesign({json.dumps({k: v for k, v in rec.items() if k != 'options'})}).design.options") == {"A": [], "B": [], "C": []}
    for bad, msg in (({"kind": "x"}, "not a site design"), ({**rec, "version": 2}, "design version 2"),
                     ({**rec, "options": {"B": [{"type": "sofa"}]}}, "option B, object 1: unknown type sofa"),
                     ({**rec, "options": {"A": [{"type": "bench", "x": 0, "z": 0, "w": 1, "h": "1", "d": 1}]}}, "object 1: h is not a number"),
                     ({**rec, "options": {"A": [{"type": "slab", "x": 0, "z": 0, "w": 1, "h": 1}]}}, "d is not a number")):
        with pytest.raises(RuntimeError, match=msg):
            run(f"B.parseDesign({json.dumps(bad)})")
    with pytest.raises(RuntimeError, match="not a site design"):
        run("B.parseDesign(null)")


def test_design_file_name(node_available):
    assert run("B.designFileName(new Date('2026-11-11T09:05:59Z'))") == "site-design-2026-11-11-09-05.json"
    assert run("B.designFileName()").startswith("site-design-20")


# ---------- alignment ----------

def test_align_two_turns_and_shifts_in_plan(node_available):
    m1, m2 = [1, 0.5, 0], [3, 0.5, 0]                     # model: along +x
    r1, r2 = [0, 0, 0], [0, 0, -4]                        # site: along -z, twice as long
    a = run(f"B.alignTwo({m1}, {m2}, {r1}, {r2})")
    assert a["yaw"] == pytest.approx(math.pi / 2) and a["ratio"] == pytest.approx(2) and a["lengthModel"] == 2 and a["lengthReal"] == 4
    assert a["t"][1] == 0
    assert run(f"B.applyAlign({json.dumps(a)}, {m1})") == pytest.approx([0, 0.5, 0], abs=1e-12)   # height kept
    p = run(f"B.applyAlign({json.dumps(a)}, {m2})")
    assert p == pytest.approx([0, 0.5, -2], abs=1e-12)                                     # along r1 -> r2
    # an arbitrary case: m1 lands on r1 in plan, direction matches
    rng = np.random.default_rng(1)
    for _ in range(5):
        m1, m2, r1, r2 = (rng.uniform(-3, 3, 3).round(3).tolist() for _ in range(4))
        a = run(f"B.alignTwo({m1}, {m2}, {r1}, {r2})")
        p1, p2 = run(f"[B.applyAlign({json.dumps(a)}, {m1}), B.applyAlign({json.dumps(a)}, {m2})]")
        assert [p1[0], p1[2]] == pytest.approx([r1[0], r1[2]], abs=1e-9) and p1[1] == pytest.approx(m1[1])
        d_got = np.array([p2[0] - p1[0], p2[2] - p1[2]])
        d_want = np.array([r2[0] - r1[0], r2[2] - r1[2]])
        assert d_got / np.linalg.norm(d_got) == pytest.approx(d_want / np.linalg.norm(d_want), abs=1e-9)
    for args in ("[0,0,0], [0.005,3,0], [0,0,0], [1,0,0]", "[0,0,0], [1,0,0], [0,0,0], [0,2,0.009]"):
        with pytest.raises(RuntimeError, match="1 cm apart"):
            run(f"B.alignTwo({args})")


def test_nearest_corner_in_plan_at_floor_height(node_available):
    r = run("B.nearestCorner([0, 0.1, 0], [3, 2.5, 2], [2.8, 2.4, 0.3])")
    assert r["corner"] == [3, 0.1, 0] and r["dist"] == pytest.approx(math.hypot(0.2, 0.3))
    assert run("B.nearestCorner([0, 0, 0], [3, 2.5, 2], [-1, 9, 3]).corner") == [0, 0, 2]
    assert run("B.nearestCorner([0, 0, 0], [3, 2.5, 2], [0.1, 0, 0.1]).corner") == [0, 0, 0]
    assert run("B.nearestCorner([0, 0, 0], [3, 2.5, 2], [2.9, 0, 1.9]).corner") == [3, 0, 2]


# ---------- integration: save -> open -> same parts ----------

def test_design_round_trip_gives_the_same_parts(node_available):
    pre = ("const d = B.emptyDesign();"
           "d.options.A = ['wall','door','window','pergola','planter','bench','panel'].map((t, i) => B.turned(B.newBlock(t, i, -i), i / 3));"
           "d.options.B = [B.resized(B.newDrawn('slab', [[0,0],[4,0],[4,3],[1,3]]), {w: 5}), B.newDrawn('walls', [[0,0],[3,0],[3,2]], 1.2)];"
           "d.options.C = B.copyOption(d, 'B', 'C').options.C;"
           "const back = B.parseDesign(JSON.stringify(B.designRecord(d, null))).design;"
           "const P = (dd) => B.OPTIONS.flatMap(k => dd.options[k].map(o => [o.type, B.blockParts(o), B.sizeLabel(o)]));")
    a, b = run("[P(d), P(back)]", pre)
    assert len(a) == 11

    def close(x, y):
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return math.isclose(x, y, abs_tol=2e-4)
        if isinstance(x, list):
            return len(x) == len(y) and all(close(i, j) for i, j in zip(x, y))
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(close(x[k], y[k]) for k in x)
        return x == y
    assert close(a, b)


# ---------- solar arrays ----------

RECT10x5 = [[0, 5], [10, 5], [10, 0], [0, 0]]                     # front edge along z = 5
PLAN_D = 1.762 * math.cos(math.radians(15))
PITCH = PLAN_D + 1.762 * math.sin(math.radians(15)) / math.tan(math.radians(25))


def test_new_array_puts_the_front_edge_on_plus_z_whichever_way_round(node_available):
    for pts in (RECT10x5, [[0, 0], [10, 0], [10, 5], [0, 5]], [[10, 5], [10, 0], [0, 0], [0, 5]]):
        a = run(f"B.newArray({json.dumps(pts)})")
        assert a["type"] == "array" and a["mat"] == "metal" and a["h"] == 0.01 and a["pv"] == {} and a["y"] == 0
        assert (a["x"], a["z"]) == (pytest.approx(5), pytest.approx(2.5))
        back = run(f"(() => {{ const a = B.newArray({json.dumps(pts)}); return a.pts.map(p => B.toRoom(a, p)); }})()")
        assert back == [pytest.approx(p, abs=1e-9) for p in pts]       # the local corners map back to the taps
        maxz = max(p[1] for p in a["pts"])
        assert a["pts"][0][1] == pytest.approx(maxz) and a["pts"][1][1] == pytest.approx(maxz)
        assert a["base"] == {"w": pytest.approx(a["w"]), "d": pytest.approx(a["d"])}
    assert [run(f"B.newArray({json.dumps(p)})")["w"] for p in (RECT10x5, [[10, 5], [10, 0], [0, 0], [0, 5]])] == [pytest.approx(10), pytest.approx(5)]
    assert run("B.newArray([[10, 5], [10, 0], [0, 0], [0, 5]]).d") == pytest.approx(10)
    assert run("B.newArray([[0,0],[1,0],[1,1]], {tiltDeg: 5}).pv") == {"tiltDeg": 5}
    assert run("B.newDrawn('array', [[0, 0], [2, 0], [2, 2]]).type") == "array"
    with pytest.raises(RuntimeError, match="3 or more roof corners"):
        run("B.newArray([[0, 0], [1, 0]])")
    with pytest.raises(RuntimeError, match="front edge is too short"):
        run("B.newArray([[0, 0], [0.04, 0], [1, 1]])")
    assert run("B.newArray([[0, 0], [0.05, 0], [1, 1]]).type") == "array"


def test_segs_cross_and_rect_in_poly(node_available):
    assert run("B.segsCross([0,0],[2,2],[0,2],[2,0])") is True
    assert run("B.segsCross([0,0],[1,1],[1,1],[2,0])") is False          # touching at an end is not crossing
    assert run("B.segsCross([0,0],[1,0],[0,1],[1,1])") is False
    assert run("B.segsCross([0,0],[2,0],[1,-1],[1,1])") is True
    assert run("B.segsCross([0,0],[2,0],[1,0.5],[1,1])") is False         # c-d stops short of a-b
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert run(f"B.rectInPoly(1, 9, 1, 9, {sq})") is True
    assert run(f"B.rectInPoly(-1, 9, 1, 9, {sq})") is False and run(f"B.rectInPoly(1, 9, 1, 11, {sq})") is False
    assert run(f"B.rectInPoly(20, 21, 20, 21, {sq})") is False                 # wholly outside, no side crossing
    left = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 6], [8, 6], [8, 4], [0, 4]]   # a slot cut in from the left
    assert run(f"B.rectInPoly(7, 9, 1, 9, {left})") is False                  # only the rect's left side is crossed
    right = [[0, 0], [10, 0], [10, 4], [2, 4], [2, 6], [10, 6], [10, 10], [0, 10]]
    assert run(f"B.rectInPoly(1, 3, 1, 9, {right})") is False                 # only its right side is crossed
    u = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 2], [4, 2], [4, 10], [0, 10]]
    assert run(f"B.rectInPoly(1, 9, 5, 6, {u})") is False               # all 4 corners inside, the slot cuts through
    assert run(f"B.rectInPoly(1, 3, 5, 6, {u})") is True and run(f"B.rectInPoly(1, 9, 0.5, 1.5, {u})") is True


def layout(pts, pv=None):
    return run(f"B.arrayLayout(B.newArray({json.dumps(pts)}, {json.dumps(pv or {})}))")


def test_array_layout_rows_and_columns_on_a_rectangle(node_available):
    L = layout(RECT10x5)                       # 9 m clear: 7 portrait panels 1.134 m wide, 2 cm apart; 4 m deep: 1 row
    assert (L["n"], L["rows"]) == (7, 1)
    assert (L["pw"], L["sl"]) == (1.134, 1.762) and L["tilt"] == pytest.approx(math.radians(15))
    assert L["planD"] == pytest.approx(PLAN_D) and L["pitch"] == pytest.approx(PITCH)
    assert L["rowGap"] == pytest.approx(PITCH - PLAN_D)
    xs = [p[0] for p in L["panels"]]
    assert xs[0] == pytest.approx(-5 + 0.5 + 0.567) and all(b - a == pytest.approx(1.154) for a, b in zip(xs, xs[1:]))
    assert all(p[1] == pytest.approx(2.5 - 0.5 - PLAN_D / 2) for p in L["panels"])      # front row hugs the setback
    assert L["kWp"] == pytest.approx(3.08) and L["kwhDay"] == pytest.approx(3.08 * 5 * 0.8)
    deep = layout([[0, 8.2], [10, 8.2], [10, 0], [0, 0]])         # 7.2 m clear >= planD + 2 pitch = 7.06 m
    assert (deep["n"], deep["rows"]) == (21, 3)
    zs = sorted({round(p[1], 6) for p in deep["panels"]}, reverse=True)
    assert zs[0] - zs[1] == pytest.approx(PITCH) and zs[1] - zs[2] == pytest.approx(PITCH)
    assert layout([[0, 8], [10, 8], [10, 0], [0, 0]])["rows"] == 2   # 7.0 m clear < 7.06 m
    exact = 0.5 * 2 + 7 * 1.134 + 6 * 0.02                            # a width that fits 7 exactly
    assert layout([[0, 5], [exact, 5], [exact, 0], [0, 0]])["n"] == 7
    assert layout([[0, 5], [exact - 0.001, 5], [exact - 0.001, 0], [0, 0]])["n"] == 6


def test_array_layout_settings_change_the_fill(node_available):
    land = layout(RECT10x5, {"orient": "landscape"})
    assert (land["pw"], land["sl"]) == (1.762, 1.134) and land["n"] == 10        # 5 a row, 2 rows
    flat = layout(RECT10x5, {"tiltDeg": 0})
    assert flat["rowGap"] == 0.02 and flat["pitch"] == pytest.approx(1.782) and flat["n"] == 14
    assert layout(RECT10x5, {"setback": 0})["n"] == 16                 # 8 a row, 2 rows
    assert layout(RECT10x5, {"setback": 2.4})["n"] == 0               # 0.2 m deep left: nothing fits
    narrow = layout([[0, 10], [1.5, 10], [1.5, 0], [0, 0]])          # deep but 0.5 m clear across: rows tried, none fill
    assert (narrow["n"], narrow["rows"]) == (0, 0)
    low = layout([[0, 8.2], [10, 8.2], [10, 0], [0, 0]], {"sunElevDeg": 15})
    assert low["pitch"] > PITCH and low["rows"] == 2
    w = layout(RECT10x5, {"module": {"watts": 500}, "sunHours": 4, "derate": 1})
    assert w["kWp"] == pytest.approx(3.5) and w["kwhDay"] == pytest.approx(14) and w["pw"] == 1.134
    assert layout([[0, 100], [100, 100], [100, 0], [0, 0]])["n"] == run("B.MAX_PANELS") == 300


def test_array_layout_keeps_every_panel_inside_an_l_shaped_roof(node_available):
    pts = [[0, 0], [12, 0], [12, 8], [6, 8], [6, 4], [0, 4]]       # front edge z = 0, no roof at x < 6, z > 4
    r = run(f"""(() => {{ const a = B.newArray({json.dumps(pts)}), L = B.arrayLayout(a), sb = 0.5, e = 1e-4;
        return {{ n: L.n, rows: L.rows,
                  inside: L.panels.every(([x, z]) => B.rectInPoly(x - L.pw / 2 - sb + e, x + L.pw / 2 + sb - e, z - L.planD / 2 - sb + e, z + L.planD / 2 + sb - e, a.pts)),
                  room: L.panels.map(p => B.toRoom(a, p)) }}; }})()""")
    assert r["n"] >= 10 and r["rows"] >= 2 and r["inside"] is True
    assert all(x > 6 or z < 4 for x, z in r["room"])                   # nothing over the missing corner
    assert any(x < 6 for x, z in r["room"]) and any(z > 4 for x, z in r["room"])


def test_pv_tuned_clamps_and_keeps_the_old_object(node_available):
    r = run("(() => { const a = B.newArray([[0,0],[4,0],[4,3]]); const b = B.pvTuned(a, {tiltDeg: 60}); return [a.pv, b.pv, b.type, b.id === a.id]; })()")
    assert r == [{}, {"tiltDeg": 45}, "array", True]
    assert run("B.pvTuned({pv: {orient: 'landscape'}}, {tiltDeg: -5}).pv") == {"orient": "landscape", "tiltDeg": 0}
    assert run("B.pvTuned({}, {tiltDeg: 20}).pv") == {"tiltDeg": 20}
    assert run("B.pvTuned({}, {setback: 0.12345}).pv") == {"setback": 0.12}
    assert run("B.pvTuned({}, {setback: -1}).pv.setback") == 0 and run("B.pvTuned({}, {setback: 9}).pv.setback") == 3
    assert run("B.pvTuned({}, {setback: 2.5}).pv.setback") == 2.5
    assert run("B.pvTuned({}, {orient: 'sideways'}).pv.orient") == "portrait"
    assert run("B.pvTuned({}, {orient: 'landscape'}).pv.orient") == "landscape"
    assert run("B.pvTuned({pv: {tiltDeg: 10}}, {setback: 1}).pv") == {"tiltDeg": 10, "setback": 1}
    assert run("B.pvOf({pv: {tiltDeg: 7, module: {watts: 500}}})") == {
        "orient": "portrait", "tiltDeg": 7, "setback": 0.5, "sunElevDeg": 25, "colGap": 0.02, "sunHours": 5, "derate": 0.8,
        "module": {"w": 1.134, "l": 1.762, "watts": 500}}
    assert run("B.pvOf({}).module") == {"w": 1.134, "l": 1.762, "watts": 440}


def span(pts):
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) - min(xs), max(zs) - min(zs)


def test_solar_totals_labels_and_picking(node_available):
    pre = f"const a = B.newArray({json.dumps(RECT10x5)}), b = B.newArray({json.dumps(RECT10x5)}, {{orient: 'landscape'}}), w = B.newBlock('wall')"
    t = run("B.solarTotals([a, w, b])", pre)
    assert (t["arrays"], t["n"]) == (2, 17) and t["kWp"] == pytest.approx(7.48) and t["kwhDay"] == pytest.approx(29.92)
    assert run("B.solarTotals([w])", pre) == {"arrays": 0, "n": 0, "kWp": 0, "kwhDay": 0}
    assert run("B.totalsLine(B.solarTotals([a, w, b]))", pre) == "17 panels, 7.48 kWp, about 29.9 kWh/day"
    assert run("B.totalsLine({n: 1, kWp: 0.44, kwhDay: 1.76})") == "1 panel, 0.44 kWp, about 1.8 kWh/day"
    assert run("B.sizeLabel(a)", pre) == "7 panels, 3.08 kWp, about 12.3 kWh/day"
    assert run("B.objectName(a)", pre) == "Solar array"
    assert run("[B.hitsObject(a, [5, 2.5]), B.hitsObject(a, [9.9, 4.9]), B.hitsObject(a, [11, 2.5])]", pre) == [True, True, False]
    assert span(run("B.scaledPts({...a, w: 20, d: 10})", pre)) == (pytest.approx(20), pytest.approx(10))


def test_array_parts_are_the_sheet_and_three_boxes_a_panel(node_available):
    a = run(f"B.newArray({json.dumps(RECT10x5)})")
    ps, L = parts(a), run(f"B.arrayLayout({json.dumps(a)})")
    assert ps[0]["kind"] == "prism" and ps[0]["h"] == 0.01 and ps[0]["y0"] == 0 and ps[0]["mat"] == "main"
    assert ps[0]["pts"] == [pytest.approx(p) for p in a["pts"]]
    assert len(ps) == 1 + 3 * 7
    t, half = math.radians(15), 1.762 / 2
    pivot = 0.15 + half * math.sin(t) + 0.035
    for i, (x, z) in enumerate(L["panels"]):
        pv, front, rear = ps[1 + 3 * i: 4 + 3 * i]
        assert pv["mat"] == "pv" and pv["c"] == pytest.approx([x, pivot, z]) and pv["s"] == pytest.approx([1.134, 0.035, 1.762])
        assert pv["rx"] == pytest.approx(t)
        assert front["mat"] == rear["mat"] == "main" and "rx" not in front
        assert front["c"][2] > z > rear["c"][2] and front["s"][1] < rear["s"][1]       # low leg at the front (+z)
        assert front["c"][1] == pytest.approx(front["s"][1] / 2) and rear["c"][1] == pytest.approx(rear["s"][1] / 2)
        assert rear["s"][1] - front["s"][1] == pytest.approx(2 * 0.8 * half * math.sin(t))
        assert front["s"][1] + rear["s"][1] == pytest.approx(2 * pivot)
        assert front["c"][2] - z == pytest.approx(0.8 * half * math.cos(t)) and z - rear["c"][2] == pytest.approx(0.8 * half * math.cos(t))
        assert front["s"][0] == pytest.approx(0.9 * 1.134) and rear["s"][2] == front["s"][2] == 0.04 and front["c"][0] == pytest.approx(x)
    assert len(parts({**a, "pv": {"setback": 2.4}})) == 1


def test_arrays_save_open_and_copy_with_their_settings(node_available):
    pre = (f"const a = B.pvTuned(B.newArray({json.dumps(RECT10x5)}), {{orient: 'landscape', tiltDeg: 10}});"
           "const d = {...B.emptyDesign(), options: {A: [a], B: [], C: []}};"
           "const r = B.parseDesign(JSON.parse(JSON.stringify(B.designRecord(d, null))))")
    got = run("[r.record.options.A[0].pv, r.design.options.A[0].type, B.sizeLabel(r.design.options.A[0]) === B.sizeLabel(a)]", pre)
    assert got == [{"orient": "landscape", "tiltDeg": 10}, "array", True]
    c = run("(() => { const c = B.copyOption(d, 'A', 'B'); c.options.B[0].pv.tiltDeg = 30; return [a.pv.tiltDeg, c.options.B[0].pv.tiltDeg, c.options.B[0].id !== a.id]; })()", pre)
    assert c == [10, 30, True]
    assert run("'pv' in B.copyOption({...B.emptyDesign(), options: {A: [B.newBlock('wall')], B: [], C: []}}, 'A', 'B').options.B[0]") is False


def test_array_edge_cases_the_mutation_check_found(node_available):
    # newDrawn('array') hands over to newArray (its pv and its local, edge-aligned corners), not the slab path
    assert run("B.newDrawn('array', [[0, 0], [2, 0], [2, 2]]).pv") == {}
    # a front edge at 45 degrees: the object turns so both ends of that edge sit on its +z side
    diag = [[0, 0], [2, 2], [4, 0], [2, -2]]
    a = run(f"B.newArray({json.dumps(diag)})")
    assert a["pts"][0][1] == pytest.approx(a["pts"][1][1]) and a["w"] == pytest.approx(math.hypot(2, 2))
    # resized deeper than wide: depth scales on its own
    pre = f"const r = B.newArray({json.dumps(RECT10x5)})"
    assert span(run("B.scaledPts({...r, w: 20, d: 5})", pre)) == (pytest.approx(20), pytest.approx(5))
    # picking follows the L outline, not its bounding box
    ell = [[0, 4], [4, 4], [4, 0], [2, 0], [2, 2], [0, 2]]
    pre = f"const l = B.newArray({json.dumps(ell)})"
    assert run("[B.hitsObject(l, [3, 1]), B.hitsObject(l, [0.5, 0.5])]", pre) == [True, False]
    # a T touch (one end on the other segment) is not a crossing
    assert run("B.segsCross([0,0],[2,0],[1,0],[1,1])") is False
    # a slot cut in from below crosses only the rect's first (z0) side
    below = [[-1, -1], [1.5, -1], [1.5, 1], [2.5, 1], [2.5, -1], [5, -1], [5, 3], [-1, 3]]
    assert run(f"B.rectInPoly(0, 4, 0, 2, {below})") is False
    assert run(f"B.rectInPoly(0, 4, 1.5, 2, {below})") is True
