# AR/VR: Hybrid scan-to-CAD prototype

Global rules in `~/.claude/CLAUDE.md` apply here (no em dashes, tests for every
new/changed function, progress counters + ETA, checkpoint/resume, detached
long jobs, resumable downloads, Kimi K3 delegation via opencode).

## Project goal

Turn phone photos of a real place into dimensionally accurate, editable CAD,
as the base for an AR solar-panel design product. Geometry-first, not
photorealism: a clean roof plane beats a million-triangle mesh.

## Current scope (prototype 1)

- Environment: flat building rooftop.
- Capture: phone photos (40-100, ~70-80% overlap), processed offline.
- Output: STEP (surfaces and solids).
- Target accuracy: 5-10 cm, to be validated, never assumed.

## Pipeline

1. Capture photos + independent tape/laser measurements (length, width, a diagonal).
2. COLMAP: camera poses (sparse). Dense: OpenMVS (tools/openmvs), since COLMAP 4.2.1 patch_match_stereo crashes (0xC0000409) on this RTX 5050.
3. Open3D (Python 3.11 venv): downsample, outlier removal, RANSAC plane fitting.
4. Scale + align: photogrammetry scale is arbitrary until set from known dimensions: AprilTags (rooftop/tags.py, scripts/make_tags.py + tag_scale.py, 36h11, 160 mm default) or a tape length.
5. CadQuery (OpenCascade, in the venv) builds solids, exports STEP; ezdxf exports DXF. Opened in AutoCAD (user has AutoCAD, NOT FreeCAD).
6. Validate against independent measurements (dimension, edge, obstacle error separately).

## Layout

```
data/<site>/images/          raw photos, never modified
data/<site>/measurements.json independent ground truth
data/<site>/colmap/          COLMAP project
outputs/<site>/              cleaned clouds, previews, .step, .dxf
rooftop/                     library: cloud.py (Open3D), cad.py (CadQuery/ezdxf)
rooftop/phase2.py (auto panel layout on the AutoCAD roof -> STEP + GLB + rooftop_solar.json sidecar the viewer reads; L-shaped footprint via RooftopSpec notch_length/notch_width, rooftop structures (stair, store, tanks) with keep-outs = footprint + winter shadow), render.py (STEP -> 3D PNG); preview/vr.html (phone viewer for Phase 2 and Phase 3 (`?phase=3`): Orbit, Cardboard VR, AR at table size or full size with Move/Exit; photoreal scene); scripts/phase2_solar.py, render_cad.py
rooftop/phase3.py + scripts/phase3_preview.py: Phase 3 model for vr.html?phase=3 (design_panels.py output + the ETH3D terrace laser scan as a toggleable LiDAR layer, preview/models/phase3_terrace.*)
rooftop/sketchup_sample.py + scripts/make_sketchup_sample.py: a 3 x 2.7 x 2.4 m pergola with six panels written as SketchUp-style exports (.dae in inches with Z_UP, .obj/.mtl with a "# File units" header) to preview/models/sample_sketchup_pergola.*; tagar.html + model.js read SketchUp exports (.skp must be exported to .dae or .obj first): .dae via ColladaLoader, .obj picked together with its .mtl and textures (LoadingManager.setURLModifier matches basenames), units "auto" = as the file says; the "Try a SketchUp export" button (m-sketchup) loads the sample .dae
rooftop/geometry.py, layout.py, panels.py, ar_export.py, validate.py: levelling, outline/keep-outs/(tilted) panel grid, STEP/PLY/PNG/GLB export, laser-scan validation
scripts/                     inspect_cloud.py, export_cad.py, design_panels.py, validate_eth3d.py, make_tags.py, tag_scale.py, make_tag_js.py (CLIs)
preview/                     tabletop AR site (GitHub Pages; index.html just redirects to tagar.html; old roof pages removed, see git history): tagar.html + tagrect.js + panel.js + phoneemu.js (desktop phone emulator: `__tagar.emuStart/emuRun` render tags from a shaken virtual phone into the real detection; tabletop AR locks to 2, 3 or 4 AprilTags of any ids, counted automatically (tagrect.js chooseTags/fitFromThree; 2 tags at opposite corners or side by side = the square beyond them), works on a table or a wall (tagCorners3D/surfacePlane: tag plane from the printed size, hit-test plane only when it agrees; plumbAxes: v down the wall; wall arrays mount flush), follows them when moved, tilted panels stand on a rack (panel.js mountParts) with soft shadows, auto fit = the panel fills the tag rectangle (or a round 1:N), drag stays inside it (panel.js moveBox); a panel array (rows x columns typed in, panel.js gridLayout/clampCount, scale and auto fit act on the whole array, HUD power: peak kW, kWh/day, avg W per hour from a sun-hours input via panel.js energy()), ACES tone mapping, sun colour/strength eased to the WebXR light estimate (panel.js smoothLight), Standard face = 1024 px canvas with chamfered half-cut cells (chamferCell), per-cell tint (cellShade), fingers, 9 busbars, bevelled frame; two types: Standard and BIPV shingle (bipv_shingle.jpg face on a glass laminate: MeshPhysicalMaterial clearcoat, heightMap bump, printed border, teal edge, real-room reflections from ARCore's cube map when the phone offers one (RoomEnvironment otherwise), a camera look (grain, vignette, softer render) so the 3D matches the camera image, and a contact shadow under the panel; debug hooks `__tagar.closeUp/panelWorld`), long-press drag + two-finger twist; snap.js = Photo button: camera + panel composite with a caption, Save/Share), tags.html + tagcheck.js (in-browser AprilTag capture check; tag36h11.js generated by make_tag_js.py, js-aruco2 2.0.0 vendored)
tools/openmvs/               OpenMVS 2.4.0 (DensifyPointCloud, Viewer.exe = .ply handler)
data/eth3d/{terrace,facade}/ ETH3D with metric poses + laser scans (ground truth)
tests/                       pytest, run with .venv\Scripts\python -m pytest
tools/colmap/                COLMAP 4.2.1 CUDA build (bin\colmap.exe, COLMAP.bat)
```

## Conventions

- CAD units: millimetres. Point cloud units are arbitrary until scaled.
- Python 3.11 venv at .venv (system default python is 3.14, use .venv).
- Do not validate the scan against a CAD model built from the same data (circular).
- Human-assisted first (user taps/corrects roof corners, enters one known length),
  automate edge/corner/obstacle detection later.
- Out of scope for now: UWB, live AR scanning, neural nets, drones, relocalization.
- Demo videos of tagar.html (OBS recipe and pitfalls: global CLAUDE.md section 18): the phone
  emulator only finds 160 mm tags from about 1.2-1.35 m (not 1.6-3 m), so keep virtual tags about
  1.0 x 0.6 m apart and the eye close; `__tagar.setRoom({..., seeThrough: false})` starts the room
  solid (it defaults to see-through, which makes a solid-then-see-through demo look reversed).
