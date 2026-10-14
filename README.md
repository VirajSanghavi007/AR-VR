# Rooftop scan-to-CAD prototype

Phone photos -> COLMAP -> Open3D planes -> STEP/DXF for AutoCAD.

## Setup (already done on this machine)

- `.venv` (Python 3.11): open3d, numpy, scipy, matplotlib, cadquery (OpenCascade, STEP), ezdxf (DXF), pytest
- `tools/colmap/` COLMAP 4.2.1 CUDA build (dense reconstruction needs CUDA)

## Workflow

1. Put photos in `data/rooftop_01/images/`, real measurements (mm) in `data/rooftop_01/measurements.json`.
2. Run `tools/colmap/.../COLMAP.bat`, new project, images = `data/rooftop_01/images`,
   database/workspace in `data/rooftop_01/colmap`, then Reconstruction > Automatic Reconstruction.
3. `.venv\Scripts\python scripts\inspect_cloud.py rooftop_01` : cleans cloud, finds planes,
   writes `outputs/rooftop_01/cleaned_cloud.ply` and `roof_preview.ply`.
4. `.venv\Scripts\python scripts\export_cad.py rooftop_01` : writes `rooftop.step` and `rooftop.dxf`.
5. In AutoCAD: `IMPORT` the .step (solids) or `OPEN` the .dxf (one layer per element).

## Tests

`.venv\Scripts\python -m pytest`

## Solar panel layout (scripts/design_panels.py)

`.venv\Scripts\python scripts\design_panels.py eth3d/terrace` (add `--unit-scale <m per unit>` for non-metric clouds)

Steps: clean cloud, level (up = largest plane normal, sign from camera centres), split near-horizontal
planes into connected patches, pick the largest (`--roof-index` to choose another), concave-hull outline,
keep-outs = anything 0.15-3 m above the roof (+0.3 m clearance), 0.5 m edge setback, best-fit panel grid
(portrait vs landscape, grid offsets, aligned to the roof's long edge).

Outputs in `outputs/<site>/`: `panels_plan.png` (top-down plan), `panels_overlay.ply` (scan + blue panels +
red outline, double-click to view), `roof_panels.step` (roof, keep-outs, panels as solids in mm, AutoCAD IMPORT),
`panels_summary.json`.

ETH3D terrace dense cloud: `colmap image_undistorter` on `dslr_calibration_jpg` (metric poses), then OpenMVS
`InterfaceCOLMAP` + `DensifyPointCloud` (COLMAP's own patch_match_stereo crashes on this machine).

### Tilted rows, roof choice, validation, AR

- Tilted rows: `design_panels.py <site> --tilt 15 --latitude 47.4 --facing-deg -70`. Rows run perpendicular
  to `--facing-deg` (direction panels face, in the plan's 2D frame), spaced so the winter-solstice noon shadow
  clears the next row. A scan has no north: take the facing from a compass bearing on site.
- Roof choice: the candidate table is printed; `--select highest` (roof above ground) or `--roof-index N`.
- Accuracy vs laser scan: `scripts\validate_eth3d.py eth3d/terrace` writes `validation.json`
  (cloud accuracy, plane tilt/height error, outline IoU/edge error, panel corner height error).
- AR: `roof_panels.glb` (glTF 2.0, metres, Y-up, origin = roof-plane origin). Our own writer:
  Open3D's .glb writer emits invalid buffer views.

## 3D / AR preview

- Online (GitHub Pages, works with "View in AR" on phones): see the repository's Pages link.
- Local, any Windows PC with Python 3: double-click `run_preview.bat` (serves `preview/` on http://localhost:8765 and opens it).
- Update the models: rerun `design_panels.py`, copy `outputs/<site>/roof_panels.glb` (3D view) and
  `panels_ar.glb` (AR: panels only, no roof slab) into `preview/`, push; the Pages workflow republishes.
- Panels are generated, not downloaded: aluminium frame + glass with a 120 half-cut cell texture, at the
  exact layout size. AR offers life size (outdoors) or tabletop 1:20.
- Corner-tap placement (`preview/ar.html`, WebXR + three.js, Android Chrome): tap real roof corner A, then B
  (long edge of the outline, shown on a map); the design snaps onto them with outline and keep-outs drawn.
  "True size" keeps metres and reports tapped vs design edge length; "Fit to taps" scales to the taps
  (demo on any open space). Anchor data: `outputs/<site>/panels_anchor.json` -> `preview/terrace_anchor.json`.
- Live layout (default mode of `ar.html`): tap A and B along the sunny edge, then every other roof corner in
  order (any count, concave/L-shapes fine; close by tapping near A or "Close roof"). "Add obstacle": tap around
  a tank/stair room (2 taps = box), panels keep the clearance (default 0.3 m) from it. `preview/roofgrid.js` checks each candidate panel (inside the roof, >= setback from its edges, >= clearance from obstacles) and fills the roof with real-size
  panels (1.722 x 1.134 m, never scaled; count follows the roof), tilt slider 0-90 deg with winter-shadow row
  gaps from the latitude. HUD shows panel count, kWp, measured side lengths and area.
  Desktop check without a phone: open `ar.html` and run `__solar.simulate(corners, obstacles)` in the console; `__solar.tap([x,y,z])` / `press(id)` drive the tap flow.

## AprilTags for real-world scale

Photos alone give a model with arbitrary size. Tags of known size fix that without a tape.

1. `.venv\Scripts\python scripts\make_tags.py` writes `outputs/apriltags_36h11_160mm.pdf` (ids 0-3, one per A4 page).
   Print at 100 % / Actual size; the black square must measure 160 mm (check the 100 mm line).
2. Lay 2-4 tags flat on the roof, spread out, each fully visible in several photos. Take the photos as usual.
3. Run COLMAP (sparse model), then
   `.venv\Scripts\python scripts\tag_scale.py rooftop_01 --tag-size 0.16`
   -> `outputs/rooftop_01/tag_scale.json`: metres per model unit, how well the tags agree (%), and
   tag-to-tag distances to spot-check with a tape.
4. `.venv\Scripts\python scripts\design_panels.py rooftop_01 --unit-scale tags`

Camera models: PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE
(others: run `colmap image_undistorter` first). Text or binary COLMAP models.
