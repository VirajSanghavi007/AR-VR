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

## Tabletop AR site (GitHub Pages)

- Online: https://virajsanghavi007.github.io/AR-VR/ (opens `tagar.html`, the tabletop AR on AprilTags;
  `tags.html` checks printed tags). The Pages workflow publishes `preview/` on every push to main.
- Tags: lay 2, 3 or 4 tags (36h11, any ids, each id once); the page counts them itself. 4 or 3 tags mark corners
  of the area; 2 tags sit at opposite corners or side by side (then the area is the square on that side, away
  from the phone). In the desktop emulator 6 cm tags locked an 80 x 50 cm layout from ~0.75 m, 5 cm from ~0.55 m;
  the 16 cm print is the safe default.
- Table or wall: before the lock each tag's own pose comes from its printed size (tagrect.js tagCorners3D) and a
  plane is fitted through all tag corners (surfacePlane). The phone's detected surface is used only when it agrees
  within 20° (a table); a plain wall is rarely detected and the hit lands on the floor, so there the tags' plane is
  used. On a wall the array hangs flush (no rack), rows run level and columns run down the wall (plumbAxes).
- Tilted panels stand on a rail-and-leg rack and cast soft shadows on the table.
- Panels: type the number of rows and columns (1-10 each, on the start page or in the AR bar; panel.js
  gridLayout), the array is laid out again keeping its heading. − / + / Fit scale the whole array (auto fit fills the tag rectangle). A two-finger twist (and
  Panel 15°) turns the picked panel, or the whole array about its centre when none is picked. The HUD shows peak kW,
  kWh/day and the average W per hour (24 h and daylight) from the "Peak sun hours / day" input with 20 % losses:
  a rough estimate, not a yield simulation.
- Local, any Windows PC with Python 3: double-click `run_preview.bat` (serves `preview/` on http://localhost:8765 and opens it).
- The older roof pages (3D terrace viewer, 2-corner roof AR with a panel grid) were removed from the site;
  they are in git history before the commit that removed them. `rooftop/ar_export.py` still writes the GLB files.

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
