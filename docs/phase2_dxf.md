# Phase 2 drawing standard (DXF layer rules)

`rooftop/dxf_read.py` reads a rooftop drawing that follows these rules. The sample
`data/sample_roof/roof.dxf` (made by `scripts/make_sample_dxf.py`) follows them and
`data/sample_roof/measurements.json` holds its true values.

Units: set the drawing's unit to millimetres (`UNITS` in AutoCAD; the reader refuses a
drawing with no unit unless `--units` is given; cm, m, in and ft are converted). The
reader moves the origin to the roof outline's south-west (min x, min y) corner.

| Layer | Entity | Meaning |
|---|---|---|
| ROOF_OUTLINE | one closed polyline | roof footprint: a rectangle, or an L with the +x, +y corner cut away |
| PARAPET | one closed polyline + text `H=1000` | parapet inner edge (one even distance inside the outline) and height |
| STAIR, STORE | closed rectangles + text `H=2700` | rooftop rooms |
| DOOR | line on a room edge | door position and width |
| TANK | circle + text `H=1500 S=600` | tank height and stand height (top = H + S) |
| DRAIN | point | roof drains |
| PIPE | line/polyline + text `D=50` | pipes (diameter, mm) |
| NORTH | one line, tail to tip | north direction (default +y) |
| DIMENSIONS | linear or aligned dimensions | ground truth for the accuracy check |
| SITE_INFO | text `LAT=19.07 LON=72.88 TZ=5.5 FLOORS=3 FLOOR_H=3200` | location (degrees, LON east positive, TZ = hours from UTC; LON and TZ drive the sun on the customer page, default Mumbai) and building height |

ROOF_OUTLINE and PARAPET are required; the rest are optional. A label belongs to the
nearest shape on its own layer. Polylines must be straight (no arcs) and closed where
the table says closed. Blocks (INSERTs) are not read yet: explode them first.

## Other layer names: layers.json

```json
{"ROOF_OUTLINE": "A-ROOF-OTLN", "TANK": ["P-TANK", "P-TANK-OH"]}
```

Unlisted layers keep their standard name. Layer names match without regard to case.

```
python scripts/phase2_solar.py --dxf drawing.dxf --layers layers.json
```
