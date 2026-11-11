# Model credits

- `duplex_living_room.glb`: the living room (space A102) of the Duplex Apartment architectural model,
  from BSI (2020) "Duplex Apartment Test Files," buildingSMART International,
  https://github.com/buildingsmart-community/Community-Sample-Test-Files (file
  `IFC 2.3.0.1 (IFC 2x3)/Duplex Apartment/Duplex_A_20110907.ifc`), licensed CC BY 4.0.
  Changes: one room clipped out of the building (2 cm skin of walls, floor and ceiling kept) and
  converted from IFC to GLB with `scripts/ifc_room_to_glb.py`.
- `office_scan_room.dae`, `.obj`, `.mtl`: the "office" scene of the ETH3D high-resolution multi-view
  benchmark (T. Schoeps et al., "A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera
  Videos," CVPR 2017), its two laser scans (https://www.eth3d.net/data/office_dslr_scan_eval.7z),
  licensed CC BY-NC-SA 4.0. Changes: merged, levelled, squared to the walls, cropped to the room, Poisson-meshed
  and decimated to 60k triangles, floor and ceiling replaced by fitted planes, written as SketchUp-style COLLADA
  and OBJ with `scripts/scan_room_to_sketchup.py`. Non-commercial use only.
- `sample_room.glb`, `rooftop_solar.glb`: made in this project (`scripts/make_room_cad.py`, `scripts/phase2_solar.py`).
