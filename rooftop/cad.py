"""Build a rooftop (slab + parapets) as CAD solids and export STEP / DXF.

Units are millimetres. STEP opens in AutoCAD via IMPORT; DXF via OPEN.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    name: str
    origin: tuple   # (x, y, z) min corner
    size: tuple     # (dx, dy, dz)


@dataclass(frozen=True)
class RooftopSpec:
    length: float
    width: float
    slab_thickness: float = 150
    parapet_height: float = 900
    parapet_thickness: float = 200
    notch_length: float = 0   # L-shaped building: this corner (+x, +y) is cut away
    notch_width: float = 0

    def __post_init__(self):
        for k in ("length", "width", "slab_thickness", "parapet_height", "parapet_thickness"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive")
        if 2 * self.parapet_thickness >= min(self.length, self.width):
            raise ValueError("Parapets are thicker than the roof allows")
        if self.notch_length < 0 or self.notch_width < 0 or (self.notch_length > 0) != (self.notch_width > 0):
            raise ValueError("notch_length and notch_width must both be positive, or both zero")
        if self.notch_length and (self.notch_length >= self.length - 2 * self.parapet_thickness
                                  or self.notch_width >= self.width - 2 * self.parapet_thickness):
            raise ValueError("The notch leaves no roof")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in keys})


def rooftop_boxes(s: RooftopSpec):
    """Slab plus four non-overlapping parapet walls sitting on it."""
    L, W, t, h, p = s.length, s.width, s.slab_thickness, s.parapet_height, s.parapet_thickness
    return [
        Box("RoofSlab", (0, 0, 0), (L, W, t)),
        Box("FrontParapet", (0, 0, t), (L, p, h)),
        Box("BackParapet", (0, W - p, t), (L, p, h)),
        Box("LeftParapet", (0, p, t), (p, W - 2 * p, h)),
        Box("RightParapet", (L - p, p, t), (p, W - 2 * p, h)),
    ]


def export_step(boxes, path):
    import cadquery as cq  # heavy import, keep lazy

    asm = cq.Assembly(name="Rooftop")
    for b in boxes:
        solid = cq.Workplane("XY").box(*b.size, centered=False).translate(b.origin)
        asm.add(solid, name=b.name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    asm.export(str(path), exportType="STEP")
    return path


def export_dxf(boxes, path):
    """Each box as a closed 3D polyface mesh on its own layer."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for b in boxes:
        (x, y, z), (dx, dy, dz) = b.origin, b.size
        v = [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
             (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)]
        doc.layers.add(b.name)
        mesh = msp.add_polyface(dxfattribs={"layer": b.name})
        for f in faces:
            mesh.append_face([v[i] for i in f])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
    return path
