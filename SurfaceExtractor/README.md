# Surface Material Extractor  —  V01

Standalone companion tool to Starling Classifier v2.

Reads **classified drone photogrammetry point clouds** (LAS/LAZ with class 2 = ground) and extracts **2D polygons of surface materials** — grass, asphalt/tarmac, concrete, painted markings — ready for import into AutoCAD / Civil3D / Revit for LOD 200 topography workflows.

## What it does

```
Classified LAZ (class 2 ground) → surface polygons
                    │
                    ▼
     ┌─────────────────────────────┐
     │ 1. Filter to class 2 ground │
     │ 2. Auto-remove parked cars  │
     │ 3. Per-point material class │
     │    (VARI + brightness +     │
     │     saturation thresholds)  │
     │ 4. Rasterise to 0.25 m grid │
     │ 5. Morphological cleanup    │
     │ 6. Vectorise per class      │
     │ 7. Simplify + filter small  │
     │ 8. Export DXF + GeoJSON     │
     └─────────────────────────────┘
                    │
                    ▼
      DXF (layered by material)
      GeoJSON (attribute per polygon)
      CSV summary (areas per class)
```

## Output classes

| DXF layer | Meaning | AutoCAD colour |
|---|---|---|
| `SURF_GRASS`             | Vegetation / lawn areas | 3 (green) |
| `SURF_ASPHALT_TARMAC`    | Dark hard surface (roads, parking) | 8 (dark grey) |
| `SURF_CONCRETE`          | Light hard surface (kerbs, slabs) | 9 (light grey) |
| `SURF_PAINTED`           | Painted markings (parking lines, text) | 2 (yellow) |
| `SURF_OTHER`             | Unclassified / mixed | 6 (magenta) |

## Requirements

- Python 3.12 + venv (uses same libraries as Starling Classifier v2)
- Input: a classified LAZ/LAS file OR folder of classified tiles
- Point Data Format with RGB (formats 2, 3, 5, 7, 8, 10)

## Launch

```cmd
start_gui.bat
```

## Standalone use

This tool assumes ground classification is already done. Run **Starling Classifier v2** first if your input isn't classified yet.
