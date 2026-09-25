"""
Export material polygons to DXF (AutoCAD/Civil3D/Revit-ready) and GeoJSON.
"""

import json
from pathlib import Path
from .material_classifier import (
    MATERIAL_NAMES, MAT_GRASS, MAT_ASPHALT, MAT_CONCRETE, MAT_PAINTED, MAT_OTHER,
)


# DXF layer names + AutoCAD Color Index (ACI)
LAYER_CONFIG = {
    MAT_GRASS:    ("SURF_GRASS",           3),   # green
    MAT_ASPHALT:  ("SURF_ASPHALT_TARMAC",  8),   # dark grey
    MAT_CONCRETE: ("SURF_CONCRETE",        9),   # light grey
    MAT_PAINTED:  ("SURF_PAINTED",         2),   # yellow
    MAT_OTHER:    ("SURF_OTHER",           6),   # magenta
}


def write_dxf(
    polygons_by_class: dict,
    out_path: str | Path,
) -> Path:
    """
    Write polygons to a layered DXF (AutoCAD 2010 format).
    polygons_by_class: {int class_code: [shapely Polygon, ...]}
    """
    import ezdxf
    out_path = Path(out_path).with_suffix(".dxf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Create layers up front
    for code, (layer_name, color) in LAYER_CONFIG.items():
        try:
            doc.layers.add(layer_name, color=color)
        except Exception:
            pass

    for code, polys in polygons_by_class.items():
        if not polys:
            continue
        layer_name = LAYER_CONFIG.get(code, ("SURF_UNCLASSIFIED", 7))[0]
        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            # Exterior ring
            ext_coords = list(poly.exterior.coords)
            if len(ext_coords) >= 3:
                msp.add_lwpolyline(
                    ext_coords,
                    close=True,
                    dxfattribs={"layer": layer_name},
                )
            # Interior holes — draw as separate polylines on same layer
            for ring in poly.interiors:
                coords = list(ring.coords)
                if len(coords) >= 3:
                    msp.add_lwpolyline(
                        coords,
                        close=True,
                        dxfattribs={"layer": layer_name},
                    )

    doc.saveas(str(out_path))
    return out_path


def write_geojson(
    polygons_by_class: dict,
    out_path: str | Path,
    epsg: int | None = None,
) -> Path:
    """
    Write polygons as GeoJSON FeatureCollection with material attribute.
    Optional EPSG for the CRS block (e.g. 27700 for OSGB36).
    """
    from shapely.geometry import mapping
    out_path = Path(out_path).with_suffix(".geojson")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for code, polys in polygons_by_class.items():
        material_name = MATERIAL_NAMES.get(code, f"CLASS_{code}")
        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            geom = mapping(poly)
            props = {
                "material": material_name,
                "material_code": int(code),
                "area_m2": float(poly.area),
            }
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": geom,
            })

    fc = {"type": "FeatureCollection", "features": features}
    if epsg is not None:
        fc["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)

    return out_path


def write_csv_summary(
    polygons_by_class: dict,
    out_path: str | Path,
) -> Path:
    """
    Write per-class summary + per-polygon detail CSV.
    """
    import csv
    out_path = Path(out_path).with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["material", "poly_id", "area_m2", "x_centroid", "y_centroid"])
        pid = 0
        for code, polys in polygons_by_class.items():
            name = MATERIAL_NAMES.get(code, f"CLASS_{code}")
            for poly in polys:
                if poly is None or poly.is_empty:
                    continue
                pid += 1
                cx, cy = poly.centroid.x, poly.centroid.y
                w.writerow([name, pid, f"{poly.area:.3f}", f"{cx:.3f}", f"{cy:.3f}"])
    return out_path
