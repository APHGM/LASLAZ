"""
End-to-end surface extraction pipeline.

Input : a classified LAS/LAZ (must have ground class 2)
Output: DXF + GeoJSON + CSV summary of surface-material polygons.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import laspy

from .material_classifier import (
    MaterialParams, classify_materials, summarise,
    MAT_GRASS, MAT_ASPHALT, MAT_CONCRETE, MAT_PAINTED, MAT_OTHER,
)
from .car_remover import detect_cars_by_geometry, expand_car_mask_downward
from .rasterize_vectorize import rasterise_points, extract_polygons
from .exporters import write_dxf, write_geojson, write_csv_summary


ASPRS_GROUND = 2


@dataclass
class PipelineParams:
    # Grid
    cell_size_m: float = 0.25
    # Polygon post-processing
    min_polygon_area_m2: float = 1.0
    simplify_tolerance_m: float = 0.20
    morph_open_iter: int = 1
    morph_close_iter: int = 2
    # Cars
    remove_cars: bool = True
    car_z_min: float = 0.30
    car_z_max: float = 2.20
    car_dbscan_eps: float = 0.40
    car_dbscan_min_pts: int = 30
    car_footprint_min_m2: float = 2.0
    car_footprint_max_m2: float = 25.0
    # Materials
    material_params: MaterialParams = field(default_factory=MaterialParams)
    # Which classes to export
    export_grass: bool = True
    export_asphalt: bool = True
    export_concrete: bool = True
    export_painted: bool = True
    export_other: bool = False
    # CRS
    epsg: int | None = 27700   # OSGB36 default for user's UK data


def _selected_class_codes(params: PipelineParams) -> list[int]:
    codes = []
    if params.export_grass:    codes.append(MAT_GRASS)
    if params.export_asphalt:  codes.append(MAT_ASPHALT)
    if params.export_concrete: codes.append(MAT_CONCRETE)
    if params.export_painted:  codes.append(MAT_PAINTED)
    if params.export_other:    codes.append(MAT_OTHER)
    return codes


def _read_las_and_ground(
    file_path: Path,
    log_fn: Callable[[str], None],
):
    """Read LAZ, return (xyz, class, R, G, B, NIR-or-None)."""
    log_fn(f"Reading {file_path.name} ...")
    las = laspy.read(str(file_path))
    xyz = np.stack([las.x.copy(), las.y.copy(), las.z.copy()], axis=1)
    cls = np.asarray(las.classification, dtype=np.uint8)

    R = np.asarray(getattr(las, "red",   np.array([])), dtype=np.uint16)
    G = np.asarray(getattr(las, "green", np.array([])), dtype=np.uint16)
    B = np.asarray(getattr(las, "blue",  np.array([])), dtype=np.uint16)
    try:
        NIR = np.asarray(las.nir, dtype=np.uint16)
        if len(NIR) != len(xyz):
            NIR = None
    except (AttributeError, ValueError):
        NIR = None

    if len(R) != len(xyz) or len(G) != len(xyz) or len(B) != len(xyz):
        raise RuntimeError(
            f"{file_path.name} has no RGB channels — surface extraction needs "
            f"a coloured photogrammetry point cloud (LAS point format 2/3/5/7/8/10)."
        )

    log_fn(f"  {len(xyz):,} points, ground (class 2) = "
           f"{int((cls == ASPRS_GROUND).sum()):,}, "
           f"RGB=yes, NIR={'yes' if NIR is not None else 'no'}")
    return xyz, cls, R, G, B, NIR


def _compute_nz_from_ground(xyz: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    """
    Interpolate ground surface from class-2 points, return nZ per point.
    Uses scipy LinearNDInterpolator with a Nearest fallback outside the hull.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    ground_pts = xyz[ground_mask]
    if len(ground_pts) < 3:
        return xyz[:, 2] - xyz[:, 2].min()
    lin = LinearNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])
    ground_z = lin(xyz[:, :2])
    outside = np.isnan(ground_z)
    if outside.any():
        near = NearestNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])
        ground_z[outside] = near(xyz[outside, :2])
    return xyz[:, 2] - ground_z


def extract_surface_polygons(
    file_path: str | Path,
    out_dir: str | Path,
    params: PipelineParams | None = None,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda a, b: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> dict:
    """
    Run the whole pipeline on one classified LAZ file.
    Returns a summary dict.
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if params is None:
        params = PipelineParams()

    xyz, cls, R, G, B, NIR = _read_las_and_ground(file_path, log_fn)
    if cancelled_fn():
        return {"status": "cancelled"}
    progress_fn(1, 5)

    ground_mask = (cls == ASPRS_GROUND)
    n_ground = int(ground_mask.sum())
    if n_ground < 100:
        raise RuntimeError(
            f"{file_path.name} has too few ground points ({n_ground}). "
            f"Run Starling Classifier first to produce class 2 ground."
        )

    # ── Car removal (optional) ────────────────────────────────────────────
    if params.remove_cars:
        log_fn("Detecting parked cars ...")
        nz = _compute_nz_from_ground(xyz, ground_mask)
        car_mask = detect_cars_by_geometry(
            xyz, nz,
            car_z_min=params.car_z_min,
            car_z_max=params.car_z_max,
            dbscan_eps=params.car_dbscan_eps,
            dbscan_min_pts=params.car_dbscan_min_pts,
            footprint_min_m2=params.car_footprint_min_m2,
            footprint_max_m2=params.car_footprint_max_m2,
        )
        n_car_cluster = int(car_mask.sum())
        car_mask = expand_car_mask_downward(xyz, car_mask, xy_radius=0.30)
        n_car_total = int(car_mask.sum())
        log_fn(f"  Cars: {n_car_cluster:,} cluster pts + shadow → "
               f"{n_car_total:,} pts excluded")
    else:
        car_mask = np.zeros(len(xyz), dtype=bool)
        n_car_total = 0
    progress_fn(2, 5)
    if cancelled_fn():
        return {"status": "cancelled"}

    # ── Classify surface materials on GROUND points only ─────────────────
    active_mask = ground_mask & ~car_mask
    if not active_mask.any():
        raise RuntimeError("No surface-eligible points after car removal")

    log_fn(f"Classifying materials on {int(active_mask.sum()):,} ground points ...")
    materials_active = classify_materials(
        R[active_mask], G[active_mask], B[active_mask],
        NIR=NIR[active_mask] if NIR is not None else None,
        params=params.material_params,
    )
    stats = summarise(materials_active)
    log_fn(f"  Material breakdown: " + "  ".join(
        f"{name}={cnt:,}" for name, cnt in stats.items() if cnt > 0
    ))
    progress_fn(3, 5)
    if cancelled_fn():
        return {"status": "cancelled"}

    # ── Rasterise + vectorise ────────────────────────────────────────────
    log_fn(f"Rasterising at {params.cell_size_m} m and extracting polygons ...")
    class_raster, x0, y0 = rasterise_points(
        xyz[active_mask, :2], materials_active,
        cell_size=params.cell_size_m,
        n_classes=6,
    )
    log_fn(f"  Raster: {class_raster.shape[0]} rows × {class_raster.shape[1]} cols")

    polys_by_class = extract_polygons(
        class_raster, x0, y0, params.cell_size_m,
        class_codes=_selected_class_codes(params),
        min_area_m2=params.min_polygon_area_m2,
        simplify_tol_m=params.simplify_tolerance_m,
        open_iter=params.morph_open_iter,
        close_iter=params.morph_close_iter,
    )
    n_polys_total = sum(len(v) for v in polys_by_class.values())
    log_fn(f"  Extracted {n_polys_total} polygons total")
    progress_fn(4, 5)
    if cancelled_fn():
        return {"status": "cancelled"}

    # ── Write outputs ────────────────────────────────────────────────────
    stem = file_path.stem
    dxf_path = write_dxf(polys_by_class, out_dir / f"{stem}_surfaces.dxf")
    log_fn(f"  DXF written: {dxf_path}")
    gj_path = write_geojson(polys_by_class, out_dir / f"{stem}_surfaces.geojson",
                            epsg=params.epsg)
    log_fn(f"  GeoJSON written: {gj_path}")
    csv_path = write_csv_summary(polys_by_class, out_dir / f"{stem}_surfaces.csv")
    log_fn(f"  CSV summary written: {csv_path}")
    progress_fn(5, 5)

    return {
        "status": "processed",
        "file": file_path.name,
        "n_points": int(len(xyz)),
        "n_ground": n_ground,
        "n_cars_excluded": n_car_total,
        "material_stats": stats,
        "n_polygons": n_polys_total,
        "dxf": str(dxf_path),
        "geojson": str(gj_path),
        "csv": str(csv_path),
    }


def extract_folder(
    tile_dir: str | Path,
    out_dir: str | Path,
    params: PipelineParams | None = None,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda a, b: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> list[dict]:
    """
    Batch-process every LAS/LAZ in a folder.
    Returns list of per-file result dicts.
    """
    tile_dir = Path(tile_dir)
    out_dir = Path(out_dir)
    files = sorted(list(tile_dir.glob("*.laz")) + list(tile_dir.glob("*.las")))
    log_fn(f"Found {len(files)} file(s) in {tile_dir}")
    results = []
    for i, f in enumerate(files):
        if cancelled_fn():
            log_fn("Cancelled.")
            break
        log_fn(f"\n[{i+1}/{len(files)}] {f.name}")
        try:
            r = extract_surface_polygons(
                f, out_dir, params,
                log_fn=log_fn,
                progress_fn=lambda a, b: progress_fn(i * 5 + a, len(files) * 5),
                cancelled_fn=cancelled_fn,
            )
            results.append(r)
        except Exception as e:
            log_fn(f"  ERROR: {type(e).__name__}: {e}")
            results.append({"status": "error", "file": f.name, "reason": str(e)})
    return results
