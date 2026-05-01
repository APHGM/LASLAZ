import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

from .las_io import read_laz, read_laz_bbox, write_classified_laz, parse_tile_coords
from .ground_classifier import classify_ground, classify_ground_csf
from .bird_detector import detect_bird_contacts, BirdCluster


@dataclass
class ProcessParams:
    tile_size: float = 100.0
    buffer_m: float = 5.0
    # Ground method: "grid" or "csf"
    ground_method: str = "csf"
    # Grid method params
    ground_cell_size: float = 0.5
    ground_threshold: float = 0.50
    smooth_passes: int = 5
    # CSF method params
    csf_cloth_resolution: float = 0.5
    csf_class_threshold: float = 0.10
    csf_rigidness: int = 2
    csf_iterations: int = 500
    csf_slope_smooth: bool = True
    csf_voxel_size: float = 0.15
    # Bird detection
    nz_min: float = 0.02
    nz_max: float = 0.40
    dbscan_eps: float = 0.30
    dbscan_min_pts: int = 8
    max_footprint_m2: float = 1.0
    min_footprint_m2: float = 0.005


def _build_tile_index(tile_dir: Path) -> dict[tuple[int, int], Path]:
    index = {}
    for f in sorted(tile_dir.glob("*.la[sz]")):
        coords = parse_tile_coords(f.name)
        if coords:
            index[coords] = f
    return index


def _load_neighbours(
    tile_index: dict,
    east: int, north: int,
    tile_size: int,
    buffer: float,
    core_xmin, core_xmax, core_ymin, core_ymax
) -> np.ndarray:
    """Load buffer-zone points from the 8 neighbouring tiles."""
    buf_xmin = core_xmin - buffer
    buf_xmax = core_xmax + buffer
    buf_ymin = core_ymin - buffer
    buf_ymax = core_ymax + buffer

    offsets = [
        (-tile_size, 0), (tile_size, 0),
        (0, -tile_size), (0, tile_size),
        (-tile_size, -tile_size), (-tile_size, tile_size),
        (tile_size, -tile_size), (tile_size, tile_size),
    ]

    parts = []
    for de, dn in offsets:
        key = (east + de, north + dn)
        if key in tile_index:
            pts = read_laz_bbox(
                tile_index[key],
                buf_xmin, buf_xmax,
                buf_ymin, buf_ymax
            )
            if len(pts):
                parts.append(pts)

    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3))


def process_all_tiles(
    tile_dir: str | Path,
    out_dir: str | Path,
    params: ProcessParams,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda i, n: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> Path:
    """
    Process every tile in tile_dir. Returns path to the CSV summary file.
    """
    tile_dir = Path(tile_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_index = _build_tile_index(tile_dir)
    tiles = sorted(tile_index.items())
    total = len(tiles)
    log_fn(f"Found {total} tiles to process.")

    all_clusters: list[dict] = []
    csv_path = out_dir / "bird_contacts.csv"

    for i, ((east, north), tile_path) in enumerate(tiles):
        if cancelled_fn():
            log_fn("Cancelled by user.")
            break

        log_fn(f"[{i+1}/{total}] Processing {tile_path.name} ...")

        try:
            xyz_core, las_src = read_laz(tile_path)
        except Exception as e:
            log_fn(f"  ERROR reading {tile_path.name}: {e}")
            progress_fn(i + 1, total)
            continue

        if len(xyz_core) == 0:
            log_fn(f"  Skipping empty tile.")
            progress_fn(i + 1, total)
            continue

        core_xmin = xyz_core[:, 0].min()
        core_xmax = xyz_core[:, 0].max()
        core_ymin = xyz_core[:, 1].min()
        core_ymax = xyz_core[:, 1].max()

        # Load neighbour buffer
        tile_size_int = int(params.tile_size)
        xyz_buf = _load_neighbours(
            tile_index, east, north, tile_size_int,
            params.buffer_m,
            core_xmin, core_xmax, core_ymin, core_ymax
        )

        # Combine for ground classification
        if len(xyz_buf):
            xyz_all = np.concatenate([xyz_core, xyz_buf], axis=0)
        else:
            xyz_all = xyz_core

        # Ground classification on full (core + buffer)
        try:
            if params.ground_method == "csf":
                ground_mask_all, nz_all = classify_ground_csf(
                    xyz_all,
                    cloth_resolution=params.csf_cloth_resolution,
                    class_threshold=params.csf_class_threshold,
                    rigidness=params.csf_rigidness,
                    iterations=params.csf_iterations,
                    slope_smooth=params.csf_slope_smooth,
                    csf_voxel_size=params.csf_voxel_size,
                )
            else:
                ground_mask_all, nz_all = classify_ground(
                    xyz_all,
                    cell_size=params.ground_cell_size,
                    ground_threshold=params.ground_threshold,
                    smooth_passes=params.smooth_passes,
                )
        except Exception as e:
            log_fn(f"  ERROR in ground classification: {e}")
            progress_fn(i + 1, total)
            continue

        # Slice back to core points only
        n_core = len(xyz_core)
        ground_mask = ground_mask_all[:n_core]
        nz = nz_all[:n_core]

        # Diagnostic: log ground surface range to help tune parameters
        nz_finite = nz[np.isfinite(nz)]
        log_fn(f"  nZ range: {nz_finite.min():.2f} to {nz_finite.max():.2f} m  "
               f"| Z range: {xyz_core[:,2].min():.2f} to {xyz_core[:,2].max():.2f} m")

        # Bird detection on core points
        try:
            classification, clusters = detect_bird_contacts(
                xyz_core, nz, ground_mask,
                nz_min=params.nz_min,
                nz_max=params.nz_max,
                dbscan_eps=params.dbscan_eps,
                dbscan_min_pts=params.dbscan_min_pts,
                max_footprint_m2=params.max_footprint_m2,
                min_footprint_m2=params.min_footprint_m2,
                log_fn=log_fn,
            )
        except Exception as e:
            log_fn(f"  ERROR in bird detection: {e}")
            progress_fn(i + 1, total)
            continue

        # Write classified tile
        out_path = out_dir / f"{tile_path.stem}_classified.las"
        try:
            write_classified_laz(las_src, xyz_core, classification, out_path)
        except Exception as e:
            log_fn(f"  ERROR writing output: {e}")

        n_ground = int(ground_mask.sum())
        n_bird = int((classification == 20).sum())
        log_fn(f"  Ground={n_ground:,}  Bird-contact pts={n_bird}  Clusters={len(clusters)}")

        for c in clusters:
            row = asdict(c)
            row["tile"] = tile_path.name
            all_clusters.append(row)

        progress_fn(i + 1, total)

    # Write CSV summary
    if all_clusters:
        fieldnames = list(all_clusters[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_clusters)
        log_fn(f"CSV summary written: {csv_path}")
    else:
        log_fn("No bird contacts found across all tiles.")

    return csv_path
