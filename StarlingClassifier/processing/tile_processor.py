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
    # Output format: "las" or "laz"
    output_format: str = "laz"
    # LAS version: "1.2" or "1.4"
    las_version: str = "1.4"
    # Ground method: "grid" or "csf"
    ground_method: str = "csf"
    # Grid method params
    ground_cell_size: float = 0.5
    ground_threshold: float = 0.50
    smooth_passes: int = 5
    # CSF method params
    csf_cloth_resolution: float = 0.30
    csf_class_threshold: float = 0.05
    csf_rigidness: int = 1
    csf_iterations: int = 1000
    csf_slope_smooth: bool = True
    csf_voxel_size: float = 0.10
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


def _write_tile_summary(
    tile_results: list[dict],
    out_dir: Path,
    log_fn: Callable[[str], None],
    tile_dir: Path
):
    """
    Write processing summary report, delete input LAZ files for processed tiles,
    and print a summary table.
    """
    processed = [t for t in tile_results if t["status"] == "PROCESSED"]
    skipped = [t for t in tile_results if t["status"] == "SKIPPED"]
    
    log_fn("\n" + "="*80)
    log_fn("PROCESSING SUMMARY")
    log_fn("="*80)
    
    # Summary counts
    log_fn(f"Total tiles: {len(tile_results)}")
    log_fn(f"  ✓ PROCESSED: {len(processed)}")
    log_fn(f"  ✗ SKIPPED:   {len(skipped)}")
    
    # Table of results
    log_fn("\n" + "-"*120)
    log_fn(f"{'Tile Name':<25} {'Status':<12} {'Ground Pts':<15} {'Bird Pts':<12} {'Clusters':<10} {'Notes':<50}")
    log_fn("-"*120)
    
    for tile_info in tile_results:
        notes = tile_info["reason"] if tile_info["reason"] else "OK"
        log_fn(
            f"{tile_info['tile']:<25} "
            f"{tile_info['status']:<12} "
            f"{tile_info['n_ground']:<15,} "
            f"{tile_info['n_bird']:<12} "
            f"{tile_info['n_clusters']:<10} "
            f"{notes:<50}"
        )
    
    log_fn("-"*120)
    
    # Delete input LAZ files for processed tiles
    deleted_count = 0
    for tile_info in processed:
        tile_file = tile_dir / tile_info["tile"]
        if tile_file.exists():
            try:
                tile_file.unlink()
                deleted_count += 1
            except Exception as e:
                log_fn(f"  WARNING: Could not delete {tile_file.name}: {e}")
    
    log_fn(f"\nDeleted {deleted_count} input LAZ files for processed tiles.")
    log_fn(f"Kept {len(skipped)} input LAZ files for skipped tiles (requires review).")
    
    # Write summary to file
    summary_file = out_dir / "processing_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("PROCESSING SUMMARY\n")
        f.write("="*120 + "\n")
        f.write(f"Total tiles: {len(tile_results)}\n")
        f.write(f"  ✓ PROCESSED: {len(processed)}\n")
        f.write(f"  ✗ SKIPPED:   {len(skipped)}\n\n")
        
        f.write("-"*120 + "\n")
        f.write(f"{'Tile Name':<25} {'Status':<12} {'Ground Pts':<15} {'Bird Pts':<12} {'Clusters':<10} {'Notes':<50}\n")
        f.write("-"*120 + "\n")
        
        for tile_info in tile_results:
            notes = tile_info["reason"] if tile_info["reason"] else "OK"
            f.write(
                f"{tile_info['tile']:<25} "
                f"{tile_info['status']:<12} "
                f"{tile_info['n_ground']:<15,} "
                f"{tile_info['n_bird']:<12} "
                f"{tile_info['n_clusters']:<10} "
                f"{notes:<50}\n"
            )
        
        f.write("-"*120 + "\n")
        f.write(f"\nDeleted {deleted_count} input LAZ files for processed tiles.\n")
        f.write(f"Kept {len(skipped)} input LAZ files for skipped tiles (requires review).\n")
    
    log_fn(f"Summary written to: {summary_file}")
    log_fn("="*80)


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
    
    # Track processing status: (tile_name, status, reason, n_ground, n_bird, n_clusters)
    tile_results: list[dict] = []

    for i, ((east, north), tile_path) in enumerate(tiles):
        if cancelled_fn():
            log_fn("Cancelled by user.")
            break

        log_fn(f"[{i+1}/{total}] Processing {tile_path.name} ...")
        
        result = {
            "tile": tile_path.name,
            "status": "UNKNOWN",
            "reason": "",
            "n_ground": 0,
            "n_bird": 0,
            "n_clusters": 0
        }

        try:
            xyz_core, las_src = read_laz(tile_path)
        except Exception as e:
            log_fn(f"  ERROR reading {tile_path.name}: {e}")
            result["status"] = "SKIPPED"
            result["reason"] = f"Read error: {e}"
            tile_results.append(result)
            progress_fn(i + 1, total)
            continue

        if len(xyz_core) == 0:
            log_fn(f"  Skipping empty tile.")
            result["status"] = "SKIPPED"
            result["reason"] = "Empty tile"
            tile_results.append(result)
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
            result["status"] = "SKIPPED"
            result["reason"] = f"Ground classification error: {e}"
            tile_results.append(result)
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
            result["status"] = "SKIPPED"
            result["reason"] = f"Bird detection error: {e}"
            tile_results.append(result)
            progress_fn(i + 1, total)
            continue

        n_ground = int(ground_mask.sum())
        n_bird = int((classification == 20).sum())
        n_clusters = len(clusters)
        
        # Check if bird detection was skipped due to high candidate fraction
        if n_clusters == 0 and n_bird == 0:
            # Could be either no birds found OR detection skipped
            # We infer from log context but mark as processed (detection was run)
            result["status"] = "SKIPPED"
            result["reason"] = "High candidate fraction or no birds detected"
        else:
            result["status"] = "PROCESSED"
            result["reason"] = ""
        
        result["n_ground"] = n_ground
        result["n_bird"] = n_bird
        result["n_clusters"] = n_clusters
        
        log_fn(f"  Ground={n_ground:,}  Bird-contact pts={n_bird}  Clusters={n_clusters}")

        # Always write — ground classification alone is valuable output, 
        # and we want to capture zero-bird cases in the CSV summary. 
        # The classification array will have 20 for bird contacts, 2 for ground, and 1 for unclassified points.
        # Write classified tile
        out_path = out_dir / f"{tile_path.stem}_classified.las"
        try:
            written = write_classified_laz(
                las_src, xyz_core, classification, out_path,
                output_format=params.output_format,
                las_version=params.las_version
            )
            log_fn(f"  WROTE: {written}  (ground={int(ground_mask.sum()):,}, birds={n_bird})")
            # Keep existing status so SKIPPED tiles still report the reason in CSV
            if result.get("status") != "SKIPPED":
                result["status"] = "PROCESSED"
        except Exception as e:
            log_fn(f"  ERROR writing output: {e}")
            result["status"] = "WRITE_FAILED"
            result["reason"] = f"Write error: {e}"
            tile_results.append(result)
            progress_fn(i + 1, total)
            continue

        for c in clusters:
            row = asdict(c)
            row["tile"] = tile_path.name
            all_clusters.append(row)

        tile_results.append(result)
        progress_fn(i + 1, total)

    # Write CSV summary
    if all_clusters:
        fieldnames = list(all_clusters[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_clusters)
        log_fn(f"CSV summary written: {csv_path}")
    else:
        log_fn("No bird contacts found across all tiles.")
    
    # Create processing summary report
    _write_tile_summary(tile_results, out_dir, log_fn, tile_dir)
    
    return csv_path
