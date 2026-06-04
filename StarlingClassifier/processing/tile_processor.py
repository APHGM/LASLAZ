import numpy as np
import csv
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from .las_io import read_laz, read_laz_bbox, write_classified_laz, parse_tile_coords
from .ground_classifier import classify_ground, classify_ground_csf
from .bird_detector import detect_bird_contacts, BirdCluster
from .height_classifier import (
    classify_heights, classify_model_keypoints, ground_mask_from_classification,
)
from .ground_classifier import _compute_nz   # for nz from existing ground


# Tiles larger than this run on their own (no parallel siblings) to protect RAM
SOLO_TILE_POINT_THRESHOLD = 150_000_000


@dataclass
class ProcessParams:
    tile_size: float = 100.0
    buffer_m: float = 5.0
    # Output format: "las" or "laz"
    output_format: str = "laz"
    # LAS version: "1.2" or "1.4"
    las_version: str = "1.4"
    # Ground source: "compute" (run CSF/Grid) or "from_file" (use existing class 2)
    ground_source: str = "compute"
    # Ground method (only used when ground_source == "compute"): "grid" or "csf"
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
    # Height-based classification (TerraScan-style)
    classify_vegetation: bool = True
    veg_low_min: float = 0.10            # nZ start of low veg
    veg_low_max: float = 1.00            # boundary low/med
    veg_med_max: float = 3.00            # boundary med/high
    noise_below_ground: float = -0.10    # nZ below this = class 7 noise
    # Model key points (thinned ground for TIN building)
    classify_model_keys: bool = False
    modelkey_step_m: float = 8.0
    # Parallelism: 1 = sequential, N = ProcessPoolExecutor with N workers
    num_workers: int = 1


# ────────────────────────────────────────────────────────────────────────────
# Tile indexing & neighbour helpers
# ────────────────────────────────────────────────────────────────────────────

def _build_tile_index(tile_dir: Path) -> tuple[dict[tuple[int, int], Path], dict]:
    """
    Index all *.laz / *.las files in tile_dir.

    Files named `<easting>_<northing>.laz` get parsed grid coordinates
    (enables neighbour buffer loading at edges).

    Files with any other naming get synthetic negative keys — they're still
    processed, but no neighbour buffer is loaded for them (no neighbours
    can match a negative key).

    Returns (index, stats_dict). stats has keys 'grid' and 'misc'.
    """
    index: dict[tuple[int, int], Path] = {}
    synthetic_id = 0
    grid_count = 0
    misc_count = 0
    for f in sorted(tile_dir.glob("*.la[sz]")):
        coords = parse_tile_coords(f.name)
        if coords:
            index[coords] = f
            grid_count += 1
        else:
            index[(-1_000_000_000 - synthetic_id,
                   -1_000_000_000 - synthetic_id)] = f
            synthetic_id += 1
            misc_count += 1
    return index, {"grid": grid_count, "misc": misc_count}


def _neighbour_paths(
    tile_index: dict, east: int, north: int, tile_size: int
) -> list[Path]:
    offsets = [
        (-tile_size, 0), (tile_size, 0),
        (0, -tile_size), (0, tile_size),
        (-tile_size, -tile_size), (-tile_size, tile_size),
        (tile_size, -tile_size), (tile_size, tile_size),
    ]
    return [tile_index[(east + de, north + dn)]
            for de, dn in offsets if (east + de, north + dn) in tile_index]


def _load_neighbour_buffer(
    neighbour_paths: list[Path],
    buffer: float,
    core_xmin, core_xmax, core_ymin, core_ymax,
) -> np.ndarray:
    parts = []
    for np_path in neighbour_paths:
        pts = read_laz_bbox(
            np_path,
            core_xmin - buffer, core_xmax + buffer,
            core_ymin - buffer, core_ymax + buffer,
        )
        if len(pts):
            parts.append(pts)
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3))


# ────────────────────────────────────────────────────────────────────────────
# Single-tile worker function (pickle-safe — module-level, no closures)
# ────────────────────────────────────────────────────────────────────────────

def _process_one_tile(job: dict) -> dict:
    """
    Run ground classification + bird detection for ONE tile.
    Returns a result dict (plus log lines captured during work).
    Designed to be called either inline or in a worker subprocess.
    """
    tile_path = Path(job["tile_path"])
    neighbour_paths = [Path(p) for p in job["neighbour_paths"]]
    params: ProcessParams = job["params"]
    out_dir = Path(job["out_dir"])

    log_lines: list[str] = []
    def log(msg: str):
        log_lines.append(msg)

    result = {
        "tile": tile_path.name,
        "status": "UNKNOWN",
        "reason": "",
        "n_ground": 0,
        "n_bird": 0,
        "n_clusters": 0,
        "clusters": [],
        "log_lines": log_lines,
        "written_path": None,
    }

    log(f"Processing {tile_path.name} ...")

    # ── Read core tile ────────────────────────────────────────────────────
    try:
        xyz_core, las_src = read_laz(tile_path)
    except Exception as e:
        log(f"  ERROR reading {tile_path.name}: {e}")
        result["status"] = "SKIPPED"
        result["reason"] = f"Read error: {e}"
        return result

    if len(xyz_core) == 0:
        log("  Skipping empty tile.")
        result["status"] = "SKIPPED"
        result["reason"] = "Empty tile"
        return result

    core_xmin = xyz_core[:, 0].min()
    core_xmax = xyz_core[:, 0].max()
    core_ymin = xyz_core[:, 1].min()
    core_ymax = xyz_core[:, 1].max()

    # ── Load neighbour buffer ─────────────────────────────────────────────
    xyz_buf = _load_neighbour_buffer(
        neighbour_paths, params.buffer_m,
        core_xmin, core_xmax, core_ymin, core_ymax,
    )
    xyz_all = np.concatenate([xyz_core, xyz_buf], axis=0) if len(xyz_buf) else xyz_core

    # ── Ground classification ─────────────────────────────────────────────
    try:
        if params.ground_source == "from_file":
            # Use existing classification (class 2 + optionally class 8) from source
            try:
                src_cls_core = np.asarray(las_src.classification, dtype=np.uint8)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot read classification field from source: {e}. "
                    f"Switch Ground source to 'Compute fresh'."
                )
            gm_core = ground_mask_from_classification(src_cls_core)
            n_ground_src = int(gm_core.sum())
            log(f"  Ground source: existing classification — {n_ground_src:,} ground pts")
            if n_ground_src < 50:
                raise RuntimeError(
                    "Source file has <50 ground points — no usable surface. "
                    "Switch to 'Compute fresh'."
                )
            # nZ derived from the existing ground surface
            ground_pts = xyz_core[gm_core]
            nz_core = _compute_nz(xyz_core, ground_pts)
            ground_mask_all = gm_core
            nz_all = nz_core      # buffer not used in from_file mode
            # Skip the buffer-merge slicing — we used core only
            n_core = len(xyz_core)
            ground_mask = ground_mask_all
            nz = nz_all
        else:
            # Compute fresh — original CSF / Grid path
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
            n_core = len(xyz_core)
            ground_mask = ground_mask_all[:n_core]
            nz = nz_all[:n_core]
    except Exception as e:
        log(f"  ERROR in ground classification: {e}")
        result["status"] = "SKIPPED"
        result["reason"] = f"Ground classification error: {e}"
        return result

    nz_finite = nz[np.isfinite(nz)]
    log(f"  nZ range: {nz_finite.min():.2f} to {nz_finite.max():.2f} m  "
        f"| Z range: {xyz_core[:,2].min():.2f} to {xyz_core[:,2].max():.2f} m")

    # ── Bird detection ────────────────────────────────────────────────────
    try:
        classification, clusters = detect_bird_contacts(
            xyz_core, nz, ground_mask,
            nz_min=params.nz_min,
            nz_max=params.nz_max,
            dbscan_eps=params.dbscan_eps,
            dbscan_min_pts=params.dbscan_min_pts,
            max_footprint_m2=params.max_footprint_m2,
            min_footprint_m2=params.min_footprint_m2,
            log_fn=log,
        )
    except Exception as e:
        log(f"  ERROR in bird detection: {e}")
        result["status"] = "SKIPPED"
        result["reason"] = f"Bird detection error: {e}"
        return result

    n_ground = int(ground_mask.sum())
    n_bird = int((classification == 20).sum())
    n_clusters = len(clusters)

    # ── Height-based classification (TerraScan-style) ─────────────────────
    if params.classify_vegetation:
        try:
            classification = classify_heights(
                classification, nz,
                veg_low_min=params.veg_low_min,
                veg_low_max=params.veg_low_max,
                veg_med_max=params.veg_med_max,
                noise_below_ground=params.noise_below_ground,
            )
            n_low_veg  = int((classification == 3).sum())
            n_med_veg  = int((classification == 4).sum())
            n_high_veg = int((classification == 5).sum())
            n_noise    = int((classification == 7).sum())
            log(f"  Veg/Noise: low={n_low_veg:,}  med={n_med_veg:,}  "
                f"high={n_high_veg:,}  noise={n_noise:,}")
        except Exception as e:
            log(f"  WARNING: vegetation classification failed: {e}")

    # ── Model keypoints (thinned ground for TIN) ─────────────────────────
    n_modelkey = 0
    if params.classify_model_keys:
        try:
            classification = classify_model_keypoints(
                xyz_core, classification,
                step_m=params.modelkey_step_m,
            )
            n_modelkey = int((classification == 8).sum())
            log(f"  Model keypoints (class 8): {n_modelkey:,} at {params.modelkey_step_m}m step")
        except Exception as e:
            log(f"  WARNING: model keypoint classification failed: {e}")

    if n_clusters == 0 and n_bird == 0:
        result["status"] = "SKIPPED"
        result["reason"] = "High candidate fraction or no birds detected"
    else:
        result["status"] = "PROCESSED"

    result["n_ground"] = n_ground
    result["n_bird"] = n_bird
    result["n_clusters"] = n_clusters
    log(f"  Ground={n_ground:,}  Bird-contact pts={n_bird}  Clusters={n_clusters}")

    # Serialise clusters for return (dataclass → dict)
    for c in clusters:
        row = asdict(c)
        row["tile"] = tile_path.name
        result["clusters"].append(row)

    # ── Write classified tile (always) ────────────────────────────────────
    out_path = out_dir / f"{tile_path.stem}_classified.las"
    try:
        written = write_classified_laz(
            las_src, xyz_core, classification, out_path,
            output_format=params.output_format,
            las_version=params.las_version,
        )
        log(f"  WROTE: {written}  (ground={n_ground:,}, birds={n_bird})")
        result["written_path"] = str(written)
        if result["status"] != "SKIPPED":
            result["status"] = "PROCESSED"
    except Exception as e:
        log(f"  ERROR writing output: {e}")
        result["status"] = "WRITE_FAILED"
        result["reason"] = f"Write error: {e}"

    return result


# ────────────────────────────────────────────────────────────────────────────
# Summary writer (unchanged behaviour)
# ────────────────────────────────────────────────────────────────────────────

def _write_tile_summary(
    tile_results: list[dict],
    out_dir: Path,
    log_fn: Callable[[str], None],
    tile_dir: Path,
):
    processed = [t for t in tile_results if t["status"] == "PROCESSED"]
    skipped = [t for t in tile_results if t["status"] != "PROCESSED"]

    log_fn("\n" + "=" * 80)
    log_fn("PROCESSING SUMMARY")
    log_fn("=" * 80)
    log_fn(f"Total tiles: {len(tile_results)}")
    log_fn(f"  ✓ PROCESSED: {len(processed)}")
    log_fn(f"  ✗ SKIPPED:   {len(skipped)}")
    log_fn("\n" + "-" * 120)
    log_fn(f"{'Tile Name':<25} {'Status':<12} {'Ground Pts':<15} "
           f"{'Bird Pts':<12} {'Clusters':<10} {'Notes':<50}")
    log_fn("-" * 120)
    for t in tile_results:
        notes = t["reason"] if t["reason"] else "OK"
        log_fn(
            f"{t['tile']:<25} {t['status']:<12} "
            f"{t['n_ground']:<15,} {t['n_bird']:<12} "
            f"{t['n_clusters']:<10} {notes:<50}"
        )
    log_fn("-" * 120)

    deleted = 0
    for t in processed:
        tf = tile_dir / t["tile"]
        if tf.exists():
            try:
                tf.unlink()
                deleted += 1
            except Exception as e:
                log_fn(f"  WARNING: Could not delete {tf.name}: {e}")
    log_fn(f"\nDeleted {deleted} input LAZ files for processed tiles.")
    log_fn(f"Kept {len(skipped)} input LAZ files for skipped tiles (requires review).")

    summary_file = out_dir / "processing_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("PROCESSING SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(f"Total tiles: {len(tile_results)}\n")
        f.write(f"  ✓ PROCESSED: {len(processed)}\n")
        f.write(f"  ✗ SKIPPED:   {len(skipped)}\n\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Tile Name':<25} {'Status':<12} {'Ground Pts':<15} "
                f"{'Bird Pts':<12} {'Clusters':<10} {'Notes':<50}\n")
        f.write("-" * 120 + "\n")
        for t in tile_results:
            notes = t["reason"] if t["reason"] else "OK"
            f.write(
                f"{t['tile']:<25} {t['status']:<12} "
                f"{t['n_ground']:<15,} {t['n_bird']:<12} "
                f"{t['n_clusters']:<10} {notes:<50}\n"
            )
        f.write("-" * 120 + "\n")
        f.write(f"\nDeleted {deleted} input LAZ files for processed tiles.\n")
        f.write(f"Kept {len(skipped)} input LAZ files for skipped tiles (requires review).\n")
    log_fn(f"Summary written to: {summary_file}")
    log_fn("=" * 80)


# ────────────────────────────────────────────────────────────────────────────
# Single-file path — skip tiling entirely
# ────────────────────────────────────────────────────────────────────────────

def process_single_file(
    file_path: str | Path,
    out_dir: str | Path,
    params: ProcessParams,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda i, n: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> Path:
    """
    Process ONE LAZ/LAS file as a single unit — no tiling, no buffer loading.
    Use when:
      * File is small (< ~300 M points, fits in RAM)
      * File is a single scan with no grid neighbours
      * You don't want the tiling overhead

    Returns path to the CSV summary (may be empty if no birds found).
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    log_fn(f"Single-file mode (no tiling): {file_path.name}")
    try:
        import laspy
        with laspy.open(str(file_path)) as r:
            log_fn(f"  Point count: {r.header.point_count:,}")
    except Exception:
        pass

    job = {
        "tile_path": str(file_path),
        "neighbour_paths": [],          # no neighbours
        "params": params,
        "out_dir": str(out_dir),
    }

    if cancelled_fn():
        log_fn("Cancelled before start.")
        return out_dir / "bird_contacts.csv"

    result = _process_one_tile(job)

    # Stream the buffered logs
    for line in result["log_lines"]:
        log_fn(line)

    progress_fn(1, 1)

    # ── Write CSV summary ────────────────────────────────────────────────
    csv_path = out_dir / "bird_contacts.csv"
    if result["clusters"]:
        fieldnames = list(result["clusters"][0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["clusters"])
        log_fn(f"CSV summary written: {csv_path}")
    else:
        log_fn("No bird contacts found.")

    # ── Mini summary (no tile-deletion step in single-file mode) ─────────
    log_fn("\n" + "=" * 60)
    log_fn(f"Status:    {result['status']}")
    log_fn(f"Ground:    {result['n_ground']:,}")
    log_fn(f"Birds:     {result['n_bird']:,}  ({result['n_clusters']} clusters)")
    if result["reason"]:
        log_fn(f"Note:      {result['reason']}")
    if result["written_path"]:
        log_fn(f"Output:    {result['written_path']}")
    log_fn("=" * 60)

    return csv_path


# ────────────────────────────────────────────────────────────────────────────
# Public entry point — dispatches sequential or parallel
# ────────────────────────────────────────────────────────────────────────────

def process_all_tiles(
    tile_dir: str | Path,
    out_dir: str | Path,
    params: ProcessParams,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda i, n: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> Path:
    tile_dir = Path(tile_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_index, idx_stats = _build_tile_index(tile_dir)
    tiles = sorted(tile_index.items())
    total = len(tiles)
    log_fn(f"Found {total} tiles to process "
           f"({idx_stats['grid']} grid-named, {idx_stats['misc']} other).")
    if idx_stats["grid"] == 0 and idx_stats["misc"] > 0:
        log_fn("  NOTE: tiles don't follow <easting>_<northing>.laz convention — "
               "neighbour-buffer loading is disabled for these tiles "
               "(each tile processed standalone, no edge smoothing).")

    # Build per-tile jobs (so workers don't need the tile_index)
    tile_size_int = int(params.tile_size)
    jobs = []
    for (east, north), tile_path in tiles:
        jobs.append({
            "tile_path": str(tile_path),
            "neighbour_paths": [str(p) for p in
                                _neighbour_paths(tile_index, east, north, tile_size_int)],
            "params": params,
            "out_dir": str(out_dir),
            "_size": tile_path.stat().st_size,
        })

    # Decide parallelism per tile (giant tiles run solo)
    sequential = params.num_workers <= 1
    if not sequential:
        # Heuristic: 1 LAZ byte ≈ 6-10 points → use file size as proxy
        # 150M point threshold ≈ ~25MB LAZ at typical compression
        # Better to gate on actual point count: read header
        log_fn(f"Parallel mode: {params.num_workers} workers "
               f"(tiles ≥{SOLO_TILE_POINT_THRESHOLD:,} pts will run solo)")

    all_clusters: list[dict] = []
    tile_results: list[dict] = []
    completed = 0

    if sequential:
        for job in jobs:
            if cancelled_fn():
                log_fn("Cancelled by user.")
                break
            result = _process_one_tile(job)
            for line in result["log_lines"]:
                log_fn(line)
            tile_results.append({
                k: v for k, v in result.items()
                if k not in ("log_lines", "clusters", "written_path")
            })
            all_clusters.extend(result["clusters"])
            completed += 1
            progress_fn(completed, total)
    else:
        # Separate giants (run sequentially) from normals (parallel)
        giants, normals = [], []
        for job in jobs:
            # Use laspy header to get exact point count (fast — only reads header)
            try:
                import laspy
                with laspy.open(job["tile_path"]) as r:
                    pt_count = r.header.point_count
            except Exception:
                pt_count = 0
            if pt_count >= SOLO_TILE_POINT_THRESHOLD:
                giants.append(job)
            else:
                normals.append(job)

        if giants:
            log_fn(f"  {len(giants)} large tile(s) will run sequentially after parallel batch")

        # Parallel batch first
        with ProcessPoolExecutor(max_workers=params.num_workers) as pool:
            futures = {pool.submit(_process_one_tile, job): job for job in normals}
            for fut in as_completed(futures):
                if cancelled_fn():
                    log_fn("Cancelling — waiting for in-flight workers…")
                    for f in futures:
                        f.cancel()
                    break
                try:
                    result = fut.result()
                except Exception as e:
                    job = futures[fut]
                    log_fn(f"WORKER CRASHED on {Path(job['tile_path']).name}: {e}")
                    completed += 1
                    progress_fn(completed, total)
                    continue
                for line in result["log_lines"]:
                    log_fn(line)
                tile_results.append({
                    k: v for k, v in result.items()
                    if k not in ("log_lines", "clusters", "written_path")
                })
                all_clusters.extend(result["clusters"])
                completed += 1
                progress_fn(completed, total)

        # Then giants, in-process (less overhead, full RAM available)
        for job in giants:
            if cancelled_fn():
                break
            log_fn(f"\n[solo] large tile {Path(job['tile_path']).name}")
            result = _process_one_tile(job)
            for line in result["log_lines"]:
                log_fn(line)
            tile_results.append({
                k: v for k, v in result.items()
                if k not in ("log_lines", "clusters", "written_path")
            })
            all_clusters.extend(result["clusters"])
            completed += 1
            progress_fn(completed, total)

    # ── Write CSV summary ─────────────────────────────────────────────────
    csv_path = out_dir / "bird_contacts.csv"
    if all_clusters:
        fieldnames = list(all_clusters[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_clusters)
        log_fn(f"CSV summary written: {csv_path}")
    else:
        log_fn("No bird contacts found across all tiles.")

    _write_tile_summary(tile_results, out_dir, log_fn, tile_dir)
    return csv_path
