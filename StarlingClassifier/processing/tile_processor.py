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
    # Parallelism: 1 = sequential, N = ProcessPoolExecutor with N workers
    num_workers: int = 1


# ────────────────────────────────────────────────────────────────────────────
# Tile indexing & neighbour helpers
# ────────────────────────────────────────────────────────────────────────────

def _build_tile_index(tile_dir: Path) -> dict[tuple[int, int], Path]:
    index = {}
    for f in sorted(tile_dir.glob("*.la[sz]")):
        coords = parse_tile_coords(f.name)
        if coords:
            index[coords] = f
    return index


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
        log(f"  ERROR in ground classification: {e}")
        result["status"] = "SKIPPED"
        result["reason"] = f"Ground classification error: {e}"
        return result

    n_core = len(xyz_core)
    ground_mask = ground_mask_all[:n_core]
    nz = nz_all[:n_core]

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

    tile_index = _build_tile_index(tile_dir)
    tiles = sorted(tile_index.items())
    total = len(tiles)
    log_fn(f"Found {total} tiles to process.")

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
