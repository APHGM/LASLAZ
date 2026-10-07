import numpy as np
import csv
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from .las_io import (
    read_laz, read_laz_bbox, write_classified_laz, parse_tile_coords,
    read_color_bands, has_rgb, has_nir,
)
from .ground_classifier import classify_ground, classify_ground_csf
from .ground_classifier import (
    classify_ground_csf_streaming, nz_from_grid,
    STREAMING_POINT_THRESHOLD,
)
from .bird_detector import detect_bird_contacts, BirdCluster
from .height_classifier import (
    classify_heights, classify_model_keypoints, ground_mask_from_classification,
)
from .ground_classifier import _compute_nz   # for nz from existing ground
from .photogrammetry_classifier import (
    vegetation_mask as _photo_vegetation_mask,
    detect_buildings as _photo_detect_buildings,
    CLASS_BUILDING,
)


def _system_profile() -> dict:
    """
    Derive all RAM/CPU tuning constants from the current machine at import time.
    No hardcoded assumptions about RAM size — scales from 8 GB laptops to
    128 GB+ workstations automatically.

    Realistic peak memory per point:
        ~160 bytes  (xyz float64 arrays + CSF cloth mesh + DBSCAN working set
                     + height/normal arrays + output buffers)

    Resource caps (leave headroom for the OS and other user processes):
        RAM  : 50 % of total installed RAM  (hard ceiling)
        CPU  : 50 % of logical cores        (hard ceiling, min 1)

    Within the processing budget:
        Each active worker gets an equal share.
        Solo tiles (run alone) may use up to 50 % of the processing budget.

    Tile size is also capped for performance: tiles larger than ~20 M points
    take very long in DBSCAN/CSF even when they fit in RAM.
    """
    # 250 bytes/pt: measured peak across xyz float64 (24) + CSF internals +
    # neighbour-buffer loading + RGB bands + height/nZ arrays + output buffers.
    # Using 160 caused 4× overcommit on dense photogrammetry tiles.
    BYTES_PER_PT = 250
    # Fraction of AVAILABLE (free) RAM to use — leave headroom for the OS and
    # other running applications (ArcGIS, AutoCAD, Revit, etc.).
    AVAIL_FRACTION = 0.55
    # Hard ceiling: never exceed this fraction of TOTAL installed RAM,
    # regardless of how much is free right now.
    TOTAL_FRACTION = 0.50
    CPU_FRACTION   = 0.50   # cap at 50 % of logical CPU cores

    try:
        import psutil
        mem      = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total     / (1024 ** 3)
    except ImportError:
        avail_gb = 8.0
        total_gb = 16.0

    cpu       = os.cpu_count() or 4
    max_cpu   = max(1, int(cpu * CPU_FRACTION))
    # Primary signal is available RAM — avoids overcommit when other apps
    # are already occupying most of total RAM.
    budget_gb = min(avail_gb * AVAIL_FRACTION, total_gb * TOTAL_FRACTION)

    # ── Per-tile target ──────────────────────────────────────────────────
    # 25 % of the processing budget per tile so multiple workers can run.
    # Floor 2 M pts (avoids thousands of tiny tiles on very low-RAM systems).
    # Ceil 20 M pts (DBSCAN/CSF become very slow above this regardless of RAM).
    target_pts = int(budget_gb * 0.25 * (1024 ** 3) / BYTES_PER_PT)
    target_pts = max(2_000_000, min(20_000_000, target_pts))

    # ── Solo tile threshold ──────────────────────────────────────────────
    # A tile this large runs without parallel siblings to avoid RAM contention.
    # 50 % of the budget → one tile may use half the headroom.
    solo_pts = int(budget_gb * 0.50 * (1024 ** 3) / BYTES_PER_PT)
    solo_pts = max(target_pts * 3, solo_pts)   # always well above target

    # ── Max parallel workers ─────────────────────────────────────────────
    # How many target-sized tiles fit in the budget at the same time,
    # further capped to CPU_FRACTION of logical cores.
    per_tile_gb  = max(0.2, target_pts * BYTES_PER_PT / (1024 ** 3))
    ram_workers  = max(1, int(budget_gb / per_tile_gb))
    max_workers  = min(ram_workers, max_cpu)

    return {
        "avail_gb":        avail_gb,
        "total_gb":        total_gb,
        "budget_gb":       budget_gb,
        "cpu":             cpu,
        "max_cpu":         max_cpu,
        "bytes_per_pt":    BYTES_PER_PT,
        "target_pts":      target_pts,
        "solo_pts":        solo_pts,
        "max_workers":     max_workers,
    }


# Computed once at import; available to the GUI and other modules.
SYSTEM = _system_profile()
SOLO_TILE_POINT_THRESHOLD  = SYSTEM["solo_pts"]
_DEFAULT_TARGET_PTS_PER_TILE = SYSTEM["target_pts"]


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
    # Probable-ground band: points within this extra margin beyond the ground
    # threshold on both sides get class 12 instead of veg/noise/unclassified.
    # 0.0 = disabled.  E.g. 0.10 → confident ground ± class_thr, probable ± (class_thr+0.10)
    probable_ground_band: float = 0.0
    # Model key points (thinned ground for TIN building)
    classify_model_keys: bool = False
    modelkey_step_m: float = 8.0
    # Parallelism: 1 = sequential, N = ProcessPoolExecutor with N workers
    num_workers: int = 1
    # Post-processing: merge all *_classified.laz output tiles into ONE final LAZ
    merge_output_to_single: bool = True
    # Custom name for the merged output (blank = auto based on input)
    merged_output_name: str = ""

    # ── v2.0 — Drone photogrammetry mode ────────────────────────────────
    # "lidar" (default, original behaviour) or "photogrammetry"
    point_cloud_source: str = "lidar"
    # Vegetation index for photogrammetry mode: "auto" | "ndvi" | "vari" | "exg"
    vegetation_index: str = "auto"
    ndvi_threshold: float = 0.20
    vari_threshold: float = 0.15
    exg_threshold: float = 15.0
    # Building detection (photogrammetry only)
    detect_buildings_flag: bool = True
    building_min_area_m2: float = 20.0
    building_min_height_agl: float = 2.5
    building_plane_threshold: float = 0.10
    building_cluster_eps: float = 1.0
    building_min_cluster_pts: int = 300
    # Photogrammetry-specific ground defaults (applied when point_cloud_source == "photogrammetry")
    # These override csf_* fields at runtime — CSF gets tighter for smoother SfM surfaces
    photo_csf_cloth_resolution: float = 2.0
    photo_csf_class_threshold: float = 0.05
    photo_csf_rigidness: int = 3
    photo_csf_iterations: int = 500
    # Exclude buildings from final output (useful for Revit topography extraction)
    exclude_buildings_from_output: bool = False

    # ── v2.1 — SLAM / indoor-outdoor mode ───────────────────────────────
    # "slam" activates SLAM-specific CSF defaults and z_grid smoothing.
    # Coarser cloth + wider threshold + median smooth handle wall-catch artefacts
    # where the scanner only saw walls (no floor), making the lowest voxel point
    # a wall point rather than a floor point.
    slam_csf_cloth_resolution: float = 0.50   # coarser → less wall snagging
    slam_csf_class_threshold: float = 0.20    # ±20 cm ground band
    slam_csf_rigidness: int = 1
    slam_csf_iterations: int = 500
    slam_zgrid_smooth_cells: int = 3          # median filter radius on z_grid (cells)
    # Vertical-surface rejection: voxel cells whose Z range exceeds this value
    # are walls/columns and excluded from CSF ground candidates.
    # Ground/floor cells span ~0.05 m of noise; walls span 1–4 m.
    # 0.0 = disabled.  Recommended: 0.50 m for SLAM.
    slam_z_range_max: float = 0.50


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

def _process_one_tile(job: dict, inline_log_fn=None) -> dict:
    """
    Run ground classification + bird detection for ONE tile.
    Returns a result dict (plus log lines captured during work).
    Designed to be called either inline or in a worker subprocess.
    Pass inline_log_fn only for inline (non-subprocess) calls so logs
    stream in real-time instead of buffering until the function returns.
    """
    tile_path = Path(job["tile_path"])
    neighbour_paths = [Path(p) for p in job["neighbour_paths"]]
    params: ProcessParams = job["params"]
    out_dir = Path(job["out_dir"])

    log_lines: list[str] = []
    def log(msg: str):
        log_lines.append(msg)
        if inline_log_fn is not None:
            inline_log_fn(msg)

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

    # ── PHOTOGRAMMETRY PRE-PASS: vegetation mask from colour indices ────
    # (Runs on CORE points only. Sets veg_mask_core / building_mask_core
    # so they can be applied to the classification array after ground.)
    veg_mask_core = np.zeros(len(xyz_core), dtype=bool)
    building_mask_core = np.zeros(len(xyz_core), dtype=bool)
    photo_mode = (params.point_cloud_source == "photogrammetry")

    if photo_mode:
        try:
            bands = read_color_bands(las_src)
            log(f"  Photogrammetry mode — bands present: "
                f"RGB={'yes' if has_rgb(bands) else 'no'}, "
                f"NIR={'yes' if has_nir(bands) else 'no'}")
            veg_mask_core, used_idx = _photo_vegetation_mask(
                bands,
                index_choice=params.vegetation_index,
                ndvi_threshold=params.ndvi_threshold,
                vari_threshold=params.vari_threshold,
                exg_threshold=params.exg_threshold,
            )
            if veg_mask_core is None:
                log("  WARNING: no colour bands available — vegetation pre-mask disabled")
                veg_mask_core = np.zeros(len(xyz_core), dtype=bool)
            else:
                log(f"  Vegetation index used: {used_idx.upper()}  "
                    f"→ {int(veg_mask_core.sum()):,} pts flagged as vegetation "
                    f"({100 * veg_mask_core.sum() / len(xyz_core):.1f}% of tile)")
        except Exception as e:
            log(f"  WARNING: vegetation index failed: {type(e).__name__}: {e}")
            veg_mask_core = np.zeros(len(xyz_core), dtype=bool)

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
                # Photogrammetry mode: use tighter CSF params for smoother SfM surfaces
                if photo_mode:
                    cloth_res  = params.photo_csf_cloth_resolution
                    class_thr  = params.photo_csf_class_threshold
                    rigidness  = params.photo_csf_rigidness
                    iterations = params.photo_csf_iterations
                    log(f"  CSF (photogrammetry): cloth={cloth_res} thr={class_thr} "
                        f"rigid={rigidness} iter={iterations}")
                else:
                    cloth_res  = params.csf_cloth_resolution
                    class_thr  = params.csf_class_threshold
                    rigidness  = params.csf_rigidness
                    iterations = params.csf_iterations
                ground_mask_all, nz_all = classify_ground_csf(
                    xyz_all,
                    cloth_resolution=cloth_res,
                    class_threshold=class_thr,
                    rigidness=rigidness,
                    iterations=iterations,
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
        err_kind = type(e).__name__
        err_msg = str(e) or "(no message — likely out of memory or native crash)"
        log(f"  ERROR in ground classification: {err_kind}: {err_msg}")
        if "memory" in err_kind.lower() or "MemoryError" in err_kind:
            log("  ↳ Out of RAM. Re-run with 'Skip tiling' UNCHECKED to tile first, "
                "or tile manually and re-process as Tile folder.")
        result["status"] = "SKIPPED"
        result["reason"] = f"Ground classification error: {err_kind}: {err_msg}"
        return result

    nz_finite = nz[np.isfinite(nz)]
    log(f"  nZ range: {nz_finite.min():.2f} to {nz_finite.max():.2f} m  "
        f"| Z range: {xyz_core[:,2].min():.2f} to {xyz_core[:,2].max():.2f} m")

    # ── Bird detection (LiDAR only — SfM birds are unreliable) ────────────
    clusters: list = []
    if photo_mode:
        # In photogrammetry mode, initialise classification without birds
        classification = np.full(len(xyz_core), 1, dtype=np.uint8)
        classification[ground_mask] = 2
        log("  Bird detection: skipped (photogrammetry mode)")
    else:
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

    # ── PHOTOGRAMMETRY: apply vegetation mask + detect buildings ─────────
    if photo_mode:
        # Apply vegetation from colour index (protect ground)
        if veg_mask_core.any():
            veg_candidates = veg_mask_core & ~ground_mask
            classification[veg_candidates] = 3   # tentative low veg (heights
                                                  # step below will refine)
        # Building detection — needs ground reference (nz), so runs here
        if params.detect_buildings_flag:
            try:
                # Interpolated ground Z per point
                z_ground_ref = xyz_core[:, 2] - nz
                candidate_for_building = ~veg_mask_core & ~ground_mask
                building_mask_core = _photo_detect_buildings(
                    xyz_core,
                    candidate_mask=candidate_for_building,
                    z_ground_ref=z_ground_ref,
                    min_area_m2=params.building_min_area_m2,
                    min_height_agl=params.building_min_height_agl,
                    plane_threshold=params.building_plane_threshold,
                    cluster_eps=params.building_cluster_eps,
                    min_cluster_pts=params.building_min_cluster_pts,
                )
                n_bldg = int(building_mask_core.sum())
                log(f"  Buildings (class {CLASS_BUILDING}): {n_bldg:,} pts "
                    f"({100 * n_bldg / max(len(xyz_core), 1):.1f}% of tile)")
                classification[building_mask_core] = CLASS_BUILDING
            except Exception as e:
                log(f"  WARNING: building detection failed: "
                    f"{type(e).__name__}: {e}")

    # ── Height-based classification (TerraScan-style) ─────────────────────
    if params.classify_vegetation:
        try:
            # Protect buildings from being overwritten by height classification
            protect_extra = (CLASS_BUILDING,) if photo_mode else ()
            _gnd_thr = (params.slam_csf_class_threshold if slam_mode
                        else params.photo_csf_class_threshold if photo_mode
                        else params.csf_class_threshold)
            classification = classify_heights(
                classification, nz,
                veg_low_min=params.veg_low_min,
                veg_low_max=params.veg_low_max,
                veg_med_max=params.veg_med_max,
                noise_below_ground=params.noise_below_ground,
                probable_ground_band=params.probable_ground_band,
                ground_class_thr=_gnd_thr,
                protect_classes=(2, 20) + protect_extra,
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
# Streaming path — 2-pass classify, never loads full point cloud
# ────────────────────────────────────────────────────────────────────────────

def _classify_streaming(
    file_path: Path,
    out_dir: Path,
    params: "ProcessParams",
    log_fn: Callable,
    progress_fn: Callable,
    cancelled_fn: Callable,
) -> Path:
    """
    Classify a large LAZ/LAS file without loading all points into RAM.

    Pass 1 — ground surface (streaming voxel thin + CSF):
        Streams the file in 5 M-point chunks. Each chunk updates a tiny
        2-D grid (rows × cols cells) with the lowest Z seen per voxel cell.
        After the full file: ~1 M ground candidates → CSF → ground raster.
        Peak RAM: ≈ 150 MB  (grid + one chunk).

    Pass 2 — classify + write:
        Streams the file again. For each chunk: look up nZ from the ground
        raster (O(N)), assign classes 2/3/4/5/7 by nZ thresholds, write the
        classified chunk to the output file.
        Peak RAM: ≈ 250 MB  (two chunks in flight + ground raster).

    Bird detection is skipped for files above STREAMING_POINT_THRESHOLD
    (typically 30 M points) — at that density the per-m² coverage is already
    very high and near-ground clusters from a scan this dense are rarely
    genuine bird contacts. A log note is emitted.
    """
    import laspy
    import math

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_path.stem}_classified.{params.output_format}"

    photo_mode = (params.point_cloud_source == "photogrammetry")
    slam_mode  = (params.point_cloud_source == "slam")

    # ── PASS 1: streaming voxel thin + CSF ───────────────────────────────
    log_fn(f"\n  [Stream Pass 1] Building ground surface from {file_path.name} ...")
    if slam_mode:
        log_fn("  Source: SLAM — using SLAM CSF params + z_grid median smoothing")
    progress_fn(0, 10)

    try:
        if params.ground_source == "from_file":
            # Ground already classified — we still need to read it, but only
            # the classification field, which laspy loads lazily.  Fall back to
            # the normal single-file path for this case (it's already fast).
            log_fn("  Ground source = from_file: streaming not needed — reading ground pts only.")
            import laspy as _lp
            with _lp.open(str(file_path)) as r:
                las = r.read()
            src_cls = np.asarray(las.classification, dtype=np.uint8)
            ground_pts = np.stack([las.x[src_cls == 2], las.y[src_cls == 2],
                                   las.z[src_cls == 2]], axis=1)
            if len(ground_pts) < 50:
                raise RuntimeError("Source has <50 class-2 pts — switch to 'Compute fresh'")
            z_grid, x_min, y_min, cell_size = None, None, None, None
            # Build grid from existing ground points
            from .ground_classifier import _fill_nan
            xmn = float(las.x.min()); ymn = float(las.y.min())
            xmx = float(las.x.max()); ymx = float(las.y.max())
            cell_size = params.csf_cloth_resolution
            cols = int(np.ceil((xmx - xmn) / cell_size)) + 2
            rows = int(np.ceil((ymx - ymn) / cell_size)) + 2
            gci = np.clip(((ground_pts[:,0]-xmn)/cell_size).astype(np.int32), 0, cols-1)
            gri = np.clip(((ground_pts[:,1]-ymn)/cell_size).astype(np.int32), 0, rows-1)
            zs = np.zeros((rows, cols), dtype=np.float64)
            zc = np.zeros((rows, cols), dtype=np.int32)
            np.add.at(zs, (gri, gci), ground_pts[:,2])
            np.add.at(zc, (gri, gci), 1)
            z_grid = np.where(zc > 0, zs / np.maximum(zc, 1), np.nan)
            if np.isnan(z_grid).any():
                z_grid = _fill_nan(z_grid)
            x_min, y_min = xmn, ymn
            del las, ground_pts
            ground_class_thr = params.csf_class_threshold
        else:
            if params.ground_method == "csf":
                if slam_mode:
                    cloth_res  = params.slam_csf_cloth_resolution
                    class_thr  = params.slam_csf_class_threshold
                    rigidness  = params.slam_csf_rigidness
                    iterations = params.slam_csf_iterations
                elif photo_mode:
                    cloth_res  = params.photo_csf_cloth_resolution
                    class_thr  = params.photo_csf_class_threshold
                    rigidness  = params.photo_csf_rigidness
                    iterations = params.photo_csf_iterations
                else:
                    cloth_res  = params.csf_cloth_resolution
                    class_thr  = params.csf_class_threshold
                    rigidness  = params.csf_rigidness
                    iterations = params.csf_iterations
                ground_class_thr = class_thr
            else:
                # Grid minimum: build it streaming too — just min-Z grid, no CSF
                cloth_res  = params.ground_cell_size
                class_thr  = params.ground_threshold
                rigidness  = 1
                iterations = 1
                ground_class_thr = params.ground_threshold

            z_grid, x_min, y_min, cell_size = classify_ground_csf_streaming(
                file_path    = str(file_path),
                extra_pts    = None,
                cloth_resolution = cloth_res,
                class_threshold  = class_thr,
                rigidness        = rigidness,
                iterations       = iterations,
                slope_smooth     = params.csf_slope_smooth,
                csf_voxel_size   = params.csf_voxel_size,
                z_range_max      = params.slam_z_range_max if slam_mode else 0.0,
                log_fn           = log_fn,
            )

            # ── SLAM: median-smooth z_grid to suppress wall-catch spikes ──────
            if slam_mode and params.slam_zgrid_smooth_cells > 0:
                from scipy.ndimage import median_filter
                r = params.slam_zgrid_smooth_cells
                log_fn(f"  SLAM: median-smoothing z_grid with radius={r} cells ({r*cell_size:.2f} m)")
                z_grid = median_filter(z_grid, size=2 * r + 1, mode="nearest")

    except Exception as e:
        log_fn(f"  ERROR in streaming ground classification: {e}")
        raise

    progress_fn(5, 10)
    if cancelled_fn():
        return out_dir / "bird_contacts.csv"

    # ── PASS 2: stream classify + write ───────────────────────────────────
    log_fn(f"\n  [Stream Pass 2] Classifying + writing {out_path.name} ...")

    CHUNK = 5_000_000
    with laspy.open(str(file_path)) as reader:
        header     = reader.header
        pt_count   = header.point_count
        chunks_tot = math.ceil(pt_count / CHUNK)

        with laspy.open(str(out_path), mode="w", header=header) as writer:
            done = 0
            for chunk in reader.chunk_iterator(CHUNK):
                if cancelled_fn():
                    log_fn("  Cancelled during streaming write.")
                    return out_dir / "bird_contacts.csv"

                x  = np.asarray(chunk.x, dtype=np.float64)
                y  = np.asarray(chunk.y, dtype=np.float64)
                z  = np.asarray(chunk.z, dtype=np.float64)
                nz = nz_from_grid(x, y, z, z_grid, x_min, y_min, cell_size)

                cls = np.full(len(x), 1, dtype=np.uint8)   # 1 = unclassified

                # Ground — uses source-specific threshold resolved in Pass 1
                gnd = np.abs(nz) <= ground_class_thr
                cls[gnd] = 2

                if params.classify_vegetation:
                    above = ~gnd
                    pg_band = params.probable_ground_band
                    if pg_band > 0.0:
                        # Probable ground: just beyond the confident threshold on both sides
                        pg = above & (np.abs(nz) <= ground_class_thr + pg_band)
                        cls[pg] = 12
                        unresolved = above & ~pg
                    else:
                        unresolved = above
                    cls[unresolved & (nz >= params.veg_low_min) & (nz < params.veg_low_max)]  = 3
                    cls[unresolved & (nz >= params.veg_low_max) & (nz < params.veg_med_max)]  = 4
                    cls[unresolved & (nz >= params.veg_med_max)]                               = 5
                    # Only flag noise for non-ground, non-probable-ground points
                    cls[unresolved & (nz < params.noise_below_ground)]                         = 7

                chunk.classification = cls
                writer.write_points(chunk)

                done += 1
                progress_fn(5 + int(done / chunks_tot * 4), 10)
                if done % 10 == 0:
                    log_fn(f"    Pass 2: {done}/{chunks_tot} chunks written")

    progress_fn(9, 10)
    log_fn(f"  Streaming classify complete → {out_path}")
    log_fn(f"  (Bird detection skipped — file has >{STREAMING_POINT_THRESHOLD:,} pts; "
           f"re-tile to <30 M pts per tile if bird detection is needed)")

    # Write an empty CSV so callers that expect a CSV path don't break
    csv_path = out_dir / "bird_contacts.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            _csv.writer(f).writerow(
                ["tile", "cluster_id", "x", "y", "z", "nz_mean",
                 "n_points", "footprint_m2"]
            )

    progress_fn(10, 10)
    return csv_path


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

    log_fn(f"Single-file mode requested: {file_path.name}")
    pt_count = 0
    try:
        import laspy
        with laspy.open(str(file_path)) as r:
            pt_count = r.header.point_count
            log_fn(f"  Point count: {pt_count:,}")
    except Exception:
        pass

    # ── RAM pre-flight check ────────────────────────────────────────────────
    use_streaming = False
    auto_tile     = False
    est_gb        = 0.0
    if pt_count > 0:
        est_gb = (pt_count * SYSTEM["bytes_per_pt"]) / (1024 ** 3)
        log_fn(
            f"  Estimated RAM need: ~{est_gb:.1f} GB  |  "
            f"Total: {SYSTEM['total_gb']:.1f} GB  |  "
            f"Available: {SYSTEM['avail_gb']:.1f} GB  |  "
            f"Budget (55% avail / 50% total cap): {SYSTEM['budget_gb']:.1f} GB"
        )
        if pt_count >= STREAMING_POINT_THRESHOLD:
            # Any file ≥ 30M points always uses streaming regardless of RAM
            # estimate. Windows "Available" includes standby/cached pages that
            # aren't instantly free, making the estimate unreliable for large
            # files. Streaming peak RAM ≈ 400 MB regardless of file size.
            use_streaming = True
        elif est_gb > SYSTEM["budget_gb"]:
            # Smaller file but still over budget — fall back to tiling.
            auto_tile = True

    # ── Streaming path: 2-pass classify without loading full cloud ──────
    if use_streaming:
        log_fn("")
        log_fn(
            f"  File has {pt_count:,} pts — using streaming classify "
            f"(≥{STREAMING_POINT_THRESHOLD:,} pt threshold; peak RAM ≈ 400 MB)."
        )
        return _classify_streaming(
            file_path=file_path,
            out_dir=out_dir,
            params=params,
            log_fn=log_fn,
            progress_fn=progress_fn,
            cancelled_fn=cancelled_fn,
        )

    # ── Auto-tile fallback (file is small but still over budget) ─────────
    if auto_tile:
        log_fn("")
        log_fn(
            f"  ⚠ File needs ~{est_gb:.1f} GB but budget is {SYSTEM['budget_gb']:.1f} GB — "
            f"switching to auto-tiling (single-file mode overridden)."
        )
        target_pts_per_tile = SYSTEM["target_pts"]
        try:
            import laspy as _lp, math
            with _lp.open(str(file_path)) as r:
                h = r.header
                area_m2 = max(1.0, (h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]))
            density = pt_count / area_m2
            tile_area = target_pts_per_tile / max(density, 1.0)
            tile_size = max(25.0, min(200.0, round(math.sqrt(tile_area) / 5) * 5))
        except Exception:
            tile_size = 100.0

        log_fn(f"  Auto-tile size: {tile_size:.0f} m")
        from .tiler import tile_file as _tile_file
        tile_dir = file_path.parent / f"Tile_{file_path.stem}_auto"
        _tile_file(
            in_path=str(file_path),
            out_dir=str(tile_dir),
            tile_size=tile_size,
            log_fn=log_fn,
            progress_fn=progress_fn,
            cancelled_fn=cancelled_fn,
            write_boundary_dxf=False,   # written to out_dir below instead
        )
        if cancelled_fn():
            log_fn("Cancelled during auto-tiling.")
            return out_dir / "bird_contacts.csv"
        log_fn("\n  Auto-tiling complete — running full tile pipeline...\n")
        params.tile_size = tile_size
        if not params.merged_output_name.strip():
            params.merged_output_name = f"{file_path.stem}_classified.laz"
        return process_all_tiles(
            tile_dir=tile_dir,
            out_dir=out_dir,
            params=params,
            log_fn=log_fn,
            progress_fn=progress_fn,
            cancelled_fn=cancelled_fn,
        )

    job = {
        "tile_path": str(file_path),
        "neighbour_paths": [],          # no neighbours
        "params": params,
        "out_dir": str(out_dir),
    }

    if cancelled_fn():
        log_fn("Cancelled before start.")
        return out_dir / "bird_contacts.csv"

    result = _process_one_tile(job, inline_log_fn=log_fn)

    # Logs were already streamed in real-time via inline_log_fn; no re-emit needed.

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

_CHECKPOINT_FILE = "_starling_checkpoint.json"


def _ckpt_load(out_dir: Path) -> dict:
    """Load checkpoint from out_dir. Returns empty dict if none exists."""
    p = out_dir / _CHECKPOINT_FILE
    if not p.exists():
        return {}
    try:
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ckpt_save(out_dir: Path, data: dict) -> None:
    import json, datetime
    data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        (out_dir / _CHECKPOINT_FILE).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        pass   # never crash on checkpoint write failure


def process_all_tiles(
    tile_dir: str | Path,
    out_dir: str | Path,
    params: ProcessParams,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda i, n: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> Path:
    import datetime

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

    # ── Checkpoint: resume support ───────────────────────────────────────
    ckpt = _ckpt_load(out_dir)
    already_done: set[str] = set(ckpt.get("completed", []))
    if already_done:
        log_fn(f"  Checkpoint found: {len(already_done)}/{total} tiles already done "
               f"— resuming from where it left off.")
    ckpt.update({
        "tile_dir": str(tile_dir),
        "total": total,
        "completed": sorted(already_done),
        "started_at": ckpt.get("started_at",
                               datetime.datetime.now().isoformat(timespec="seconds")),
        "status": "in_progress",
    })
    _ckpt_save(out_dir, ckpt)

    # Build per-tile jobs — skip tiles already completed in a previous run
    tile_size_int = int(params.tile_size)
    jobs = []
    skipped = 0
    for (east, north), tile_path in tiles:
        if tile_path.name in already_done:
            skipped += 1
            continue
        jobs.append({
            "tile_path": str(tile_path),
            "neighbour_paths": [str(p) for p in
                                _neighbour_paths(tile_index, east, north, tile_size_int)],
            "params": params,
            "out_dir": str(out_dir),
            "_size": tile_path.stat().st_size,
        })
    if skipped:
        log_fn(f"  Skipping {skipped} already-completed tile(s). "
               f"{len(jobs)} remaining.")

    # ── RAM-aware worker cap ─────────────────────────────────────────────
    # Use actual median tile point count from headers to refine the cap;
    # fall back to SYSTEM["max_workers"] which is already RAM-derived.
    effective_workers = min(params.num_workers, SYSTEM["max_workers"])
    if params.num_workers > effective_workers:
        log_fn(
            f"  Worker count reduced {params.num_workers} → {effective_workers} "
            f"(RAM budget {SYSTEM['budget_gb']:.0f} GB on this machine)"
        )
    if effective_workers > 1 and len(jobs) > 1:
        try:
            import laspy as _lp
            sample_pts = []
            for j in jobs[:min(8, len(jobs))]:
                try:
                    with _lp.open(j["tile_path"]) as _r:
                        sample_pts.append(_r.header.point_count)
                except Exception:
                    pass
            if sample_pts:
                median_pts    = sorted(sample_pts)[len(sample_pts) // 2]
                per_tile_gb   = max(0.2, median_pts * SYSTEM["bytes_per_pt"] / (1024 ** 3))
                actual_cap    = max(1, int(SYSTEM["budget_gb"] / per_tile_gb))
                actual_cap    = min(actual_cap, SYSTEM["cpu"])
                if actual_cap < effective_workers:
                    log_fn(
                        f"  Tile-size refinement: {effective_workers} → {actual_cap} workers "
                        f"(median tile {median_pts/1e6:.1f} M pts, "
                        f"~{per_tile_gb:.1f} GB each)"
                    )
                    effective_workers = actual_cap
        except Exception:
            pass

    # Decide parallelism per tile (giant tiles run solo)
    sequential = effective_workers <= 1
    if not sequential:
        log_fn(f"Parallel mode: {effective_workers} workers "
               f"(tiles ≥{SOLO_TILE_POINT_THRESHOLD:,} pts will run solo)")

    all_clusters: list[dict] = []
    tile_results: list[dict] = []
    completed = skipped   # count already-done tiles in progress bar

    def _mark_done(tile_path_str: str) -> None:
        name = Path(tile_path_str).name
        already_done.add(name)
        ckpt["completed"] = sorted(already_done)
        ckpt["completed_count"] = len(already_done)
        _ckpt_save(out_dir, ckpt)

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
            _mark_done(job["tile_path"])
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
        import time as _time
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            # Track submission time per tile so heartbeat can show elapsed
            in_flight: dict = {}   # future → (tile_name, start_ts)
            futures = {}
            n_workers = effective_workers
            n_jobs = len(normals)

            log_fn(f"  Dispatching {n_jobs} tile(s) to a queue with {n_workers} "
                   f"concurrent worker slot(s)...")
            for job in normals:
                fut = pool.submit(_process_one_tile, job)
                tname = Path(job["tile_path"]).name
                futures[fut] = job
                in_flight[fut] = (tname, _time.time())
            log_fn(f"  All {n_jobs} tile(s) queued. {n_workers} will run at a "
                   f"time; others wait for a free worker slot.")

            last_heartbeat = _time.time()
            HEARTBEAT_SEC = 60

            for fut in as_completed(futures):
                # Periodic heartbeat — show queue state
                now = _time.time()
                if now - last_heartbeat > HEARTBEAT_SEC:
                    pending = [(name, now - start)
                               for f, (name, start) in in_flight.items()
                               if not f.done()]
                    if pending:
                        # Actually running = at most n_workers of the pending futures
                        n_running = min(len(pending), n_workers)
                        n_queued  = max(0, len(pending) - n_workers)
                        # Longest-elapsed pending are almost certainly the ones running
                        pending_sorted = sorted(pending, key=lambda x: -x[1])
                        running_list = pending_sorted[:n_workers]
                        running_str = ", ".join(
                            f"{n} ({s/60:.0f}m)" for n, s in running_list
                        )
                        log_fn(
                            f"  ⏱ running: {running_str}   |   "
                            f"queued: {n_queued}   |   "
                            f"completed: {completed}/{total}"
                        )
                    last_heartbeat = now

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
                    in_flight.pop(fut, None)
                    completed += 1
                    progress_fn(completed, total)
                    continue

                # Remove from in-flight tracker and log completion time
                tname, tstart = in_flight.pop(fut, (Path(futures[fut]["tile_path"]).name, now))
                elapsed = now - tstart
                n_remaining = len(in_flight)
                n_queued  = max(0, n_remaining - n_workers)
                log_fn(f"  ✓ done  {tname}   ({elapsed/60:.1f} min)   |   "
                       f"remaining: {n_remaining}  "
                       f"(queued: {n_queued} · running: {min(n_remaining, n_workers)})")

                for line in result["log_lines"]:
                    log_fn(line)
                tile_results.append({
                    k: v for k, v in result.items()
                    if k not in ("log_lines", "clusters", "written_path")
                })
                all_clusters.extend(result["clusters"])
                completed += 1
                _mark_done(futures[fut]["tile_path"])
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
            _mark_done(job["tile_path"])
            progress_fn(completed, total)

    # ── Mark checkpoint complete ──────────────────────────────────────────
    ckpt["status"] = "complete"
    _ckpt_save(out_dir, ckpt)
    log_fn(f"  Checkpoint complete: all {total} tiles processed.")

    # ── Tile index DXF → output folder ───────────────────────────────────
    try:
        from .tiler import write_tile_boundaries_dxf as _write_dxf
        dxf_path = out_dir / "tile_index.dxf"
        _write_dxf(tile_dir, out_path=dxf_path, log_fn=log_fn)
    except Exception as _e:
        log_fn(f"  WARNING: tile index DXF not written: {_e}")

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

    # ── Optional: merge classified tiles into ONE single LAZ ──────────────
    if params.merge_output_to_single:
        classified = sorted(Path(out_dir).glob("*_classified.la[sz]"))
        if len(classified) >= 1:
            from .tiler import merge_las_files as _merge
            src_stem = Path(tile_dir).name
            name = params.merged_output_name.strip() or f"{src_stem}_classified.laz"
            merged_out = Path(out_dir) / name
            log_fn(f"\nMerging {len(classified)} classified tiles → {merged_out.name}")
            try:
                _merge(
                    in_paths=classified,
                    out_path=merged_out,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                    cancelled_fn=cancelled_fn,
                )
                shards_dir = Path(out_dir) / f"_shards_{src_stem}"
                shards_dir.mkdir(exist_ok=True)
                moved = 0
                for t in classified:
                    try:
                        t.rename(shards_dir / t.name)
                        moved += 1
                    except Exception:
                        pass
                log_fn(f"  Moved {moved} shard tiles to: {shards_dir}")
                log_fn(f"  FINAL SINGLE OUTPUT: {merged_out}")
            except Exception as e:
                log_fn(f"  WARNING: could not merge output tiles: {e}")

    return csv_path
