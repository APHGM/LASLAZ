"""
Photogrammetry-specific classification helpers.

For drone photogrammetry point clouds we CANNOT rely on multi-return
information (there is none — it's single-return by construction). We compensate
with:
  1. A vegetation index (NDVI when NIR available, VARI/ExG when RGB-only) to
     mask green plants BEFORE ground classification.
  2. Building detection via horizontal plane RANSAC (open3d) so wide flat
     rooftops don't get labelled as ground.

Both stages run BEFORE the standard CSF ground pass so the cloth simulator only
sees ground candidates.
"""

import numpy as np


# Vegetation index defaults (adjustable via ProcessParams)
NDVI_THRESHOLD_DEFAULT = 0.20
VARI_THRESHOLD_DEFAULT = 0.15
EXG_THRESHOLD_DEFAULT  = 15.0

# ASPRS class codes (mirror the ones in height_classifier)
CLASS_BUILDING = 6


# ────────────────────────────────────────────────────────────────────────────
# Vegetation indices — pure numpy, no dependencies
# ────────────────────────────────────────────────────────────────────────────

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    True NDVI = (NIR - Red) / (NIR + Red).
    Requires the NIR channel (LAS point format 8 or 10).
    Result is a float32 array in [-1, +1].
    """
    r = red.astype(np.float32)
    n = nir.astype(np.float32)
    denom = n + r
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denom > 0, (n - r) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_vari(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    VARI (Visible Atmospherically Resistant Index) = (G - R) / (G + R - B).
    Best RGB-only vegetation index for mixed lighting conditions.
    Result is a float32 array — typically in [-1, +1].
    """
    r = red.astype(np.float32)
    g = green.astype(np.float32)
    b = blue.astype(np.float32)
    denom = g + r - b
    with np.errstate(divide="ignore", invalid="ignore"):
        vari = np.where(np.abs(denom) > 1e-6, (g - r) / denom, 0.0)
    # Clip to sensible range — VARI can spike outside [-1,1] when denom ~ 0
    return np.clip(vari, -2.0, 2.0).astype(np.float32)


def compute_exg(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    ExG (Excess Green) = 2G - R - B.
    Simple, robust for green vegetation. Result is float32 in same units as input.
    """
    r = red.astype(np.float32)
    g = green.astype(np.float32)
    b = blue.astype(np.float32)
    return (2.0 * g - r - b).astype(np.float32)


def vegetation_mask(
    bands: dict[str, np.ndarray | None],
    index_choice: str = "auto",
    ndvi_threshold: float = NDVI_THRESHOLD_DEFAULT,
    vari_threshold: float = VARI_THRESHOLD_DEFAULT,
    exg_threshold: float = EXG_THRESHOLD_DEFAULT,
) -> tuple[np.ndarray | None, str]:
    """
    Compute a boolean vegetation mask using the best available index.

    index_choice:
        "auto"  — NDVI if NIR present, else VARI, else ExG
        "ndvi"  — force NDVI (requires NIR band; returns (None, "unavailable") if missing)
        "vari"  — force VARI (requires RGB)
        "exg"   — force ExG (requires RGB)

    Returns (mask, used_index_name):
        mask: bool array or None if no colour data available
        used_index_name: "ndvi" | "vari" | "exg" | "unavailable"
    """
    r, g, b, n = bands.get("red"), bands.get("green"), bands.get("blue"), bands.get("nir")
    has_rgb = all(x is not None for x in (r, g, b))
    has_nir = n is not None

    choice = index_choice.lower()

    if choice == "auto":
        if has_nir and r is not None:
            idx = compute_ndvi(r, n)
            return idx > ndvi_threshold, "ndvi"
        if has_rgb:
            idx = compute_vari(r, g, b)
            return idx > vari_threshold, "vari"
        return None, "unavailable"

    if choice == "ndvi":
        if not (has_nir and r is not None):
            return None, "unavailable"
        return compute_ndvi(r, n) > ndvi_threshold, "ndvi"

    if choice == "vari":
        if not has_rgb:
            return None, "unavailable"
        return compute_vari(r, g, b) > vari_threshold, "vari"

    if choice == "exg":
        if not has_rgb:
            return None, "unavailable"
        return compute_exg(r, g, b) > exg_threshold, "exg"

    return None, "unavailable"


# ────────────────────────────────────────────────────────────────────────────
# Building detection — pure numpy plane RANSAC + sklearn DBSCAN
# ────────────────────────────────────────────────────────────────────────────
# open3d was used originally but crashes worker processes on Windows due to
# native MKL / OpenMP conflicts. Rewrote to pure numpy so it's fully safe
# in ProcessPoolExecutor workers on all platforms.


def _numpy_ransac_plane(
    points: np.ndarray,
    distance_threshold: float,
    num_iterations: int = 500,
    seed: int | None = None,
) -> tuple[tuple[float, float, float, float] | None, np.ndarray]:
    """
    Pure-numpy RANSAC plane fit.
    Returns (plane_coefs (a,b,c,d) with ax+by+cz+d=0, inlier_indices).
    Never touches native libraries — safe in multiprocessing workers.
    """
    n = len(points)
    if n < 3:
        return None, np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    best_count = 0
    best_plane = None
    best_inliers = np.array([], dtype=np.int64)

    pts64 = points.astype(np.float64, copy=False)

    for _ in range(num_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts64[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:      # colinear sample → skip
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        # Signed distance of every point to this plane, then take abs
        distances = np.abs(pts64 @ normal + d)
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > best_count:
            best_count = len(inliers)
            a, b, c = normal
            best_plane = (float(a), float(b), float(c), float(d))
            best_inliers = inliers

    return best_plane, best_inliers


def _voxel_downsample_idx(
    pts: np.ndarray,
    voxel_size: float,
    max_pts: int,
) -> np.ndarray:
    """
    Return indices into pts that form a voxel-downsampled subset.
    Falls back to uniform random sampling when the voxel grid still
    exceeds max_pts (avoids ballooning RAM on ultra-dense tiles).
    """
    if len(pts) <= max_pts:
        return np.arange(len(pts), dtype=np.int64)

    # Assign each point to a voxel cell and keep one representative per cell.
    origin = pts[:, :3].min(axis=0)
    keys = ((pts[:, :3] - origin) / voxel_size).astype(np.int32)
    # Pack (ix, iy, iz) into a single int64 key — fast unique lookup.
    K0 = int(keys[:, 0].max()) + 1
    K1 = int(keys[:, 1].max()) + 1
    flat_key = keys[:, 0].astype(np.int64) * (K1 * (int(keys[:, 2].max()) + 1)) \
             + keys[:, 1].astype(np.int64) * (int(keys[:, 2].max()) + 1) \
             + keys[:, 2].astype(np.int64)
    _, first_occ = np.unique(flat_key, return_index=True)
    if len(first_occ) > max_pts:
        # Still too many voxels — uniform subsample
        rng = np.random.default_rng(42)
        first_occ = rng.choice(first_occ, size=max_pts, replace=False)
    return first_occ.astype(np.int64)


def detect_buildings(
    xyz: np.ndarray,
    candidate_mask: np.ndarray,
    z_ground_ref: np.ndarray | None = None,
    min_area_m2: float = 20.0,
    min_height_agl: float = 2.5,
    plane_threshold: float = 0.10,
    cluster_eps: float = 1.0,
    min_cluster_pts: int = 300,
    max_planes_per_tile: int = 8,
    ransac_iterations: int = 500,
    max_ransac_pts: int = 2_000_000,
) -> np.ndarray:
    """
    Identify large horizontal planar clusters (rooftops) using pure numpy
    RANSAC and sklearn DBSCAN.  Worker-safe on Windows.

    max_ransac_pts controls how many candidate points are passed to each
    RANSAC call.  When the candidate set is larger it is voxel-downsampled
    first; inliers are then recovered from the FULL candidate set so no
    classification coverage is lost.  This keeps peak allocations bounded
    regardless of tile density (fixes MemoryError on 30–60 M point tiles).

    Returns building_mask (bool array, len == len(xyz)).
    """
    n_total = len(xyz)
    building_mask = np.zeros(n_total, dtype=bool)

    if not candidate_mask.any():
        return building_mask

    active = candidate_mask.copy()
    if z_ground_ref is not None:
        agl = xyz[:, 2] - z_ground_ref
        active &= agl > min_height_agl

    active_idx = np.where(active)[0]
    if len(active_idx) < min_cluster_pts:
        return building_mask

    # sklearn DBSCAN — already used elsewhere in the codebase, safe in workers
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return building_mask

    # Full-resolution candidate array (float32 to halve memory vs float64;
    # plane fitting uses float64 internally on the small downsampled subset).
    full_pts = xyz[active_idx].astype(np.float32)
    # Mask into full_pts that tracks which points are still "available"
    # (not yet consumed by a found plane).
    available = np.ones(len(full_pts), dtype=bool)

    for _ in range(max_planes_per_tile):
        avail_idx = np.where(available)[0]
        if len(avail_idx) < min_cluster_pts:
            break

        pts_avail = full_pts[avail_idx]

        # ── Downsample for RANSAC plane fitting ──────────────────────────
        # Voxel size: aim for ~max_ransac_pts points from the bounding box.
        if len(pts_avail) > max_ransac_pts:
            bbox = pts_avail[:, :3].max(axis=0) - pts_avail[:, :3].min(axis=0)
            vol = float(np.prod(np.maximum(bbox, 1e-3)))
            vox = max(0.05, (vol / max_ransac_pts) ** (1.0 / 3.0))
            ds_local_idx = _voxel_downsample_idx(pts_avail, vox, max_ransac_pts)
            ransac_pts = pts_avail[ds_local_idx].astype(np.float64)
        else:
            ds_local_idx = None
            ransac_pts = pts_avail.astype(np.float64)

        plane_model, inlier_ds_idx = _numpy_ransac_plane(
            ransac_pts,
            distance_threshold=plane_threshold,
            num_iterations=ransac_iterations,
        )

        if plane_model is None or len(inlier_ds_idx) < min_cluster_pts:
            break

        a, b, c, d_coef = plane_model
        norm_len = (a * a + b * b + c * c) ** 0.5
        if norm_len == 0:
            break
        nz_component = abs(c) / norm_len

        if nz_component < 0.90:
            # Not a horizontal plane — remove these points and try again.
            if ds_local_idx is not None:
                available[avail_idx[ds_local_idx[inlier_ds_idx]]] = False
            else:
                available[avail_idx[inlier_ds_idx]] = False
            continue

        # ── Recover full-resolution inliers from ALL available pts ───────
        # Compute distances in float32 chunks to avoid one giant allocation.
        CHUNK = 2_000_000
        normal_f32 = np.array([a, b, c], dtype=np.float32) / float(norm_len)
        d_f32 = float(d_coef) / float(norm_len)
        inlier_full_local = []
        for start in range(0, len(pts_avail), CHUNK):
            chunk = pts_avail[start:start + CHUNK, :3]
            dist = np.abs(chunk @ normal_f32 + d_f32)
            hits = np.where(dist < plane_threshold)[0]
            if len(hits):
                inlier_full_local.append(hits + start)
        if not inlier_full_local:
            break
        inlier_full_local = np.concatenate(inlier_full_local)

        if len(inlier_full_local) < min_cluster_pts:
            available[avail_idx[inlier_full_local]] = False
            continue

        # ── Cluster inliers in XY to separate individual roofs ───────────
        inlier_pts_xy = pts_avail[inlier_full_local, :2]
        try:
            labels = DBSCAN(
                eps=cluster_eps,
                min_samples=min_cluster_pts,
                algorithm="ball_tree",
                n_jobs=1,
            ).fit_predict(inlier_pts_xy)
        except Exception:
            labels = np.zeros(len(inlier_pts_xy), dtype=np.int32)

        for lbl in set(labels.tolist()) - {-1}:
            cluster_mask = labels == lbl
            if int(cluster_mask.sum()) < min_cluster_pts:
                continue
            pts_xy = inlier_pts_xy[cluster_mask]
            span_x = float(pts_xy[:, 0].max() - pts_xy[:, 0].min())
            span_y = float(pts_xy[:, 1].max() - pts_xy[:, 1].min())
            if (span_x * span_y) < min_area_m2:
                continue
            local = avail_idx[inlier_full_local[cluster_mask]]
            building_mask[active_idx[local]] = True

        # Mark all full-res inliers as consumed for the next iteration.
        available[avail_idx[inlier_full_local]] = False

    return building_mask
