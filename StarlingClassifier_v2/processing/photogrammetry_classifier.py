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
) -> np.ndarray:
    """
    Identify large horizontal planar clusters (rooftops) using pure numpy
    RANSAC and sklearn DBSCAN.  Worker-safe on Windows.

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

    # ── Iterative plane extraction ──
    remaining_pts = xyz[active_idx].astype(np.float64)
    remaining_local_idx = np.arange(len(remaining_pts))

    for _ in range(max_planes_per_tile):
        if len(remaining_local_idx) < min_cluster_pts:
            break

        plane_model, inlier_local_idx = _numpy_ransac_plane(
            remaining_pts[remaining_local_idx],
            distance_threshold=plane_threshold,
            num_iterations=ransac_iterations,
        )

        if plane_model is None:
            break
        if len(inlier_local_idx) < min_cluster_pts:
            break

        a, b, c, _d = plane_model
        norm_len = (a * a + b * b + c * c) ** 0.5
        if norm_len == 0:
            break
        nz_component = abs(c) / norm_len   # verticality of the normal

        if nz_component < 0.90:            # not horizontal → discard, don't cluster
            keep_mask = np.ones(len(remaining_local_idx), dtype=bool)
            keep_mask[inlier_local_idx] = False
            remaining_local_idx = remaining_local_idx[keep_mask]
            continue

        # Cluster inliers in XY to separate individual roofs
        inlier_pts_xy = remaining_pts[remaining_local_idx[inlier_local_idx]][:, :2]
        try:
            labels = DBSCAN(
                eps=cluster_eps,
                min_samples=min_cluster_pts,
                algorithm="ball_tree",
                n_jobs=1,      # workers are already parallel; keep DBSCAN single-thread
            ).fit_predict(inlier_pts_xy)
        except Exception:
            labels = np.zeros(len(inlier_pts_xy), dtype=np.int32)

        for lbl in set(labels.tolist()) - {-1}:
            cluster_mask = labels == lbl
            if int(cluster_mask.sum()) < min_cluster_pts:
                continue
            pts_xy = inlier_pts_xy[cluster_mask]
            span_x = pts_xy[:, 0].max() - pts_xy[:, 0].min()
            span_y = pts_xy[:, 1].max() - pts_xy[:, 1].min()
            if (span_x * span_y) < min_area_m2:
                continue
            local = remaining_local_idx[inlier_local_idx[cluster_mask]]
            building_mask[active_idx[local]] = True

        # Remove ALL inliers from remaining so next iteration finds another plane
        keep_mask = np.ones(len(remaining_local_idx), dtype=bool)
        keep_mask[inlier_local_idx] = False
        remaining_local_idx = remaining_local_idx[keep_mask]

    return building_mask
