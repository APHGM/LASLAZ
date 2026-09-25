"""
Geometric filter to remove parked cars from ground/surface classification.

Cars in drone photogrammetry ground clouds show up as:
  - Point clusters sitting ~0.4–2.0 m above the local ground surface
  - Rectangular footprint ~3–20 m² (small car to van/SUV)
  - Higher colour saturation than road/pavement (they're painted)

Approach:
  1. Find points with nZ (height above ground) in the car band
  2. Cluster them spatially in XY (DBSCAN)
  3. Reject clusters that are too big (buildings), too small (debris/kerb)
  4. Return mask of points belonging to car-shaped clusters

Note: this runs on ALL points from the tile (not just ground). Any point that
is part of a car cluster gets marked for exclusion from surface polygons.
"""

import numpy as np


def detect_cars_by_geometry(
    xyz: np.ndarray,
    nz: np.ndarray,
    car_z_min: float = 0.30,
    car_z_max: float = 2.20,
    dbscan_eps: float = 0.40,
    dbscan_min_pts: int = 30,
    footprint_min_m2: float = 2.0,
    footprint_max_m2: float = 25.0,
) -> np.ndarray:
    """
    Return a bool mask (True = point belongs to a car-shaped cluster).

    Args:
        xyz     : Nx3 array of point coordinates
        nz      : per-point height above ground surface
        car_z_min, car_z_max: nZ band for car body height
        dbscan_eps, dbscan_min_pts: DBSCAN clustering parameters
        footprint_min_m2, footprint_max_m2: XY bounding-box area limits
    """
    n = len(xyz)
    car_mask = np.zeros(n, dtype=bool)

    candidate_mask = np.isfinite(nz) & (nz >= car_z_min) & (nz <= car_z_max)
    idx = np.where(candidate_mask)[0]
    if len(idx) < dbscan_min_pts:
        return car_mask

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return car_mask

    pts_xy = xyz[idx, :2]
    labels = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_pts,
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(pts_xy)

    for lbl in set(labels.tolist()) - {-1}:
        members_local = np.where(labels == lbl)[0]
        if len(members_local) < dbscan_min_pts:
            continue
        pxy = pts_xy[members_local]
        span_x = pxy[:, 0].max() - pxy[:, 0].min()
        span_y = pxy[:, 1].max() - pxy[:, 1].min()
        area = span_x * span_y
        if area < footprint_min_m2 or area > footprint_max_m2:
            continue
        # Passed all filters — mark as car
        car_mask[idx[members_local]] = True

    return car_mask


def expand_car_mask_downward(
    xyz: np.ndarray,
    car_mask: np.ndarray,
    xy_radius: float = 0.30,
) -> np.ndarray:
    """
    For each car cluster, also mark any point directly BELOW it (within xy_radius)
    as part of the car — this catches ground points obscured by / underneath the
    car body that would otherwise leave a hole in the surface polygon.

    Uses a KDTree lookup in XY only.
    """
    if not car_mask.any():
        return car_mask
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return car_mask

    tree = cKDTree(xyz[car_mask, :2])
    dists, _ = tree.query(xyz[:, :2], k=1, workers=-1)
    within = dists <= xy_radius
    return car_mask | within
