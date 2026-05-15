import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import Callable


# ASPRS classification codes
CLASS_UNCLASSIFIED = 1
CLASS_GROUND = 2
CLASS_BIRD_CONTACT = 20   # user-defined

# If more than this fraction of tile points are candidates, ground is wrong
MAX_CANDIDATE_FRACTION = 0.05   # 5%
MAX_CANDIDATE_ABS = 500_000     # hard cap before voxel downsample


@dataclass
class BirdCluster:
    cluster_id: int
    point_count: int
    x_centroid: float
    y_centroid: float
    z_centroid: float
    nz_mean: float
    footprint_m2: float


def _voxel_downsample(pts: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one representative point per voxel cell.
    Returns (downsampled_pts, mapping) where mapping[i] = original index of kept point.
    """
    keys = np.floor(pts / voxel).astype(np.int64)
    # Cantor-style hash to get unique voxel id
    k = keys[:, 0] * 1_000_003 + keys[:, 1]
    _, first = np.unique(k, return_index=True)
    return pts[first], first


def detect_bird_contacts(
    xyz: np.ndarray,
    nz: np.ndarray,
    ground_mask: np.ndarray,
    nz_min: float = 0.02,
    nz_max: float = 0.40,
    dbscan_eps: float = 0.30,
    dbscan_min_pts: int = 8,
    max_footprint_m2: float = 1.0,
    min_footprint_m2: float = 0.005,
    log_fn: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[BirdCluster]]:
    """
    Detect starling ground-contact points.

    Returns:
        classification  — uint8 array per point
        clusters        — list of BirdCluster metadata
    """
    n = len(xyz)
    classification = np.full(n, CLASS_UNCLASSIFIED, dtype=np.uint8)
    classification[ground_mask] = CLASS_GROUND

    candidate_mask = (nz >= nz_min) & (nz <= nz_max) & ~ground_mask
    candidate_idx = np.where(candidate_mask)[0]
    n_candidates = len(candidate_idx)

    clusters: list[BirdCluster] = []

    if n_candidates < dbscan_min_pts:
        return classification, clusters

    # Sanity check: if >5% of points are candidates, ground classification is unreliable
    candidate_fraction = n_candidates / n
    if candidate_fraction > MAX_CANDIDATE_FRACTION:
        log_fn(
            f"  WARNING: {n_candidates:,} candidate pts ({candidate_fraction:.1%} of tile) — "
            f"ground surface may be incorrect. Bird detection skipped for this tile. "
            f"Try CSF method or adjust ground threshold."
        )
        return classification, clusters

    pts_2d = xyz[candidate_idx, :2]

    # Voxel downsample if still too many for DBSCAN
    voxel_idx = None
    if n_candidates > MAX_CANDIDATE_ABS:
        voxel_size = dbscan_eps / 3.0
        pts_down, local_keep = _voxel_downsample(pts_2d, voxel_size)
        log_fn(f"  Voxel downsampled {n_candidates:,} → {len(pts_down):,} candidates "
               f"(voxel={voxel_size:.3f}m)")
        voxel_idx = local_keep
        pts_for_dbscan = pts_down
    else:
        pts_for_dbscan = pts_2d

    log_fn(f"  Running DBSCAN on {len(pts_for_dbscan):,} candidate points ...")
    labels = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_pts,
        algorithm="ball_tree",
        n_jobs=-1
    ).fit_predict(pts_for_dbscan)

    unique_labels = set(labels) - {-1}
    log_fn(f"  DBSCAN found {len(unique_labels)} raw clusters")

    for lbl in unique_labels:
        if voxel_idx is not None:
            # Map voxel cluster back to full candidate set via spatial proximity
            vox_members_local = voxel_idx[labels == lbl]
            vox_pts = pts_2d[vox_members_local]
            # All candidates within eps of any voxel member belong to this cluster
            from scipy.spatial import cKDTree
            tree = cKDTree(vox_pts)
            dists, _ = tree.query(pts_2d, k=1, workers=-1)
            full_local = np.where(dists <= dbscan_eps)[0]
            members = candidate_idx[full_local]
        else:
            members = candidate_idx[labels == lbl]

        pts = xyz[members]
        nz_vals = nz[members]

        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        footprint = x_range * y_range

        if footprint < min_footprint_m2 or footprint > max_footprint_m2:
            continue

        classification[members] = CLASS_BIRD_CONTACT
        clusters.append(BirdCluster(
            cluster_id=int(lbl),
            point_count=len(members),
            x_centroid=float(pts[:, 0].mean()),
            y_centroid=float(pts[:, 1].mean()),
            z_centroid=float(pts[:, 2].mean()),
            nz_mean=float(nz_vals.mean()),
            footprint_m2=float(footprint),
        ))

    return classification, clusters
