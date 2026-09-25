import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter


def classify_ground_csf(
    xyz: np.ndarray,
    cloth_resolution: float = 0.5,
    class_threshold: float = 0.10,
    rigidness: int = 2,
    iterations: int = 500,
    slope_smooth: bool = True,
    time_step: float = 0.65,
    csf_voxel_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify ground using Cloth Simulation Filter.

    CSF runs on a voxel-thinned subset (default 0.15m voxel) then the
    resulting ground surface is interpolated back to all original points
    via a fast grid lookup — O(N) regardless of point count.

    Returns:
        ground_mask  — bool array, True = ground point
        nz           — float array, height above ground for every point
    """
    import CSF

    # ── 1. Voxel thin: keep lowest Z point per voxel cell ────────────────
    thin_idx = _voxel_thin(xyz, csf_voxel_size)
    xyz_thin = xyz[thin_idx]

    # ── 2. Run CSF on thinned cloud ───────────────────────────────────────
    csf = CSF.CSF()
    csf.params.bSloopSmooth = slope_smooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.class_threshold = class_threshold
    csf.params.rigidness = rigidness
    csf.params.interations = iterations
    csf.params.time_step = time_step

    csf.setPointCloud(xyz_thin)

    grp = CSF.VecInt()
    off_grp = CSF.VecInt()
    csf.do_filtering(grp, off_grp, exportCloth=False)

    # ── 3. Build ground surface grid and lookup nZ for every raw point ────
    ground_thin_idx = thin_idx[np.array(grp, dtype=np.int64)]
    ground_pts = xyz[ground_thin_idx]

    # Use grid cell = cloth_resolution (same scale as CSF cloth mesh)
    nz = _compute_nz(xyz, ground_pts, cell_size=cloth_resolution)

    # Original points within class_threshold of surface = ground
    ground_mask = np.abs(nz) <= class_threshold

    return ground_mask, nz


def _voxel_thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """
    Return indices of one point per voxel (the lowest Z in each cell).
    Fully vectorised — no Python loops, O(N log N) sort.
    """
    x_min = float(xyz[:, 0].min())
    y_min = float(xyz[:, 1].min())

    ci = ((xyz[:, 0] - x_min) / voxel).astype(np.int64)
    ri = ((xyz[:, 1] - y_min) / voxel).astype(np.int64)
    cols = int(ci.max()) + 1
    keys = ri * cols + ci

    # Sort by (cell_key, Z) — first occurrence per key is the min-Z point
    order = np.lexsort((xyz[:, 2], keys))
    keys_sorted = keys[order]

    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]

    return order[first].astype(np.int64)


def _compute_nz(
    xyz: np.ndarray,
    ground_pts: np.ndarray,
    cell_size: float = 0.5,
) -> np.ndarray:
    """
    Fast grid-based nZ: rasterise ground_pts onto a grid, fill gaps with
    nearest-cell propagation, then do an O(N) array lookup for every raw point.

    Replaces LinearNDInterpolator which was O(N log M) — unusable on 77M+ pts.
    """
    if len(ground_pts) < 3:
        return xyz[:, 2] - xyz[:, 2].min()

    x_min = min(float(xyz[:, 0].min()), float(ground_pts[:, 0].min()))
    y_min = min(float(xyz[:, 1].min()), float(ground_pts[:, 1].min()))
    x_max = float(xyz[:, 0].max())
    y_max = float(xyz[:, 1].max())

    cols = int(np.ceil((x_max - x_min) / cell_size)) + 2
    rows = int(np.ceil((y_max - y_min) / cell_size)) + 2

    gci = np.clip(((ground_pts[:, 0] - x_min) / cell_size).astype(np.int32), 0, cols - 1)
    gri = np.clip(((ground_pts[:, 1] - y_min) / cell_size).astype(np.int32), 0, rows - 1)

    z_sum   = np.zeros((rows, cols), dtype=np.float64)
    z_count = np.zeros((rows, cols), dtype=np.int32)
    np.add.at(z_sum,   (gri, gci), ground_pts[:, 2])
    np.add.at(z_count, (gri, gci), 1)

    z_grid = np.where(z_count > 0, z_sum / np.maximum(z_count, 1), np.nan)

    # Fill NaN cells so every lookup returns a sensible value
    if np.isnan(z_grid).any():
        z_grid = _fill_nan(z_grid)

    # O(N) lookup for all raw points
    ci = np.clip(((xyz[:, 0] - x_min) / cell_size).astype(np.int32), 0, cols - 1)
    ri = np.clip(((xyz[:, 1] - y_min) / cell_size).astype(np.int32), 0, rows - 1)

    return xyz[:, 2] - z_grid[ri, ci]


def classify_ground(
    xyz: np.ndarray,
    cell_size: float = 1.0,
    ground_threshold: float = 0.30,
    smooth_passes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify ground points using Progressive Grid Minimum.

    Returns:
        ground_mask  — bool array, True = ground point
        nz           — float array, height above ground for every point
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    cols = int(np.ceil((x_max - x_min) / cell_size)) + 1
    rows = int(np.ceil((y_max - y_min) / cell_size)) + 1

    col_idx = np.clip(((x - x_min) / cell_size).astype(np.int32), 0, cols - 1)
    row_idx = np.clip(((y - y_min) / cell_size).astype(np.int32), 0, rows - 1)

    # Build minimum Z grid
    z_min_grid = np.full((rows, cols), np.inf, dtype=np.float64)
    np.minimum.at(z_min_grid, (row_idx, col_idx), z)
    z_min_grid[z_min_grid == np.inf] = np.nan

    # Progressive morphological smoothing to remove buildings/objects
    z_smooth = _progressive_smooth(z_min_grid, smooth_passes)

    # Ensure no NaN remain (possible when smooth_passes == 0)
    if np.isnan(z_smooth).any():
        z_smooth = _fill_nan(z_smooth)

    # O(N) direct grid lookup — no scipy interpolator needed
    ground_z = z_smooth[row_idx, col_idx]

    nz = z - ground_z
    ground_mask = (nz >= -ground_threshold) & (nz <= ground_threshold)

    return ground_mask, nz


def _progressive_smooth(grid: np.ndarray, passes: int) -> np.ndarray:
    """
    Progressive morphological opening: erosion (min) then dilation (max).
    Removes elevated objects (buildings, trees) while preserving ground surface.
    Each pass uses a wider window to handle larger structures.
    """
    result = grid.copy()
    window = 3
    for _ in range(passes):
        nan_mask = np.isnan(result)
        if nan_mask.any():
            filled = _fill_nan(result)
        else:
            filled = result
        eroded = minimum_filter(filled, size=window)
        result = maximum_filter(eroded, size=window)
        window += 2
    return result


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN cells with the nearest non-NaN neighbour using scipy distance transform.
    Much faster than NearestNDInterpolator for dense grids.
    """
    from scipy.ndimage import distance_transform_edt
    valid = ~np.isnan(grid)
    if not valid.any():
        return np.zeros_like(grid)
    # distance_transform_edt + indices gives the nearest-valid-cell for each NaN
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    filled = grid.copy()
    filled[~valid] = grid[nearest[0][~valid], nearest[1][~valid]]
    return filled
