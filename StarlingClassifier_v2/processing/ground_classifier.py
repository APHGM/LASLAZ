import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
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
    resulting ground surface is interpolated back to all original points.
    This keeps CSF fast regardless of raw point density.

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

    # ── 3. Interpolate ground surface back to all points ──────────────────
    ground_thin_idx = thin_idx[np.array(grp, dtype=np.int64)]
    ground_pts = xyz[ground_thin_idx]

    nz = _compute_nz(xyz, ground_pts)

    # Original points within class_threshold of surface = ground
    ground_mask = np.abs(nz) <= class_threshold

    return ground_mask, nz


def _voxel_thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """
    Return indices of one point per voxel (the lowest Z in each cell).
    Much faster than random sampling — keeps ground-level points.
    """
    x_min = xyz[:, 0].min()
    y_min = xyz[:, 1].min()

    ci = ((xyz[:, 0] - x_min) / voxel).astype(np.int32)
    ri = ((xyz[:, 1] - y_min) / voxel).astype(np.int32)

    # Unique cell key
    cols = int(ci.max()) + 1
    keys = ri * cols + ci

    # For each cell keep the point with lowest Z
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    z_sorted = xyz[order, 2]

    _, first_occ = np.unique(keys_sorted, return_index=True)

    # Among duplicates per cell, pick lowest Z
    # (unique already gives first; re-sort by Z within each cell via argmin)
    keep = []
    boundaries = np.concatenate([first_occ, [len(keys_sorted)]])
    for i in range(len(first_occ)):
        seg = slice(boundaries[i], boundaries[i + 1])
        local_min = np.argmin(z_sorted[seg])
        keep.append(order[boundaries[i] + local_min])

    return np.array(keep, dtype=np.int64)


def _compute_nz(xyz: np.ndarray, ground_pts: np.ndarray) -> np.ndarray:
    """Interpolate ground surface from classified ground points and return nZ."""
    if len(ground_pts) < 3:
        return xyz[:, 2] - xyz[:, 2].min()

    lin = LinearNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])
    ground_z = lin(xyz[:, :2])

    outside = np.isnan(ground_z)
    if outside.any():
        near = NearestNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])
        ground_z[outside] = near(xyz[outside, :2])

    return xyz[:, 2] - ground_z


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

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    cols = int(np.ceil((x_max - x_min) / cell_size)) + 1
    rows = int(np.ceil((y_max - y_min) / cell_size)) + 1

    col_idx = ((x - x_min) / cell_size).astype(np.int32)
    row_idx = ((y - y_min) / cell_size).astype(np.int32)

    # Build minimum Z grid
    z_min_grid = np.full((rows, cols), np.inf, dtype=np.float64)
    np.minimum.at(z_min_grid, (row_idx, col_idx), z)

    # Replace empty cells with NaN
    z_min_grid[z_min_grid == np.inf] = np.nan

    # Progressive morphological smoothing to remove buildings/objects
    z_smooth = _progressive_smooth(z_min_grid, smooth_passes)

    # Interpolate ground surface at every point location
    ground_z = _interpolate_surface(z_smooth, x_min, y_min, cell_size, x, y)

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
        # Erosion: push surface down past buildings/trees
        eroded = minimum_filter(filled, size=window)
        # Dilation: fill back up to ground level (buildings don't come back)
        result = maximum_filter(eroded, size=window)
        window += 2
    return result


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells with nearest non-NaN value."""
    rows, cols = np.indices(grid.shape)
    valid = ~np.isnan(grid)
    if not valid.any():
        return np.zeros_like(grid)
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(
        np.stack([rows[valid], cols[valid]], axis=1),
        grid[valid]
    )
    filled = grid.copy()
    filled[~valid] = interp(
        np.stack([rows[~valid], cols[~valid]], axis=1)
    )
    return filled


def _interpolate_surface(
    z_grid: np.ndarray,
    x_min: float, y_min: float,
    cell_size: float,
    x_pts: np.ndarray, y_pts: np.ndarray
) -> np.ndarray:
    """Interpolate the ground raster at arbitrary point locations."""
    rows, cols = np.indices(z_grid.shape)
    valid = ~np.isnan(z_grid)

    gx = x_min + cols[valid] * cell_size
    gy = y_min + rows[valid] * cell_size
    gz = z_grid[valid]

    # Linear interpolation inside convex hull, nearest outside
    lin = LinearNDInterpolator(np.stack([gx, gy], axis=1), gz)
    ground_z = lin(np.stack([x_pts, y_pts], axis=1))

    # Fill points outside convex hull with nearest
    outside = np.isnan(ground_z)
    if outside.any():
        near = NearestNDInterpolator(np.stack([gx, gy], axis=1), gz)
        ground_z[outside] = near(np.stack([x_pts[outside], y_pts[outside]], axis=1))

    return ground_z
