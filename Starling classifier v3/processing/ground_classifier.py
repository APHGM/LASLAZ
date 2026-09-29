import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

# ── Streaming threshold ───────────────────────────────────────────────────────
# Files larger than this use the 2-pass streaming path instead of loading
# everything into RAM. 30 M points ≈ 720 MB raw XYZ — comfortably below the
# ~2 GB working headroom most machines can spare without eviction pressure.
STREAMING_POINT_THRESHOLD = 30_000_000


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


# ── Streaming path (large files / low RAM) ───────────────────────────────────

def nz_from_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_grid: np.ndarray,
    x_min: float,
    y_min: float,
    cell_size: float,
) -> np.ndarray:
    """
    Per-chunk nZ lookup against a pre-built 2-D ground raster.
    Uses bilinear interpolation between the 4 surrounding grid cells so
    the ground surface is smooth rather than stepped at cell boundaries.
    O(N) — safe to call on huge arrays.
    """
    rows, cols = z_grid.shape

    # Fractional grid coordinates
    fx_raw = (x - x_min) / cell_size
    fy_raw = (y - y_min) / cell_size

    # Floor cell (top-left of the 2×2 neighbourhood)
    ci0 = np.clip(fx_raw.astype(np.int32), 0, cols - 2)
    ri0 = np.clip(fy_raw.astype(np.int32), 0, rows - 2)
    ci1 = ci0 + 1
    ri1 = ri0 + 1

    # Sub-cell fractions (0..1)
    tx = fx_raw - ci0
    ty = fy_raw - ri0

    # Bilinear blend of the four corners
    z_interp = (z_grid[ri0, ci0] * (1.0 - tx) * (1.0 - ty) +
                z_grid[ri0, ci1] * tx          * (1.0 - ty) +
                z_grid[ri1, ci0] * (1.0 - tx) * ty          +
                z_grid[ri1, ci1] * tx          * ty)

    return z - z_interp


def classify_ground_csf_streaming(
    file_path: str,
    extra_pts: "np.ndarray | None" = None,
    cloth_resolution: float = 0.30,
    class_threshold: float = 0.05,
    rigidness: int = 1,
    iterations: int = 1000,
    slope_smooth: bool = True,
    time_step: float = 0.65,
    csf_voxel_size: float = 0.10,
    chunk_size: int = 5_000_000,
    z_range_max: float = 0.0,
    log_fn=None,
) -> tuple:
    """
    Two-pass streaming ground classification — never loads the full point cloud.

    Pass 1 (this function):
        Stream the file in chunks of `chunk_size` points.
        For each chunk update a tiny 2-D grid (rows × cols cells) that tracks
        the LOWEST and HIGHEST Z seen per voxel cell.
        After all chunks: reject cells whose Z range exceeds z_range_max (vertical
        surfaces such as walls span several metres; flat ground spans < 0.1 m).
        Remaining cells (~1 M ground candidates) → CSF → 2-D ground-elevation raster.

    Pass 2 (caller):
        Use `nz_from_grid()` to classify each new chunk in O(N).

    Peak RAM this function needs:
        grid (30 MB for 95 m × 95 m @ 0.10 m) + one chunk (~120 MB) ≈ 150 MB.

    Args:
        z_range_max: maximum Z spread (m) within a voxel cell to be considered a
            ground candidate.  0.0 = disabled (all cells passed to CSF).
            Recommended: 0.50 m for SLAM / indoor-outdoor scans.

    Returns: (z_grid, x_min, y_min, cell_size)
        z_grid    — 2-D float64 raster of ground elevation
        x_min/y_min — origin of the raster (metres)
        cell_size — raster pixel size (= cloth_resolution)
    """
    import laspy
    import CSF

    # ── Read bounds from header (no points loaded) ────────────────────────
    with laspy.open(file_path) as f:
        h = f.header
        x_min, y_min = float(h.x_min), float(h.y_min)
        x_max, y_max = float(h.x_max), float(h.y_max)
        pt_count     = h.point_count

    if extra_pts is not None and len(extra_pts):
        x_min = min(x_min, float(extra_pts[:, 0].min()))
        y_min = min(y_min, float(extra_pts[:, 1].min()))
        x_max = max(x_max, float(extra_pts[:, 0].max()))
        y_max = max(y_max, float(extra_pts[:, 1].max()))

    # ── Voxel grid — tracks min_z, max_z, and representative x/y per cell ─
    v = csf_voxel_size
    cols_v = int(np.ceil((x_max - x_min) / v)) + 1
    rows_v = int(np.ceil((y_max - y_min) / v)) + 1
    use_zrange = z_range_max > 0.0
    grid_mb = rows_v * cols_v * (32 if use_zrange else 24) / 1e6

    if log_fn:
        n_chunks = (pt_count + chunk_size - 1) // chunk_size
        zr_tag = f", z_range_max={z_range_max:.2f} m" if use_zrange else ""
        log_fn(f"  Streaming voxel thin: {rows_v}×{cols_v} grid = {grid_mb:.0f} MB  "
               f"({pt_count:,} pts in {n_chunks} chunks of {chunk_size:,}{zr_tag})")

    min_z  = np.full((rows_v, cols_v), np.inf,   dtype=np.float64)
    best_x = np.zeros((rows_v, cols_v),           dtype=np.float64)
    best_y = np.zeros((rows_v, cols_v),           dtype=np.float64)
    if use_zrange:
        max_z = np.full((rows_v, cols_v), -np.inf, dtype=np.float64)

    def _update_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """
        Update per-cell min_z (and optionally max_z) for a batch of points.
        Within the batch, resolve ties per voxel via argsort so the lowest-Z
        point per cell wins.
        """
        ci = np.clip(((x - x_min) / v).astype(np.int32), 0, cols_v - 1)
        ri = np.clip(((y - y_min) / v).astype(np.int32), 0, rows_v - 1)

        # Within this batch: keep the lowest-Z point per cell
        flat     = ri.astype(np.int64) * cols_v + ci
        order    = np.argsort(z)
        f_sorted = flat[order]
        _, first = np.unique(f_sorted, return_index=True)
        idx      = order[first]
        u_ri, u_ci = ri[idx], ci[idx]
        u_z, u_x, u_y = z[idx], x[idx], y[idx]

        upd = u_z < min_z[u_ri, u_ci]
        min_z [u_ri[upd], u_ci[upd]] = u_z[upd]
        best_x[u_ri[upd], u_ci[upd]] = u_x[upd]
        best_y[u_ri[upd], u_ci[upd]] = u_y[upd]

        # Track max Z (all points, not just per-cell winners) for wall detection
        if use_zrange:
            _, first_all = np.unique(flat, return_index=True)
            # Use np.maximum.at for full per-cell maximum across all batch points
            np.maximum.at(max_z, (ri, ci), z)

    # ── Stream: neighbour buffer first, then main file ────────────────────
    if extra_pts is not None and len(extra_pts):
        _update_grid(extra_pts[:, 0], extra_pts[:, 1], extra_pts[:, 2])

    done = 0
    with laspy.open(file_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            _update_grid(
                np.asarray(chunk.x, dtype=np.float64),
                np.asarray(chunk.y, dtype=np.float64),
                np.asarray(chunk.z, dtype=np.float64),
            )
            done += 1
            if log_fn and done % 5 == 0:
                log_fn(f"    Voxel thin: chunk {done} done")

    # ── Extract ground candidates (one per voxel cell) ────────────────────
    valid = min_z < np.inf

    # Wall/vertical-surface filter: cells whose Z range exceeds z_range_max are
    # vertical structures (walls, columns, door frames).  Ground and floor cells
    # span only a few centimetres of noise; wall cells span 1–4 m.
    if use_zrange:
        z_range = max_z - min_z
        wall_mask = z_range[valid] >= z_range_max
        flat_mask = ~wall_mask
        bx = best_x[valid][flat_mask]
        by = best_y[valid][flat_mask]
        mz = min_z  [valid][flat_mask]
        n_wall = int(wall_mask.sum())
        if log_fn:
            log_fn(f"  Z-range filter (>{z_range_max:.2f} m): removed {n_wall:,} wall/vertical cells")
        del max_z, z_range, wall_mask, flat_mask
    else:
        bx = best_x[valid]
        by = best_y[valid]
        mz = min_z[valid]

    candidates = np.column_stack([bx, by, mz])
    n_cand     = len(candidates)
    if log_fn:
        log_fn(f"  {n_cand:,} ground candidates ({100*n_cand/pt_count:.1f}% of file)")

    if n_cand < 3:
        raise RuntimeError("Too few ground candidates after streaming voxel thin")

    # Free the large grid arrays before CSF
    del min_z, best_x, best_y, valid, bx, by, mz

    # ── CSF on candidates ─────────────────────────────────────────────────
    if log_fn:
        log_fn(f"  Running CSF on {n_cand:,} candidates ...")
    csf = CSF.CSF()
    csf.params.bSloopSmooth    = slope_smooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.class_threshold  = class_threshold
    csf.params.rigidness        = rigidness
    csf.params.interations      = iterations
    csf.params.time_step        = time_step
    csf.setPointCloud(candidates)
    grp = CSF.VecInt(); off_grp = CSF.VecInt()
    csf.do_filtering(grp, off_grp, exportCloth=False)

    ground_pts = candidates[np.array(grp, dtype=np.int64)]
    if log_fn:
        log_fn(f"  CSF: {len(ground_pts):,} ground points identified")
    del candidates

    # ── Build 2-D ground elevation raster ─────────────────────────────────
    cell  = cloth_resolution
    cols  = int(np.ceil((x_max - x_min) / cell)) + 2
    rows  = int(np.ceil((y_max - y_min) / cell)) + 2

    gci   = np.clip(((ground_pts[:, 0] - x_min) / cell).astype(np.int32), 0, cols - 1)
    gri   = np.clip(((ground_pts[:, 1] - y_min) / cell).astype(np.int32), 0, rows - 1)
    z_sum = np.zeros((rows, cols), dtype=np.float64)
    z_cnt = np.zeros((rows, cols), dtype=np.int32)
    np.add.at(z_sum, (gri, gci), ground_pts[:, 2])
    np.add.at(z_cnt, (gri, gci), 1)

    z_grid = np.where(z_cnt > 0, z_sum / np.maximum(z_cnt, 1), np.nan)
    if np.isnan(z_grid).any():
        z_grid = _fill_nan(z_grid)

    if log_fn:
        log_fn(f"  Ground raster: {rows}×{cols} cells @ {cell} m/cell  "
               f"({rows*cols*8/1e6:.0f} MB)")

    return z_grid, x_min, y_min, cell
