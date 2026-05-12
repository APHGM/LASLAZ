import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree


def _get_grid_seeds(xyz, cell_size):
    """Find lowest point in each grid cell."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, y_min = x.min(), y.min()
    
    cols = int(np.ceil((x.max() - x_min) / cell_size)) + 1
    col_idx = ((x - x_min) / cell_size).astype(np.int32)
    row_idx = ((y - y_min) / cell_size).astype(np.int32)
    
    flat_idx = row_idx * cols + col_idx
    order = np.lexsort((z, flat_idx))
    sorted_flat_idx = flat_idx[order]
    
    _, first_occ = np.unique(sorted_flat_idx, return_index=True)
    return xyz[order[first_occ]]


def _interpolate_surface(seeds, x, y):
    """Create a surface from seeds and interpolate at x, y."""
    if len(seeds) < 3:
        return np.full_like(x, seeds[:, 2].min() if len(seeds) > 0 else 0)
        
    lin = LinearNDInterpolator(seeds[:, :2], seeds[:, 2])
    z_surf = lin(x, y)
    
    nan_mask = np.isnan(z_surf)
    if nan_mask.any():
        near = NearestNDInterpolator(seeds[:, :2], seeds[:, 2])
        z_surf[nan_mask] = near(x[nan_mask], y[nan_mask])
    return z_surf


def classify_ground(
    xyz: np.ndarray,
    cell_size: float = 0.25,
    ground_threshold: float = 0.15,
    object_sensitivity: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Multi-Level Ground Classifier (TerraScan Style).
    
    Processes the cloud at 3 scales to 'grow' the ground surface:
    1. Coarse (10m): Establishes absolute base ground.
    2. Medium (2.5m): Captures major terrain features.
    3. Fine (User): Captures high-detail surface.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # --- LEVEL 1: Coarse (Base Ground) ---
    # Find base ground on 10m grid - effectively ignores all buildings
    seeds_l1 = _get_grid_seeds(xyz, 10.0)
    surf_l1 = _interpolate_surface(seeds_l1, x, y)
    
    # --- LEVEL 2: Medium (Terrain Features) ---
    # Find terrain features on 2.5m grid
    # Accept points only if they are within a reasonable distance of L1 base
    seeds_l2_raw = _get_grid_seeds(xyz, 2.5)
    l2_ref_z = _interpolate_surface(seeds_l1, seeds_l2_raw[:, 0], seeds_l2_raw[:, 1])
    # Allow 2m vertical deviation for L2 - captures hills but not roofs
    valid_l2 = seeds_l2_raw[:, 2] < l2_ref_z + 2.0 
    seeds_l2 = seeds_l2_raw[valid_l2]
    
    if len(seeds_l2) < 5: seeds_l2 = seeds_l1 # Fallback
    surf_l2 = _interpolate_surface(seeds_l2, x, y)
    
    # --- LEVEL 3: Fine (Detail Capture) ---
    # Final refinement on user-specified grid
    seeds_l3_raw = _get_grid_seeds(xyz, cell_size)
    l3_ref_z = _interpolate_surface(seeds_l2, seeds_l3_raw[:, 0], seeds_l3_raw[:, 1])
    
    # Adaptive sensitivity: tighter on flat ground, relaxed on L2 features
    # Here we use the user's object_sensitivity
    valid_l3 = seeds_l3_raw[:, 2] < l3_ref_z + object_sensitivity
    seeds_l3 = seeds_l3_raw[valid_l3]
    
    if len(seeds_l3) < 5: seeds_l3 = seeds_l2 # Fallback
    
    # --- FINAL ADAPTIVE CLASSIFICATION ---
    final_surf = _interpolate_surface(seeds_l3, x, y)
    nz = z - final_surf
    
    # Adaptive threshold: allow slightly more room if the local neighborhood is rough
    # (Simplified: just use user threshold but ensure it's robust)
    ground_mask = (nz >= -0.10) & (nz <= ground_threshold)
    
    return ground_mask, nz


def classify_ground_csf(*args, **kwargs):
    return None, None
