"""
Height-based classification — mirrors TerraScan macro behaviour.

Applied AFTER bird detection so birds keep their class 20 label.
Uses nZ (height above ground) which is already computed during
the ground classification stage.

ASPRS classes assigned:
    2  ground          (unchanged)
    3  low veg         (veg_low_min ≤ nZ < veg_low_max)
    4  med veg         (veg_low_max ≤ nZ < veg_med_max)
    5  high veg        (nZ ≥ veg_med_max)
    7  low noise       (nZ ≤ noise_below_ground, i.e. below surface)
    8  model key       (thinned ground sample — for TIN building)
    12 probable ground (just beyond ground threshold — uncertain, review manually)
    20 bird contact    (set by bird detector before us)
"""

import numpy as np


# Class codes
CLASS_UNCLASSIFIED    = 1
CLASS_GROUND          = 2
CLASS_LOW_VEG         = 3
CLASS_MED_VEG         = 4
CLASS_HIGH_VEG        = 5
CLASS_LOW_NOISE       = 7
CLASS_MODEL_KEY       = 8
CLASS_PROBABLE_GROUND = 12
CLASS_BIRD_CONTACT    = 20


def classify_heights(
    classification: np.ndarray,
    nz: np.ndarray,
    veg_low_min: float = 0.10,
    veg_low_max: float = 1.00,
    veg_med_max: float = 3.00,
    noise_below_ground: float = -0.10,
    probable_ground_band: float = 0.0,
    ground_class_thr: float = 0.05,
    protect_classes: tuple[int, ...] = (CLASS_GROUND, CLASS_BIRD_CONTACT),
) -> np.ndarray:
    """
    Apply height-band vegetation + noise classification.

    Points already labelled as `protect_classes` are NOT overwritten —
    so ground stays ground, birds stay birds. Everything else above ground
    gets veg classes by nZ; everything below ground becomes low noise.

    When probable_ground_band > 0, points within (ground_class_thr,
    ground_class_thr + probable_ground_band) of the surface on both sides
    are tagged class 12 (probable ground) instead of veg/noise/unclassified.

    Returns the updated classification array (modified in place + returned).
    """
    classification = classification.copy()
    protected = np.isin(classification, protect_classes)
    candidates = ~protected

    finite = np.isfinite(nz)
    cand = candidates & finite

    # Probable ground band (class 12) — beyond confident ground on both sides
    if probable_ground_band > 0.0:
        pg = cand & (np.abs(nz) <= ground_class_thr + probable_ground_band)
        classification[pg] = CLASS_PROBABLE_GROUND
        # Protect probable-ground from veg/noise overwrite below
        protected2 = protected | pg
        cand = (~protected2) & finite

    # Low noise — below ground surface (never overwrites ground or probable-ground)
    low_noise = cand & (nz <= noise_below_ground)
    classification[low_noise] = CLASS_LOW_NOISE

    # Low vegetation
    low_veg = cand & (nz >= veg_low_min) & (nz < veg_low_max)
    classification[low_veg] = CLASS_LOW_VEG

    # Medium vegetation
    med_veg = cand & (nz >= veg_low_max) & (nz < veg_med_max)
    classification[med_veg] = CLASS_MED_VEG

    # High vegetation
    high_veg = cand & (nz >= veg_med_max)
    classification[high_veg] = CLASS_HIGH_VEG

    return classification


def classify_model_keypoints(
    xyz: np.ndarray,
    classification: np.ndarray,
    step_m: float = 8.0,
    from_class: int = CLASS_GROUND,
    to_class: int = CLASS_MODEL_KEY,
) -> np.ndarray:
    """
    Thin the ground class into a 'model key' subset suitable for TIN building.
    For each step_m × step_m cell, the LOWEST ground point is kept as a
    model keypoint; other ground points stay class 2.

    Equivalent to TerraScan's FnScanClassifyModelKey for a uniform grid.
    Returns the updated classification array.
    """
    classification = classification.copy()
    ground_mask = classification == from_class
    if not ground_mask.any():
        return classification

    g_idx = np.where(ground_mask)[0]
    pts = xyz[g_idx]

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    col = ((pts[:, 0] - x_min) / step_m).astype(np.int64)
    row = ((pts[:, 1] - y_min) / step_m).astype(np.int64)
    cols = int(col.max()) + 1
    key = row * cols + col

    # For each cell, find index of point with min Z
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    z_sorted = pts[order, 2]

    _, first = np.unique(key_sorted, return_index=True)
    boundaries = np.concatenate([first, [len(key_sorted)]])

    keepers_local = []
    for i in range(len(first)):
        seg = slice(boundaries[i], boundaries[i + 1])
        local_min = np.argmin(z_sorted[seg])
        keepers_local.append(order[boundaries[i] + local_min])
    keepers_local = np.array(keepers_local, dtype=np.int64)

    # Map back to original indexing and re-label
    keeper_global = g_idx[keepers_local]
    classification[keeper_global] = to_class
    return classification


def ground_mask_from_classification(
    source_classification: np.ndarray,
    ground_class: int = CLASS_GROUND,
    extra_ground_classes: tuple[int, ...] = (CLASS_MODEL_KEY,),
) -> np.ndarray:
    """
    Build a ground mask from an already-classified source.
    Treats both class 2 (ground) and class 8 (model key) as ground.
    """
    classes = (ground_class,) + tuple(extra_ground_classes)
    return np.isin(source_classification, classes)
