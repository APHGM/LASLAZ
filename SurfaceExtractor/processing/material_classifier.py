"""
Per-point surface-material classifier using RGB (and NIR if reliable) indices.

Assumes source is a photogrammetry point cloud where GROUND has already been
classified (class 2). We only look at those points to decide what MATERIAL
each ground point represents.

Materials (custom class codes used internally, mapped to DXF layers on export):
    0  UNKNOWN / SKIP
    1  GRASS               — vegetation cover, playing fields, verges
    2  ASPHALT / TARMAC    — dark low-saturation hard surface
    3  CONCRETE            — light low-saturation hard surface
    4  PAINTED             — very bright markings (parking lines, arrows, text)
    5  OTHER               — mixed / ambiguous
"""

import numpy as np
from dataclasses import dataclass


# Material codes (internal — not ASPRS)
MAT_UNKNOWN  = 0
MAT_GRASS    = 1
MAT_ASPHALT  = 2
MAT_CONCRETE = 3
MAT_PAINTED  = 4
MAT_OTHER    = 5

MATERIAL_NAMES = {
    MAT_UNKNOWN:  "UNKNOWN",
    MAT_GRASS:    "GRASS",
    MAT_ASPHALT:  "ASPHALT_TARMAC",
    MAT_CONCRETE: "CONCRETE",
    MAT_PAINTED:  "PAINTED",
    MAT_OTHER:    "OTHER",
}


@dataclass
class MaterialParams:
    # Vegetation
    vari_grass_threshold: float = 0.10       # VARI > this → grass
    ndvi_grass_threshold: float = 0.20       # NDVI > this → grass (if NIR reliable)
    use_ndvi_if_available: bool = False      # False: safer default — many drone
                                             #        NIR bands are unreliable
    # Brightness thresholds (0–65535 → normalised to 0–255)
    dark_asphalt_max: float   = 100.0        # I < this → asphalt candidate
    light_concrete_min: float = 160.0        # I > this → concrete candidate
    painted_min: float        = 220.0        # I > this + high sat → painted
    # Saturation (0–1): hard surfaces are LOW saturation, painted lines vary
    hardsurf_max_sat: float   = 0.20         # asphalt/concrete < this
    painted_min_sat: float    = 0.15         # painted markings usually saturated


def _to_0_255(band: np.ndarray) -> np.ndarray:
    """Normalise laspy RGB (uint16 0–65535) to a 0–255 float array."""
    b = band.astype(np.float32)
    # laspy uint16 range → divide by 257 to hit 0–255
    return np.clip(b / 257.0, 0, 255)


def _compute_hsv_i_s(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    """Return (intensity, saturation) in normalised space, avoiding div-by-zero."""
    r = R / 255.0
    g = G / 255.0
    b = B / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    intensity = (r + g + b) / 3.0 * 255.0           # 0–255
    sat = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)  # 0–1
    return intensity.astype(np.float32), sat.astype(np.float32)


def _vari(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    r = R.astype(np.float32)
    g = G.astype(np.float32)
    b = B.astype(np.float32)
    denom = g + r - b
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(denom) > 1e-6, (g - r) / denom, 0.0)
    return np.clip(v, -2.0, 2.0).astype(np.float32)


def _ndvi(R: np.ndarray, NIR: np.ndarray) -> np.ndarray:
    r = R.astype(np.float32)
    n = NIR.astype(np.float32)
    denom = n + r
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom > 0, (n - r) / denom, 0.0)
    return v.astype(np.float32)


def classify_materials(
    R: np.ndarray, G: np.ndarray, B: np.ndarray,
    NIR: np.ndarray | None = None,
    params: MaterialParams | None = None,
) -> np.ndarray:
    """
    Return an int8 array of material codes, one per input point.
    R, G, B, (NIR): uint16 arrays from laspy (0–65535 range).
    """
    if params is None:
        params = MaterialParams()

    R8 = _to_0_255(R)
    G8 = _to_0_255(G)
    B8 = _to_0_255(B)

    intensity, sat = _compute_hsv_i_s(R8, G8, B8)
    vari = _vari(R8, G8, B8)

    n = len(R)
    materials = np.full(n, MAT_OTHER, dtype=np.int8)

    # ── Grass — vegetation index first ─────────────────────────────────
    if NIR is not None and params.use_ndvi_if_available:
        ndvi = _ndvi(R, NIR)
        grass_mask = ndvi > params.ndvi_grass_threshold
    else:
        grass_mask = vari > params.vari_grass_threshold
    materials[grass_mask] = MAT_GRASS

    # Below rules only apply to non-grass points
    non_grass = ~grass_mask

    # ── Painted markings — very bright + saturated ─────────────────────
    painted_mask = non_grass & (intensity > params.painted_min) & (sat > params.painted_min_sat)
    materials[painted_mask] = MAT_PAINTED

    # ── Concrete — light, low saturation ───────────────────────────────
    concrete_mask = (
        non_grass & ~painted_mask &
        (intensity > params.light_concrete_min) &
        (sat < params.hardsurf_max_sat)
    )
    materials[concrete_mask] = MAT_CONCRETE

    # ── Asphalt/Tarmac — dark, low saturation ──────────────────────────
    asphalt_mask = (
        non_grass & ~painted_mask & ~concrete_mask &
        (intensity < params.dark_asphalt_max) &
        (sat < params.hardsurf_max_sat)
    )
    materials[asphalt_mask] = MAT_ASPHALT

    # Everything else stays MAT_OTHER
    return materials


def summarise(materials: np.ndarray) -> dict[str, int]:
    """Count per-class point totals for logging."""
    return {
        MATERIAL_NAMES[code]: int((materials == code).sum())
        for code in (MAT_GRASS, MAT_ASPHALT, MAT_CONCRETE,
                     MAT_PAINTED, MAT_OTHER, MAT_UNKNOWN)
    }
