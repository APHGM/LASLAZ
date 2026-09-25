import laspy
import numpy as np
from pathlib import Path


def read_laz(filepath: str | Path) -> tuple[np.ndarray, object]:
    """Read LAZ/LAS file. Returns (Nx3 xyz array, laspy header)."""
    las = laspy.read(str(filepath))
    xyz = np.stack([
        las.x.copy(),
        las.y.copy(),
        las.z.copy()
    ], axis=1)
    return xyz, las


def read_color_bands(source_las) -> dict[str, np.ndarray | None]:
    """
    Return a dict of colour-band arrays present in the source.

    Keys always present (None when the band doesn't exist in the file):
        'red'   – uint16 array or None
        'green' – uint16 array or None
        'blue'  – uint16 array or None
        'nir'   – uint16 array or None  (only in point formats 8 and 10)

    Values are returned as-is from laspy (typically 0–65535).
    """
    bands: dict[str, np.ndarray | None] = {
        "red": None, "green": None, "blue": None, "nir": None
    }
    for band_name in ("red", "green", "blue", "nir"):
        try:
            arr = getattr(source_las, band_name)
            if arr is not None and len(arr) > 0:
                bands[band_name] = np.asarray(arr)
        except (AttributeError, ValueError):
            pass
    return bands


def has_rgb(bands: dict) -> bool:
    return all(bands.get(c) is not None for c in ("red", "green", "blue"))


def has_nir(bands: dict) -> bool:
    return bands.get("nir") is not None


# ────────────────────────────────────────────────────────────────────────────
# Auto-detect helpers (v2.1)
# ────────────────────────────────────────────────────────────────────────────

def estimate_optimal_tile_size(
    filepath: str | Path,
    target_pts_per_tile: int = 30_000_000,
    min_tile_m: float = 10.0,
    max_tile_m: float = 200.0,
    round_to_m: float = 5.0,
) -> tuple[float, dict]:
    """
    Compute a memory-safe tile size in metres from point density.

    Reads only the LAS/LAZ header — very fast, no point loading.
    Returns (tile_size_m, info_dict) where info_dict has diagnostics.
    """
    import math
    with laspy.open(str(filepath)) as r:
        h = r.header
        pc = int(h.point_count)
        dx = float(h.maxs[0] - h.mins[0])
        dy = float(h.maxs[1] - h.mins[1])

    area = max(dx * dy, 1.0)
    density = pc / area if pc > 0 else 0.0

    if density <= 0:
        tile_size = 100.0
    else:
        tile_area = target_pts_per_tile / density
        tile_size = math.sqrt(tile_area)
        tile_size = round(tile_size / round_to_m) * round_to_m
        tile_size = max(min_tile_m, min(max_tile_m, tile_size))

    return float(tile_size), {
        "point_count": pc,
        "area_m2": area,
        "density_pts_m2": density,
        "target_pts_per_tile": target_pts_per_tile,
        "chosen_tile_size_m": tile_size,
    }


def guess_scan_type(
    filepath: str | Path,
    sample_size: int = 100_000,
) -> tuple[str, str, dict]:
    """
    Determine whether a LAS/LAZ is likely LiDAR or drone photogrammetry.

    Reads only the first sample_size points from the header + first chunk —
    never loads the whole file, so it is safe even on billion-point files.

    Returns (source_type, reason, diagnostics_dict).
    source_type ∈ {"lidar", "photogrammetry", "unknown"}.
    """
    with laspy.open(str(filepath)) as r:
        n_total = int(r.header.point_count)
        info: dict = {"point_count": n_total}
        if n_total == 0:
            return "unknown", "empty point cloud", info
        # Read only the first min(sample_size, n_total) points
        n_read = min(sample_size, n_total)
        chunk = r.read_points(n_read)

    # ── Intensity ──
    intensity = None
    try:
        intensity = np.asarray(chunk.intensity, dtype=np.float32)
    except (AttributeError, ValueError):
        intensity = None

    # ── NIR ──
    nir_present = False
    try:
        nir = np.asarray(chunk["nir"])
        nir_present = len(nir) > 0 and int(nir.max()) > 0
    except (AttributeError, KeyError, ValueError):
        nir_present = False

    # ── RGB ──
    rgb_present = False
    r = g = b = None
    try:
        r = np.asarray(chunk.red, dtype=np.float32)
        g = np.asarray(chunk.green, dtype=np.float32)
        b = np.asarray(chunk.blue, dtype=np.float32)
        rgb_present = (
            len(r) > 0 and
            (float(r.max()) > 0 or float(g.max()) > 0 or float(b.max()) > 0)
        )
    except (AttributeError, ValueError):
        rgb_present = False

    info["has_nir"] = nir_present
    info["has_rgb"] = rgb_present

    # ── Intensity-based decision ──
    if intensity is None or len(intensity) == 0:
        info["intensity_status"] = "missing"
        # No intensity at all + RGB present → photogrammetry
        if rgb_present:
            return "photogrammetry", "no intensity field, RGB present", info
        return "unknown", "no intensity field and no RGB", info

    n_unique = int(len(np.unique(intensity)))
    std = float(intensity.std())
    mean_v = float(intensity.mean())
    info.update({
        "intensity_unique": n_unique,
        "intensity_std": std,
        "intensity_mean": mean_v,
    })

    if n_unique <= 1:
        return "photogrammetry", f"intensity is constant (value={mean_v:.0f})", info
    if std < 5.0:
        return "photogrammetry", f"very narrow intensity range (std={std:.1f})", info

    # If we have RGB, test correlation with intensity (photogrammetry copies
    # RGB brightness into intensity in some pipelines)
    if rgb_present and r is not None:
        try:
            rgb_bright = (r + g + b) / 3.0
            # Guard against constant arrays that produce NaN
            if intensity.std() > 0 and rgb_bright.std() > 0:
                corr = float(np.corrcoef(intensity, rgb_bright)[0, 1])
                info["intensity_vs_rgb_corr"] = corr
                if abs(corr) > 0.95:
                    return ("photogrammetry",
                            f"intensity correlates with RGB brightness (r={corr:.2f})",
                            info)
        except Exception:
            pass

    if std > 50:
        return ("lidar",
                f"varied intensity (std={std:.0f}, mean={mean_v:.0f})",
                info)
    return ("unknown",
            f"marginal intensity variability (std={std:.1f})",
            info)


def read_laz_bbox(filepath: str | Path, xmin, xmax, ymin, ymax) -> np.ndarray:
    """Read only points within a bounding box (for buffer loading)."""
    xyz, _ = read_laz(filepath)
    mask = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
    )
    return xyz[mask]


def write_classified_laz(
    source_las,
    xyz: np.ndarray,
    classification: np.ndarray,
    out_path: str | Path,
    output_format: str = "laz",
    las_version: str = "1.4",
) -> Path:
    """Write output LAS/LAZ with classification field.

    Returns the actual written file path so the caller can log it.
    Tolerates GUI display labels like 'LAZ (compressed)' or 'LAS (uncompressed)'.
    Raises a clear error if LAZ write fails because the lazrs backend is missing.
    """
    # Tolerate display labels from the GUI like "LAZ (compressed)"
    fmt_norm = (output_format or "laz").strip().lower()
    fmt = "laz" if "laz" in fmt_norm else "las"

    # Normalise version — "match" means use the source's version
    ver_norm = (las_version or "1.4").strip().lower()
    if ver_norm == "match":
        src_ver = source_las.header.version
        ver_norm = f"{src_ver.major}.{src_ver.minor}"
    elif not ver_norm.startswith("1."):
        ver_norm = "1.4"

    # Build header preserving point format ID, extra dims, and VLRs
    # (so RGB / GPS time / scanner metadata survive)
    header = laspy.LasHeader(
        point_format=source_las.header.point_format.id,   # use ID, not object
        version=ver_norm,
    )
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales

    # Preserve extra dimensions defined on the source
    for ed in getattr(source_las.header.point_format, "extra_dimensions", []):
        try:
            header.add_extra_dim(ed)
        except Exception:
            pass

    # Preserve VLRs (RGB metadata, projection info, scanner custom blocks)
    try:
        header.vlrs.extend(source_las.header.vlrs)
    except Exception:
        pass

    out = laspy.LasData(header=header)

    # ── Preserve EVERY field by copying the packed point record ──────────
    # This single line keeps X/Y/Z, intensity, RGB, GPS time, return number,
    # scan angle, etc. — anything the source has. We DON'T re-set X/Y/Z
    # because they're already correct (xyz came from this same source).
    # Re-assigning out.x = ... after out.points = ... breaks laspy's
    # internal property bindings (lowercase x vs uppercase X field).
    if len(classification) != len(source_las.points):
        raise ValueError(
            f"classification length ({len(classification)}) does not match "
            f"source point count ({len(source_las.points)})."
        )
    out.points = source_las.points.copy()
    out.classification = classification.astype(np.uint8)

    # ── Back-fill LAS 1.2/1.3 legacy point-count fields ───────────────────
    # laspy 2.6.x writes the extended (LAS 1.4) point count correctly but
    # leaves the legacy 32-bit field at 0, which trips up lasinfo and any
    # older LAS-1.2-only reader. Populate it explicitly when it fits.
    n_pts = len(out.points)
    if n_pts < (1 << 32):
        try:
            out.header.point_count = n_pts
        except Exception:
            pass
        try:
            # All-single-return common case; if you need finer return-number
            # accounting, recompute from out.return_number instead.
            import numpy as _np
            counts = _np.zeros(5, dtype=_np.uint32)
            rn = _np.asarray(out.return_number, dtype=_np.int32)
            for i in range(1, 6):
                counts[i - 1] = int((rn == i).sum())
            out.header.number_of_points_by_return = counts
        except Exception:
            pass

    out_path = Path(out_path).with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        out.write(str(out_path))
    except Exception as e:
        if fmt == "laz":
            raise RuntimeError(
                f"LAZ write failed for {out_path.name}. "
                f"Install backend:  pip install lazrs   "
                f"OR switch Output Format to LAS. Original error: {e}"
            ) from e
        raise

    return out_path


def parse_tile_coords(filename: str) -> tuple[int, int] | None:
    """Parse easting, northing from filename like '498000_171500.laz'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            return None
    return None
