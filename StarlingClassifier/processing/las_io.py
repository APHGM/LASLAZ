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
