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

    # Normalise version to "1.x"
    ver_norm = (las_version or "1.4").strip()
    if not ver_norm.startswith("1."):
        ver_norm = "1.4"

    header = laspy.LasHeader(
        point_format=source_las.header.point_format,
        version=ver_norm,
    )
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales

    out = laspy.LasData(header=header)
    out.x = xyz[:, 0]
    out.y = xyz[:, 1]
    out.z = xyz[:, 2]
    out.classification = classification.astype(np.uint8)

    for dim in source_las.point_format.dimension_names:
        if dim in ("X", "Y", "Z", "classification"):
            continue
        try:
            setattr(out, dim, getattr(source_las, dim))
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
