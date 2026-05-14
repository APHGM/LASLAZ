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
    las_version: str = "1.4"
):
    """Write output LAS or LAZ with classification field.
    
    Args:
        source_las: Source LAS data object
        xyz: Nx3 array of coordinates
        classification: Array of classification values
        out_path: Output file path
        output_format: Output format - 'las' or 'laz' (default: 'laz')
        las_version: LAS version - '1.2' or '1.4' (default: '1.4')
    """
    # Parse LAS version
    version_parts = tuple(map(int, las_version.split('.')))
    
    header = laspy.LasHeader(
        point_format=source_las.header.point_format,
        version=version_parts
    )
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales

    out = laspy.LasData(header=header)
    out.x = xyz[:, 0]
    out.y = xyz[:, 1]
    out.z = xyz[:, 2]
    out.classification = classification.astype(np.uint8)

    # Copy other standard fields if present
    for dim in source_las.point_format.dimension_names:
        if dim in ("X", "Y", "Z", "classification"):
            continue
        try:
            setattr(out, dim, getattr(source_las, dim))
        except Exception:
            pass

    out.write(str(out_path))


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
