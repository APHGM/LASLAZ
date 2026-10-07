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


# ── E57 support ───────────────────────────────────────────────────────────────

def e57_scan_count(filepath: str | Path) -> int:
    """Return the number of scans in an E57 file."""
    import pye57
    e57 = pye57.E57(str(filepath))
    count = e57.scan_count
    e57.close()
    return count


_E572LAS_PATHS = [
    r"C:\LAStools\bin\e572las.exe",
    r"C:\Program Files\LAStools\bin\e572las.exe",
    r"C:\lastools\bin\e572las.exe",
]


def _find_e572las() -> str | None:
    import sys, shutil
    # When running as a PyInstaller bundle, check alongside the EXE first
    if hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "e572las.exe"
        if bundled.exists():
            return str(bundled)
        # Also check next to the frozen executable itself
        exe_dir = Path(sys.executable).parent
        alongside = exe_dir / "e572las.exe"
        if alongside.exists():
            return str(alongside)
    # System PATH
    found = shutil.which("e572las") or shutil.which("e572las.exe")
    if found:
        return found
    # Common LAStools install locations
    for p in _E572LAS_PATHS:
        if Path(p).exists():
            return p
    return None


def convert_e57_to_laz(
    filepath: str | Path,
    out_path: str | Path | None = None,
    log_fn=print,
) -> Path:
    """
    Convert an E57 file to a single LAZ file.

    Primary path: e572las.exe (LAStools free tool) — handles any size file
    in one pass with minimal RAM.  Falls back to pye57 chunked reader when
    e572las is not installed.

    Returns the path to the written LAZ.
    """
    import pye57
    from pye57.e57 import COORDINATE_SYSTEMS, SUPPORTED_CARTESIAN_POINT_FIELDS

    import subprocess as _sub, os as _os

    E57_CONVERT_CHUNK = 5_000_000   # points per chunk — ~400 MB peak RAM

    filepath = Path(filepath)
    if out_path is None:
        out_path = filepath.with_suffix(".laz")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_fn(f"Converting E57 -> LAZ: {filepath.name}")

    # ── Fast path: e572las.exe (LAStools) ────────────────────────────────
    e572las = _find_e572las()
    if e572las:
        log_fn(f"  Using e572las.exe (LAStools) for conversion")
        args = [e572las, str(filepath), "-o", str(out_path)]
        proc = _sub.Popen(
            args, stdout=_sub.PIPE, stderr=_sub.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env={**_os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in proc.stdout:
            log_fn(f"  {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            log_fn(f"  WARNING: e572las.exe failed (exit {proc.returncode}), "
                   f"falling back to pye57 ...")
        elif out_path.exists():
            log_fn(f"  Done -> {out_path.name}")
            return out_path

    log_fn(f"  Using pye57 chunked reader ...")

    e57 = pye57.E57(str(filepath))
    try:
        n_scans = e57.scan_count
        log_fn(f"  {n_scans} scan(s) found in E57")

        # ── Collect scan metadata (no point data loaded yet) ─────────────
        scan_meta = []
        for i in range(n_scans):
            hdr = e57.get_header(i)
            pt_count = hdr.point_count
            available = set(hdr.point_fields)
            want_intensity = "intensity" in available
            want_colors    = all(c in available for c in
                                 ("colorRed", "colorGreen", "colorBlue"))
            log_fn(f"  Scan {i+1}/{n_scans}: {pt_count:,} pts  "
                   f"fields={sorted(available)}")
            scan_meta.append((hdr, pt_count, want_intensity, want_colors))

        has_rgb       = any(m[3] for m in scan_meta)
        has_intensity = any(m[2] for m in scan_meta)
        fmt_id = 7 if has_rgb else 6

        # ── Build field list for chunk buffers ───────────────────────────
        # Use Cartesian fields + optional extras + invalid-state flag
        base_fields   = list(SUPPORTED_CARTESIAN_POINT_FIELDS.keys())  # X/Y/Z
        extra_fields  = []
        if has_intensity:
            extra_fields.append("intensity")
        if has_rgb:
            extra_fields += ["colorRed", "colorGreen", "colorBlue"]
        valid_field = "cartesianInvalidState"

        # ── Pass 1: read first chunk of each scan to get XYZ extents ─────
        x_min = y_min = z_min = np.inf
        for i, (hdr, pt_count, _, _) in enumerate(scan_meta):
            # Build minimal buffers (XYZ + valid-state only) at chunk size
            sample_fields = base_fields + [valid_field]
            sample_fields = [f for f in sample_fields if f in hdr.point_fields]
            sdata, sbuf = e57.make_buffers(sample_fields,
                                           min(E57_CONVERT_CHUNK, pt_count))
            reader = hdr.points.reader(sbuf)
            n = reader.read()
            reader.close()
            if n == 0:
                continue
            sx = sdata["cartesianX"][:n].astype(np.float64)
            sy = sdata["cartesianY"][:n].astype(np.float64)
            sz = sdata["cartesianZ"][:n].astype(np.float64)
            if valid_field in sdata:
                vld = ~sdata[valid_field][:n].astype(bool)
            else:
                vld = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
            # Apply pose so extents are in global frame
            if hdr.has_pose() and vld.any():
                xyz = np.stack([sx[vld], sy[vld], sz[vld]]).T
                xyz = e57.to_global(xyz, hdr.rotation, hdr.translation)
                x_min = min(x_min, float(xyz[:, 0].min()))
                y_min = min(y_min, float(xyz[:, 1].min()))
                z_min = min(z_min, float(xyz[:, 2].min()))
            elif vld.any():
                x_min = min(x_min, float(sx[vld].min()))
                y_min = min(y_min, float(sy[vld].min()))
                z_min = min(z_min, float(sz[vld].min()))

        if np.isinf(x_min):
            raise RuntimeError("E57 file contained no valid points")

        # ── Build laspy writer ───────────────────────────────────────────
        laz_hdr = laspy.LasHeader(point_format=fmt_id, version="1.4")
        laz_hdr.offsets = np.array([x_min, y_min, z_min])
        laz_hdr.scales  = np.array([0.001, 0.001, 0.001])

        log_fn(f"  Writing {out_path.name}  "
               f"(LAS 1.4, fmt {fmt_id}, "
               f"{'RGB+' if has_rgb else ''}{'intensity' if has_intensity else 'XYZ-only'}) ...")

        total_written = 0
        with open(str(out_path), "wb") as _f, \
             laspy.LasWriter(_f, header=laz_hdr, do_compress=True) as writer:
            for i, (hdr, pt_count, want_intensity, want_colors) in enumerate(scan_meta):
                log_fn(f"  Scan {i+1}/{n_scans}: converting {pt_count:,} pts ...")

                # Build chunk buffers with only fields present in this scan
                chunk_fields = [f for f in base_fields if f in hdr.point_fields]
                if want_intensity:
                    chunk_fields.append("intensity")
                if want_colors:
                    chunk_fields += [c for c in
                                     ("colorRed", "colorGreen", "colorBlue")
                                     if c in hdr.point_fields]
                if valid_field in hdr.point_fields:
                    chunk_fields.append(valid_field)

                data, buffers = e57.make_buffers(chunk_fields, E57_CONVERT_CHUNK)
                reader = hdr.points.reader(buffers)

                pts_done = 0
                try:
                    while True:
                        n = reader.read()
                        if n == 0:
                            break

                        # Valid mask
                        if valid_field in data:
                            vld = ~data[valid_field][:n].astype(bool)
                        else:
                            vld = (np.isfinite(data["cartesianX"][:n]) &
                                   np.isfinite(data["cartesianY"][:n]) &
                                   np.isfinite(data["cartesianZ"][:n]))

                        if not vld.any():
                            pts_done += n
                            continue

                        x = data["cartesianX"][:n][vld].astype(np.float64)
                        y = data["cartesianY"][:n][vld].astype(np.float64)
                        z = data["cartesianZ"][:n][vld].astype(np.float64)

                        # Apply pose transform
                        if hdr.has_pose():
                            xyz = np.stack([x, y, z]).T
                            xyz = e57.to_global(xyz, hdr.rotation, hdr.translation)
                            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

                        nv = len(x)
                        pts = laspy.ScaleAwarePointRecord.zeros(nv, header=laz_hdr)
                        pts.x = x
                        pts.y = y
                        pts.z = z

                        if want_intensity and "intensity" in data:
                            iv = data["intensity"][:n][vld].astype(np.float64)
                            pts.intensity = (np.clip(iv, 0.0, 1.0) * 65535).astype(np.uint16)

                        if want_colors and has_rgb:
                            for ch, attr in [("colorRed",   "red"),
                                             ("colorGreen", "green"),
                                             ("colorBlue",  "blue")]:
                                if ch in data:
                                    cv = data[ch][:n][vld].astype(np.float64)
                                    setattr(pts, attr,
                                            (np.clip(cv, 0.0, 1.0) * 65535).astype(np.uint16))

                        writer.write_points(pts)
                        total_written += nv
                        pts_done += n

                        pct = int(100 * pts_done / pt_count)
                        log_fn(f"    {pct}%  ({total_written:,} pts written total)")
                finally:
                    reader.close()

    finally:
        e57.close()

    if total_written == 0:
        raise RuntimeError("E57 file contained no valid points after conversion")

    log_fn(f"  Done: {total_written:,} points -> {out_path.name}")
    return out_path
