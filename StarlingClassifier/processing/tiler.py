"""
Stream a single huge LAZ file into NxN m tiles using laspy chunked reads.
RAM-safe — never loads the whole file at once.
"""
import laspy
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Callable


def _clone_header(src_header: laspy.LasHeader) -> laspy.LasHeader:
    """
    Build a new LasHeader that preserves point format (incl. RGB),
    scales, offsets, extra dimensions, and VLRs from the source.
    Required so drone LiDAR colour, GPS time, and scanner metadata
    survive the tiling step.
    """
    new_header = laspy.LasHeader(
        point_format=src_header.point_format.id,   # use ID, not the object
        version=src_header.version,
    )
    new_header.scales = src_header.scales
    new_header.offsets = src_header.offsets

    # Copy extra dimensions (e.g. NIR for some scanners, custom attributes)
    for ed in getattr(src_header.point_format, "extra_dimensions", []):
        try:
            new_header.add_extra_dim(ed)
        except Exception:
            pass

    # Copy VLRs — RGB and other metadata often live here
    try:
        new_header.vlrs.extend(src_header.vlrs)
    except Exception:
        pass

    # Copy EVLRs (LAS 1.4 only)
    try:
        if hasattr(src_header, "evlrs") and src_header.evlrs:
            new_header.evlrs.extend(src_header.evlrs)
    except Exception:
        pass

    return new_header


def merge_las_files(
    in_paths: list[str | Path],
    out_path: str | Path,
    chunk_size: int = 5_000_000,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda done, total: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> Path:
    """
    Stream-merge multiple LAS/LAZ files into one output file.

    Uses the FIRST file's header (version, point format, scales, offsets,
    VLRs) as the template. All other files must have a *compatible* point
    format — same dimension layout. Different scales/offsets are tolerated
    because laspy converts via the scaled x/y/z properties.

    RAM-safe — reads each source in chunks of `chunk_size` points and writes
    them straight into the output, never materialising the whole cloud.
    """
    if not in_paths:
        raise ValueError("merge_las_files: in_paths is empty")
    paths = [Path(p) for p in in_paths]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Inspect headers, compute totals ──────────────────────────────────
    total_pts = 0
    for p in paths:
        with laspy.open(str(p)) as r:
            total_pts += r.header.point_count
    log_fn(f"Merging {len(paths)} files — total {total_pts:,} points → {out_path.name}")

    # ── Use first file as header template ────────────────────────────────
    with laspy.open(str(paths[0])) as r0:
        header_template = _clone_header(r0.header)

    done = 0
    written_count = 0
    with laspy.open(str(out_path), mode="w", header=header_template) as writer:
        for i, p in enumerate(paths):
            if cancelled_fn():
                log_fn("Merge cancelled.")
                break

            log_fn(f"  [{i+1}/{len(paths)}] {p.name}")
            try:
                with laspy.open(str(p)) as r:
                    src_fmt_id = r.header.point_format.id
                    template_fmt_id = header_template.point_format.id
                    if src_fmt_id != template_fmt_id:
                        log_fn(f"    WARNING: point format {src_fmt_id} differs from "
                               f"template {template_fmt_id} — skipping (incompatible)")
                        continue

                    for chunk in r.chunk_iterator(chunk_size):
                        if cancelled_fn():
                            break
                        writer.write_points(chunk)
                        written_count += len(chunk)
                        done += len(chunk)
                        progress_fn(done, total_pts)
            except Exception as e:
                log_fn(f"    ERROR reading {p.name}: {e}")
                continue

    log_fn(f"Merge complete: {written_count:,} points written to {out_path}")
    return out_path


def tile_file(
    in_path: str,
    out_dir: str,
    tile_size: float = 100.0,
    chunk_size: int = 5_000_000,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda done, total: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
    write_boundary_dxf: bool = True,
) -> Path:
    """Tile a LAS/LAZ file into spatial NxN m chunks. Returns out_dir as Path."""
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fn(f"Opening {in_path.name} ...")
    with laspy.open(str(in_path)) as reader:
        header = reader.header
        log_fn(
            f"  {header.point_count:,} points  "
            f"X {header.mins[0]:.1f}-{header.maxs[0]:.1f}  "
            f"Y {header.mins[1]:.1f}-{header.maxs[1]:.1f}"
        )

        writers: dict[tuple[int, int], laspy.LasWriter] = {}
        counts: dict[tuple[int, int], int] = defaultdict(int)
        total_pts = header.point_count

        try:
            done = 0
            for chunk in reader.chunk_iterator(chunk_size):
                if cancelled_fn():
                    log_fn("Tiling cancelled.")
                    break

                x = chunk.x
                y = chunk.y
                tx = (np.floor(x / tile_size) * tile_size).astype(np.int64)
                ty = (np.floor(y / tile_size) * tile_size).astype(np.int64)

                key = tx * 10_000_000 + ty
                uniq, inv = np.unique(key, return_inverse=True)

                for k_idx, _k in enumerate(uniq):
                    mask = inv == k_idx
                    east = int(tx[mask][0])
                    north = int(ty[mask][0])
                    tile_key = (east, north)

                    if tile_key not in writers:
                        out_path = out_dir / f"{east}_{north}.laz"
                        new_header = _clone_header(header)
                        writers[tile_key] = laspy.open(
                            str(out_path), mode="w", header=new_header
                        )

                    sub = chunk[mask]
                    writers[tile_key].write_points(sub)
                    counts[tile_key] += int(mask.sum())

                done += len(chunk)
                progress_fn(done, total_pts)
                pct = 100 * done / total_pts
                log_fn(f"  {done:,}/{total_pts:,} ({pct:.1f}%)   tiles open: {len(writers)}")
        finally:
            for w in writers.values():
                w.close()

    log_fn(f"\n{len(writers)} tiles written to {out_dir}")
    for k, n in sorted(counts.items()):
        log_fn(f"  {k[0]}_{k[1]}.laz  {n:,} pts")

    if write_boundary_dxf and len(writers) > 0:
        try:
            write_tile_boundaries_dxf(out_dir, log_fn=log_fn)
        except Exception as e:
            log_fn(f"  WARNING: boundary DXF generation failed: {e}")

    return out_dir


def write_tile_boundaries_dxf(
    tile_dir: str | Path,
    out_path: str | Path | None = None,
    use_bbox: bool = True,
    layer_name: str = "TILE_BOUNDARIES",
    add_labels: bool = True,
    log_fn: Callable[[str], None] = print,
) -> Path:
    """
    Generate a DXF showing tile boundaries — equivalent to
        lasboundary64 -i *.laz -use_bb -odir b -cores 8
    but writes DXF (CAD-friendly) instead of shapefile.

    use_bbox=True   → rectangle from each tile's LAS header min/max (fast)
    use_bbox=False  → reads point cloud and computes 2D convex hull (slower,
                      tighter boundary). NOT YET IMPLEMENTED — falls back to bbox.

    Returns path to the written DXF.
    """
    import ezdxf
    import laspy

    tile_dir = Path(tile_dir)
    if out_path is None:
        out_path = tile_dir / f"{tile_dir.name}_boundaries.dxf"
    out_path = Path(out_path).with_suffix(".dxf")

    doc = ezdxf.new(dxfversion="R2010")   # AutoCAD 2010 — widely compatible
    msp = doc.modelspace()
    doc.layers.add(layer_name, color=3)   # green

    if add_labels:
        doc.layers.add(f"{layer_name}_LABELS", color=2)   # yellow

    count = 0
    for f in sorted(tile_dir.glob("*.la[sz]")):
        try:
            with laspy.open(str(f)) as r:
                h = r.header
                xmin, ymin = h.mins[0], h.mins[1]
                xmax, ymax = h.maxs[0], h.maxs[1]
        except Exception as e:
            log_fn(f"  WARNING: cannot read {f.name}: {e}")
            continue

        # Closed rectangle as LWPOLYLINE
        msp.add_lwpolyline(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            close=True,
            dxfattribs={"layer": layer_name},
        )

        if add_labels:
            # Tile name centered in cell
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            # Text height ~3% of tile size for sensible scale
            th = max((xmax - xmin), (ymax - ymin)) * 0.03
            msp.add_text(
                f.stem,
                dxfattribs={
                    "layer": f"{layer_name}_LABELS",
                    "height": th,
                },
            ).set_placement((cx, cy), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

        count += 1

    doc.saveas(str(out_path))
    log_fn(f"DXF boundary file written: {out_path}  ({count} tile rectangles)")
    return out_path
