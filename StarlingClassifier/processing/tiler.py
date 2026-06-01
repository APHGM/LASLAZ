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


def tile_file(
    in_path: str,
    out_dir: str,
    tile_size: float = 100.0,
    chunk_size: int = 5_000_000,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda done, total: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
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

    return out_dir
