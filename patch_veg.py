"""
patch_veg.py  —  Apply height-based vegetation classification to already-classified
                 LAZ/LAS tiles that have correct class-2 (ground) but are missing
                 classes 3 / 4 / 5 / 7 due to the slam_mode NameError bug.

Two streaming passes — never loads the full file into RAM.
  Pass 1: collect class-2 point positions → build 2-D ground-elevation grid
  Pass 2: stream all points, compute nZ from grid, assign veg/noise classes,
          write to output file

Usage
-----
  python patch_veg.py <path> [<path> ...]  [options]

  <path> can be:
    - a single .laz / .las file
    - a folder  (searched recursively for *_classified.laz and *_classified.las)

Options
  --veg-low-min F    nZ start of low-veg  (default 0.10 m)
  --veg-low-max F    nZ boundary low/med  (default 1.00 m)
  --veg-med-max F    nZ boundary med/high (default 3.00 m)
  --noise-below F    nZ below this → class 7 noise (default -0.10 m)
  --ground-thr  F    nZ within ±this → keep as class 2 (default 0.05 m)
  --chunk       N    points per streaming chunk (default 5000000)
  --overwrite        write back to the SAME file (default: write *_vegpatch.laz
                     alongside the input — safer for first run)
  --workers     N    parallel files (default: 4)
  --pattern     P    glob pattern to search in folders (default: *_classified.laz)

Examples
  # Patch everything in five project folders, write alongside (safe default)
  python patch_veg.py P:\\proj1 P:\\proj2 P:\\proj3 P:\\proj4 P:\\proj5

  # Overwrite in-place with custom veg thresholds
  python patch_veg.py P:\\proj1 --overwrite --veg-low-min 0.15 --veg-med-max 2.5

  # Patch a single merged file
  python patch_veg.py P:\\proj1\\output\\merged_classified.laz --overwrite
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ─── height classification ────────────────────────────────────────────────────

def _classify_heights(cls, nz, veg_low_min, veg_low_max, veg_med_max,
                      noise_below, ground_thr, protect=(2, 20)):
    """Assign veg / noise classes by nZ.  Protected classes are left untouched."""
    cls = cls.copy()
    protected = np.zeros(len(cls), dtype=bool)
    for c in protect:
        protected |= cls == c
    cand = ~protected
    finite = np.isfinite(nz)
    c = cand & finite
    cls[c & (nz <= noise_below)]                             = 7   # low noise
    cls[c & (nz >= veg_low_min) & (nz < veg_low_max)]       = 3   # low veg
    cls[c & (nz >= veg_low_max) & (nz < veg_med_max)]       = 4   # med veg
    cls[c & (nz >= veg_med_max)]                             = 5   # high veg
    return cls


# ─── ground grid helpers ──────────────────────────────────────────────────────

def _fill_nan(grid):
    """Nearest-neighbour fill for NaN cells using scipy distance transform."""
    from scipy.ndimage import distance_transform_edt
    valid = ~np.isnan(grid)
    if not valid.any():
        return np.zeros_like(grid)
    _, idx = distance_transform_edt(~valid, return_indices=True)
    out = grid.copy()
    out[~valid] = grid[idx[0][~valid], idx[1][~valid]]
    return out


# ─── per-file worker ──────────────────────────────────────────────────────────

def _patch_one(args):
    """
    Patch a single file.  Called in a worker process.
    Returns (path_str, elapsed_s, n_pts, classes_added) or raises.
    """
    import laspy

    (src, dst, veg_low_min, veg_low_max, veg_med_max,
     noise_below, ground_thr, chunk_size, cell_size) = args

    src = Path(src)
    dst = Path(dst)
    t0 = time.time()

    # ── Pass 1: build ground grid from class-2 points ─────────────────────
    with laspy.open(str(src)) as reader:
        h = reader.header
        xmin, ymin = float(h.mins[0]), float(h.mins[1])
        xmax, ymax = float(h.maxs[0]), float(h.maxs[1])
        src_fmt   = h.point_format.id
        src_ver   = h.version
        src_scl   = h.scales
        src_off   = h.offsets
        try:
            src_vlrs = list(h.vlrs)
        except Exception:
            src_vlrs = []

        ncols = int(math.ceil((xmax - xmin) / cell_size)) + 2
        nrows = int(math.ceil((ymax - ymin) / cell_size)) + 2
        z_sum   = np.zeros((nrows, ncols), dtype=np.float64)
        z_count = np.zeros((nrows, ncols), dtype=np.int32)

        for chunk in reader.chunk_iterator(chunk_size):
            mask = np.asarray(chunk.classification, dtype=np.uint8) == 2
            if not mask.any():
                continue
            xs = np.asarray(chunk.x, dtype=np.float64)[mask]
            ys = np.asarray(chunk.y, dtype=np.float64)[mask]
            zs = np.asarray(chunk.z, dtype=np.float64)[mask]
            ci = np.clip(((xs - xmin) / cell_size).astype(np.int32), 0, ncols - 1)
            ri = np.clip(((ys - ymin) / cell_size).astype(np.int32), 0, nrows - 1)
            np.add.at(z_sum,   (ri, ci), zs)
            np.add.at(z_count, (ri, ci), 1)

    n_gnd_cells = int((z_count > 0).sum())
    if n_gnd_cells == 0:
        raise RuntimeError("No class-2 (ground) points found — cannot build ground surface")

    z_grid = np.where(z_count > 0, z_sum / np.maximum(z_count, 1), np.nan)
    if np.isnan(z_grid).any():
        z_grid = _fill_nan(z_grid)

    def _nz(xs, ys, zs):
        ci = np.clip(((xs - xmin) / cell_size).astype(np.int32), 0, ncols - 1)
        ri = np.clip(((ys - ymin) / cell_size).astype(np.int32), 0, nrows - 1)
        return zs - z_grid[ri, ci]

    # ── Pass 2: stream classify + write ───────────────────────────────────
    ver_str = f"{src_ver.major}.{src_ver.minor}"
    new_hdr = laspy.LasHeader(point_format=src_fmt, version=ver_str)
    new_hdr.scales  = src_scl
    new_hdr.offsets = src_off
    try:
        new_hdr.vlrs.extend(src_vlrs)
    except Exception:
        pass

    do_compress = dst.suffix.lower() == ".laz"
    n_pts = 0
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 20: 0}

    # Write to a temp file first so the source is never left half-written
    tmp_path = dst.with_suffix(".tmp_patch")
    try:
        with laspy.open(str(src)) as reader, \
             open(str(tmp_path), "wb") as _fout, \
             laspy.LasWriter(_fout, header=new_hdr, do_compress=do_compress) as writer:

            for chunk in reader.chunk_iterator(chunk_size):
                xs  = np.asarray(chunk.x, dtype=np.float64)
                ys  = np.asarray(chunk.y, dtype=np.float64)
                zs  = np.asarray(chunk.z, dtype=np.float64)
                nz  = _nz(xs, ys, zs)
                cls = np.asarray(chunk.classification, dtype=np.uint8)

                cls = _classify_heights(
                    cls, nz,
                    veg_low_min=veg_low_min,
                    veg_low_max=veg_low_max,
                    veg_med_max=veg_med_max,
                    noise_below=noise_below,
                    ground_thr=ground_thr,
                )
                chunk.classification = cls
                writer.write_points(chunk)
                n_pts += len(xs)
                for c in class_counts:
                    class_counts[c] += int((cls == c).sum())

        # Atomic rename
        if dst.exists():
            dst.unlink()
        tmp_path.rename(dst)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    veg = class_counts[3] + class_counts[4] + class_counts[5]
    noise = class_counts[7]
    elapsed = time.time() - t0
    return str(src), elapsed, n_pts, veg, noise, class_counts


# ─── file discovery ───────────────────────────────────────────────────────────

def _find_files(paths, pattern):
    found = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob(pattern)))
        else:
            print(f"  WARNING: {p} does not exist — skipped")
    # De-duplicate preserving order
    seen = set()
    out = []
    for f in found:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Patch missing veg/noise classes in already-classified LAZ/LAS files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="+", help="File(s) or folder(s) to patch")
    ap.add_argument("--veg-low-min", type=float, default=0.10,
                    metavar="M", help="nZ start of low-veg class 3  (default 0.10 m)")
    ap.add_argument("--veg-low-max", type=float, default=1.00,
                    metavar="M", help="nZ boundary low/med  (default 1.00 m)")
    ap.add_argument("--veg-med-max", type=float, default=3.00,
                    metavar="M", help="nZ boundary med/high (default 3.00 m)")
    ap.add_argument("--noise-below", type=float, default=-0.10,
                    metavar="M", help="nZ below this → class 7 noise (default -0.10 m)")
    ap.add_argument("--ground-thr",  type=float, default=0.05,
                    metavar="M", help="nZ within ±this stays class 2 (default 0.05 m)")
    ap.add_argument("--cell-size",   type=float, default=1.0,
                    metavar="M", help="Ground grid cell size (default 1.0 m)")
    ap.add_argument("--chunk",       type=int,   default=5_000_000,
                    metavar="N", help="Points per streaming chunk (default 5000000)")
    ap.add_argument("--overwrite",   action="store_true",
                    help="Write back to the same file (default: write *_vegpatch.laz alongside)")
    ap.add_argument("--workers",     type=int,   default=4,
                    metavar="N", help="Parallel files (default 4)")
    ap.add_argument("--pattern",     default="*_classified.laz",
                    metavar="GLOB", help="Pattern for folder search (default *_classified.laz)")
    args = ap.parse_args()

    files = _find_files(args.paths, args.pattern)
    if not files:
        print(f"No files matched pattern '{args.pattern}' in: {args.paths}")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s) to patch.")
    print(f"  veg thresholds: low {args.veg_low_min}–{args.veg_low_max} m  "
          f"med {args.veg_low_max}–{args.veg_med_max} m  high ≥{args.veg_med_max} m")
    print(f"  noise below: {args.noise_below} m   ground band: ±{args.ground_thr} m")
    print(f"  output: {'overwrite in-place' if args.overwrite else 'write *_vegpatch.laz alongside'}")
    print(f"  workers: {args.workers}   chunk: {args.chunk:,} pts\n")

    # Build task list
    tasks = []
    for src in files:
        if args.overwrite:
            dst = src
        else:
            dst = src.with_name(src.stem.removesuffix("_classified") + "_classified_vegpatch" + src.suffix)
        tasks.append((
            str(src), str(dst),
            args.veg_low_min, args.veg_low_max, args.veg_med_max,
            args.noise_below, args.ground_thr,
            args.chunk, args.cell_size,
        ))

    ok = err = 0
    total_pts = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_src = {pool.submit(_patch_one, t): t[0] for t in tasks}
        for fut in as_completed(future_to_src):
            src = future_to_src[fut]
            try:
                src_path, elapsed, n_pts, veg, noise, counts = fut.result()
                ok += 1
                total_pts += n_pts
                print(f"  ✓ {Path(src_path).name}  "
                      f"({n_pts:,} pts  {elapsed/60:.1f} min)  "
                      f"veg={veg:,}  noise={noise:,}  "
                      f"[gnd={counts[2]:,}  low={counts[3]:,}  "
                      f"med={counts[4]:,}  high={counts[5]:,}  bird={counts[20]:,}]")
            except Exception as e:
                err += 1
                print(f"  ✗ {Path(src).name}  ERROR: {e}")

    print(f"\nDone: {ok} patched, {err} failed — {total_pts:,} total points processed.")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
