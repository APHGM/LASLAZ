"""
Batch-process multiple LAS/LAZ files through the Starling Classifier pipeline.

Usage:
    python batch_process.py <input_folder> [options]

Each *.las / *.laz file in <input_folder> is processed independently.
Output for each file goes to <input_folder>/<stem>_output/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from processing.tile_processor import ProcessParams, process_single_file


def log_prefix(file_index: int, total: int, stem: str) -> str:
    return f"[{file_index}/{total}] {stem}"


def main():
    ap = argparse.ArgumentParser(description="Batch-process LAS/LAZ files")
    ap.add_argument("input_folder", help="Folder containing *.las / *.laz files")
    ap.add_argument("--out-root", default="",
                    help="Root output folder (default: alongside each input file)")
    ap.add_argument("--tile-size", type=float, default=100.0, help="Tile size in metres")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel tile workers per file (default 1)")
    ap.add_argument("--ground-method", choices=["csf", "grid"], default="csf")
    ap.add_argument("--no-merge", action="store_true",
                    help="Do not merge classified tiles into one output file")
    ap.add_argument("--output-format", choices=["laz", "las"], default="laz")
    args = ap.parse_args()

    in_folder = Path(args.input_folder)
    if not in_folder.is_dir():
        print(f"ERROR: Not a directory: {in_folder}")
        sys.exit(1)

    files = sorted(in_folder.glob("*.la[sz]"))
    if not files:
        print(f"ERROR: No *.las / *.laz files found in {in_folder}")
        sys.exit(1)

    total = len(files)
    print(f"\nFound {total} file(s) to process in: {in_folder}")
    print("=" * 70)

    params = ProcessParams(
        tile_size=args.tile_size,
        ground_method=args.ground_method,
        num_workers=args.workers,
        merge_output_to_single=not args.no_merge,
        output_format=args.output_format,
    )

    results = []
    overall_start = time.time()

    for i, las_path in enumerate(files, 1):
        if args.out_root:
            out_dir = Path(args.out_root) / las_path.stem
        else:
            out_dir = las_path.parent / f"{las_path.stem}_output"

        print(f"\n{log_prefix(i, total, las_path.name)}")
        print(f"  Output → {out_dir}")
        print("-" * 70)

        t0 = time.time()
        try:
            ext = args.output_format
            params.merged_output_name = f"{las_path.stem}_classified.{ext}"
            process_single_file(
                file_path=las_path,
                out_dir=out_dir,
                params=params,
                log_fn=lambda msg: print(f"  {msg}"),
                progress_fn=lambda done, tot: print(
                    f"  Progress: {done}/{tot} tiles", end="\r"
                ),
            )
            elapsed = time.time() - t0
            # Verify output was actually written
            expected = out_dir / f"{las_path.stem}_classified.{ext}"
            if expected.exists() and expected.stat().st_size > 0:
                status = "OK"
            else:
                # Check for any classified output
                outputs = list(out_dir.glob(f"*_classified.{ext}"))
                if outputs and outputs[0].stat().st_size > 0:
                    status = "OK"
                else:
                    status = "WRITE_FAILED"
            results.append((las_path.name, status, elapsed))
            print(f"\n  Done in {elapsed/60:.1f} min  [{status}]")
        except Exception as e:
            elapsed = time.time() - t0
            results.append((las_path.name, f"FAILED: {e}", elapsed))
            print(f"\n  ERROR: {e}")

    # Final summary
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for name, status, elapsed in results:
        print(f"  {name:<40} {status:<12}  {elapsed/60:.1f} min")
    print("-" * 70)
    print(f"Total time: {total_elapsed/60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
