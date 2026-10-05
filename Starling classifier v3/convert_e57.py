"""
Batch E57 → LAZ converter
--------------------------
Converts all E57 files in a folder (or a single E57 file) to LAZ.
The output LAZ is written alongside the E57 unless --out is specified.

Usage:
    python convert_e57.py  <path>                   # single file or folder
    python convert_e57.py  <path>  --out <folder>   # custom output folder
    python convert_e57.py  <path>  --recursive       # scan subfolders too
"""

import sys
import time
import argparse
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def convert_one(e57_path: Path, out_dir: Path | None) -> Path | None:
    sys.path.insert(0, str(Path(__file__).parent))
    from processing.las_io import convert_e57_to_laz

    if out_dir is None:
        laz_path = e57_path.with_suffix(".laz")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        laz_path = out_dir / (e57_path.stem + ".laz")

    if laz_path.exists():
        _log(f"  SKIP (already exists): {laz_path.name}")
        return laz_path

    t0 = time.time()
    try:
        convert_e57_to_laz(str(e57_path), out_path=str(laz_path), log_fn=_log)
        elapsed = time.time() - t0
        size_mb = laz_path.stat().st_size / 1e6
        _log(f"  OK  {laz_path.name}  ({size_mb:.0f} MB, {elapsed:.0f}s)")
        return laz_path
    except Exception as e:
        _log(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser(description="Batch E57 → LAZ converter")
    ap.add_argument("path", help="E57 file or folder containing E57 files")
    ap.add_argument("--out", default=None, help="Output folder (default: alongside each E57)")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders")
    args = ap.parse_args()

    p = Path(args.path)
    out_dir = Path(args.out) if args.out else None

    if p.is_file():
        if p.suffix.lower() != ".e57":
            _log(f"ERROR: not an E57 file: {p}")
            sys.exit(1)
        files = [p]
    elif p.is_dir():
        pattern = "**/*.e57" if args.recursive else "*.e57"
        files = sorted(p.glob(pattern))
        if not files:
            _log(f"No E57 files found in {p}")
            sys.exit(0)
    else:
        _log(f"ERROR: path not found: {p}")
        sys.exit(1)

    _log(f"Found {len(files)} E57 file(s)")
    ok, fail = 0, 0
    for i, f in enumerate(files, 1):
        _log(f"\n[{i}/{len(files)}] {f.name}")
        result = convert_one(f, out_dir)
        if result:
            ok += 1
        else:
            fail += 1

    _log(f"\n{'='*60}")
    _log(f"Done: {ok} converted, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
