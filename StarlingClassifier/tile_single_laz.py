"""
CLI wrapper around processing.tiler.tile_file.

Usage:
    python tile_single_laz.py <input.laz> <out_dir> [--tile 100] [--chunk 5000000]
"""
import argparse
import sys
from pathlib import Path

# processing/ is right next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from processing.tiler import tile_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to source .laz/.las")
    ap.add_argument("out_dir", help="Output folder for tiles")
    ap.add_argument("--tile", type=float, default=100.0, help="Tile size m")
    ap.add_argument("--chunk", type=int, default=5_000_000,
                    help="Points per read chunk (lower = less RAM)")
    ap.add_argument("--no-dxf", action="store_true",
                    help="Skip writing the tile-boundary DXF after tiling")
    args = ap.parse_args()
    tile_file(args.input, args.out_dir, args.tile, args.chunk,
              write_boundary_dxf=not args.no_dxf)
