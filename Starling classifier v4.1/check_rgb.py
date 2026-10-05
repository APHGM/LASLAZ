"""
Diagnostic: confirm RGB survived a tiling operation.
Usage:
    python check_rgb.py <source.laz> <tile.laz>
"""
import sys
import laspy


def report(label: str, path: str):
    print(f"\n=== {label}: {path} ===")
    las = laspy.read(path)
    print(f"  Point format ID: {las.point_format.id}")
    print(f"  LAS version: {las.header.version}")
    print(f"  Point count: {len(las.points):,}")
    print(f"  Dimensions: {las.point_format.dimension_names}")
    has_rgb = all(hasattr(las, c) for c in ("red", "green", "blue"))
    print(f"  Has RGB:    {has_rgb}")
    if has_rgb:
        print(f"    red   sample: {las.red[:5].tolist()}")
        print(f"    green sample: {las.green[:5].tolist()}")
        print(f"    blue  sample: {las.blue[:5].tolist()}")
        print(f"    red   range : {int(las.red.min())} – {int(las.red.max())}")
    print(f"  VLR count: {len(las.header.vlrs)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_rgb.py <source.laz> <tile.laz>")
        sys.exit(1)
    report("SOURCE", sys.argv[1])
    report("TILE  ", sys.argv[2])
