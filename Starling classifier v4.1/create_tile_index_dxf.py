"""
Create a tile-index DXF from a folder of LAS/LAZ files.

Each file's bounding box is drawn as a closed rectangle on the TILE_BOUNDARIES
layer, with a label (filename stem) centred in each cell.

Usage:
    python create_tile_index_dxf.py <folder> [options]

Examples:
    python create_tile_index_dxf.py D:\Data\tiles
    python create_tile_index_dxf.py D:\Data\tiles --out D:\Data\tile_index.dxf
    python create_tile_index_dxf.py D:\Data\tiles --no-labels --recursive
"""

import argparse
import sys
from pathlib import Path


def create_tile_index(
    folder: Path,
    out_path: Path,
    recursive: bool = False,
    add_labels: bool = True,
    layer_name: str = "TILE_BOUNDARIES",
    label_layer: str = "TILE_LABELS",
) -> int:
    """
    Read every *.las / *.laz in `folder` and write a tile-index DXF.

    Returns the number of tiles written.
    """
    try:
        import ezdxf
        import ezdxf.enums
    except ImportError:
        print("ERROR: ezdxf is not installed.  Run:  pip install ezdxf")
        sys.exit(1)

    try:
        import laspy
    except ImportError:
        print("ERROR: laspy is not installed.  Run:  pip install laspy[lazrs]")
        sys.exit(1)

    pattern = "**/*.la[sz]" if recursive else "*.la[sz]"
    files = sorted(folder.glob(pattern))

    if not files:
        print(f"No *.las / *.laz files found in {folder}")
        return 0

    print(f"Found {len(files)} file(s) — reading headers ...")

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    doc.layers.add(layer_name, color=3)      # green rectangles
    if add_labels:
        doc.layers.add(label_layer, color=2)  # yellow text

    count = 0
    skipped = 0
    for f in files:
        try:
            with laspy.open(str(f)) as r:
                h = r.header
                xmin, ymin = float(h.mins[0]), float(h.mins[1])
                xmax, ymax = float(h.maxs[0]), float(h.maxs[1])
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            skipped += 1
            continue

        if xmin == xmax or ymin == ymax:
            print(f"  SKIP {f.name}: degenerate bbox ({xmin},{ymin}) – ({xmax},{ymax})")
            skipped += 1
            continue

        msp.add_lwpolyline(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            close=True,
            dxfattribs={"layer": layer_name},
        )

        if add_labels:
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            th = max(xmax - xmin, ymax - ymin) * 0.03
            msp.add_text(
                f.stem,
                dxfattribs={"layer": label_layer, "height": th},
            ).set_placement((cx, cy), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

        count += 1
        print(f"  {f.name:<50}  ({xmin:.1f},{ymin:.1f}) – ({xmax:.1f},{ymax:.1f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))
    print(f"\nDXF written: {out_path}")
    print(f"  Tiles included : {count}")
    if skipped:
        print(f"  Skipped        : {skipped}")
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Build a tile-index DXF from a folder of LAS/LAZ files."
    )
    ap.add_argument("folder", help="Folder containing *.las / *.laz files")
    ap.add_argument(
        "--out", default="",
        help="Output DXF path (default: <folder>/tile_index.dxf)"
    )
    ap.add_argument(
        "--recursive", "-r", action="store_true",
        help="Also search sub-folders"
    )
    ap.add_argument(
        "--no-labels", action="store_true",
        help="Omit filename labels inside each rectangle"
    )
    ap.add_argument(
        "--layer", default="TILE_BOUNDARIES",
        help="DXF layer name for rectangles (default: TILE_BOUNDARIES)"
    )
    ap.add_argument(
        "--label-layer", default="TILE_LABELS",
        help="DXF layer name for text labels (default: TILE_LABELS)"
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: Not a directory: {folder}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else folder / "tile_index.dxf"
    if out_path.is_dir():
        out_path = out_path / "tile_index.dxf"
    if not out_path.suffix:
        out_path = out_path.with_suffix(".dxf")

    create_tile_index(
        folder=folder,
        out_path=out_path,
        recursive=args.recursive,
        add_labels=not args.no_labels,
        layer_name=args.layer,
        label_layer=args.label_layer,
    )


if __name__ == "__main__":
    main()
