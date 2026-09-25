"""
fix_microstation_blocks.py

Fixes duplicate block definitions created by MicroStation DWG export.

MicroStation creates separate block definitions (e.g. Tree_KOREC_1, Tree_KOREC_2 ...)
for each uniquely-scaled cell placement.  AutoCAD shows them as clutter in the block
list.  This script:

  1. Identifies numbered variants  (BaseName_N)  that have a corresponding base block.
  2. Computes the scale ratio baked into each variant by comparing bounding boxes.
  3. Updates every INSERT of a variant to use the base block name, multiplying the
     existing INSERT scale by the ratio so the visual size is unchanged.
  4. Saves the result to a new DXF file; the unused variant definitions can then
     be removed with PURGE inside AutoCAD.

Workflow
--------
  AutoCAD  →  Save As DXF
  python fix_microstation_blocks.py  your_drawing.dxf
  AutoCAD  →  Open  your_drawing_fixed.dxf  →  PURGE  →  Save As DWG

Requirements
------------
  pip install ezdxf
"""

import re
import sys
from pathlib import Path

import ezdxf
from ezdxf import bbox as ezdxf_bbox


# ---------------------------------------------------------------------------
# Bounding-box helper
# ---------------------------------------------------------------------------

def _block_size(doc, block_name: str):
    """Return (width, height) of a block definition, or None if empty/unknown."""
    try:
        block = doc.blocks[block_name]
        extents = ezdxf_bbox.extents([block], fast=True)
        if extents.has_data:
            size = extents.size
            return size.x, size.y
    except Exception:
        pass
    return None


def _scale_ratio(doc, base_name: str, num_name: str):
    """
    Estimate the uniform scale factor baked into *num_name* relative to *base_name*.
    Falls back to 1.0 if extents cannot be computed.
    """
    base_size = _block_size(doc, base_name)
    num_size  = _block_size(doc, num_name)

    if base_size and num_size:
        bw, bh = base_size
        nw, nh = num_size
        # Use X dimension; fall back to Y if X is degenerate
        if bw > 1e-9:
            return nw / bw
        if bh > 1e-9:
            return nh / bh

    return 1.0


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def fix_duplicate_blocks(input_path: str, output_path: str | None = None):
    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else \
                  input_path.parent / (input_path.stem + "_fixed" + input_path.suffix)

    print(f"Reading  : {input_path}")
    doc = ezdxf.readfile(str(input_path))

    # Pattern: ends with  _<digits>
    suffix_re   = re.compile(r'^(.+)_(\d+)$')
    all_blocks  = {b.name for b in doc.blocks}

    # Build remap table  { variant_name -> (base_name, scale_ratio) }
    remap: dict[str, tuple[str, float]] = {}

    for blk_name in sorted(all_blocks):
        m = suffix_re.match(blk_name)
        if not m:
            continue
        base_name = m.group(1)
        if base_name not in all_blocks:
            continue                          # no base block → skip

        ratio = _scale_ratio(doc, base_name, blk_name)
        remap[blk_name] = (base_name, ratio)
        print(f"  {blk_name!s:<60s}  ->  {base_name}  (scale ×{ratio:.6g})")

    if not remap:
        print("No numbered duplicate blocks found.  Nothing to do.")
        return

    # Walk every INSERT in model space, paper space, AND inside block definitions
    # (handles nested references too).
    updated = 0
    for container in doc.blocks:
        for entity in container:
            if entity.dxftype() != "INSERT":
                continue
            blk_name = entity.dxf.name
            if blk_name not in remap:
                continue

            base_name, ratio = remap[blk_name]

            # Read current scales (default 1.0 per DXF spec)
            cur_x = entity.dxf.get("xscale", 1.0)
            cur_y = entity.dxf.get("yscale", 1.0)
            cur_z = entity.dxf.get("zscale", 1.0)

            entity.dxf.name   = base_name
            entity.dxf.xscale = cur_x * ratio
            entity.dxf.yscale = cur_y * ratio
            entity.dxf.zscale = cur_z * ratio
            updated += 1

    print(f"\nUpdated {updated} INSERT reference(s).")
    doc.saveas(str(output_path))
    print(f"Saved    : {output_path}")
    print("\nNext step: open the DXF in AutoCAD, run PURGE → All, then Save As DWG.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:  python fix_microstation_blocks.py  input.dxf  [output.dxf]")
        sys.exit(1)

    fix_duplicate_blocks(
        input_path  = sys.argv[1],
        output_path = sys.argv[2] if len(sys.argv) > 2 else None,
    )
