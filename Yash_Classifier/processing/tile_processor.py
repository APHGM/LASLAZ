import numpy as np
import time
import laspy
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from .las_io import write_classified_laz
from .ground_classifier import classify_ground


@dataclass
class ProcessParams:
    auto_mode: bool = True
    grid_cell_size: float = 0.25
    ground_threshold: float = 0.15
    object_sensitivity: float = 0.50
    output_format: str = "laz"  # "las" or "laz"
    same_location: bool = True  # Export to same folder as input
    las_version: str = "1.4"  # "1.2" or "1.4"


def auto_analyze(xyz: np.ndarray, log_fn: Callable[[str], None]) -> tuple[float, float, float]:
    """Perform a quick pre-scan of the point cloud to determine optimal parameters."""
    n = len(xyz)
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    area = max(1.0, x_range * y_range)
    density = n / area
    
    if density > 500: auto_cell = 0.20
    elif density > 100: auto_cell = 0.35
    else: auto_cell = 0.60
    
    auto_thr = 0.15
    auto_sens = 0.60
    
    log_fn(f"    - Auto-Detected: Detail={auto_cell}m, Thickness={auto_thr:.2f}m, Sensitivity={auto_sens}m")
    return float(auto_cell), float(auto_thr), float(auto_sens)


def process_all_files(
    input_dir: str | Path,
    out_dir: str | Path,
    params: ProcessParams,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda i, n: None,
    cancelled_fn: Callable[[], bool] = lambda: False,
) -> None:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    
    # Determine output directory
    if params.same_location:
        out_dir = input_dir
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("**/*.la[sz]")))
    total_files = len(files)

    for i, file_path in enumerate(files):
        if cancelled_fn(): break
        log_fn(f"\n[{i+1}/{total_files}] Processing {file_path.name}...")
        
        try:
            with laspy.open(str(file_path)) as f:
                n_pts = f.header.point_count
                bbox = [f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max]
                
                # Chunking decision: if > 25M points, process in spatial blocks
                use_spatial_chunks = n_pts > 25_000_000
                chunk_size = 150.0 # meters
                buffer = 15.0
                
                log_fn("  Loading points into memory...")
                las_full = f.read()
                xyz_full = np.stack([las_full.x, las_full.y, las_full.z], axis=1)
                
                if not use_spatial_chunks:
                    if params.auto_mode:
                        cell, thr, sens = auto_analyze(xyz_full, log_fn)
                    else:
                        cell, thr, sens = params.grid_cell_size, params.ground_threshold, params.object_sensitivity
                    
                    ground_mask, _ = classify_ground(xyz_full, cell, thr, sens)
                    classification_full = np.full(n_pts, 1, dtype=np.uint8)
                    classification_full[ground_mask] = 2
                else:
                    log_fn(f"  MASSIVE FILE ({n_pts:,} points). Using Spatial Chunking (150m blocks)...")
                    
                    nx = int(np.ceil((bbox[1] - bbox[0]) / chunk_size))
                    ny = int(np.ceil((bbox[3] - bbox[2]) / chunk_size))
                    total_chunks = nx * ny
                    
                    classification_full = np.full(n_pts, 1, dtype=np.uint8)
                    
                    # Auto-analyze on the full cloud once
                    if params.auto_mode:
                        cell, thr, sens = auto_analyze(xyz_full, log_fn)
                    else:
                        cell, thr, sens = params.grid_cell_size, params.ground_threshold, params.object_sensitivity

                    for cid in range(total_chunks):
                        if cancelled_fn(): break
                        
                        ty, tx = cid // nx, cid % nx
                        c_xmin, c_ymin = bbox[0] + tx * chunk_size, bbox[2] + ty * chunk_size
                        c_xmax, c_ymax = c_xmin + chunk_size, c_ymin + chunk_size
                        
                        mask_buffered = (
                            (xyz_full[:, 0] >= c_xmin - buffer) & (xyz_full[:, 0] < c_xmax + buffer) &
                            (xyz_full[:, 1] >= c_ymin - buffer) & (xyz_full[:, 1] < c_ymax + buffer)
                        )
                        
                        n_chunk_pts = np.sum(mask_buffered)
                        if n_chunk_pts == 0: continue
                        
                        log_fn(f"    -> Block {cid+1}/{total_chunks} ({n_chunk_pts:,} pts)...")
                        
                        xyz_chunk = xyz_full[mask_buffered]
                        ground_mask_chunk, _ = classify_ground(xyz_chunk, cell, thr, sens)
                        
                        mask_core_local = (
                            (xyz_chunk[:, 0] >= c_xmin) & (xyz_chunk[:, 0] < c_xmax) &
                            (xyz_chunk[:, 1] >= c_ymin) & (xyz_chunk[:, 1] < c_ymax)
                        )
                        
                        global_indices = np.where(mask_buffered)[0]
                        core_global_indices = global_indices[mask_core_local]
                        classification_full[core_global_indices[ground_mask_chunk[mask_core_local]]] = 2

                # Save Final
                output_ext = "laz" if params.output_format.lower() == "laz" else "las"
                log_fn(f"  Saving classified {output_ext.upper()} (LAS {params.las_version})...")
                out_path = out_dir / f"{file_path.stem}_ground.{output_ext}"
                write_classified_laz(las_full, xyz_full, classification_full, out_path, params.output_format, params.las_version)
                log_fn(f"  SUCCESS! Total Ground Points: {int((classification_full==2).sum()):,}")

        except Exception as e:
            log_fn(f"  ERROR processing {file_path.name}: {e}")

        progress_fn(i + 1, total_files)

    log_fn("\nBatch processing finished.")
