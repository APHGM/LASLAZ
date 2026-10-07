# Starling Ground-Contact Classifier — v4.2

A Windows desktop application (PyQt6) that classifies LiDAR, SLAM, and drone
photogrammetry point clouds — ground, vegetation, buildings, low noise, probable
ground, and starling bird-contact points.

---

## Version History

| Version | Highlights |
|---|---|
| **v4.2** | Probable ground (class 12), slope noise fix, always-stream ≥30M pts |
| **v4.1** | E57 input support, E57→LAZ batch tab, DXF to output folder |
| **v4.0** | Batch Process tab, multi-file queue, auto-detect per file |
| **v3 / v0.4** | Streaming 2-pass classify (handles 400M+ pts, ~400 MB RAM), SLAM mode, z-range wall filter, bilinear ground interpolation |
| **v2.0** | Drone photogrammetry mode, RGB/NIR vegetation index, building detection |
| **v1.x** | Initial release — CSF ground, DBSCAN bird detection, tile pipeline |

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Fresh Install on a New Machine](#fresh-install-on-a-new-machine)
4. [Input Modes](#input-modes)
5. [Output Classes](#output-classes)
6. [GUI Reference](#gui-reference)
7. [Point Cloud Source Modes](#point-cloud-source-modes)
8. [Streaming Classification](#streaming-classification)
9. [Batch Processing](#batch-processing)
10. [E57 → LAZ Conversion](#e57--laz-conversion)
11. [Recommended Settings](#recommended-settings)
12. [Building a Standalone EXE](#building-a-standalone-exe)
13. [Troubleshooting](#troubleshooting)
14. [Architecture Notes](#architecture-notes)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| Python | 3.10 | **3.12** (tested) |
| RAM | 16 GB | **64–128 GB** for 300M+ point clouds |
| CPU | 4 cores | 8+ cores |
| Disk | 50 GB free | 200+ GB for large datasets |

> **Large files:** The streaming path handles files of any size with ~400 MB peak RAM.
> Files ≥ 30M points are automatically routed to streaming — no configuration needed.

---

## Quick Start

```cmd
start_gui.bat
```

Or for E57 batch conversion without the full GUI:

```cmd
convert_e57.bat
```

---

## Fresh Install on a New Machine

### Step 1 — Install Python 3.12
Download from https://www.python.org/downloads/windows/ — tick **"Add Python to PATH"**.

### Step 2 — Create a virtual environment
```cmd
cd D:\YourPath\
py -3.12 -m venv arunpy
arunpy\Scripts\activate
```

### Step 3 — Copy the project folder
Copy the entire `Starling classifier v4.1\` folder to your machine.

### Step 4 — Install dependencies
```cmd
pip install -r requirements.txt
```

`requirements.txt`:
```
PyQt6
laspy[lazrs]
numpy
scipy
scikit-learn
cloth-simulation-filter
pye57
```

### Step 5 — Update venv paths in launcher scripts
Edit `start_gui.bat`, `build.bat`, `convert_e57.bat` — change:
```bat
set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
```
to your venv's `python.exe`.

### Step 6 — Test
```cmd
start_gui.bat
```
The GUI should open within ~5 seconds.

---

## Input Modes

The **Classify** tab supports three input modes:

| Mode | Use when |
|---|---|
| **Single file** | One LAZ/LAS/E57 file — any size (streaming handles 400M+ pts) |
| **Tile folder** | Folder of pre-tiled LAZ files named `<easting>_<northing>.laz` |
| **Multi-file merge** | Several files to merge then classify as one |

For E57 files: the app auto-converts to LAZ first (subprocess-isolated so a bad E57 won't crash the GUI), then classifies.

---

## Output Classes

| Class | Name | Notes |
|---|---|---|
| 1 | Unclassified | Points outside all bands |
| **2** | **Ground** | Confident ground (within CSF class threshold) |
| 3 | Low vegetation | 0.10 – 1.00 m above ground |
| 4 | Medium vegetation | 1.00 – 3.00 m |
| 5 | High vegetation | ≥ 3.00 m |
| 6 | Building | Photogrammetry mode only |
| 7 | Low noise | Below ground surface |
| 8 | Model keypoints | Thinned ground for TIN — optional |
| **12** | **Probable ground** | Just beyond ground threshold — uncertain, review manually |
| 20 | Bird contact | DBSCAN clusters near ground |

**Class 12 — Probable Ground** (new in v4.2): Points slightly outside the confident
ground band on both sides (above and below). Useful on slopes and SLAM scans where
the CSF cloth may not perfectly follow transitions. Review in CloudCompare or
TerraSolid and merge to class 2 if genuinely ground.

---

## GUI Reference

### Classify Tab

#### Input / Output
| Field | Notes |
|---|---|
| Input mode | Single file / Tile folder / Multi-file merge |
| Output folder | Where classified files and CSVs are written |
| Skip tiling | Process the whole file as one (recommended for single files ≤ RAM) |
| Tile size | Auto (density-based) or manual. Used when Skip tiling is off. |

#### Point Cloud Source
Select the scanner type — the app uses source-specific CSF defaults:

| Source | Cloth res | Class threshold | Notes |
|---|---|---|---|
| **LiDAR** | 0.30 m | ±5 cm | Default for airborne / terrestrial |
| **Photogrammetry** | 2.00 m | ±5 cm | Tighter CSF, vegetation index, building detection |
| **SLAM / Indoor** | 0.50 m | ±20 cm | Wall rejection, median z_grid smoothing |

#### Ground Classification
| Option | Description |
|---|---|
| Compute fresh | Run CSF or Grid Minimum (default) |
| Use existing class 2 | Skip CSF, use pre-classified ground from scanner software |

**CSF parameters:**
| Field | Default | Notes |
|---|---|---|
| Cloth resolution | 0.30 m | Smaller = more detail, slower |
| Class threshold | 0.05 m | ±distance from cloth = ground band |
| Rigidness | 1 | 1=slopes, 2=mixed, 3=flat |
| Iterations | 1000 | More = slower but more accurate |
| Slope smooth | ✓ | Recommended — prevents cloth snagging |
| Pre-thin voxel | 0.10 m | Reduces CSF input while preserving ground |

#### Height Classification
| Field | Default | Notes |
|---|---|---|
| Low veg starts at nZ | 0.10 m | Points below this are ground-adjacent |
| Low / Med boundary | 1.00 m | |
| Med / High boundary | 3.00 m | |
| Low noise threshold | −0.10 m | Points further below ground = class 7 |
| Tag probable ground | Off | Enable to mark uncertain slope points as class 12 |
| Probable ground band | 0.10 m | Extra margin beyond ground threshold on both sides |
| Model keypoints | Off | Class 8 — thinned ground for TIN building |
| Keypoint grid step | 8 m | Mirrors TerraScan FnScanClassifyModelKey |

#### Bird Detection
| Field | Default | Notes |
|---|---|---|
| Min height (nZ) | 0.02 m | Exclude ground noise |
| Max height (nZ) | 0.40 m | Taller than a starling = not a bird |
| DBSCAN eps | 0.30 m | Cluster spatial extent (bird body size) |
| DBSCAN min pts | 8 | Minimum points to form a cluster |
| Min footprint | 0.005 m² | Reject dust/noise |
| Max footprint | 1.0 m² | Reject kerbs, signs, vehicles |

---

## Point Cloud Source Modes

### LiDAR (default)
Standard airborne or terrestrial LiDAR. CSF cloth resolution 0.30 m, ±5 cm ground band.

### Photogrammetry
Dense SfM point clouds from drones (Recap, Agisoft, Pix4D). Activates:
- Vegetation index from RGB (NDVI with NIR, VARI without, ExG fallback)
- Building detection via RANSAC plane fitting → class 6 before CSF
- Tighter cloth (2.0 m) suited to smoother photogrammetry surfaces
- Bird detection disabled (too many false positives in dense RGB clouds)

### SLAM / Indoor-Outdoor
Leica BLK, Faro Focus, NavVis, Matterport-style scans. Activates:
- Coarser cloth (0.50 m) — less wall snagging
- Wider ground threshold (±20 cm) — handles floor surface variation
- **Z-range wall filter**: voxel cells spanning > 0.50 m in Z are walls/columns, excluded before CSF
- **Median z_grid smoothing**: 3-cell radius smooth on the ground raster to handle gaps from filtered walls
- Workers forced to 1 (SLAM scans are typically single large files)

---

## Streaming Classification

Files ≥ **30 million points** are automatically processed in streaming mode — no
configuration needed.

**How it works (2-pass):**

**Pass 1 — Ground surface:**
Stream the file in 5M-point chunks. Each chunk updates a 2D voxel grid (0.10 m cells)
tracking minimum Z per cell. After all chunks: ~1M ground candidates → CSF → ground
elevation raster (z_grid).

**Pass 2 — Classify all points:**
Stream again. For each chunk, compute nZ by bilinear interpolation of z_grid. Assign
ground / vegetation / probable-ground / noise classes and write to the output file.

**Peak RAM:** ~400 MB regardless of file size (tested on 883M point E57).

**Note:** Bird detection is skipped in streaming mode. Re-tile to <30M pts per tile
if bird detection is needed.

---

## Batch Processing

The **Batch Process** tab processes a folder of LAZ/LAS files sequentially.

1. Browse to a folder of LAZ/LAS files
2. Click **Scan Folder** — shows filenames and point counts
3. Configure output (alongside each source file, or a custom root folder)
4. Tick **Skip tiling** if files are already clean (recommended)
5. Click **Start Batch**

Classification settings (source type, CSF params, vegetation thresholds) are taken
from the **Classify** tab.

Large files ≥ 30M points automatically use streaming — no RAM budget needed.

---

## E57 → LAZ Conversion

Recap, Faro, Leica, and other scanners export E57 format. The app handles E57 in two ways:

### 1. In the Classify tab
Select an `.e57` file directly. It is converted to LAZ first (subprocess-isolated)
then classified normally.

### 2. E57 → LAZ tab (batch)
Dedicated converter for bulk E57 conversion before classification:

1. Browse to a folder of E57 files
2. Optionally set a separate output folder (default: alongside each E57)
3. Click **Scan** then **Convert All**

All scans inside each E57 are merged into a single LAZ. RGB and intensity are
preserved when available. Scan pose transforms are applied so coordinates are in the
project's global frame.

### 3. Standalone command-line converter
```cmd
convert_e57.bat
```
Or directly:
```cmd
python convert_e57.py "P:\path\to\file.e57"
python convert_e57.py "P:\folder"  --out "Q:\output"  --recursive
```

---

## Recommended Settings

### LiDAR — default (outdoor terrestrial / airborne)
```
Source:           LiDAR
Cloth resolution: 0.30 m
Class threshold:  0.05 m  (±5 cm)
Rigidness:        1
Iterations:       1000
Pre-thin voxel:   0.10 m
Probable ground:  Off (or 0.10 m band on sloped sites)
```

### SLAM — indoor / outdoor walk-through scan
```
Source:           SLAM / Indoor-outdoor scan
(All CSF params are set automatically by the app)
Probable ground:  On, 0.15 m band  (slope transitions)
```

### Photogrammetry — drone RGB/NIR
```
Source:           Drone photogrammetry
Vegetation index: Auto
Building detect:  On
(All other params set automatically)
```

### Speed mode — large dense tiles
```
Cloth resolution: 0.50 m   (fewer cloth particles)
Class threshold:  0.10 m
Iterations:       300
Pre-thin voxel:   0.20 m
```

---

## Building a Standalone EXE

```cmd
build.bat
```

Produces `dist\StarlingClassifier\StarlingClassifier.exe` (~200–350 MB).

**Key build settings (`build.spec`):**
- `pye57` must be included with `collect_all("pye57")` for E57 support
- `lazrs` included as hidden import for LAZ writing
- Heavy packages excluded: `torch`, `tensorflow`, `cupy`, `nvidia*`, `matplotlib`

---

## Troubleshooting

### E57 file crashes the app silently
pye57's C++ library (libE57Format) can segfault on malformed files. The GUI now
runs E57 conversion in a subprocess — if it crashes, the GUI stays alive and reports
the exit code. Test the file with the standalone converter first:
```cmd
python convert_e57.py "path\to\file.e57"
```

### Push to GitHub fails with HTTP 500
The repo contains a large file (`Classification.7z`, ~1.3 GB) committed to history.
GitHub blocks files > 100 MB. Use `git filter-repo` to remove it from history, or
push only the `Starling classifier v4.1\` folder content.

### Memory at 99% during batch
Ensure the updated code is running (v4.2+). Files ≥ 30M points now always use
streaming regardless of available RAM estimate. Restart the GUI after updating.

### Ground points classified as low noise on slopes
Fixed in v4.2. The noise check previously overrode ground classification on steep
slope transitions. Update to v4.2 and re-run. Optionally enable **Tag probable ground**
to flag uncertain slope boundary points as class 12 for manual review.

### "LAZ write failed"
```cmd
D:\YourPath\arunpy\Scripts\pip.exe install lazrs
```

### CSF stdout lines appear in console
Expected — CSF's C++ layer writes directly to stdout. Harmless.

### Processing very slow
- Switch to Speed mode settings (cloth 0.50 m, 300 iterations)
- For SLAM scans: workers are forced to 1 — this is intentional
- Streaming path (≥30M pts) is always sequential (single pass per file)

---

## Architecture Notes

### Per-tile pipeline (non-streaming)
```
LAZ tile + 5 m buffer from 8 neighbours
    │
    ▼
Voxel-thin → CSF cloth simulation → ground raster
    │
    ▼
nZ per point (bilinear interpolation of ground raster)
    │
    ▼
Height classification (ground / veg / probable-ground / noise)
    │
    ▼
Bird detection: DBSCAN on 0.02–0.40 m nZ candidates
    │
    ▼
Write classified LAZ + bird_contacts.csv
```

### Streaming pipeline (≥30M points)
```
Pass 1: stream in 5M-pt chunks → 2D voxel min-Z grid → CSF → z_grid raster
Pass 2: stream again → bilinear nZ lookup → classify → write chunk
Peak RAM: ~400 MB
```

### Folder layout
```
Starling classifier v4.1/
├── main.py                  # GUI entry point
├── convert_e57.py           # Standalone E57 → LAZ batch converter
├── convert_e57.bat          # Batch launcher for E57 converter
├── start_gui.bat            # Launch GUI
├── build.bat                # PyInstaller build
├── build.spec               # PyInstaller spec
├── requirements.txt
├── README.md
├── gui/
│   └── main_window.py       # PyQt6 window, WorkerThread, BatchWorkerThread, E57WorkerThread
└── processing/
    ├── tile_processor.py    # ProcessParams, process_single_file, _classify_streaming, process_all_tiles
    ├── ground_classifier.py # classify_ground_csf, classify_ground_csf_streaming, nz_from_grid
    ├── tiler.py             # tile_file, write_tile_boundaries_dxf
    ├── height_classifier.py # classify_heights (veg + noise + probable-ground)
    ├── bird_detector.py     # detect_bird_contacts, DBSCAN
    ├── las_io.py            # read_laz, write_classified_laz, convert_e57_to_laz
    └── photogrammetry_classifier.py  # vegetation index, building detection
```

---

## License

Internal tool — distribution per your organisation's policy.
