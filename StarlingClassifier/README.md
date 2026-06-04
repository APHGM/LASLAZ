# Starling Ground-Contact Classifier

A Windows desktop application (PyQt6) that detects **starling bird ground-contact
points** in LiDAR point clouds — terrestrial or airborne.

It works by:
1. Classifying **ground** with the Cloth Simulation Filter (CSF) or a Grid-Minimum
   morphological filter
2. Finding **small clusters of points just above the ground surface** (height + footprint
   filtered, bird-sized only) using DBSCAN
3. Writing classified LAS/LAZ files (ASPRS classes: `2 = ground`, `20 = bird contact`)
   plus a CSV of bird-cluster centroids

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start (Existing Setup)](#quick-start-existing-setup)
3. [Fresh Install on a New Machine](#fresh-install-on-a-new-machine)
4. [Running the App](#running-the-app)
5. [GUI Reference](#gui-reference)
6. [Recommended Parameter Settings](#recommended-parameter-settings)
7. [Folder Layout](#folder-layout)
8. [Building a Standalone EXE](#building-a-standalone-exe)
9. [Distributing the EXE](#distributing-the-exe)
10. [Troubleshooting](#troubleshooting)
11. [Architecture Notes](#architecture-notes)
12. [Known Limitations](#known-limitations)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| Python | 3.10 | **3.12** (tested) |
| RAM | 16 GB | **32 – 64 GB** for large terrestrial scans |
| CPU | 4 cores | 8+ cores (parallel processing) |
| Disk | 50 GB free | 200+ GB for big datasets + intermediate tiles |

Tested with: Python 3.12, Windows 11, 64 GB RAM, Intel i7.

---

## Quick Start (Existing Setup)

If everything is already installed:

```cmd
:: Just launch the GUI
start_gui.bat
```

If you need to tile a huge LAZ first, then launch the GUI:
```cmd
run.bat
```

---

## Fresh Install on a New Machine

### Step 1 — Install Python 3.12
Download from https://www.python.org/downloads/windows/ — **tick "Add Python to PATH"** during install.

### Step 2 — Create a virtual environment
```cmd
cd D:\YourPath\
py -3.12 -m venv arunpy
arunpy\Scripts\activate
```

### Step 3 — Clone or copy the project folder
Copy the entire `StarlingClassifier\` folder to your new machine.

### Step 4 — Install dependencies
```cmd
cd D:\YourPath\StarlingClassifier
pip install -r requirements.txt
```

`requirements.txt` contents:
```
PyQt6
laspy[lazrs]
numpy
scipy
scikit-learn
cloth-simulation-filter
```

> **Note:** `lazrs` is the LAZ compression backend — `laspy[lazrs]` installs both together. If LAZ writing ever fails with a backend error, run `pip install lazrs` explicitly.

### Step 5 — Update the venv path in the launcher scripts
Edit `start_gui.bat`, `run.bat`, and `build.bat`. Change the line:
```bat
set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
```
to point at your new venv's `python.exe`.

### Step 6 — Test it
```cmd
start_gui.bat
```
The GUI should open within ~5 seconds.

---

## Running the App

There are **three ways** to run depending on your input:

### Option A — Single LAZ file (small to medium, <300 M points)
Best for typical terrestrial scans that fit in RAM.

1. Launch GUI (`start_gui.bat`)
2. Input mode: **"Single LAZ file"**
3. Browse to your `.las` or `.laz` file
4. **Tick "Skip tiling — process whole file as one"** ✅ (recommended)
5. Choose an output folder
6. Click **Start Processing**

### Option B — Single LAZ file with auto-tiling (very large, >300 M points)
For scans too big to fit in RAM.

1. Launch GUI
2. Input mode: **"Single LAZ file"**
3. Browse to your file
4. **Untick "Skip tiling"** so the file is split first
5. Set tile size (default 100 m; use 25 m for very dense scans)
6. Choose output folder, click Start

The app tiles into `Tile_<filename>\` next to your input, then processes each tile.

### Option C — Pre-tiled folder of LAZ tiles
If you already have tiles named `<easting>_<northing>.laz`.

1. Launch GUI
2. Input mode: **"Tile folder"**
3. Browse to the folder
4. Choose output folder, click Start

### Output
- `<tilename>_classified.laz` — LAS/LAZ with ASPRS classification codes:
  | Class | Meaning |
  |---|---|
  | 1 | Unclassified |
  | **2** | **Ground** |
  | **3** | **Low vegetation** (0.10 – 1.0 m above ground) |
  | **4** | **Medium vegetation** (1.0 – 3.0 m) |
  | **5** | **High vegetation** (≥ 3.0 m) |
  | **7** | **Low noise** (below ground surface) |
  | **8** | **Model keypoints** (thinned ground for TIN) — optional |
  | **20** | **Bird contact** |
- `bird_contacts.csv` — one row per detected bird cluster (centroid X/Y/Z, point count, footprint)
- `processing_summary.txt` — human-readable processing report

---

## GUI Reference

### Input / Output
| Field | Notes |
|---|---|
| Input mode | "Tile folder" or "Single LAZ file" |
| Buffer (m) | Edge buffer when processing tiles (5 m default) |
| Parallel workers | 1 = sequential. Higher = process N tiles in parallel processes. Auto-default = `cpu_count / 2`. |
| Output format | LAZ (compressed, recommended) or LAS (uncompressed) |
| LAS version | 1.4 (newer, recommended) or 1.2 (compatibility) |

### Ground Classification
| Source | When to use |
|---|---|
| **Compute fresh** | Default. Runs CSF or Grid Minimum on the points. |
| **Use existing classification 2** | When the LAS already has ground classified (e.g. from TerraScan / scanner software). Skips CSF, reads class 2 directly, only runs vegetation/bird detection on top. |

| Method | When to use |
|---|---|
| **CSF** (Cloth Simulation Filter) | **Recommended** for "Compute fresh". Handles slopes, vegetation, buildings well. |
| **Grid Minimum** | Fallback for very flat ground or when CSF is unavailable. Pure numpy/scipy. |

### Height Classification (new — TerraScan-style)
| Field | Description |
|---|---|
| Classify vegetation by height | Toggle — runs height-band classification using `nZ` from ground stage |
| Low veg starts at nZ | Default 0.10 m (excludes ground noise) |
| Low / Med boundary | Default 1.00 m |
| Med / High boundary | Default 3.00 m |
| Low noise threshold | Default −0.10 m (points below ground = class 7) |
| Classify model keypoints | Toggle — keeps lowest ground point per N × N grid cell as class 8 |
| Model keypoint grid step | Default 8 m (matches TerraScan macro convention) |

These mirror the TerraScan macro functions `FnScanClassifyHgtGrd`, `FnScanClassifyLow`,
and `FnScanClassifyModelKey`.

### Bird Contact Detection
| Field | Description |
|---|---|
| Min height above ground (m) | Lowest nZ to consider as bird (default 0.02 — excludes ground noise) |
| Max height above ground (m) | Highest nZ to consider (default 0.40 — taller than this isn't a bird) |
| DBSCAN eps (m) | Cluster spatial extent (default 0.30 — bird body size) |
| DBSCAN min points | Minimum points to form a cluster (default 8) |
| Min cluster footprint (m²) | Reject clusters smaller than this (default 0.005) |
| Max cluster footprint (m²) | Reject clusters bigger than this (default 1.0) |

---

## Recommended Parameter Settings

### CSF — tight ground band (recommended defaults)
```
Cloth resolution:  0.30 m
Class threshold:   0.05 m
Rigidness:         1   (1=slope, 2=mixed, 3=flat)
Iterations:        800–1000
Pre-thin voxel:    0.10 m
Slope smooth:      ✓ enabled
```

### CSF — speed mode (dense data, large tiles)
```
Cloth resolution:  0.50 m   (4× fewer cloth particles)
Class threshold:   0.10 m
Rigidness:         1
Iterations:        150–300  (much faster)
Pre-thin voxel:    0.20 m   (4× fewer input points)
Slope smooth:      ✓ enabled
```

### Bird detection — tighter false-positive control
```
Max height above ground: 0.15 m   (instead of 0.40 — starling standing height)
Max cluster footprint:   0.30 m²  (instead of 1.0 — kerbs/sigs/vehicles excluded)
DBSCAN min points:       15       (instead of 8 — denser cluster required)
```

---

## Folder Layout

```
StarlingClassifier/
├── main.py                  # GUI entry point
├── tile_single_laz.py       # CLI tiler (streams LAZ → tiles)
├── start_gui.bat            # Double-click to launch GUI
├── run.bat                  # CLI tile + GUI flow (asks for paths)
├── build.bat                # Clean PyInstaller build
├── package.bat              # Zip dist/ for sharing
├── build.spec               # PyInstaller spec (lean ~200 MB build)
├── requirements.txt         # pip dependencies
├── README.md                # this file
├── CONTEXT.md               # AI handoff / brief for new chats
├── Sterling.ico             # app icon
├── setting.txt              # saved parameter values
├── output/                  # default output location (created at runtime)
├── dist/                    # PyInstaller output
├── gui/
│   └── main_window.py       # PyQt6 window + WorkerThread
└── processing/
    ├── tiler.py             # chunked LAZ → tiles
    ├── tile_processor.py    # tile loop + ProcessParams dataclass + parallel dispatch
    ├── ground_classifier.py # CSF + Grid Min implementations
    ├── bird_detector.py     # height filter + DBSCAN + voxel downsample
    └── las_io.py            # read_laz / write_classified_laz
```

---

## Building a Standalone EXE

For distribution to colleagues who don't have Python.

```cmd
build.bat
```

This:
1. Cleans previous `dist/` and `build/` folders
2. Re-runs PyInstaller with `--clean --noconfirm` against `build.spec`
3. Produces `dist\StarlingClassifier\StarlingClassifier.exe`

**Build time:** ~3–6 minutes.
**Output size:** ~200–300 MB (whole `dist/StarlingClassifier/` folder).

### Build settings in `build.spec`
- **Excludes** `torch`, `tensorflow`, `cupy`, `nvidia*`, `matplotlib`, `PyQt5`, `PySide6`, etc. — these would balloon the build to 5+ GB if included.
- **Does NOT exclude `pandas`** — the `cloth-simulation-filter` package imports pandas internally; excluding it breaks CSF.
- **Includes** `lazrs` as a hidden import so LAZ writing works.
- **UTF-8 forced** via `PYTHONIOENCODING` env var so ✓/✗ characters in logs don't crash.

---

## Distributing the EXE

```cmd
package.bat
```

This zips the `dist\StarlingClassifier\` folder into `StarlingClassifier_<YYYY-MM-DD>.zip`
(plus `README.txt` for recipients).

### Recipient instructions
1. Extract the ZIP to any folder (e.g. `C:\Apps\StarlingClassifier\`)
2. **Extract the WHOLE folder, not just the .exe**
3. Double-click `StarlingClassifier.exe`
4. No Python installation required

### Distribution tips
- File size ~200–300 MB — email systems will block; use OneDrive/SharePoint/WeTransfer link
- Some antivirus tools flag PyInstaller exes as unknown (false-positive) — ask IT to whitelist if blocked
- First launch takes ~5–15 sec (Windows scanning bundled DLLs); faster after that

---

## Troubleshooting

### "FATAL ERROR: 'charmap' codec can't encode character …"
Unicode encoding crash. Already mitigated by `sys.stdout.reconfigure(encoding="utf-8")` in `main.py` and `encoding="utf-8"` on file writes. If it recurs, check for new ✓/✗/° characters in log strings.

### "LAZ write failed"
The `lazrs` backend isn't installed in the active Python env.
```cmd
D:\YourPath\arunpy\Scripts\pip.exe show lazrs
```
Verify the `Location:` line matches your venv. If not, reinstall in the venv:
```cmd
D:\YourPath\arunpy\Scripts\pip.exe install lazrs
```
Or just switch **Output Format** to "LAS (uncompressed)" in the GUI.

### "WARNING: high candidate fraction — ground surface may be incorrect"
Common on terrestrial scans. The 5% guard was tuned for airborne LiDAR. Either:
- Raise `MAX_CANDIDATE_FRACTION` in `processing/bird_detector.py` to `0.30`
- Or accept that bird detection skipped for that tile (ground classification still succeeded)

### Build size > 1 GB
Heavy dependency leaking through. Add it to `excludes` in `build.spec`. Common culprits: `torch`, `cupy`, `nvidia`, `tune_sklearn`, `ray`.

### Processing taking 3+ hours, progress bar stuck
- Check Task Manager: if RAM is ~maxed, you're swapping → reduce parallel workers
- CSF iterations × cloth resolution is the slowness driver — switch to "speed mode" settings above

### Parallel mode shows no per-tile logs until tile finishes
Expected. Worker subprocess logs are buffered and only flush when the tile returns. Reduce CSF iterations or use Workers = 1 if you want live progress.

### Pylance "value is not a known attribute of None"
Static type checker false-positives — the code runs fine. Either:
- Add `# type: ignore` comments
- Or add `# pyright: reportOptionalMemberAccess=false` at top of file

---

## Architecture Notes

### Pipeline (per tile)
```
LAZ tile + buffer points from 8 neighbours
       │
       ▼
Voxel-thin to 0.10–0.20 m (CSF needs sparser input)
       │
       ▼
CSF cloth simulation → ground / non-ground indices
       │
       ▼
Interpolate ground surface back to ALL original points → nZ per point
       │
       ▼
Filter: 0.02 m < nZ < 0.40 m AND not ground = bird candidates
       │
       ▼
Voxel downsample candidates if > 500K (memory protection)
       │
       ▼
DBSCAN on XY → bird clusters
       │
       ▼
Footprint filter (0.005 – 1.0 m² typical bird size)
       │
       ▼
Write LAZ with class 2 (ground) + class 20 (bird) + class 1 (other)
```

### Parallelism model
- `ProcessPoolExecutor` with N workers (configurable in GUI)
- Each worker handles one tile end-to-end (CSF + DBSCAN)
- Tiles ≥150 M points run **solo** (no parallel siblings — RAM protection)
- Logs buffered per worker, flushed when tile completes
- `multiprocessing.freeze_support()` call in `main.py` is REQUIRED for PyInstaller exe

### Edge handling
Each tile loads a configurable buffer (default 5 m) from its 8 grid neighbours.
Ground classification uses tile + buffer; bird detection uses core tile only.
This prevents ground-surface discontinuities at tile boundaries.

Tile filenames must follow `<easting>_<northing>.laz` convention
(e.g. `498000_171500.laz`) so neighbours can be found by ±tile_size on each coord.

---

## Known Limitations

- **Dense low scrub** (<10 cm tall) merges into ground class — geometry alone can't separate
  it from real ground. Would need intensity / return-count post-processing.
- **Very dense tiles** (>500 M points in a 100×100 m tile) — tile smaller (25 m) first.
- **CSF is slow** on huge dense data (1 hour+ for 100M+ point tile) — use speed-mode settings
  or tile smaller.
- **No GPU acceleration.** cuML DBSCAN would give 10–50× speedup but requires NVIDIA + CUDA + cuML install.
- **CSF C++ stdout** (`[0] Configuring terrain...` lines) bypasses the log capture in parallel
  mode — appears interleaved between workers.
- **Tile name parser** assumes integer eastings/northings split by underscore.

---

## Resuming Work / AI Handoff

The file `CONTEXT.md` in this folder is a condensed brief specifically for handing
the project to a new AI chat or developer. Paste its contents into a new conversation
along with what you want to change, and they'll have full context without
re-discovering it.

---

## Version History

- **v1.0** — Initial GUI + sequential tile processing (CSF + DBSCAN)
- **v1.1** — Added Grid Minimum fallback, fixed morphological opening bug
- **v1.2** — Added voxel thinning for CSF speedup, memory-safe DBSCAN
- **v1.3** — Added single-file mode with auto-tiling, build pipeline
- **v1.4** — Added LAS/LAZ output format choice, LAS version selection
- **v1.5** — Added parallel processing with auto-throttle for giant tiles
- **v1.6** — Added "Skip tiling" mode for small-to-medium files
- **v1.7** — Drone-LiDAR fix: RGB and extra dimensions now preserved through tiling and write
- **v1.8** — TerraScan-style height classification (low/med/high veg, low noise, model keypoints) + "Use existing classification 2" ground source

---

## License

Internal tool — distribution per your organisation's policy.
