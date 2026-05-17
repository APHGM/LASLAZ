# Starling Ground-Contact Classifier — Project Context

> Hand-off document. Use this to brief a new chat / new AI / new developer
> so they don't have to start from scratch.

---

## What this tool does

Classifies **starling bird ground-contact points** in terrestrial LiDAR scans.

Pipeline:
1. **Tile** a single huge LAZ (>1 B points) or read pre-tiled folder
2. **Ground classification** via CSF (Cloth Simulation Filter) or Grid Minimum fallback
3. **Bird detection** — points 0.02–0.40 m above ground, clustered with DBSCAN,
   filtered by cluster footprint (bird-sized: 0.005–1.0 m²)
4. **Output** per-tile `.las` (ASPRS class 2 = ground, 20 = bird contact) +
   `bird_contacts.csv` with centroid XYZ, count, footprint per cluster

---

## Stack

| Component | Library |
|---|---|
| GUI | PyQt6 |
| LAS/LAZ I/O | laspy 2.x + lazrs |
| Ground (primary) | `cloth-simulation-filter` (CSF) |
| Ground (fallback) | numpy + scipy (grid min + morphological opening) |
| Clustering | sklearn DBSCAN (with voxel pre-thin for memory) |
| Build | PyInstaller |

Python venv: `D:\VSCode_Working\Python\arunpy\` (Python 3.12)

---

## Folder layout

```
D:\VSCode_Working\Python\Classification\Classification\
├── main.py                  ← GUI entry point
├── tile_single_laz.py       ← CLI tiler (streams LAZ → tiles)
├── start_gui.bat            ← double-click to launch GUI
├── run.bat                  ← CLI tile + GUI flow
├── build.spec               ← PyInstaller spec (lean build, ~200 MB)
├── requirements.txt
├── README.md
├── CONTEXT.md               ← this file
├── gui\
│   └── main_window.py       ← PyQt6 window + WorkerThread
└── processing\
    ├── tiler.py             ← chunked LAZ tiler
    ├── tile_processor.py    ← tile loop + ProcessParams dataclass
    ├── ground_classifier.py ← CSF + Grid Min implementations
    ├── bird_detector.py     ← height filter + DBSCAN + voxel downsample
    └── las_io.py            ← read_laz / write_classified_laz
```

---

## Recommended CSF settings (tight ground band)

```
Cloth resolution:  0.30 m
Class threshold:   0.05 m
Rigidness:         1   (1=slope, 2=mixed, 3=flat)
Iterations:        800
Pre-thin voxel:    0.10 m
Slope smooth:      enabled
```

These are the **defaults in both `ProcessParams` and the GUI spinboxes**.

---

## Bird detection settings

```
nZ min:              0.02 m
nZ max:              0.40 m
DBSCAN eps:          0.30 m
DBSCAN min points:   8
Min footprint:       0.005 m²
Max footprint:       1.0 m²
```

---

## Key design decisions / things that took us hours

1. **PDAL doesn't work in this env** — use pure Python.
2. **CSF package quirks**: pip name is `cloth-simulation-filter`, import name is `CSF`.
   `params.interations` (sic — typo in upstream library, do NOT fix to `iterations`).
3. **CSF on raw 5M+ point tiles hangs** → always voxel-thin to ~0.10–0.15 m
   before passing to CSF, then interpolate the resulting ground surface
   back to all original points.
4. **Tile edge handling** — each tile loads a 5 m buffer from the 8 neighbouring
   tiles before ground classification, then bird detection runs on core area only.
   Tile filenames follow `<easting>_<northing>.laz` convention (e.g.
   `498000_171500.laz`) so neighbours = ±tile_size on each coord.
5. **DBSCAN MemoryError fix**: if candidate count > 500K, run DBSCAN on a
   voxel-downsampled version (eps/3 voxel) and expand results back via KDTree.
   If candidate fraction > 5% of tile, skip — means ground is wrong.
6. **Grid Minimum bug fixed**: original used `min_filter → uniform_filter` which
   pushed surface DOWN below ground. Correct combo is
   `min_filter → max_filter` (morphological opening).
7. **PyInstaller bloat**: arunpy venv has torch/cupy/CUDA libs which balloon
   the build to 5+ GB. `build.spec` excludes them aggressively → ~200 MB.

---

## How to launch

| Method | Use case |
|---|---|
| Double-click `start_gui.bat` | Normal GUI launch |
| Double-click `run.bat` | CLI tile + GUI two-step (prompts for paths) |
| `python tile_single_laz.py in.laz out_dir --tile 100` | Tile only, scripted |
| Build exe: `pyinstaller --noconfirm --clean build.spec` | Distribution build |

GUI has two input modes (top dropdown):
- **Tile folder** — point at pre-tiled `*.laz` directory
- **Single LAZ file** — auto-tiles into `Tile_<filename>/` next to source

---

## Known limitations / open items

- Dense low scrub (<10 cm tall) merges into ground class — can't separate
  with geometry alone. Would need intensity / return-count post-pass.
- Very dense tiles (>500 M pts in 100×100 m) — tile smaller (25 m) first.
- No GPU acceleration yet. cuML DBSCAN would give 10–50× speedup on
  bird detection but requires NVIDIA + cuML install.
- Tile name parser assumes integer eastings/northings.

---

## To resume work in a new chat

Paste this whole file as your first message, plus:
1. What you want to change/add
2. Any new log output or error messages
3. The current `ProcessParams` values if you've tuned them

That's enough context for any AI to pick up where we left off.
