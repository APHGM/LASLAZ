# LASLAZ — LiDAR & Point Cloud Tools

Python tools for processing LAS/LAZ/E57 point clouds on Windows.

---

## Projects

### [Starling Ground-Contact Classifier v4.2](Starling%20classifier%20v4.1/README.md)

A Windows desktop application (PyQt6) that classifies LiDAR, SLAM, and drone
photogrammetry point clouds — ground, vegetation, buildings, low noise, probable
ground (class 12), and starling bird-contact points.

**Key features:**
- Streaming 2-pass classification — handles 400M+ point files at ~400 MB RAM
- SLAM / indoor-outdoor scan mode (Leica BLK, Faro Focus, NavVis)
- Drone photogrammetry mode (RGB/NIR vegetation index, building detection)
- Batch Process tab — multi-file queue with auto streaming for large files
- E57 → LAZ conversion tab — Faro, Leica, Recap export support
- Probable ground (class 12) — flags uncertain slope boundary points for review
- Bird contact detection — DBSCAN clusters near ground surface (class 20)

**Output classes:** ground (2), low/med/high veg (3/4/5), building (6),
low noise (7), model keypoints (8), probable ground (12), bird contact (20)

→ [Full documentation](Starling%20classifier%20v4.1/README.md)

---

### Other folders

| Folder | Description |
|---|---|
| `Classification/` | Earlier classification experiments |
| `StarlingClassifier/` | Legacy v1–v2 builds |
| `StarlingClassifier_v2/` | v2.x photogrammetry edition |
| `SurfaceExtractor/` | Surface material extraction pipeline |
| `tools/` | Utility scripts |

---

## Requirements

- Windows 10/11, Python 3.12
- See `Starling classifier v4.1/requirements.txt` for dependencies
