# Starling Ground-Contact Classifier

A PyQt6 desktop application for detecting starling ground-contact points in LiDAR point cloud data.

## Features

- **Ground classification** via Cloth Simulation Filter (CSF) or Grid Minimum
- **Bird-contact detection** using DBSCAN clustering on near-ground points
- **Tile-based processing** with automatic neighbour buffering for seamless results
- **CSV summary** of detected bird-contact clusters with centroid coordinates and footprint

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Build Standalone EXE

```bash
pip install pyinstaller
pyinstaller --noconfirm --clean build.spec
```

The output is in `dist/StarlingClassifier/`.

## Usage

1. Select a folder of `.laz` tiles
2. Choose an output folder
3. Adjust ground classification and bird detection parameters
4. Click **Start Processing**
