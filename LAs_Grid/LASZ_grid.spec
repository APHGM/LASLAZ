# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the LAS/LAZ Grid Batch Processor.

Build:
    pyinstaller --noconfirm --clean LASZ_grid.spec
"""
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

hiddenimports = [
    "lazrs",
    "laspy",
    "numpy",
]

binaries = collect_dynamic_libs("lazrs")

excludes = [
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras", "jax", "jaxlib",
    "cupy", "nvidia",
    "cupyx", "cupy_backends",
    "matplotlib", "pandas", "IPython", "jupyter", "notebook",
    "seaborn", "plotly", "bokeh",
    "PIL", "Pillow",
    "PyQt5", "PyQt6", "PySide6", "PySide2", "wx",
    "scipy", "sklearn", "scikit-learn",
    "shapely", "geopandas", "rasterio", "fiona", "pyarrow", "pyproj",
]

a = Analysis(
    ["LASZ_grid.py"],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LASZGridBatch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LASZGridBatch",
)
