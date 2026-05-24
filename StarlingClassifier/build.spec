# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — Starling Classifier.
Run:  pyinstaller --noconfirm --clean build.spec
"""
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# Force UTF-8 in the bundled exe so unicode chars (✓, °, ²) don't crash
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

block_cipher = None

hiddenimports = [
    "sklearn.neighbors._partition_nodes",
    "sklearn.utils._cython_blas",
    "sklearn.utils._typedefs",
    "sklearn.tree._utils",
    "scipy.spatial.transform._rotation_groups",
    "scipy._lib.array_api_compat.numpy.fft",
    "lazrs",
]
hiddenimports += collect_submodules("CSF")

binaries = collect_dynamic_libs("CSF")

# Large frameworks we definitely don't use.
# NOTE: do NOT add `pandas` here — the cloth-simulation-filter (CSF) package
# imports pandas internally during init. Excluding pandas breaks CSF.
excludes = [
    # Deep learning
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras", "jax", "jaxlib",
    "cupy", "nvidia",
    # Visualisation / notebooks
    "matplotlib", "IPython", "jupyter", "notebook",
    "seaborn", "plotly", "bokeh",
    "PIL", "Pillow",
    # Other GUI toolkits
    "PyQt5", "PySide6", "PySide2", "tkinter", "wx",
    # Heavy geospatial libs we don't call
    "shapely", "geopandas", "rasterio", "fiona", "pyarrow",
    # Sklearn add-ons (often pull in torch/ray)
    "tune_sklearn", "ray", "skopt", "scikitplot",
]

a = Analysis(
    ["main.py"],
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
    name="StarlingClassifier",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=r"D:\VSCode_Working\Python\Classification\StarlingClassifier\Sterling.ico",
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
    name="StarlingClassifier",
)
