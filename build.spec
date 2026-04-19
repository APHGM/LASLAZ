# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — lean build of Starling Classifier.
Run:  pyinstaller --noconfirm --clean build.spec
"""
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

block_cipher = None

hiddenimports = [
    "sklearn.neighbors._partition_nodes",
    "sklearn.utils._cython_blas",
    "sklearn.tree._utils",
    "scipy.spatial.transform._rotation_groups",
    "scipy._lib.array_api_compat.numpy.fft",
]
hiddenimports += collect_submodules("CSF")

binaries = collect_dynamic_libs("CSF")

excludes = [
    # ── Large DL frameworks (not used, saves ~5 GB) ───────────────────
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras", "jax", "jaxlib",
    "cupy", "cupy_backends", "cuda", "cudf", "cuml",
    "nvidia", "nvfuser",
    # ── Heavy viz/data packages (not used) ────────────────────────────
    "matplotlib", "pandas",
    "IPython", "jupyter", "notebook",
    "seaborn", "plotly", "bokeh",
    "pyarrow", "shapely", "geopandas", "rasterio",
    "PIL", "Pillow",
    # ── Alternative GUI bindings ──────────────────────────────────────
    "PyQt5", "PySide6", "PySide2", "tkinter", "wx",
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
    upx=True,             # set False if UPX not installed
    console=False,        # True to keep a console window (useful for debugging)
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
    upx=True,
    upx_exclude=[],
    name="StarlingClassifier",
)
