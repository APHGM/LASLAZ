# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — Starling Classifier.
Run:  pyinstaller --noconfirm --clean build.spec
"""
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

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

# Only exclude large frameworks that are definitely not used
excludes = [
    # DL frameworks
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras", "jax", "jaxlib",
    "cupy", "nvidia",
    # Viz / notebooks
    "matplotlib", "pandas", "IPython", "jupyter", "notebook",
    "seaborn", "plotly", "bokeh",
    "PIL", "Pillow",
    # GUI toolkits we don't use
    "PyQt5", "PySide6", "PySide2", "tkinter", "wx",
    # Geo
    "shapely", "geopandas", "rasterio", "fiona", "pyarrow",
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
    icon='D:\VSCode_Working\Python\Classification\StarlingClassifier\Sterling.ico',
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
