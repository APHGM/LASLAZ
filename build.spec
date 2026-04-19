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
    # ── THE BIG ONES (torch + CUDA ~5GB) ──────────────────────────────
    "torch", "torchvision", "torchaudio",
    "cupy", "cupy_backends", "cuda", "cudf", "cuml",
    "nvidia", "nvfuser",
    "tensorflow", "keras", "jax", "jaxlib",
    # ── Heavy scientific libs not used ────────────────────────────────
    "matplotlib", "pandas", "pandas_profiling",
    "IPython", "jupyter", "jupyter_client", "notebook", "ipykernel",
    "seaborn", "plotly", "bokeh", "altair",
    "sympy", "statsmodels", "patsy",
    "pyarrow", "fastparquet",
    "pyproj", "shapely", "fiona", "geopandas", "rasterio",
    "lxml", "bs4", "html5lib",
    "PIL", "Pillow",
    # ── Alternative Qt / GUI bindings ─────────────────────────────────
    "PyQt5", "PySide6", "PySide2", "tkinter", "wx",
    # ── scikit-learn addons / hyperparam tuning ───────────────────────
    "tune_sklearn", "ray", "skopt", "scikitplot",
    "optuna", "hyperopt", "mlflow",
    # ── Dev / doc / test tooling ──────────────────────────────────────
    "pytest", "unittest", "docutils", "sphinx", "lib2to3",
    "pydantic", "hypothesis", "nose",
    # ── Rarely-needed stdlib extras ───────────────────────────────────
    "http.cookiejar", "xmlrpc", "pydoc_data",
    "curses", "turtle", "turtledemo",
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
