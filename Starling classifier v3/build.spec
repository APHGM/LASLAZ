# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — Starling Classifier v2.0 (Photogrammetry Edition).
Run:  pyinstaller --noconfirm --clean build.spec
"""
import os
from PyInstaller.utils.hooks import (
    collect_submodules, collect_dynamic_libs, collect_data_files, collect_all,
)

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
    # scipy._external is the current location (scipy._lib path removed — no longer exists)
    "scipy._external.array_api_compat.numpy.fft",
    "psutil",
]
hiddenimports += collect_submodules("CSF")
# Pick up ALL scipy array_api_compat submodules regardless of scipy version
try:
    hiddenimports += collect_submodules("scipy._external.array_api_compat")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("scipy._lib.array_api_compat")
except Exception:
    pass
# Explicit local-package submodules — PyInstaller's static analysis sometimes
# misses these on multiprocessing spawn or dynamic imports
hiddenimports += collect_submodules("gui")
hiddenimports += collect_submodules("processing")

# CSF is a single .pyd extension, not a package — collect_dynamic_libs doesn't apply
binaries = []
datas = []

# lazrs — Rust native extension; collect_all picks up the .pyd binary
lazrs_datas, lazrs_bins, lazrs_hidden = collect_all("lazrs")
datas    += lazrs_datas
binaries += lazrs_bins
hiddenimports += lazrs_hidden

# laszip — C++ LASzip backend; fallback when lazrs can't decompress a variant
try:
    laszip_datas, laszip_bins, laszip_hidden = collect_all("laszip")
    datas    += laszip_datas
    binaries += laszip_bins
    hiddenimports += laszip_hidden
except Exception:
    pass

# laspy compression backends
hiddenimports += collect_submodules("laspy")

# ezdxf ships font definitions, default DXF templates, and uses pyparsing
# internally — collect_all grabs submodules + data files + binaries together.
# Filter out ezdxf.addons.browser — it needs PySide6/PyQt5; we use PyQt6.
ezdxf_datas, ezdxf_bins, ezdxf_hidden = collect_all("ezdxf")
ezdxf_hidden = [m for m in ezdxf_hidden if "ezdxf.addons.browser" not in m]
datas    += ezdxf_datas
binaries += ezdxf_bins
hiddenimports += ezdxf_hidden
# pyparsing is a runtime dependency of ezdxf
hiddenimports += ["pyparsing"]

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
    # open3d — removed in v2.1 (native lib crashed workers on Windows).
    # Building detection now uses pure numpy RANSAC + sklearn DBSCAN.
    "open3d",
    # ezdxf GUI browser addon needs PySide6 or PyQt5 — we use PyQt6, skip it
    "ezdxf.addons.browser",
    # pyproj is a transitive dep we never call — exclude to suppress data warning
    "pyproj",
]

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],   # so gui/, processing/ are seen as packages
    binaries=binaries,
    datas=datas,
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
    name="StarlingClassifier_v3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=os.path.join(SPECPATH, "Sterling.ico"),
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
    name="StarlingClassifier_v4.1",
)
