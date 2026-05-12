# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'pandas', 'matplotlib', 'tkinter', 'PIL', 'notebook', 'IPython', 'unittest', 'cv2', 'sklearn', 'pyproj', 'lxml', 'pyarrow', 'cryptography', 'PyQt6.QtNetwork', 'PyQt6.QtPdf', 'PyQt6.QtSvg', 'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtBluetooth', 'PyQt6.QtMultimedia', 'PyQt6.QtPositioning', 'PyQt6.QtSensors', 'PyQt6.QtWebChannel', 'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebSetup', 'PyQt6.QtWebSockets', 'PyQt6.QtXml'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='adaptive_ground_classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
