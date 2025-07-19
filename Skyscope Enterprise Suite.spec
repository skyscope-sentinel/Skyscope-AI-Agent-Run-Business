# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_application.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config'), ('assets', 'assets')],
    hiddenimports=['PyQt6.QtCore', 'PyQt6.QtWidgets', 'PyQt6.QtGui', 'PyQt6.QtCharts', 'psutil', 'autonomous_orchestrator'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Skyscope Enterprise Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/skyscope_icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Skyscope Enterprise Suite',
)
app = BUNDLE(
    coll,
    name='Skyscope Enterprise Suite.app',
    icon='assets/skyscope_icon.icns',
    bundle_identifier=None,
)
