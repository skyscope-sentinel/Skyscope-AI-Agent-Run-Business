# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Determine icon file - use the provided assets
if sys.platform == 'darwin' and os.path.exists('Skyscope.icns'):
    icon_file = 'Skyscope.icns'
elif os.path.exists('skyscope-logo.png'):
    icon_file = 'skyscope-logo.png'
elif os.path.exists('logo.png'):
    icon_file = 'logo.png'
else:
    icon_file = None

# Core hidden imports - only include what we know exists
hidden_imports = [
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.state',
    'streamlit.components.v1',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'numpy',
    'yaml',
    'pyyaml',
    'requests',
    'cryptography',
    'cryptography.fernet',
    'json',
    'uuid',
    'datetime',
    'pathlib',
    'typing',
    'dataclasses',
    'enum',
    'logging',
    'threading',
    'queue',
    'asyncio',
    'os',
    'sys',
    'platform',
    'base64',
    'hashlib',
    'secrets',
    'random',
    'time',
    'collections',
    'functools',
    'copy',
    'io',
    'contextlib',
    'inspect',
    'importlib',
    'string',
    're',
    'math',
    'types',
    'abc',
]

# Data files - only include files that exist
datas = []
if os.path.exists('skyscope-logo.png'):
    datas.append(('skyscope-logo.png', '.'))
if os.path.exists('logo.png'):
    datas.append(('logo.png', '.'))
if os.path.exists('knowledge_base.md'):
    datas.append(('knowledge_base.md', '.'))

# Streamlit static files
try:
    import streamlit
    streamlit_path = Path(streamlit.__file__).parent
    static_path = streamlit_path / 'static'
    if static_path.exists():
        datas.append((str(static_path), 'streamlit/static'))
except ImportError:
    pass

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'pandas.tests',
        'PIL.tests',
        'test',
        'tests',
        'testing',
        'tkinter',
        'tk',
        'tcl',
        '_tkinter',
        'turtle',
        'pydoc',
        'doctest',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Skyscope Sentinel',
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
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Skyscope Sentinel',
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Skyscope Sentinel.app',
        icon=icon_file,
        bundle_identifier='com.skyscope.sentinel',
        version='1.0.0',
        info_plist={
            'CFBundleName': 'Skyscope Sentinel',
            'CFBundleDisplayName': 'Skyscope Sentinel Intelligence',
            'CFBundleIdentifier': 'com.skyscope.sentinel',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleExecutable': 'Skyscope Sentinel',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSMinimumSystemVersion': '10.14',
            'NSHumanReadableCopyright': 'Copyright © 2025 Skyscope Technologies',
        },
    )