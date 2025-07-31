#!/bin/bash

# =============================================================================
# Quick Build Script - Fixes current issues and builds the app
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_step() { echo -e "${BLUE}>>> $1${NC}"; }
echo_success() { echo -e "${GREEN}✅ $1${NC}"; }
echo_warn() { echo -e "${YELLOW}⚠️ $1${NC}"; }
echo_error() { echo -e "${RED}❌ $1${NC}"; }

APP_NAME="Skyscope Sentinel"

echo_step "Skyscope Sentinel - Quick Build & Fix"
echo "====================================="

# 1. Fix missing logo issue
echo_step "Creating logo file..."
if [[ ! -f "logo.png" ]]; then
    python3 -c "
from PIL import Image, ImageDraw, ImageFont
import os

# Create a 512x512 image
size = 512
img = Image.new('RGBA', (size, size), (30, 144, 255, 255))  # Blue background
draw = ImageDraw.Draw(img)

# Add gradient effect
for i in range(size):
    alpha = int(255 * (1 - i / (size * 2)))
    if alpha > 0:
        color = (255, 255, 255, alpha)
        draw.rectangle([0, i, size, i+1], fill=color)

# Add text
try:
    font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 48)
except:
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 48)
    except:
        font = ImageFont.load_default()

text = 'SKYSCOPE'
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (size - text_width) // 2
y = (size - text_height) // 2 - 50

# Add text with shadow
draw.text((x+2, y+2), text, fill=(0, 0, 0, 128), font=font)
draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

# Add subtitle
try:
    small_font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 20)
except:
    try:
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except:
        small_font = ImageFont.load_default()

subtitle = 'AI Sentinel Intelligence'
bbox = draw.textbbox((0, 0), subtitle, font=small_font)
sub_width = bbox[2] - bbox[0]
sub_x = (size - sub_width) // 2
sub_y = y + text_height + 20

draw.text((sub_x+1, sub_y+1), subtitle, fill=(0, 0, 0, 128), font=small_font)
draw.text((sub_x, sub_y), subtitle, fill=(200, 200, 200, 255), font=small_font)

img.save('logo.png', 'PNG')
print('✅ Logo created successfully')
"
    echo_success "Logo created"
else
    echo_success "Logo already exists"
fi

# 2. Setup virtual environment with uv if available, otherwise use venv
echo_step "Setting up Python environment..."

if command -v uv >/dev/null 2>&1; then
    echo_step "Using uv for faster package management..."
    if [[ ! -d ".venv" ]]; then
        uv venv --python 3.11 .venv
    fi
    source .venv/bin/activate
    
    # Install core dependencies with uv
    echo_step "Installing core dependencies..."
    uv pip install --upgrade pip setuptools wheel
    uv pip install streamlit pillow numpy pyyaml python-dotenv requests cryptography pyinstaller
    
    # Try to install additional dependencies
    echo_step "Installing additional dependencies..."
    uv pip install pandas matplotlib plotly tqdm colorama rich psutil || echo_warn "Some optional packages failed to install"
    
else
    echo_step "Using standard venv and pip..."
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    
    # Install core dependencies
    pip install --upgrade pip setuptools wheel
    pip install streamlit pillow numpy pyyaml python-dotenv requests cryptography pyinstaller
    pip install pandas matplotlib plotly tqdm colorama rich psutil || echo_warn "Some optional packages failed to install"
fi

echo_success "Python environment ready"

# 3. Test core imports
echo_step "Testing core imports..."
python3 -c "
import sys
modules = ['streamlit', 'PIL', 'numpy', 'yaml', 'requests', 'cryptography']
failed = []

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
        failed.append(module)

if failed:
    print(f'⚠️ Failed imports: {failed}')
    print('Some features may be limited but core functionality should work')
else:
    print('🎉 All core modules imported successfully')
"

# 4. Create macOS icon
echo_step "Creating macOS icon..."
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    mkdir -p icon.iconset
    sips -z 16 16 logo.png --out icon.iconset/icon_16x16.png 2>/dev/null
    sips -z 32 32 logo.png --out icon.iconset/icon_16x16@2x.png 2>/dev/null
    sips -z 32 32 logo.png --out icon.iconset/icon_32x32.png 2>/dev/null
    sips -z 64 64 logo.png --out icon.iconset/icon_32x32@2x.png 2>/dev/null
    sips -z 128 128 logo.png --out icon.iconset/icon_128x128.png 2>/dev/null
    sips -z 256 256 logo.png --out icon.iconset/icon_128x128@2x.png 2>/dev/null
    sips -z 256 256 logo.png --out icon.iconset/icon_256x256.png 2>/dev/null
    sips -z 512 512 logo.png --out icon.iconset/icon_256x256@2x.png 2>/dev/null
    sips -z 512 512 logo.png --out icon.iconset/icon_512x512.png 2>/dev/null
    cp logo.png icon.iconset/icon_512x512@2x.png
    iconutil -c icns icon.iconset -o app_icon.icns
    rm -rf icon.iconset
    echo_success "macOS icon created"
else
    echo_warn "macOS icon tools not available, using PNG icon"
fi

# 5. Create optimized PyInstaller spec
echo_step "Creating PyInstaller spec file..."
cat > skyscope.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Determine icon file
if sys.platform == 'darwin' and os.path.exists('app_icon.icns'):
    icon_file = 'app_icon.icns'
elif os.path.exists('logo.png'):
    icon_file = 'logo.png'
else:
    icon_file = None

# Core hidden imports
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
]

# Optional imports (add if available)
optional_imports = [
    'pandas',
    'matplotlib',
    'plotly',
    'tqdm',
    'colorama',
    'rich',
    'psutil',
]

for module in optional_imports:
    try:
        __import__(module)
        hidden_imports.append(module)
    except ImportError:
        pass

# Data files
datas = []
if os.path.exists('logo.png'):
    datas.append(('logo.png', '.'))
if os.path.exists('knowledge_base.md'):
    datas.append(('knowledge_base.md', '.'))

# Streamlit static files
try:
    import streamlit
    streamlit_path = Path(streamlit.__file__).parent
    datas.append((str(streamlit_path / 'static'), 'streamlit/static'))
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
EOF

echo_success "PyInstaller spec created"

# 6. Build the application
echo_step "Building application with PyInstaller..."
pyinstaller --clean --noconfirm skyscope.spec

if [[ $? -eq 0 ]]; then
    echo_success "Application built successfully!"
    
    # Check if app bundle was created
    if [[ -d "dist/Skyscope Sentinel.app" ]]; then
        echo_success "macOS app bundle created: dist/Skyscope Sentinel.app"
        
        # Create DMG if create-dmg is available
        if command -v create-dmg >/dev/null 2>&1; then
            echo_step "Creating DMG..."
            create-dmg \
                --volname "Skyscope Sentinel" \
                --window-pos 200 120 \
                --window-size 800 400 \
                --icon-size 100 \
                --icon "Skyscope Sentinel.app" 200 190 \
                --hide-extension "Skyscope Sentinel.app" \
                --app-drop-link 600 185 \
                "Skyscope_Sentinel_v1.0.0.dmg" \
                "dist/Skyscope Sentinel.app" 2>/dev/null || echo_warn "DMG creation had warnings but may have succeeded"
            
            if [[ -f "Skyscope_Sentinel_v1.0.0.dmg" ]]; then
                echo_success "DMG created: Skyscope_Sentinel_v1.0.0.dmg"
            fi
        fi
    fi
    
    echo ""
    echo_success "Build completed successfully!"
    echo ""
    echo "📦 Build artifacts:"
    if [[ -d "dist" ]]; then
        ls -la dist/
    fi
    echo ""
    echo "🚀 Next steps:"
    echo "1. Test the app: open 'dist/Skyscope Sentinel.app'"
    echo "2. Install from DMG: open 'Skyscope_Sentinel_v1.0.0.dmg' (if created)"
    echo "3. For development: streamlit run app.py"
    
else
    echo_error "Build failed!"
    echo "Check the error messages above for details."
    exit 1
fi