#!/bin/bash

# =============================================================================
# Skyscope Sentinel Intelligence - Final Cross-Platform Build Script
# =============================================================================
# This script creates production-ready applications for macOS, Windows, and Linux
# using modern Python packaging tools and handles all dependency issues
# =============================================================================

set -e

# Configuration
APP_NAME="Skyscope Sentinel"
APP_VERSION="1.0.0"
BUILD_DIR="builds"
DIST_DIR="distributions"

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

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*) echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

# Main build function
main() {
    echo_step "Skyscope Sentinel Intelligence - Final Build System"
    echo "========================================================="
    
    local os=$(detect_os)
    echo_step "Detected OS: $os"
    
    # Create directories
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    # Activate virtual environment
    if [[ -d ".venv" ]]; then
        echo_step "Activating uv virtual environment..."
        source .venv/bin/activate
    elif [[ -d "venv" ]]; then
        echo_step "Activating standard virtual environment..."
        source venv/bin/activate
    else
        echo_error "No virtual environment found. Please run setup first."
        exit 1
    fi
    
    # Verify core dependencies
    echo_step "Verifying core dependencies..."
    python3 -c "
import sys
required = ['streamlit', 'PIL', 'numpy', 'yaml', 'requests', 'cryptography', 'pyinstaller']
missing = []

for module in required:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing.append(module)
        print(f'❌ {module}')

if missing:
    print(f'Missing: {missing}')
    sys.exit(1)
else:
    print('🎉 All required dependencies available')
"
    
    if [[ $? -ne 0 ]]; then
        echo_error "Missing dependencies. Installing..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install streamlit pillow numpy pyyaml requests cryptography pyinstaller
        else
            pip install streamlit pillow numpy pyyaml requests cryptography pyinstaller
        fi
    fi
    
    # Verify assets
    echo_step "Verifying assets..."
    if [[ ! -f "skyscope-logo.png" ]]; then
        echo_warn "skyscope-logo.png not found"
    else
        echo_success "Logo found: skyscope-logo.png"
    fi
    
    if [[ ! -f "Skyscope.icns" ]]; then
        echo_warn "Skyscope.icns not found"
    else
        echo_success "Icon found: Skyscope.icns"
    fi
    
    # Create optimized spec file
    echo_step "Creating optimized PyInstaller spec..."
    create_spec_file "$os"
    
    # Build application
    echo_step "Building application..."
    build_application "$os"
    
    # Create distribution packages
    echo_step "Creating distribution packages..."
    create_distribution "$os"
    
    # Summary
    echo_step "Build Summary"
    echo "============="
    if [[ -d "$DIST_DIR" ]]; then
        echo_success "Distribution files:"
        ls -la "$DIST_DIR/"
    fi
    
    echo_success "Build completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    case $os in
        "macos")
            echo "1. Test: open 'dist/Skyscope Sentinel.app'"
            echo "2. Install: open '$DIST_DIR/Skyscope_Sentinel_macOS.dmg'"
            ;;
        "linux")
            echo "1. Test: ./dist/Skyscope_Sentinel/Skyscope_Sentinel"
            echo "2. Install: chmod +x '$DIST_DIR/Skyscope_Sentinel_Linux.AppImage' && ./$DIST_DIR/Skyscope_Sentinel_Linux.AppImage"
            ;;
        "windows")
            echo "1. Test: dist/Skyscope_Sentinel/Skyscope_Sentinel.exe"
            echo "2. Install: $DIST_DIR/Skyscope_Sentinel_Windows_Setup.exe"
            ;;
    esac
}

# Create optimized spec file
create_spec_file() {
    local target_os=$1
    
    cat > "skyscope_optimized.spec" << 'EOF'
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Determine icon and logo files
icon_file = None
logo_file = None

if sys.platform == 'darwin':
    if os.path.exists('Skyscope.icns'):
        icon_file = 'Skyscope.icns'
elif sys.platform == 'win32':
    if os.path.exists('Skyscope.ico'):
        icon_file = 'Skyscope.ico'

if os.path.exists('skyscope-logo.png'):
    logo_file = 'skyscope-logo.png'

# Essential hidden imports only
hidden_imports = [
    # Streamlit core
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.state',
    'streamlit.components.v1',
    'streamlit.runtime.caching',
    'streamlit.runtime.legacy_caching',
    
    # PIL/Pillow
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'PIL._imaging',
    
    # Core libraries
    'numpy',
    'yaml',
    'pyyaml',
    'requests',
    'cryptography',
    'cryptography.fernet',
    'cryptography.hazmat',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.backends',
    'cryptography.hazmat.backends.openssl',
    
    # Python standard library
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
    'concurrent.futures',
    'multiprocessing',
    'subprocess',
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
    'weakref',
    'gc',
    'pickle',
    'sqlite3',
    'csv',
    'tempfile',
    'shutil',
    'zipfile',
    'tarfile',
    'gzip',
    'urllib',
    'urllib.request',
    'urllib.parse',
    'http',
    'http.client',
    'ssl',
    'socket',
    'email',
    'mimetypes',
    'configparser',
    'argparse',
]

# Data files
datas = []

# Add logo files
if logo_file and os.path.exists(logo_file):
    datas.append((logo_file, '.'))

# Add knowledge base
if os.path.exists('knowledge_base.md'):
    datas.append(('knowledge_base.md', '.'))

# Add config files
for config_file in ['config.json', 'config.yaml', 'config.yml']:
    if os.path.exists(config_file):
        datas.append((config_file, '.'))

# Streamlit static files
try:
    import streamlit
    streamlit_path = Path(streamlit.__file__).parent
    static_path = streamlit_path / 'static'
    if static_path.exists():
        datas.append((str(static_path), 'streamlit/static'))
    
    # Add runtime files
    runtime_path = streamlit_path / 'runtime'
    if runtime_path.exists():
        datas.append((str(runtime_path), 'streamlit/runtime'))
        
except ImportError:
    pass

# Analysis
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
        # Exclude test modules
        'test',
        'tests',
        'testing',
        'unittest',
        'doctest',
        'pydoc',
        
        # Exclude GUI frameworks we don't use
        'tkinter',
        'tk',
        'tcl',
        '_tkinter',
        'turtle',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        
        # Exclude large optional libraries
        'scipy',
        'sklearn',
        'tensorflow',
        'torch',
        'cv2',
        'opencv',
        'matplotlib.tests',
        'numpy.tests',
        'pandas.tests',
        'PIL.tests',
        
        # Exclude development tools
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'distutils',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Create PYZ
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Skyscope_Sentinel',
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

# Collect all files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Skyscope_Sentinel',
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
            'CFBundleExecutable': 'Skyscope_Sentinel',
            'CFBundleIconFile': 'Skyscope.icns',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSMinimumSystemVersion': '10.14',
            'NSHumanReadableCopyright': 'Copyright © 2025 Skyscope Technologies',
            'CFBundleDocumentTypes': [],
            'NSPrincipalClass': 'NSApplication',
            'LSApplicationCategoryType': 'public.app-category.productivity',
        },
    )
EOF

    echo_success "Optimized spec file created"
}

# Build application
build_application() {
    local target_os=$1
    
    echo_step "Building with PyInstaller..."
    
    # Clean previous builds
    rm -rf build dist
    
    # Build with timeout to prevent hanging
    timeout 600 pyinstaller --clean --noconfirm skyscope_optimized.spec
    
    if [[ $? -eq 0 ]]; then
        echo_success "Application built successfully"
        
        # Verify build
        case $target_os in
            "macos")
                if [[ -d "dist/Skyscope Sentinel.app" ]]; then
                    echo_success "macOS app bundle created"
                elif [[ -d "dist/Skyscope_Sentinel" ]]; then
                    echo_success "macOS application directory created"
                else
                    echo_error "Build verification failed"
                    return 1
                fi
                ;;
            "linux"|"windows")
                if [[ -d "dist/Skyscope_Sentinel" ]]; then
                    echo_success "Application directory created"
                else
                    echo_error "Build verification failed"
                    return 1
                fi
                ;;
        esac
        
        return 0
    else
        echo_error "PyInstaller build failed"
        return 1
    fi
}

# Create distribution packages
create_distribution() {
    local os=$1
    
    case $os in
        "macos")
            create_macos_distribution
            ;;
        "linux")
            create_linux_distribution
            ;;
        "windows")
            create_windows_distribution
            ;;
    esac
}

# Create macOS distribution
create_macos_distribution() {
    echo_step "Creating macOS distribution..."
    
    local app_path=""
    if [[ -d "dist/Skyscope Sentinel.app" ]]; then
        app_path="dist/Skyscope Sentinel.app"
    elif [[ -d "dist/Skyscope_Sentinel" ]]; then
        # Create app bundle manually
        mkdir -p "dist/Skyscope Sentinel.app/Contents/MacOS"
        mkdir -p "dist/Skyscope Sentinel.app/Contents/Resources"
        
        cp -r "dist/Skyscope_Sentinel/"* "dist/Skyscope Sentinel.app/Contents/MacOS/"
        
        if [[ -f "Skyscope.icns" ]]; then
            cp "Skyscope.icns" "dist/Skyscope Sentinel.app/Contents/Resources/"
        fi
        
        # Create Info.plist
        cat > "dist/Skyscope Sentinel.app/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Skyscope Sentinel</string>
    <key>CFBundleDisplayName</key>
    <string>Skyscope Sentinel Intelligence</string>
    <key>CFBundleIdentifier</key>
    <string>com.skyscope.sentinel</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleExecutable</key>
    <string>Skyscope_Sentinel</string>
    <key>CFBundleIconFile</key>
    <string>Skyscope.icns</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>10.14</string>
</dict>
</plist>
EOF
        
        app_path="dist/Skyscope Sentinel.app"
        echo_success "App bundle created manually"
    else
        echo_error "No application found to package"
        return 1
    fi
    
    # Create DMG
    if command -v create-dmg >/dev/null 2>&1; then
        echo_step "Creating DMG with create-dmg..."
        create-dmg \
            --volname "Skyscope Sentinel" \
            --window-pos 200 120 \
            --window-size 800 400 \
            --icon-size 100 \
            --icon "Skyscope Sentinel.app" 200 190 \
            --hide-extension "Skyscope Sentinel.app" \
            --app-drop-link 600 185 \
            "$DIST_DIR/Skyscope_Sentinel_macOS.dmg" \
            "$app_path" 2>/dev/null || {
                echo_warn "create-dmg had issues, trying alternative method..."
                
                # Alternative: create simple DMG
                hdiutil create -volname "Skyscope Sentinel" -srcfolder "$app_path" -ov -format UDZO "$DIST_DIR/Skyscope_Sentinel_macOS.dmg"
            }
        
        if [[ -f "$DIST_DIR/Skyscope_Sentinel_macOS.dmg" ]]; then
            echo_success "DMG created: $DIST_DIR/Skyscope_Sentinel_macOS.dmg"
        fi
    else
        echo_warn "create-dmg not available, creating ZIP archive..."
        cd dist
        zip -r "../$DIST_DIR/Skyscope_Sentinel_macOS.zip" "Skyscope Sentinel.app"
        cd ..
        echo_success "ZIP archive created: $DIST_DIR/Skyscope_Sentinel_macOS.zip"
    fi
}

# Create Linux distribution
create_linux_distribution() {
    echo_step "Creating Linux distribution..."
    
    if [[ ! -d "dist/Skyscope_Sentinel" ]]; then
        echo_error "Linux application not found"
        return 1
    fi
    
    # Create simple tar.gz
    cd dist
    tar -czf "../$DIST_DIR/Skyscope_Sentinel_Linux.tar.gz" "Skyscope_Sentinel"
    cd ..
    echo_success "Linux archive created: $DIST_DIR/Skyscope_Sentinel_Linux.tar.gz"
    
    # Try to create AppImage if tools are available
    if command -v wget >/dev/null 2>&1; then
        echo_step "Attempting to create AppImage..."
        
        # Create AppDir structure
        local appdir="$BUILD_DIR/Skyscope_Sentinel.AppDir"
        mkdir -p "$appdir/usr/bin"
        mkdir -p "$appdir/usr/share/applications"
        mkdir -p "$appdir/usr/share/icons/hicolor/256x256/apps"
        
        # Copy application
        cp -r "dist/Skyscope_Sentinel/"* "$appdir/usr/bin/"
        
        # Copy icon
        if [[ -f "skyscope-logo.png" ]]; then
            cp "skyscope-logo.png" "$appdir/usr/share/icons/hicolor/256x256/apps/skyscope-sentinel.png"
            cp "skyscope-logo.png" "$appdir/skyscope-sentinel.png"
        fi
        
        # Create desktop file
        cat > "$appdir/usr/share/applications/skyscope-sentinel.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Skyscope Sentinel
Exec=Skyscope_Sentinel
Icon=skyscope-sentinel
Categories=Office;Development;
Comment=AI Agentic Swarm Autonomous System
Terminal=false
EOF
        
        # Copy desktop file to AppDir root
        cp "$appdir/usr/share/applications/skyscope-sentinel.desktop" "$appdir/"
        
        # Create AppRun script
        cat > "$appdir/AppRun" << 'APPRUN_EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/Skyscope_Sentinel" "$@"
APPRUN_EOF
        chmod +x "$appdir/AppRun"
        
        # Download appimagetool if not available
        if [[ ! -f "appimagetool" ]]; then
            echo_step "Downloading appimagetool..."
            wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -O appimagetool || {
                echo_warn "Could not download appimagetool"
                return 0
            }
            chmod +x appimagetool
        fi
        
        # Create AppImage
        ./appimagetool "$appdir" "$DIST_DIR/Skyscope_Sentinel_Linux.AppImage" 2>/dev/null || {
            echo_warn "AppImage creation failed, but tar.gz is available"
        }
        
        if [[ -f "$DIST_DIR/Skyscope_Sentinel_Linux.AppImage" ]]; then
            echo_success "AppImage created: $DIST_DIR/Skyscope_Sentinel_Linux.AppImage"
        fi
    fi
}

# Create Windows distribution
create_windows_distribution() {
    echo_step "Creating Windows distribution..."
    
    if [[ ! -d "dist/Skyscope_Sentinel" ]]; then
        echo_error "Windows application not found"
        return 1
    fi
    
    # Create ZIP archive
    cd dist
    zip -r "../$DIST_DIR/Skyscope_Sentinel_Windows.zip" "Skyscope_Sentinel"
    cd ..
    echo_success "Windows ZIP archive created: $DIST_DIR/Skyscope_Sentinel_Windows.zip"
}

# Run main function
main "$@"