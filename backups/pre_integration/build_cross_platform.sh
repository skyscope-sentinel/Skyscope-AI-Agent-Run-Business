#!/bin/bash

# =============================================================================
# Skyscope Sentinel Intelligence - Cross-Platform Build System
# =============================================================================
# This script builds GUI applications for macOS, Windows, and Linux
# using modern Python packaging tools (uv, conda) and PyInstaller
# =============================================================================

set -e  # Exit on any error

# --- Configuration ---
APP_NAME="Skyscope Sentinel"
APP_IDENTIFIER="com.skyscope.sentinel"
APP_VERSION="1.0.0"
PYTHON_VERSION="3.11"  # Use 3.11 for better compatibility
BUILD_DIR="builds"
DIST_DIR="dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---
echo_step() {
    echo -e "${BLUE}>>> $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warn() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies based on OS
install_system_deps() {
    local os=$(detect_os)
    echo_step "Installing system dependencies for $os..."
    
    case $os in
        "macos")
            if ! command_exists brew; then
                echo_step "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            # Install required tools
            brew install --quiet git python@3.11 node create-dmg upx || true
            echo_success "macOS dependencies installed"
            ;;
        "linux")
            if command_exists apt-get; then
                sudo apt-get update -qq
                sudo apt-get install -y git python3.11 python3.11-venv python3.11-dev \
                    nodejs npm build-essential upx-ucl fuse libfuse2 || true
            elif command_exists yum; then
                sudo yum install -y git python3.11 python3.11-venv python3.11-devel \
                    nodejs npm gcc gcc-c++ make upx fuse fuse-libs || true
            elif command_exists pacman; then
                sudo pacman -S --noconfirm git python nodejs npm base-devel upx fuse2 || true
            fi
            echo_success "Linux dependencies installed"
            ;;
        "windows")
            echo_warn "Windows detected. Please ensure you have:"
            echo "  - Python 3.11 installed"
            echo "  - Git installed"
            echo "  - Visual Studio Build Tools or equivalent"
            ;;
    esac
}

# Install uv (modern Python package manager)
install_uv() {
    if ! command_exists uv; then
        echo_step "Installing uv (modern Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        echo_success "uv installed"
    else
        echo_success "uv already installed"
    fi
}

# Install conda/miniconda
install_conda() {
    if ! command_exists conda; then
        echo_step "Installing Miniconda..."
        local os=$(detect_os)
        local installer=""
        
        case $os in
            "macos")
                if [[ $(uname -m) == "arm64" ]]; then
                    installer="Miniconda3-latest-MacOSX-arm64.sh"
                else
                    installer="Miniconda3-latest-MacOSX-x86_64.sh"
                fi
                ;;
            "linux")
                installer="Miniconda3-latest-Linux-x86_64.sh"
                ;;
            "windows")
                installer="Miniconda3-latest-Windows-x86_64.exe"
                ;;
        esac
        
        if [[ -n "$installer" ]]; then
            curl -O "https://repo.anaconda.com/miniconda/$installer"
            if [[ $os == "windows" ]]; then
                echo_warn "Please run $installer manually on Windows"
            else
                bash "$installer" -b -p "$HOME/miniconda3"
                rm "$installer"
                export PATH="$HOME/miniconda3/bin:$PATH"
                conda init bash
            fi
            echo_success "Miniconda installed"
        fi
    else
        echo_success "Conda already installed"
    fi
}

# Create conda environment
setup_conda_env() {
    echo_step "Setting up conda environment..."
    
    # Initialize conda in current shell
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/opt/miniconda3/etc/profile.d/conda.sh"
    fi
    
    # Create environment
    conda create -n skyscope python=$PYTHON_VERSION -y || true
    conda activate skyscope
    
    echo_success "Conda environment created and activated"
}

# Setup Python environment with uv
setup_uv_env() {
    echo_step "Setting up Python environment with uv..."
    
    # Create virtual environment with uv
    uv venv --python $PYTHON_VERSION .venv
    source .venv/bin/activate
    
    # Install dependencies with uv (much faster than pip)
    echo_step "Installing dependencies with uv..."
    uv pip install --upgrade pip setuptools wheel
    
    # Install core dependencies first
    uv pip install streamlit pillow numpy pyyaml python-dotenv requests cryptography
    
    # Try to install full requirements, but continue if some fail
    if [[ -f "requirements.txt" ]]; then
        echo_step "Installing from requirements.txt..."
        uv pip install -r requirements.txt || {
            echo_warn "Some optional dependencies failed to install, continuing with core packages..."
        }
    fi
    
    # Install build tools
    uv pip install pyinstaller auto-py-to-exe nuitka
    
    echo_success "Python environment setup complete"
}

# Create application assets
create_assets() {
    echo_step "Creating application assets..."
    
    # Create logo if it doesn't exist
    if [[ ! -f "logo.png" ]]; then
        echo_step "Generating default logo..."
        python3 -c "
from PIL import Image, ImageDraw, ImageFont
import os

# Create a 512x512 image with gradient background
size = 512
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Create gradient background
for i in range(size):
    alpha = int(255 * (1 - i / size))
    color = (30, 144, 255, alpha)  # Blue gradient
    draw.rectangle([0, i, size, i+1], fill=color)

# Add text
try:
    font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 60)
except:
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 60)
    except:
        font = ImageFont.load_default()

text = 'SKYSCOPE'
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (size - text_width) // 2
y = (size - text_height) // 2

# Add text with shadow
draw.text((x+2, y+2), text, fill=(0, 0, 0, 128), font=font)
draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

# Add subtitle
try:
    small_font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 24)
except:
    try:
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
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
print('Logo created: logo.png')
"
        echo_success "Default logo created"
    else
        echo_success "Logo already exists"
    fi
    
    # Create icon files for different platforms
    echo_step "Creating platform-specific icons..."
    
    # macOS icon (.icns)
    if command_exists iconutil; then
        mkdir -p icon.iconset
        sips -z 16 16 logo.png --out icon.iconset/icon_16x16.png 2>/dev/null || true
        sips -z 32 32 logo.png --out icon.iconset/icon_16x16@2x.png 2>/dev/null || true
        sips -z 32 32 logo.png --out icon.iconset/icon_32x32.png 2>/dev/null || true
        sips -z 64 64 logo.png --out icon.iconset/icon_32x32@2x.png 2>/dev/null || true
        sips -z 128 128 logo.png --out icon.iconset/icon_128x128.png 2>/dev/null || true
        sips -z 256 256 logo.png --out icon.iconset/icon_128x128@2x.png 2>/dev/null || true
        sips -z 256 256 logo.png --out icon.iconset/icon_256x256.png 2>/dev/null || true
        sips -z 512 512 logo.png --out icon.iconset/icon_256x256@2x.png 2>/dev/null || true
        sips -z 512 512 logo.png --out icon.iconset/icon_512x512.png 2>/dev/null || true
        cp logo.png icon.iconset/icon_512x512@2x.png 2>/dev/null || true
        iconutil -c icns icon.iconset -o app_icon.icns 2>/dev/null || true
        rm -rf icon.iconset
        echo_success "macOS icon created"
    fi
    
    # Windows icon (.ico) - using Pillow
    python3 -c "
try:
    from PIL import Image
    img = Image.open('logo.png')
    img.save('app_icon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
    print('Windows icon created: app_icon.ico')
except Exception as e:
    print(f'Could not create Windows icon: {e}')
" || true
}

# Create PyInstaller spec file
create_spec_file() {
    local target_os=$1
    echo_step "Creating PyInstaller spec file for $target_os..."
    
    cat > "skyscope_${target_os}.spec" << EOF
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Determine icon file based on platform
if sys.platform == 'darwin':
    icon_file = 'app_icon.icns' if os.path.exists('app_icon.icns') else None
elif sys.platform == 'win32':
    icon_file = 'app_icon.ico' if os.path.exists('app_icon.ico') else None
else:
    icon_file = 'logo.png' if os.path.exists('logo.png') else None

# Hidden imports for all dependencies
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
    'pandas',
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
    'concurrent.futures',
    'multiprocessing',
    'subprocess',
    'shutil',
    'tempfile',
    'os',
    'sys',
    'platform',
    'socket',
    'urllib',
    'urllib.request',
    'urllib.parse',
    'http',
    'http.client',
    'ssl',
    'hashlib',
    'base64',
    'hmac',
    'secrets',
    'random',
    'time',
    'calendar',
    'collections',
    'itertools',
    'functools',
    'operator',
    'copy',
    'pickle',
    'sqlite3',
    'csv',
    'io',
    'contextlib',
    'weakref',
    'gc',
    'inspect',
    'importlib',
    'pkgutil',
    'zipfile',
    'tarfile',
    'gzip',
    'bz2',
    'lzma',
    'zlib',
    'email',
    'email.mime',
    'email.mime.text',
    'email.mime.multipart',
    'email.mime.base',
    'mimetypes',
    'unicodedata',
    'locale',
    'gettext',
    'argparse',
    'configparser',
    'textwrap',
    'string',
    're',
    'fnmatch',
    'glob',
    'linecache',
    'traceback',
    'warnings',
    'types',
    'abc',
    'numbers',
    'math',
    'cmath',
    'decimal',
    'fractions',
    'statistics',
    'array',
    'bisect',
    'heapq',
    'keyword',
    'reprlib',
    'pprint',
    'enum',
]

# Data files to include
datas = []

# Add logo if it exists
if os.path.exists('logo.png'):
    datas.append(('logo.png', '.'))

# Add knowledge base if it exists
if os.path.exists('knowledge_base.md'):
    datas.append(('knowledge_base.md', '.'))

# Add any other data files
for file in ['config.json', 'README.md', 'LICENSE']:
    if os.path.exists(file):
        datas.append((file, '.'))

# Streamlit static files
try:
    import streamlit
    streamlit_path = Path(streamlit.__file__).parent
    datas.append((str(streamlit_path / 'static'), 'streamlit/static'))
    datas.append((str(streamlit_path / 'runtime'), 'streamlit/runtime'))
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
        'matplotlib',
        'scipy',
        'sklearn',
        'tensorflow',
        'torch',
        'cv2',
        'opencv',
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
    name='${APP_NAME}',
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
    name='${APP_NAME}',
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='${APP_NAME}.app',
        icon=icon_file,
        bundle_identifier='${APP_IDENTIFIER}',
        version='${APP_VERSION}',
        info_plist={
            'CFBundleName': '${APP_NAME}',
            'CFBundleDisplayName': '${APP_NAME}',
            'CFBundleIdentifier': '${APP_IDENTIFIER}',
            'CFBundleVersion': '${APP_VERSION}',
            'CFBundleShortVersionString': '${APP_VERSION}',
            'CFBundleExecutable': '${APP_NAME}',
            'CFBundleIconFile': 'app_icon.icns',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSMinimumSystemVersion': '10.14',
            'NSHumanReadableCopyright': 'Copyright © 2025 Skyscope Technologies',
            'CFBundleDocumentTypes': [],
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
        },
    )
EOF

    echo_success "PyInstaller spec file created for $target_os"
}

# Build application
build_app() {
    local target_os=$1
    echo_step "Building application for $target_os..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR/$target_os"
    
    # Build with PyInstaller
    pyinstaller --clean --noconfirm "skyscope_${target_os}.spec" \
        --distpath "$BUILD_DIR/$target_os/dist" \
        --workpath "$BUILD_DIR/$target_os/build" \
        --specpath "$BUILD_DIR/$target_os"
    
    if [[ $? -eq 0 ]]; then
        echo_success "Application built successfully for $target_os"
        return 0
    else
        echo_error "Build failed for $target_os"
        return 1
    fi
}

# Create distribution packages
create_distributions() {
    local os=$(detect_os)
    echo_step "Creating distribution packages..."
    
    case $os in
        "macos")
            create_macos_dmg
            ;;
        "linux")
            create_linux_appimage
            ;;
        "windows")
            create_windows_installer
            ;;
    esac
}

# Create macOS DMG
create_macos_dmg() {
    if [[ ! -d "$BUILD_DIR/macos/dist/${APP_NAME}.app" ]]; then
        echo_error "macOS app bundle not found"
        return 1
    fi
    
    echo_step "Creating macOS DMG..."
    
    local dmg_name="${APP_NAME// /_}_v${APP_VERSION}_macOS.dmg"
    
    if command_exists create-dmg; then
        create-dmg \
            --volname "$APP_NAME" \
            --volicon "app_icon.icns" \
            --window-pos 200 120 \
            --window-size 800 400 \
            --icon-size 100 \
            --icon "$APP_NAME.app" 200 190 \
            --hide-extension "$APP_NAME.app" \
            --app-drop-link 600 185 \
            "$DIST_DIR/$dmg_name" \
            "$BUILD_DIR/macos/dist/$APP_NAME.app"
        
        echo_success "DMG created: $DIST_DIR/$dmg_name"
    else
        echo_warn "create-dmg not available, copying app bundle instead"
        cp -r "$BUILD_DIR/macos/dist/$APP_NAME.app" "$DIST_DIR/"
    fi
}

# Create Linux AppImage
create_linux_appimage() {
    if [[ ! -d "$BUILD_DIR/linux/dist/${APP_NAME}" ]]; then
        echo_error "Linux application not found"
        return 1
    fi
    
    echo_step "Creating Linux AppImage..."
    
    # Create AppDir structure
    local appdir="$BUILD_DIR/linux/${APP_NAME}.AppDir"
    mkdir -p "$appdir/usr/bin"
    mkdir -p "$appdir/usr/share/applications"
    mkdir -p "$appdir/usr/share/icons/hicolor/512x512/apps"
    
    # Copy application
    cp -r "$BUILD_DIR/linux/dist/${APP_NAME}"/* "$appdir/usr/bin/"
    
    # Copy icon
    cp logo.png "$appdir/usr/share/icons/hicolor/512x512/apps/${APP_NAME,,}.png"
    cp logo.png "$appdir/${APP_NAME,,}.png"
    
    # Create desktop file
    cat > "$appdir/usr/share/applications/${APP_NAME,,}.desktop" << EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec=${APP_NAME}
Icon=${APP_NAME,,}
Categories=Office;Development;
Comment=AI Agentic Swarm Autonomous System
Terminal=false
EOF
    
    # Copy desktop file to AppDir root
    cp "$appdir/usr/share/applications/${APP_NAME,,}.desktop" "$appdir/"
    
    # Create AppRun script
    cat > "$appdir/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/${APP_NAME}" "$@"
EOF
    chmod +x "$appdir/AppRun"
    
    # Download appimagetool if not available
    if [[ ! -f "appimagetool" ]]; then
        echo_step "Downloading appimagetool..."
        wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -O appimagetool
        chmod +x appimagetool
    fi
    
    # Create AppImage
    local appimage_name="${APP_NAME// /_}_v${APP_VERSION}_Linux.AppImage"
    ./appimagetool "$appdir" "$DIST_DIR/$appimage_name"
    
    if [[ $? -eq 0 ]]; then
        echo_success "AppImage created: $DIST_DIR/$appimage_name"
    else
        echo_error "AppImage creation failed"
    fi
}

# Create Windows installer
create_windows_installer() {
    if [[ ! -d "$BUILD_DIR/windows/dist/${APP_NAME}" ]]; then
        echo_error "Windows application not found"
        return 1
    fi
    
    echo_step "Creating Windows installer..."
    
    # Create NSIS installer script
    cat > "installer.nsi" << EOF
!define APP_NAME "$APP_NAME"
!define APP_VERSION "$APP_VERSION"
!define APP_PUBLISHER "Skyscope Technologies"
!define APP_URL "https://skyscope.ai"
!define APP_EXECUTABLE "${APP_NAME}.exe"

Name "\${APP_NAME}"
OutFile "$DIST_DIR/${APP_NAME// /_}_v${APP_VERSION}_Windows_Setup.exe"
InstallDir "\$PROGRAMFILES64\\\${APP_NAME}"
RequestExecutionLevel admin

Page directory
Page instfiles

Section "Install"
    SetOutPath "\$INSTDIR"
    File /r "$BUILD_DIR/windows/dist/${APP_NAME}/*"
    
    CreateDirectory "\$SMPROGRAMS\\\${APP_NAME}"
    CreateShortCut "\$SMPROGRAMS\\\${APP_NAME}\\\${APP_NAME}.lnk" "\$INSTDIR\\\${APP_EXECUTABLE}"
    CreateShortCut "\$DESKTOP\\\${APP_NAME}.lnk" "\$INSTDIR\\\${APP_EXECUTABLE}"
    
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\\\${APP_NAME}" "DisplayName" "\${APP_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\\\${APP_NAME}" "UninstallString" "\$INSTDIR\uninstall.exe"
    WriteUninstaller "\$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "\$INSTDIR\*.*"
    RMDir /r "\$INSTDIR"
    Delete "\$SMPROGRAMS\\\${APP_NAME}\\\${APP_NAME}.lnk"
    Delete "\$DESKTOP\\\${APP_NAME}.lnk"
    RMDir "\$SMPROGRAMS\\\${APP_NAME}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\\\${APP_NAME}"
SectionEnd
EOF
    
    if command_exists makensis; then
        makensis installer.nsi
        echo_success "Windows installer created"
    else
        echo_warn "NSIS not available, creating ZIP archive instead"
        cd "$BUILD_DIR/windows/dist"
        zip -r "../../../$DIST_DIR/${APP_NAME// /_}_v${APP_VERSION}_Windows.zip" "${APP_NAME}"
        cd - > /dev/null
        echo_success "Windows ZIP archive created"
    fi
}

# Test the built application
test_application() {
    local target_os=$1
    echo_step "Testing application for $target_os..."
    
    case $target_os in
        "macos")
            if [[ -d "$BUILD_DIR/macos/dist/${APP_NAME}.app" ]]; then
                echo_success "macOS app bundle exists"
                # Test if executable exists
                if [[ -f "$BUILD_DIR/macos/dist/${APP_NAME}.app/Contents/MacOS/${APP_NAME}" ]]; then
                    echo_success "macOS executable found"
                else
                    echo_error "macOS executable not found"
                fi
            else
                echo_error "macOS app bundle not found"
            fi
            ;;
        "linux")
            if [[ -d "$BUILD_DIR/linux/dist/${APP_NAME}" ]]; then
                echo_success "Linux application directory exists"
                if [[ -f "$BUILD_DIR/linux/dist/${APP_NAME}/${APP_NAME}" ]]; then
                    echo_success "Linux executable found"
                else
                    echo_error "Linux executable not found"
                fi
            else
                echo_error "Linux application not found"
            fi
            ;;
        "windows")
            if [[ -d "$BUILD_DIR/windows/dist/${APP_NAME}" ]]; then
                echo_success "Windows application directory exists"
                if [[ -f "$BUILD_DIR/windows/dist/${APP_NAME}/${APP_NAME}.exe" ]]; then
                    echo_success "Windows executable found"
                else
                    echo_error "Windows executable not found"
                fi
            else
                echo_error "Windows application not found"
            fi
            ;;
    esac
}

# Main build function
main() {
    echo_step "Skyscope Sentinel Intelligence - Cross-Platform Build System"
    echo "=============================================================="
    
    local current_os=$(detect_os)
    local target_platforms=("$current_os")
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                target_platforms=("macos" "linux" "windows")
                shift
                ;;
            --platform)
                target_platforms=("$2")
                shift 2
                ;;
            --clean)
                echo_step "Cleaning build directories..."
                rm -rf "$BUILD_DIR" "$DIST_DIR"
                echo_success "Build directories cleaned"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --all              Build for all platforms"
                echo "  --platform OS      Build for specific platform (macos/linux/windows)"
                echo "  --clean            Clean build directories"
                echo "  --help             Show this help"
                exit 0
                ;;
            *)
                echo_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Create directories
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    # Install system dependencies
    install_system_deps
    
    # Install modern Python tools
    install_uv
    
    # Setup Python environment
    if command_exists conda; then
        setup_conda_env
    else
        setup_uv_env
    fi
    
    # Create assets
    create_assets
    
    # Build for each target platform
    for platform in "${target_platforms[@]}"; do
        echo_step "Building for $platform..."
        
        create_spec_file "$platform"
        
        if build_app "$platform"; then
            test_application "$platform"
        else
            echo_error "Build failed for $platform"
            continue
        fi
    done
    
    # Create distribution packages
    create_distributions
    
    # Summary
    echo_step "Build Summary"
    echo "=============="
    
    if [[ -d "$DIST_DIR" ]]; then
        echo_success "Distribution files created in $DIST_DIR:"
        ls -la "$DIST_DIR"
    fi
    
    echo_success "Cross-platform build completed!"
    echo ""
    echo "Next steps:"
    echo "1. Test the applications on their respective platforms"
    echo "2. Sign the applications for distribution (if needed)"
    echo "3. Upload to distribution channels"
}

# Run main function with all arguments
main "$@"