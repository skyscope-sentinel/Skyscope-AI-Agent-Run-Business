#!/bin/bash
#
# BUILD_MACOS_APP.sh
#
# Build script for creating a complete macOS application bundle
# for the Skyscope AI Agentic Swarm Business/Enterprise system
#
# This script creates a fully functional .app bundle with all dependencies
# and proper macOS integration including system tray, notifications, etc.
#
# Created: January 2025
# Author: Skyscope Sentinel Intelligence
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="Skyscope Enterprise Suite"
APP_VERSION="2.0.0"
APP_IDENTIFIER="ai.skyscope.enterprise"
BASE_DIR="$(pwd)"
VENV_DIR="$BASE_DIR/venv"
BUILD_DIR="$BASE_DIR/build"
DIST_DIR="$BASE_DIR/dist"
APP_BUNDLE_DIR="$DIST_DIR/$APP_NAME.app"

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}\n"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    fi
}

# Show welcome banner
clear
echo -e "${PURPLE}"
echo "  ███████╗██╗  ██╗██╗   ██╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗"
echo "  ██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝"
echo "  ███████╗█████╔╝  ╚████╔╝ ███████╗██║     ██║   ██║██████╔╝█████╗  "
echo "  ╚════██║██╔═██╗   ╚██╔╝  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝  "
echo "  ███████║██║  ██╗   ██║   ███████║╚██████╗╚██████╔╝██║     ███████╗"
echo "  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝"
echo "                                                                    "
echo "  ███████╗███╗   ██╗████████╗███████╗██████╗ ██████╗ ██████╗ ██╗███████╗███████╗"
echo "  ██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝"
echo "  █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝██████╔╝██████╔╝██║███████╗█████╗  "
echo "  ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔═══╝ ██╔══██╗██║╚════██║██╔══╝  "
echo "  ███████╗██║ ╚████║   ██║   ███████╗██║  ██║██║     ██║  ██║██║███████║███████╗"
echo "  ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝"
echo -e "${NC}"
echo -e "${CYAN}                      macOS App Builder v${APP_VERSION}${NC}"
echo -e "${CYAN}                      ========================${NC}"
echo ""

# Check if running on macOS
print_header "Checking Build Requirements"

if [ "$(uname)" != "Darwin" ]; then
    print_error "This script must be run on macOS to build a macOS application."
    exit 1
fi

print_success "Running on macOS"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    print_error "Virtual environment not found. Please run COMPLETE_SYSTEM_SETUP.sh first."
    exit 1
fi

print_success "Virtual environment found"

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if main application exists
if [ ! -f "main_application.py" ]; then
    print_error "main_application.py not found. Cannot build application."
    exit 1
fi

print_success "Main application found"

# Install PyInstaller if not already installed
print_info "Checking PyInstaller..."
if ! pip show pyinstaller >/dev/null 2>&1; then
    print_info "Installing PyInstaller..."
    pip install pyinstaller
fi

print_success "PyInstaller ready"

# Clean previous builds
print_header "Cleaning Previous Builds"

if [ -d "$BUILD_DIR" ]; then
    print_info "Removing previous build directory..."
    rm -rf "$BUILD_DIR"
fi

if [ -d "$DIST_DIR" ]; then
    print_info "Removing previous dist directory..."
    rm -rf "$DIST_DIR"
fi

print_success "Previous builds cleaned"

# Create application icon
print_header "Creating Application Icon"

print_info "Creating application icon..."

# Create a simple icon using Python
python3 << 'EOF'
from PIL import Image, ImageDraw
import os

# Create a 512x512 icon
size = 512
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw a gradient background
for i in range(size):
    color = int(255 * (1 - i / size))
    draw.line([(0, i), (size, i)], fill=(0, color, 255, 255))

# Draw the "S" logo
draw.text((size//4, size//4), "S", fill=(255, 255, 255, 255), font_size=size//2)

# Save as PNG
os.makedirs('assets', exist_ok=True)
img.save('assets/icon.png')
print("Icon created successfully")
EOF

# Convert PNG to ICNS for macOS
if command_exists sips; then
    print_info "Converting icon to ICNS format..."
    create_dir "assets"
    
    # Create iconset directory
    mkdir -p "assets/icon.iconset"
    
    # Generate different sizes
    sips -z 16 16 assets/icon.png --out assets/icon.iconset/icon_16x16.png
    sips -z 32 32 assets/icon.png --out assets/icon.iconset/icon_16x16@2x.png
    sips -z 32 32 assets/icon.png --out assets/icon.iconset/icon_32x32.png
    sips -z 64 64 assets/icon.png --out assets/icon.iconset/icon_32x32@2x.png
    sips -z 128 128 assets/icon.png --out assets/icon.iconset/icon_128x128.png
    sips -z 256 256 assets/icon.png --out assets/icon.iconset/icon_128x128@2x.png
    sips -z 256 256 assets/icon.png --out assets/icon.iconset/icon_256x256.png
    sips -z 512 512 assets/icon.png --out assets/icon.iconset/icon_256x256@2x.png
    sips -z 512 512 assets/icon.png --out assets/icon.iconset/icon_512x512.png
    cp assets/icon.png assets/icon.iconset/icon_512x512@2x.png
    
    # Create ICNS file
    iconutil -c icns assets/icon.iconset
    
    print_success "Application icon created"
else
    print_warning "sips command not found. Using default icon."
fi

# Create PyInstaller spec file
print_header "Creating PyInstaller Specification"

print_info "Creating PyInstaller spec file..."

cat > skyscope_enterprise.spec << EOF
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files and submodules
datas = []
hiddenimports = []

# Add PyQt6 and related modules
hiddenimports.extend([
    'PyQt6.QtCore',
    'PyQt6.QtGui', 
    'PyQt6.QtWidgets',
    'PyQt6.QtCharts',
    'PyQt6.QtWebEngine',
    'PyQt6.sip',
])

# Add scientific computing libraries
hiddenimports.extend([
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'plotly',
    'scipy',
    'scikit-learn',
])

# Add AI and ML libraries
hiddenimports.extend([
    'openai',
    'anthropic',
    'google.generativeai',
    'huggingface_hub',
    'transformers',
    'torch',
    'tensorflow',
    'langchain',
    'chromadb',
    'sentence_transformers',
])

# Add crypto and finance libraries
hiddenimports.extend([
    'ccxt',
    'web3',
    'binance',
    'yfinance',
    'ta',
])

# Add web and networking libraries
hiddenimports.extend([
    'requests',
    'aiohttp',
    'websockets',
    'fastapi',
    'uvicorn',
    'streamlit',
])

# Add system libraries
hiddenimports.extend([
    'psutil',
    'cryptography',
    'pydantic',
    'pillow',
    'opencv-python',
])

block_cipher = None

a = Analysis(
    ['main_application.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Skyscope Enterprise Suite',
)

app = BUNDLE(
    coll,
    name='$APP_NAME.app',
    icon='assets/icon.icns',
    bundle_identifier='$APP_IDENTIFIER',
    version='$APP_VERSION',
    info_plist={
        'CFBundleName': '$APP_NAME',
        'CFBundleDisplayName': '$APP_NAME',
        'CFBundleIdentifier': '$APP_IDENTIFIER',
        'CFBundleVersion': '$APP_VERSION',
        'CFBundleShortVersionString': '$APP_VERSION',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSMinimumSystemVersion': '10.14.0',
        'NSHumanReadableCopyright': '© 2025 Skyscope Sentinel Intelligence',
        'CFBundleDocumentTypes': [],
        'LSApplicationCategoryType': 'public.app-category.business',
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': True
        },
        'NSCameraUsageDescription': 'This app may use the camera for AI processing.',
        'NSMicrophoneUsageDescription': 'This app may use the microphone for voice commands.',
        'NSNetworkVolumesUsageDescription': 'This app needs network access for AI services.',
        'NSDesktopFolderUsageDescription': 'This app may access desktop files for processing.',
        'NSDocumentsFolderUsageDescription': 'This app may access documents for processing.',
        'NSDownloadsFolderUsageDescription': 'This app may access downloads for processing.',
    },
)
EOF

print_success "PyInstaller spec file created"

# Build the application
print_header "Building macOS Application"

print_info "Starting PyInstaller build process..."
print_info "This may take several minutes..."

pyinstaller --clean skyscope_enterprise.spec

if [ $? -eq 0 ]; then
    print_success "Application built successfully!"
else
    print_error "Build failed. Check the output above for errors."
    exit 1
fi

# Verify the application bundle
print_header "Verifying Application Bundle"

if [ -d "$APP_BUNDLE_DIR" ]; then
    print_success "Application bundle created: $APP_BUNDLE_DIR"
    
    # Check bundle structure
    if [ -f "$APP_BUNDLE_DIR/Contents/MacOS/Skyscope Enterprise Suite" ]; then
        print_success "Executable found in bundle"
    else
        print_warning "Executable not found in expected location"
    fi
    
    if [ -f "$APP_BUNDLE_DIR/Contents/Info.plist" ]; then
        print_success "Info.plist found in bundle"
    else
        print_warning "Info.plist not found in bundle"
    fi
    
    # Get bundle size
    BUNDLE_SIZE=$(du -sh "$APP_BUNDLE_DIR" | cut -f1)
    print_info "Application bundle size: $BUNDLE_SIZE"
    
else
    print_error "Application bundle not found at expected location"
    exit 1
fi

# Create DMG installer (optional)
print_header "Creating DMG Installer"

if command_exists hdiutil; then
    print_info "Creating DMG installer..."
    
    DMG_NAME="$DIST_DIR/Skyscope Enterprise Suite v$APP_VERSION.dmg"
    
    # Remove existing DMG
    if [ -f "$DMG_NAME" ]; then
        rm "$DMG_NAME"
    fi
    
    # Create temporary DMG
    hdiutil create -size 1g -fs HFS+ -volname "Skyscope Enterprise Suite" -srcfolder "$APP_BUNDLE_DIR" "$DMG_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "DMG installer created: $DMG_NAME"
    else
        print_warning "Failed to create DMG installer"
    fi
else
    print_warning "hdiutil not found. Skipping DMG creation."
fi

# Create launcher script
print_header "Creating Launcher Script"

print_info "Creating launcher script..."

cat > "Launch Skyscope Enterprise Suite.command" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
open "dist/Skyscope Enterprise Suite.app"
EOF

chmod +x "Launch Skyscope Enterprise Suite.command"

print_success "Launcher script created"

# Final summary
print_header "Build Complete"

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Build completed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Your Skyscope Enterprise Suite macOS application is ready:"
echo ""
echo "  📱 Application Bundle: $APP_BUNDLE_DIR"
if [ -f "$DMG_NAME" ]; then
echo "  💿 DMG Installer: $DMG_NAME"
fi
echo "  🚀 Launcher Script: Launch Skyscope Enterprise Suite.command"
echo ""
echo "To run the application:"
echo "  • Double-click the .app bundle in Finder"
echo "  • Or double-click the launcher script"
echo "  • Or run: open \"$APP_BUNDLE_DIR\""
echo ""
echo "The application includes:"
echo "  ✓ Complete GUI interface with real-time monitoring"
echo "  ✓ Debug console with live business activity output"
echo "  ✓ 10,000 AI agent orchestration system"
echo "  ✓ Autonomous income generation capabilities"
echo "  ✓ Crypto trading and portfolio management"
echo "  ✓ System tray integration"
echo "  ✓ Dark theme optimized for professional use"
echo ""
echo -e "${CYAN}Enjoy your autonomous AI business empire! 🚀${NC}"
echo ""