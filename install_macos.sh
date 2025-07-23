#!/bin/bash

# ==============================================================================
# Skyscope Sentinel Intelligence - macOS Application Packager
# ==============================================================================
# This script automates the entire process of setting up the environment,
# cloning the repository, and packaging the application into a distributable
# .dmg file for macOS (Intel x86_64).
#
# It performs the following steps:
# 1. Checks and installs necessary system dependencies like Homebrew and Git.
# 2. Checks and installs a compatible version of Python (3.11+).
# 3. Clones the application repository from GitHub.
# 4. Sets up a dedicated Python virtual environment.
# 5. Installs all required Python packages from requirements.txt.
# 6. Generates a macOS-compatible application icon.
# 7. Bundles the Python application into a standalone .app using PyInstaller.
# 8. Packages the .app bundle into a professional .dmg disk image.
# ==============================================================================

# --- Script Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Variables ---
REPO_URL="https://github.com/skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI.git"
REPO_DIR="Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI"
VENV_DIR="venv"
APP_NAME="Skyscope Sentinel"
PYTHON_VERSION_MAJOR=3
PYTHON_VERSION_MINOR=11
LOGO_FILE="logo.png" # Assumes a logo.png exists in the repo root

# --- Color Definitions for Logging ---
C_BLUE='\033[0;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_RED='\033[0;31m'
C_NC='\033[0m' # No Color

# --- Helper Functions ---
echo_step() {
    echo -e "\n${C_BLUE}>>> $1${C_NC}"
}

echo_success() {
    echo -e "${C_GREEN}✅ $1${C_NC}"
}

echo_warn() {
    echo -e "${C_YELLOW}⚠️ $1${C_NC}"
}

echo_error() {
    echo -e "${C_RED}❌ $1${C_NC}"
    exit 1
}

command_exists() {
    command -v "$1" &> /dev/null
}

# --- Main Execution ---

# 1. Check and Install Homebrew
echo_step "Checking for Homebrew..."
if ! command_exists brew; then
    echo_warn "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add Homebrew to PATH for the current session
    (echo; echo 'eval "$(/usr/local/bin/brew shellenv)"') >> ~/.zprofile
    eval "$(/usr/local/bin/brew shellenv)"
    echo_success "Homebrew installed."
else
    echo_success "Homebrew is already installed."
fi

# 2. Check and Install Dependencies (Git, Python, Node, create-dmg)
echo_step "Checking for dependencies (Git, Python, Node.js, create-dmg)..."
dependencies=("git" "python@${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" "node" "create-dmg")
for dep in "${dependencies[@]}"; do
    # For python, check with `python3` command
    if [[ "$dep" == "python@"* ]]; then
        if command_exists python3 && [[ $(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") == "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" ]]; then
            echo_success "Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} is already installed."
            continue
        else
             echo_warn "Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} not found or not the default. Installing via Homebrew..."
             brew install "$dep"
             echo_success "Python installed."
        fi
    elif ! command_exists "${dep%%@*}"; then
        echo_warn "'${dep%%@*}' not found. Installing via Homebrew..."
        brew install "$dep"
        echo_success "'${dep%%@*}' installed."
    else
        echo_success "'${dep%%@*}' is already installed."
    fi
done

# 3. Clone Repository
echo_step "Cloning the application repository..."
if [ -d "$REPO_DIR" ]; then
    echo_warn "Repository directory '$REPO_DIR' already exists. Pulling latest changes."
    cd "$REPO_DIR"
    git pull
    cd ..
else
    git clone "$REPO_URL"
    echo_success "Repository cloned."
fi
cd "$REPO_DIR"

# 4. Set up Python Virtual Environment
echo_step "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo_success "Virtual environment created."
else
    echo_success "Virtual environment already exists."
fi

# 5. Activate Virtual Environment and Install Dependencies
echo_step "Activating virtual environment and installing Python packages..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo_success "All Python dependencies installed."
else
    echo_error "requirements.txt not found in the repository."
fi
# Install PyInstaller separately to ensure it's present
pip install pyinstaller

# 6. Create macOS Application Icon (.icns)
echo_step "Creating application icon..."
if [ -f "$LOGO_FILE" ]; then
    ICONSET_DIR="Skyscope.iconset"
    mkdir -p "$ICONSET_DIR"
    sips -z 16 16     "$LOGO_FILE" --out "${ICONSET_DIR}/icon_16x16.png"
    sips -z 32 32     "$LOGO_FILE" --out "${ICONSET_DIR}/icon_16x16@2x.png"
    sips -z 32 32     "$LOGO_FILE" --out "${ICONSET_DIR}/icon_32x32.png"
    sips -z 64 64     "$LOGO_FILE" --out "${ICONSET_DIR}/icon_32x32@2x.png"
    sips -z 128 128   "$LOGO_FILE" --out "${ICONSET_DIR}/icon_128x128.png"
    sips -z 256 256   "$LOGO_FILE" --out "${ICONSET_DIR}/icon_128x128@2x.png"
    sips -z 256 256   "$LOGO_FILE" --out "${ICONSET_DIR}/icon_256x256.png"
    sips -z 512 512   "$LOGO_FILE" --out "${ICONSET_DIR}/icon_256x256@2x.png"
    sips -z 512 512   "$LOGO_FILE" --out "${ICONSET_DIR}/icon_512x512.png"
    sips -z 1024 1024 "$LOGO_FILE" --out "${ICONSET_DIR}/icon_512x512@2x.png"
    iconutil -c icns "$ICONSET_DIR"
    rm -rf "$ICONSET_DIR"
    echo_success "Application icon 'Skyscope.icns' created."
else
    echo_warn "logo.png not found. A default icon will be used."
fi

# 7. Build the Standalone Application with PyInstaller
echo_step "Building the standalone .app bundle with PyInstaller..."
# Note: Add any necessary data files with `--add-data`
# Example: --add-data "path/to/data:data"
pyinstaller --name "$APP_NAME" \
            --windowed \
            --icon="Skyscope.icns" \
            --noconfirm \
            --clean \
            --add-data "knowledge_base.md:." \
            --add-data "config.py:." \
            --hidden-import="streamlit.web.cli" \
            app.py

echo_success "Application bundle created in 'dist/$APP_NAME.app'."

# 8. Create the .dmg Disk Image
echo_step "Creating final .dmg disk image..."
DMG_FILE="${APP_NAME}.dmg"
if [ -f "$DMG_FILE" ]; then
    rm "$DMG_FILE"
fi

create-dmg --volname "$APP_NAME" \
           --volicon "Skyscope.icns" \
           --window-pos 200 120 \
           --window-size 800 400 \
           --icon-size 100 \
           --icon "$APP_NAME.app" 200 190 \
           --hide-extension "$APP_NAME.app" \
           --app-drop-link 600 185 \
           "$DMG_FILE" \
           "dist/$APP_NAME.app"

echo_success "Successfully created '$DMG_FILE'."
echo_step "Build process complete! You can find the distributable disk image in the current directory."

# --- Cleanup ---
deactivate
cd ..

exit 0
