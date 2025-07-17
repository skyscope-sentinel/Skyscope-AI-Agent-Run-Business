#!/bin/bash

# ==============================================================================
# Skyscope Sentinel Intelligence - Linux Application Packager (AppImage)
# ==============================================================================
# This script automates the entire process of setting up the environment,
# cloning the repository, and packaging the application into a portable
# AppImage for various Linux distributions.
#
# It performs the following steps:
# 1. Detects the Linux distribution and appropriate package manager.
# 2. Checks for and installs necessary system dependencies (Python, Git, FUSE).
# 3. Downloads the AppImageTool utility.
# 4. Clones the application repository from GitHub or updates it.
# 5. Sets up a dedicated Python virtual environment.
# 6. Installs all required Python packages from requirements.txt.
# 7. Bundles the application using PyInstaller.
# 8. Creates a complete AppDir structure with desktop integration files.
# 9. Packages the AppDir into a final, portable .AppImage file.
# ==============================================================================

# --- Script Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Variables ---
REPO_URL="https://github.com/skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI.git"
REPO_DIR="Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI"
VENV_DIR="venv"
APP_NAME="Skyscope-Sentinel"
APP_VERSION="1.0.0"
PYTHON_VERSION_MAJOR=3
PYTHON_VERSION_MINOR=11
LOGO_FILE="logo.png" # Assumes a logo.png exists in the repo root
APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"

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

# 1. Check for Sudo / Root privileges
echo_step "Checking for root privileges..."
if [ "$EUID" -ne 0 ]; then
    if ! command_exists sudo; then
        echo_error "sudo command not found. Please run this script as root."
    fi
    echo_warn "This script needs to install packages. You may be prompted for your password."
    SUDO="sudo"
else
    SUDO=""
fi
echo_success "Privileges check passed."

# 2. Detect Linux Distribution and Package Manager
echo_step "Detecting Linux distribution and package manager..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID=$ID
    ID_LIKE=${ID_LIKE:-$ID}
else
    echo_error "Cannot detect Linux distribution (/etc/os-release not found)."
fi

case "$ID_LIKE" in
    *debian*|*ubuntu*)
        PKG_MANAGER="apt"
        INSTALL_CMD="$SUDO apt-get update && $SUDO apt-get install -y"
        PYTHON_PACKAGES="python3 python3-pip python3-venv"
        FUSE_PACKAGE="libfuse2"
        ;;
    *fedora*|*rhel*|*centos*)
        PKG_MANAGER="dnf"
        INSTALL_CMD="$SUDO dnf install -y"
        PYTHON_PACKAGES="python3 python3-pip python3-devel"
        FUSE_PACKAGE="fuse-libs"
        ;;
    *arch*)
        PKG_MANAGER="pacman"
        INSTALL_CMD="$SUDO pacman -Syu --noconfirm"
        PYTHON_PACKAGES="python python-pip"
        FUSE_PACKAGE="fuse2"
        ;;
    *)
        echo_error "Unsupported Linux distribution: $OS_ID. Please install dependencies manually."
        ;;
esac
echo_success "Detected '$OS_ID'. Using '$PKG_MANAGER' package manager."

# 3. Install System Dependencies
echo_step "Checking and installing system dependencies..."
DEPS_TO_INSTALL=""
if ! command_exists git; then DEPS_TO_INSTALL="$DEPS_TO_INSTALL git"; fi
if ! command_exists python3; then DEPS_TO_INSTALL="$DEPS_TO_INSTALL $PYTHON_PACKAGES"; fi
if ! command_exists pip3; then DEPS_TO_INSTALL="$DEPS_TO_INSTALL $PYTHON_PACKAGES"; fi
if ! dpkg -s $FUSE_PACKAGE >/dev/null 2>&1 && ! rpm -q $FUSE_PACKAGE >/dev/null 2>&1 && ! pacman -Q $FUSE_PACKAGE >/dev/null 2>&1; then
    DEPS_TO_INSTALL="$DEPS_TO_INSTALL $FUSE_PACKAGE"
fi

if [ -n "$DEPS_TO_INSTALL" ]; then
    echo_warn "Installing missing dependencies: $DEPS_TO_INSTALL"
    $INSTALL_CMD $DEPS_TO_INSTALL
    echo_success "System dependencies installed."
else
    echo_success "All system dependencies are already satisfied."
fi

# 4. Download AppImageTool
echo_step "Downloading AppImageTool..."
if [ ! -f "appimagetool" ]; then
    wget -O appimagetool "$APPIMAGETOOL_URL"
    chmod +x appimagetool
    echo_success "AppImageTool downloaded."
else
    echo_success "AppImageTool already exists."
fi

# 5. Clone or Update Repository
echo_step "Acquiring project source code..."
if [ -d "$REPO_DIR" ]; then
    echo_warn "Repository directory exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
else
    git clone "$REPO_URL"
    echo_success "Repository cloned."
fi
cd "$REPO_DIR"

# 6. Set up Python Virtual Environment and Install Dependencies
echo_step "Setting up Python virtual environment and installing dependencies..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo_success "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
deactivate
echo_success "All Python dependencies installed in the virtual environment."

# 7. Build with PyInstaller
echo_step "Building standalone bundle with PyInstaller..."
source "$VENV_DIR/bin/activate"
pyinstaller --name "$APP_NAME" \
            --noconfirm \
            --clean \
            --windowed \
            --add-data "knowledge_base.md:." \
            --add-data "config.py:." \
            --add-data "models.py:." \
            --add-data "utils.py:." \
            --hidden-import="streamlit.web.cli" \
            --hidden-import="sklearn.utils._cython_blas" \
            --hidden-import="sklearn.neighbors._typedefs" \
            --hidden-import="sklearn.neighbors._quad_tree" \
            --hidden-import="sklearn.tree._utils" \
            --hidden-import="pkg_resources.py2_warn" \
            app.py
deactivate
echo_success "PyInstaller build complete."

# 8. Create AppDir Structure for AppImage
echo_step "Creating AppDir structure..."
APPDIR="$APP_NAME.AppDir"
if [ -d "$APPDIR" ]; then
    rm -rf "$APPDIR"
fi
mkdir -p "$APPDIR/usr/bin"

# Move PyInstaller bundle into AppDir
mv "dist/$APP_NAME" "$APPDIR/usr/bin/$APP_NAME"

# Create AppRun script
echo_step "Creating AppRun entrypoint..."
cat > "$APPDIR/AppRun" <<EOL
#!/bin/bash
HERE=\$(dirname \$(readlink -f "\${0}"))
export PATH="\${HERE}/usr/bin:\${PATH}"
exec "\${HERE}/usr/bin/$APP_NAME/$APP_NAME" "\$@"
EOL
chmod +x "$APPDIR/AppRun"
echo_success "AppRun created."

# Create .desktop file for desktop integration
echo_step "Creating .desktop file..."
if [ -f "$LOGO_FILE" ]; then
    cp "$LOGO_FILE" "$APPDIR/$APP_NAME.png"
    ICON_NAME=$APP_NAME
else
    echo_warn "logo.png not found. Using a default icon name."
    ICON_NAME="utilities-terminal" # A generic icon
fi

cat > "$APPDIR/$APP_NAME.desktop" <<EOL
[Desktop Entry]
Name=$APP_NAME
Exec=$APP_NAME
Icon=$ICON_NAME
Type=Application
Categories=Utility;Development;AI;
Comment=Skyscope Sentinel Intelligence - AI Agentic Swarm
EOL
echo_success ".desktop file created."

# 9. Build the AppImage
echo_step "Building the final AppImage..."
# Move appimagetool to the parent directory to run it
mv ../appimagetool .
ARCH=x86_64 ./appimagetool "$APPDIR"
# Move appimagetool back
mv appimagetool ..
echo_success "AppImage created successfully!"

# --- Final Output ---
FINAL_APPIMAGE_NAME="${APP_NAME}-x86_64.AppImage"
echo_step "Build process complete!"
echo -e "${C_GREEN}Your portable application is ready: ${C_YELLOW}${REPO_DIR}/${FINAL_APPIMAGE_NAME}${C_NC}"
echo "You can now run it from any compatible Linux distribution."

# --- Cleanup ---
cd ..

exit 0
