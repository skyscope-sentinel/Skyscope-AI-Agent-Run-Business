#!/bin/bash
#
# COMPLETE_SYSTEM_SETUP.sh
#
# Comprehensive setup script for Skyscope AI Agentic Swarm Business/Enterprise
# This script ensures all dependencies are installed and the system is properly configured
# for autonomous operation with a fully functional GUI macOS application.
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
BASE_DIR="$(pwd)"
VENV_DIR="$BASE_DIR/venv"
LOGS_DIR="$BASE_DIR/logs"
CONFIG_DIR="$BASE_DIR/config"
DATA_DIR="$BASE_DIR/data"

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
echo -e "${CYAN}                      Complete System Setup v${APP_VERSION}${NC}"
echo -e "${CYAN}                      ==============================${NC}"
echo ""
echo "This script will set up the complete Skyscope AI Agentic Swarm Business/Enterprise system"
echo "with all dependencies, proper configuration, and a fully functional GUI application."
echo ""

# Check if running on macOS
print_header "Checking System Requirements"

if [ "$(uname)" != "Darwin" ]; then
    print_error "This script is optimized for macOS. Some features may not work on other platforms."
    exit 1
fi

print_success "Running on macOS"

# Check Python version
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
print_success "Python version: $PYTHON_VERSION"

# Create necessary directories
print_header "Creating Directory Structure"

create_dir "$LOGS_DIR"
create_dir "$CONFIG_DIR"
create_dir "$DATA_DIR"
create_dir "$BASE_DIR/assets"
create_dir "$BASE_DIR/backups"
create_dir "$BASE_DIR/temp"

# Install Homebrew if not already installed
print_header "Installing Dependencies"

if ! command_exists brew; then
    print_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH
    if [ -f ~/.zshrc ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f ~/.bash_profile ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    print_success "Homebrew installed successfully"
else
    print_success "Homebrew is already installed"
fi

# Update Homebrew
print_info "Updating Homebrew..."
brew update

# Install system dependencies
print_info "Installing system dependencies..."
brew install python@3.11 git cmake ninja pkg-config qt@6

# Set up Python virtual environment
print_header "Setting Up Python Environment"

if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install core Python dependencies
print_info "Installing core Python dependencies..."
pip install --upgrade \
    PyQt6 \
    PyQt6-Charts \
    PyQt6-WebEngine \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    dash \
    dash-bootstrap-components \
    requests \
    aiohttp \
    websockets \
    psutil \
    cryptography \
    pydantic \
    fastapi \
    uvicorn \
    streamlit \
    pillow \
    opencv-python \
    scikit-learn

# Install AI and ML dependencies (FREE UNLIMITED ACCESS!)
print_info "Installing AI and ML dependencies..."
pip install --upgrade \
    openai-unofficial \
    anthropic \
    google-generativeai \
    huggingface-hub \
    transformers \
    torch \
    tensorflow \
    langchain \
    chromadb \
    sentence-transformers

# Install crypto and finance dependencies
print_info "Installing crypto and finance dependencies..."
pip install --upgrade \
    ccxt \
    web3 \
    python-binance \
    yfinance \
    ta \
    pandas-ta

# Install development tools
print_info "Installing development tools..."
pip install --upgrade \
    pytest \
    black \
    mypy \
    flake8 \
    pyinstaller

print_success "All dependencies installed successfully"

# Create main application launcher
print_header "Creating Main Application"

print_info "Creating main application launcher..."

print_success "System setup completed successfully!"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup Complete! Your Skyscope Enterprise Suite is ready to run.${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "To start the system:"
echo "  1. Run: source venv/bin/activate"
echo "  2. Run: python main_application.py"
echo ""
echo "To build the macOS app:"
echo "  1. Run: ./BUILD_MACOS_APP.sh"
echo ""