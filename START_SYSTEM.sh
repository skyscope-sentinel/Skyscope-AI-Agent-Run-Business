#!/bin/bash
#
# START_SYSTEM.sh
#
# Complete startup script for Skyscope AI Agentic Swarm Business/Enterprise
# This script ensures all dependencies are installed and starts the system
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
BASE_DIR="$(pwd)"
VENV_DIR="$BASE_DIR/venv"
LOGS_DIR="$BASE_DIR/logs"

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
echo -e "${CYAN}                      System Startup v2.0${NC}"
echo -e "${CYAN}                      ==================${NC}"
echo ""

# Check if running on macOS
print_header "System Check"

if [ "$(uname)" != "Darwin" ]; then
    print_warning "This system is optimized for macOS. Some features may not work correctly."
fi

print_success "System check complete"

# Check Python
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
print_success "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
print_header "Environment Setup"

if [ ! -d "$VENV_DIR" ]; then
    print_info "Virtual environment not found. Running setup..."
    if [ -f "COMPLETE_SYSTEM_SETUP.sh" ]; then
        chmod +x COMPLETE_SYSTEM_SETUP.sh
        ./COMPLETE_SYSTEM_SETUP.sh
    else
        print_error "Setup script not found. Please run COMPLETE_SYSTEM_SETUP.sh first."
        exit 1
    fi
fi

print_success "Virtual environment found"

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check critical dependencies
print_header "Dependency Check"

print_info "Checking critical dependencies..."

# Check PyQt6
if python3 -c "import PyQt6" 2>/dev/null; then
    print_success "PyQt6 is available"
else
    print_warning "PyQt6 not found. Installing..."
    pip install PyQt6 PyQt6-Charts PyQt6-WebEngine
fi

# Check psutil
if python3 -c "import psutil" 2>/dev/null; then
    print_success "psutil is available"
else
    print_warning "psutil not found. Installing..."
    pip install psutil
fi

# Check other critical packages
CRITICAL_PACKAGES=("numpy" "pandas" "requests" "cryptography")

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        print_success "$package is available"
    else
        print_warning "$package not found. Installing..."
        pip install "$package"
    fi
done

# Create necessary directories
print_header "Directory Setup"

mkdir -p logs config data assets temp backups

print_success "Directories created"

# Check main application
print_header "Application Check"

if [ ! -f "main_application.py" ]; then
    print_error "main_application.py not found. Cannot start application."
    exit 1
fi

if [ ! -f "autonomous_orchestrator.py" ]; then
    print_error "autonomous_orchestrator.py not found. Cannot start autonomous operations."
    exit 1
fi

print_success "Application files found"

# Start the application
print_header "Starting Application"

print_info "Starting Skyscope AI Agentic Swarm Business/Enterprise..."
print_info "This will launch the GUI application with real-time monitoring"
print_info "and autonomous business operations with 10,000 AI agents."
echo ""

# Set environment variables for better performance
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_ENABLE_HIGHDPI_SCALING=1

# Start the application
python3 main_application.py

# Cleanup on exit
print_header "Shutdown"
print_info "Application closed. Cleaning up..."

# Deactivate virtual environment
deactivate 2>/dev/null || true

print_success "System shutdown complete"
echo ""
echo -e "${CYAN}Thank you for using Skyscope AI Agentic Swarm Business/Enterprise!${NC}"
echo ""