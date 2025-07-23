#!/bin/bash
# 01-run-first.sh
# Purpose: Install all prerequisites for Skyscope Sentinel Intelligence platform
# Author: Grok 3, created for Miss Casey Jay Topojani
# Date: May 17, 2025
# Repository: https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business

# Exit on error
set -e

# Define variables
LOG_FILE="$HOME/skyscope_prereq_install.log"
REPO_DIR="$HOME/Skyscope-AI-Agent-Run-Business"
VENV_DIR="$REPO_DIR/venv"
ANACONDA_INSTALLER="Anaconda3-2024.06-1-Linux-x86_64.sh"
ANACONDA_URL="https://repo.anaconda.com/archive/$ANACONDA_INSTALLER"
ANACONDA_DIR="$HOME/anaconda3"
PYTHON_VERSION="3.10"  # Minimum version required by 01-main.sh
NODE_VERSION="18"  # Stable version for Next.js
OLLAMA_MODEL="gemma3:1b"

# Required pip packages from 01-main.sh
PIP_PACKAGES=(
    "fastapi==0.110.0"
    "uvicorn==0.29.0"
    "requests==2.31.0"
    "aiohttp==3.9.3"
    "beautifulsoup4==4.12.3"
    "pandas==2.2.1"
    "numpy==1.26.4"
    "matplotlib==3.8.3"
    "seaborn==0.13.2"
    "scikit-learn==1.4.1"
    "tensorflow==2.15.0"
    "torch==2.2.1"
    "prometheus-client==0.20.0"
    "python-dotenv==1.0.1"
    "vault==0.2.0"
    "websockets==12.0"
    "psutil==5.9.8"
    "loguru==0.7.2"
)

# Create log file
mkdir -p "$(dirname "$LOG_FILE")"
echo "Starting prerequisite installation at $(date)" > "$LOG_FILE"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update system
log "Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y >> "$LOG_FILE" 2>&1
sudo apt-get install -y \
    curl git wget jq unzip tar make gcc g++ cmake libssl-dev \
    build-essential software-properties-common apt-transport-https \
    ca-certificates gnupg lsb-release >> "$LOG_FILE" 2>&1

# Install Anaconda3
if [ ! -d "$ANACONDA_DIR" ]; then
    log "Downloading Anaconda3..."
    wget "$ANACONDA_URL" -O "/tmp/$ANACONDA_INSTALLER" >> "$LOG_FILE" 2>&1
    log "Installing Anaconda3..."
    bash "/tmp/$ANACONDA_INSTALLER" -b -p "$ANACONDA_DIR" >> "$LOG_FILE" 2>&1
    # Initialize conda
    "$ANACONDA_DIR/bin/conda" init bash >> "$LOG_FILE" 2>&1
    source "$HOME/.bashrc"
    rm "/tmp/$ANACONDA_INSTALLER"
else
    log "Anaconda3 already installed at $ANACONDA_DIR"
fi

# Ensure conda is in PATH
export PATH="$ANACONDA_DIR/bin:$PATH"
log "Conda version: $(conda --version)"

# Install Python 3 and pip (fallback for system Python)
log "Installing Python $PYTHON_VERSION and pip..."
sudo apt-get install -y python${PYTHON_VERSION} python3-pip python3-venv >> "$LOG_FILE" 2>&1
if ! command_exists python3; then
    log "ERROR: Python 3 installation failed"
    exit 1
fi
log "Python version: $(python3 --version)"
log "Pip version: $(pip3 --version)"

# Install Node.js and npm
if ! command_exists node; then
    log "Installing Node.js v$NODE_VERSION..."
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash - >> "$LOG_FILE" 2>&1
    sudo apt-get install -y nodejs >> "$LOG_FILE" 2>&1
else
    log "Node.js already installed: $(node --version)"
fi
log "npm version: $(npm --version)"

# Install Docker
if ! command_exists docker; then
    log "Installing Docker..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y >> "$LOG_FILE" 2>&1
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin >> "$LOG_FILE" 2>&1
    sudo usermod -aG docker "$USER"
else
    log "Docker already installed: $(docker --version)"
fi

# Install Docker Compose
if ! command_exists docker-compose; then
    log "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    log "Docker Compose already installed: $(docker-compose --version)"
fi

# Install Ollama
if ! command_exists ollama; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh >> "$LOG_FILE" 2>&1
else
    log "Ollama already installed: $(ollama --version)"
fi
# Pull Ollama model
log "Pulling Ollama model $OLLAMA_MODEL..."
ollama pull "$OLLAMA_MODEL" >> "$LOG_FILE" 2>&1

# Install kaspad
if [ ! -d "$HOME/kaspad" ]; then
    log "Installing kaspad..."
    git clone https://github.com/kaspanet/kaspad.git "$HOME/kaspad" >> "$LOG_FILE" 2>&1
    cd "$HOME/kaspad"
    go build >> "$LOG_FILE" 2>&1
    cd -
else
    log "kaspad already installed at $HOME/kaspad"
fi

# Clone repository if not present
if [ ! -d "$REPO_DIR" ]; then
    log "Cloning Skyscope-AI-Agent-Run-Business repository..."
    git clone https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.git "$REPO_DIR" >> "$LOG_FILE" 2>&1
else
    log "Repository already cloned at $REPO_DIR"
fi
cd "$REPO_DIR"

# Create virtual environment with system Python
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    log "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
log "Virtual environment activated: $(python --version)"

# Upgrade pip in virtual environment
log "Upgrading pip in virtual environment..."
pip install --upgrade pip >> "$LOG_FILE" 2>&1

# Install pip packages
log "Installing pip packages..."
for package in "${PIP_PACKAGES[@]}"; do
    log "Installing $package..."
    pip install "$package" >> "$LOG_FILE" 2>&1
done

# Verify installations
log "Verifying installations..."
command_exists docker && log "Docker installed: $(docker --version)"
command_exists docker-compose && log "Docker Compose installed: $(docker-compose --version)"
command_exists node && log "Node.js installed: $(node --version)"
command_exists npm && log "npm installed: $(npm --version)"
command_exists ollama && log "Ollama installed: $(ollama --version)"
command_exists python && log "Python installed: $(python --version)"
command_exists pip && log "Pip installed: $(pip --version)"
[ -d "$HOME/kaspad" ] && log "kaspad installed at $HOME/kaspad"

# Deactivate virtual environment
deactivate
log "Virtual environment deactivated"

# Final instructions
log "Prerequisite installation complete!"
echo "To run the Skyscope Sentinel Intelligence platform:"
echo "1. Navigate to $REPO_DIR"
echo "2. Activate the virtual environment: source $VENV_DIR/bin/activate"
echo "3. Run the main script: ./01-main.sh"
echo "4. Check logs: tail -f ~/skyscope-sentinel/logs/setup.log"
echo "5. Access the platform:"
echo "   - Frontend: http://localhost:3000"
echo "   - API: http://localhost:6000"
echo "   - Grafana: http://localhost:3001"
echo "6. Check wallet and credentials: cat ~/Skyscope_manager.txt"
echo "Installation log: $LOG_FILE"

exit 0
