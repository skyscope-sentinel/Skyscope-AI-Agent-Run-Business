#!/bin/bash
#
# RUN_COMPLETE_SYSTEM.sh
#
# Skyscope Sentinel Intelligence AI Platform - Complete System Launcher
#
# This script checks for dependencies, sets up the environment, and launches
# the complete integrated system on macOS with proper configuration.
#
# Features:
# - Comprehensive dependency checking
# - Automatic environment setup
# - API key configuration
# - Virtual environment management
# - Resource optimization
# - Error handling and recovery
# - Progress monitoring
#
# Created on: July 17, 2025
# Author: Skyscope Sentinel Intelligence
#

# Set strict error handling
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
APP_VERSION="1.0.0"
APP_IDENTIFIER="ai.skyscope.enterprise"
BASE_DIR="$HOME/SkyscopeEnterprise"
VENV_DIR="$BASE_DIR/venv"
CONFIG_DIR="$BASE_DIR/config"
DATA_DIR="$BASE_DIR/data"
LOGS_DIR="$BASE_DIR/logs"
SECURE_DIR="$BASE_DIR/secure"
BACKUP_DIR="$BASE_DIR/backups"
STRATEGIES_DIR="$BASE_DIR/strategies"
MODELS_DIR="$BASE_DIR/models"
PYTHON_MIN_VERSION="3.8.0"
MAIN_SCRIPT="FINAL_COMPLETE_INTEGRATION.py"
CONFIG_FILE="$CONFIG_DIR/system_config.json"
LOG_FILE="$LOGS_DIR/system_launcher.log"

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

# Function to ask yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    fi
}

# Function to check Python version
check_python_version() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
    print_info "Found Python version: $PYTHON_VERSION"
    
    # Compare versions (simplified)
    if [[ "$PYTHON_VERSION" < "$PYTHON_MIN_VERSION" ]]; then
        print_error "Python version must be at least $PYTHON_MIN_VERSION"
        exit 1
    fi
}

# Function to check and install required packages
check_install_packages() {
    local packages=("$@")
    local missing_packages=()
    
    for package in "${packages[@]}"; do
        if ! pip show "$package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing packages: ${missing_packages[*]}"
        if ask_yes_no "Do you want to install missing packages?"; then
            print_info "Installing missing packages..."
            pip install "${missing_packages[@]}"
            print_success "Packages installed successfully"
        else
            print_error "Required packages are missing. Cannot continue."
            exit 1
        fi
    else
        print_success "All required packages are installed"
    fi
}

# Function to setup environment variables
setup_environment_variables() {
    # Check for API keys
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY environment variable not set"
        if ask_yes_no "Do you want to set OPENAI_API_KEY now?"; then
            read -p "Enter your OpenAI API key: " api_key
            export OPENAI_API_KEY="$api_key"
            # Add to .zshrc or .bash_profile for persistence
            if [ -f "$HOME/.zshrc" ]; then
                echo "export OPENAI_API_KEY=\"$api_key\"" >> "$HOME/.zshrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                echo "export OPENAI_API_KEY=\"$api_key\"" >> "$HOME/.bash_profile"
            fi
            print_success "OPENAI_API_KEY set successfully"
        fi
    else
        print_success "OPENAI_API_KEY is set"
    fi
    
    # Check for other API keys if needed
    for key_name in "GEMINI_API_KEY" "HUGGINGFACE_API_KEY" "ANTHROPIC_API_KEY"; do
        if [ -z "${!key_name}" ]; then
            print_info "$key_name is not set (optional)"
        else
            print_success "$key_name is set"
        fi
    done
}

# Function to check system resources
check_system_resources() {
    # Check available memory
    if command_exists vm_stat; then
        AVAILABLE_MEMORY=$(vm_stat | grep "Pages free:" | awk '{print $3}' | sed 's/\.//')
        AVAILABLE_MEMORY_MB=$((AVAILABLE_MEMORY * 4096 / 1024 / 1024))
        print_info "Available memory: ${AVAILABLE_MEMORY_MB}MB"
        
        if [ "$AVAILABLE_MEMORY_MB" -lt 4000 ]; then
            print_warning "Low memory available. At least 4GB recommended."
            if ! ask_yes_no "Continue anyway?"; then
                exit 1
            fi
        fi
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_info "Available disk space: $AVAILABLE_SPACE"
    
    # Check CPU cores
    CPU_CORES=$(sysctl -n hw.ncpu)
    print_info "CPU cores: $CPU_CORES"
}

# Function to setup virtual environment
setup_virtual_environment() {
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
    
    # Install required packages
    print_info "Installing required packages..."
    pip install PyQt6 PyQt6-Charts numpy pandas matplotlib seaborn scikit-learn tensorflow torch ccxt requests docker psutil GPUtil cryptography pytest black mypy flake8 fastapi uvicorn
    pip install openai-unofficial google-generativeai huggingface_hub anthropic
    pip install python-binance web3 pycoingecko eth-brownie
    print_success "All required packages installed"
}

# Function to check for main script
check_main_script() {
    if [ ! -f "$MAIN_SCRIPT" ]; then
        print_error "Main script $MAIN_SCRIPT not found"
        exit 1
    fi
    print_success "Found main script: $MAIN_SCRIPT"
}

# Function to create default configuration
create_default_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_info "Creating default configuration..."
        create_dir "$(dirname "$CONFIG_FILE")"
        
        cat > "$CONFIG_FILE" << EOF
{
  "system": {
    "name": "Skyscope Sentinel Intelligence AI Platform",
    "version": "1.0.0",
    "max_agents": 10000,
    "log_level": "INFO",
    "debug_mode": false,
    "backup_enabled": true,
    "backup_interval": 3600,
    "health_check_interval": 300
  },
  "ai": {
    "default_provider": "openai-unofficial",
    "providers": {
      "openai-unofficial": {
        "enabled": true,
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o"
      },
      "google-gemini": {
        "enabled": false,
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-pro"
      },
      "huggingface": {
        "enabled": false,
        "api_key_env": "HUGGINGFACE_API_KEY",
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1"
      },
      "anthropic": {
        "enabled": false,
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-3-opus-20240229"
      }
    }
  },
  "resources": {
    "cpu_allocation_percentage": 80,
    "memory_allocation_percentage": 70,
    "gpu_allocation_percentage": 90,
    "disk_allocation_percentage": 50,
    "network_allocation_percentage": 60
  },
  "income": {
    "target_daily": 1000.0,
    "risk_level": "medium",
    "strategies": {
      "crypto_trading": {
        "enabled": true,
        "allocation_percentage": 25,
        "risk_level": "medium"
      },
      "mev_bot": {
        "enabled": true,
        "allocation_percentage": 20,
        "risk_level": "high"
      },
      "nft_generation": {
        "enabled": true,
        "allocation_percentage": 15,
        "risk_level": "medium"
      },
      "freelance_work": {
        "enabled": true,
        "allocation_percentage": 15,
        "risk_level": "low"
      },
      "content_creation": {
        "enabled": true,
        "allocation_percentage": 10,
        "risk_level": "low"
      },
      "social_media": {
        "enabled": true,
        "allocation_percentage": 10,
        "risk_level": "medium"
      },
      "affiliate_marketing": {
        "enabled": true,
        "allocation_percentage": 5,
        "risk_level": "low"
      }
    }
  },
  "agents": {
    "total": 10000,
    "allocation": {
      "crypto_trading": 2000,
      "mev_bot": 1000,
      "nft_generation": 2000,
      "freelance_work": 2000,
      "content_creation": 1500,
      "social_media": 1000,
      "affiliate_marketing": 500
    },
    "auto_scaling": true,
    "performance_threshold": 0.7
  },
  "wallet": {
    "secure_storage": true,
    "auto_withdrawal": false,
    "withdrawal_threshold": 1000.0,
    "withdrawal_address": "",
    "supported_currencies": [
      "BTC", "ETH", "SOL", "BNB", "USDT", "USDC"
    ]
  },
  "legal": {
    "compliance_check_enabled": true,
    "business_name": "Skyscope Sentinel Intelligence",
    "tax_tracking_enabled": true,
    "jurisdiction": "United States",
    "terms_of_service_version": "1.0.0"
  },
  "ui": {
    "theme": "dark",
    "refresh_interval": 5000,
    "chart_history": 24,
    "minimize_to_tray": true,
    "enable_notifications": true
  },
  "integration": {
    "pinokio_enabled": true,
    "pinokio_port": 42000,
    "vscode_integration": true,
    "docker_integration": true
  }
}
EOF
        print_success "Default configuration created"
    else
        print_info "Configuration file already exists"
    fi
}

# Function to optimize system for performance
optimize_system() {
    # Set process priority
    if command_exists renice; then
        print_info "Setting process priority..."
        renice -n -10 -p $$ >/dev/null 2>&1 || true
    fi
    
    # Disable sleep mode
    if command_exists caffeinate; then
        print_info "Disabling system sleep..."
        caffeinate -d -i -m -s &
        CAFFEINATE_PID=$!
        # Register trap to kill caffeinate on exit
        trap "kill $CAFFEINATE_PID 2>/dev/null || true" EXIT
    fi
}

# Function to run the main script
run_main_script() {
    print_header "Launching $APP_NAME"
    
    # Create log directory
    create_dir "$(dirname "$LOG_FILE")"
    
    print_info "Starting the system..."
    python3 "$MAIN_SCRIPT" --config "$CONFIG_FILE" "$@" 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        print_error "System exited with code $EXIT_CODE. Check logs for details."
        return $EXIT_CODE
    else
        print_success "System completed successfully"
        return 0
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
echo -e "${CYAN}                      Complete System Launcher v${APP_VERSION}${NC}"
echo -e "${CYAN}                      ============================${NC}"
echo ""
echo -e "This script will launch the complete ${PURPLE}Skyscope Enterprise Suite${NC} on your Mac,"
echo "with all components, systems, and integrations."
echo ""

# Check if running on macOS
print_header "Checking System Requirements"

if [ "$(uname)" != "Darwin" ]; then
    print_error "This script is for macOS only. Exiting."
    exit 1
fi

# Check macOS version
OS_VERSION=$(sw_vers -productVersion)
print_info "macOS Version: $OS_VERSION"

# Check if running on Intel Mac
if [ "$(uname -m)" != "x86_64" ]; then
    print_warning "This script is optimized for Intel Macs. Some features may not work correctly on Apple Silicon."
    if ! ask_yes_no "Do you want to continue anyway?"; then
        echo "Launch cancelled."
        exit 0
    fi
fi

# Main execution
print_header "Setting Up Environment"
check_python_version
setup_environment_variables
check_system_resources
setup_virtual_environment
check_main_script
create_default_config
optimize_system

# Run the main script
print_header "Running Complete System"
run_main_script "$@"
EXIT_CODE=$?

# Final message
if [ $EXIT_CODE -eq 0 ]; then
    print_header "System Completed Successfully"
    echo -e "${GREEN}The ${PURPLE}Skyscope Enterprise Suite${GREEN} has completed its execution.${NC}"
    echo -e "Check the logs at ${CYAN}$LOG_FILE${NC} for details."
else
    print_header "System Encountered Errors"
    echo -e "${RED}The ${PURPLE}Skyscope Enterprise Suite${RED} encountered errors during execution.${NC}"
    echo -e "Check the logs at ${CYAN}$LOG_FILE${NC} for details."
    echo -e "Exit code: ${RED}$EXIT_CODE${NC}"
fi

echo ""
echo -e "${PURPLE}Thank you for using Skyscope Enterprise Suite!${NC}"
echo ""

exit $EXIT_CODE
