#!/bin/bash
#
# Skyscope Enterprise Suite - Quick Run Script
#
# This script quickly sets up and runs the Skyscope Enterprise macOS GUI application
# - Checks for dependencies
# - Sets up virtual environment if needed
# - Installs required packages
# - Launches the GUI app
#

# Set strict error handling
set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="Skyscope Enterprise Suite"
BASE_DIR="$HOME/SkyscopeEnterprise"
VENV_DIR="$BASE_DIR/venv"
CONFIG_DIR="$BASE_DIR/config"
DATA_DIR="$BASE_DIR/data"
LOGS_DIR="$BASE_DIR/logs"
SECURE_DIR="$BASE_DIR/config/secure"

# Function to print messages
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    print_error "This script is for macOS only. Exiting."
    exit 1
fi

# Print welcome message
clear
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}       Skyscope Enterprise Suite - Quick Start         ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# Create base directories
print_message "Creating directory structure..."
mkdir -p "$BASE_DIR"
mkdir -p "$VENV_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$SECURE_DIR"

# Check for Homebrew and install if needed
if ! command_exists brew; then
    print_message "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
        print_error "Failed to install Homebrew. Please install it manually."
        exit 1
    }
    
    # Add Homebrew to PATH
    if [ -f ~/.zshrc ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f ~/.bash_profile ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    print_success "Homebrew installed"
else
    print_message "Homebrew already installed"
fi

# Install Python if needed
if ! command_exists python3; then
    print_message "Installing Python..."
    brew install python || {
        print_error "Failed to install Python. Please install it manually."
        exit 1
    }
    print_success "Python installed"
else
    print_message "Python already installed: $(python3 --version)"
fi

# Install Qt if needed
if ! brew list qt@6 &>/dev/null; then
    print_message "Installing Qt6..."
    brew install qt@6 || {
        print_warning "Failed to install Qt6 via Homebrew. Will try with pip later."
    }
else
    print_message "Qt6 already installed"
fi

# Set up Python virtual environment
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    print_message "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || {
        print_error "Failed to create virtual environment. Please check your Python installation."
        exit 1
    }
    print_success "Virtual environment created"
else
    print_message "Virtual environment already exists"
fi

# Activate virtual environment
print_message "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Install required packages
print_message "Installing required packages..."
pip install PyQt6 PyQt6-Charts numpy pandas matplotlib requests cryptography || {
    print_error "Failed to install required packages."
    exit 1
}

# Install AI provider packages
print_message "Installing AI provider packages..."
pip install openai-unofficial google-generativeai huggingface_hub anthropic || {
    print_warning "Some AI provider packages could not be installed. Continuing anyway."
}

# Install crypto packages
print_message "Installing crypto packages..."
pip install python-binance web3 pycoingecko || {
    print_warning "Some crypto packages could not be installed. Continuing anyway."
}

# Create main application file
print_message "Creating application files..."
mkdir -p "$BASE_DIR/skyscope/gui"
mkdir -p "$BASE_DIR/skyscope/core"
mkdir -p "$BASE_DIR/skyscope/utils"

# Create __init__.py files
touch "$BASE_DIR/skyscope/__init__.py"
touch "$BASE_DIR/skyscope/gui/__init__.py"
touch "$BASE_DIR/skyscope/core/__init__.py"
touch "$BASE_DIR/skyscope/utils/__init__.py"

# Create main application file
cat > "$BASE_DIR/main.py" << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Enterprise Suite - Main Application
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Skyscope')

# Import PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, 
        QLabel, QPushButton, QTabWidget, QStatusBar
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont
except ImportError as e:
    logger.error(f"Failed to import PyQt6: {e}")
    print("Error: PyQt6 is not installed. Please install it with: pip install PyQt6")
    sys.exit(1)

class MainWindow(QMainWindow):
    """Main application window with dark theme"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Skyscope Enterprise Suite")
        self.resize(1280, 800)
        
        # Set up theme
        self.apply_theme()
        
        # Set up UI components
        self.setup_ui()
        
        # Set up timer for status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)
    
    def apply_theme(self):
        """Apply dark theme to the application"""
        # Dark theme colors
        colors = {
            "background": "#121212",
            "card_background": "#1E1E1E",
            "primary": "#BB86FC",
            "secondary": "#03DAC6",
            "text": "#FFFFFF",
            "text_secondary": "#B3B3B3"
        }
        
        # Set application style sheet
        style_sheet = f"""
        QMainWindow, QDialog {{
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QWidget {{
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QTabWidget::pane {{
            border: 1px solid #333333;
            background-color: {colors['card_background']};
        }}
        
        QTabBar::tab {{
            background-color: {colors['background']};
            color: {colors['text_secondary']};
            padding: 8px 16px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors['card_background']};
            color: {colors['primary']};
            border-bottom: 2px solid {colors['primary']};
        }}
        
        QPushButton {{
            background-color: {colors['primary']};
            color: {colors['background']};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QPushButton:hover {{
            background-color: {colors['primary']}CC;
        }}
        
        QPushButton:pressed {{
            background-color: {colors['primary']}99;
        }}
        """
        
        self.setStyleSheet(style_sheet)
    
    def setup_ui(self):
        """Set up UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create dashboard tab
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_tab)
        
        # Add welcome label
        welcome_label = QLabel("Welcome to Skyscope Enterprise Suite")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        dashboard_layout.addWidget(welcome_label)
        
        # Add description
        description_label = QLabel(
            "Your complete AI-powered income generation system\n"
            "Use the buttons below to manage your income strategies"
        )
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_layout.addWidget(description_label)
        
        # Add start button
        start_button = QPushButton("Start Income Generation")
        start_button.setMinimumHeight(50)
        start_button.clicked.connect(self.on_start_income_generation)
        dashboard_layout.addWidget(start_button)
        
        # Add to tab widget
        self.tab_widget.addTab(dashboard_tab, "Dashboard")
        
        # Add crypto tab
        crypto_tab = QWidget()
        crypto_layout = QVBoxLayout(crypto_tab)
        crypto_layout.addWidget(QLabel("Crypto Trading Strategies"))
        self.tab_widget.addTab(crypto_tab, "Crypto")
        
        # Add agents tab
        agents_tab = QWidget()
        agents_layout = QVBoxLayout(agents_tab)
        agents_layout.addWidget(QLabel("AI Agent Management"))
        self.tab_widget.addTab(agents_tab, "Agents")
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("System Ready")
        self.status_bar.addPermanentWidget(self.status_label)
    
    def update_status(self):
        """Update status indicators"""
        self.status_label.setText("System Active - Ready for Income Generation")
    
    def on_start_income_generation(self):
        """Handle start income generation button click"""
        self.status_label.setText("Starting income generation...")
        QTimer.singleShot(2000, lambda: self.status_label.setText("Income generation active"))

def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Skyscope Enterprise Suite")
        app.setOrganizationName("Skyscope Sentinel Intelligence")
        
        main_window = MainWindow()
        main_window.show()
        
        sys.exit(app.exec())
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make the main script executable
chmod +x "$BASE_DIR/main.py"

# Launch the application
print_message "Launching Skyscope Enterprise Suite..."
cd "$BASE_DIR"
python3 main.py

# Deactivate virtual environment when app closes
deactivate

print_success "Application closed. You can run it again with: $0"
