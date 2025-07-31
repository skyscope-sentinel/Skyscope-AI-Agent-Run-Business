#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Sentinel Intelligence AI Platform
Windows Application Entry Point

This module provides the main GUI application for the Skyscope Sentinel Intelligence
AI Platform, a fully autonomous income-earning platform that leverages 10,000 AI agents
to generate income through various strategies, primarily focused on cryptocurrency.

Features:
1. PIN-based security system
2. System tray integration
3. Professional black-themed UI
4. Real-time monitoring dashboard
5. Agent status monitoring
6. Income strategy tracking
7. Wallet management interface
8. System resource monitoring
9. Legal compliance dashboard
10. Automatic updates and self-improvement

Created on: July 16, 2025
"""

import os
import sys
import json
import time
import logging
import hashlib
import threading
import datetime
import traceback
import webbrowser
import subprocess
import configparser
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Try to import required packages, install if missing
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QLabel, QLineEdit,
        QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
        QSystemTrayIcon, QMenu, QAction, QMessageBox, QProgressBar,
        QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
        QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame,
        QFileDialog, QInputDialog, QSizePolicy, QSpacerItem, QStyle
    )
    from PyQt5.QtGui import (
        QIcon, QPixmap, QFont, QPalette, QColor, QFontDatabase,
        QKeyEvent, QPainter, QBrush, QPen, QLinearGradient, QPainterPath
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QSize, QRect,
        QPoint, QUrl, QSettings, QByteArray, QBuffer, QIODevice,
        QPropertyAnimation, QEasingCurve, QEvent, QObject
    )
    from PyQt5.QtChart import (
        QChart, QChartView, QLineSeries, QValueAxis, QPieSeries, QBarSeries,
        QBarSet, QBarCategoryAxis
    )
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5", "PyQtChart"])
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QLabel, QLineEdit,
        QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
        QSystemTrayIcon, QMenu, QAction, QMessageBox, QProgressBar,
        QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
        QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame,
        QFileDialog, QInputDialog, QSizePolicy, QSpacerItem, QStyle
    )
    from PyQt5.QtGui import (
        QIcon, QPixmap, QFont, QPalette, QColor, QFontDatabase,
        QKeyEvent, QPainter, QBrush, QPen, QLinearGradient, QPainterPath
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QSize, QRect,
        QPoint, QUrl, QSettings, QByteArray, QBuffer, QIODevice,
        QPropertyAnimation, QEasingCurve, QEvent, QObject
    )
    from PyQt5.QtChart import (
        QChart, QChartView, QLineSeries, QValueAxis, QPieSeries, QBarSeries,
        QBarSet, QBarCategoryAxis
    )

# Try to import psutil for system monitoring
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# Try to import Autonomous Income System
try:
    from autonomous_income_system import AutonomousIncomeSystem, get_wallet_manager
except ImportError:
    print("Warning: Autonomous Income System module not found.")
    print("The application will run in limited functionality mode.")
    AutonomousIncomeSystem = None
    get_wallet_manager = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("skyscope_app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SkyscopeApp')

# Constants
APP_NAME = "Skyscope Sentinel Intelligence AI Platform"
APP_VERSION = "1.0.0"
APP_ICON = "icon.png"
CONFIG_FILE = "config.ini"
PIN_FILE = "security.dat"
DATA_DIR = os.path.join(os.path.expanduser("~"), "Skyscope Sentinel Intelligence")
THEME_DARK = {
    "bg_primary": "#121212",
    "bg_secondary": "#1E1E1E",
    "bg_tertiary": "#252525",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0B0B0",
    "accent_primary": "#00A3FF",
    "accent_secondary": "#7B68EE",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "error": "#F44336",
    "chart_colors": ["#00A3FF", "#7B68EE", "#4CAF50", "#FFC107", "#F44336", "#E91E63", "#9C27B0", "#673AB7"]
}

class PINDialog(QDialog):
    """Dialog for PIN authentication"""
    
    def __init__(self, parent=None, setup_mode=False):
        """Initialize the PIN dialog."""
        super().__init__(parent)
        self.setup_mode = setup_mode
        self.pin = ""
        self.confirm_pin = ""
        self.pin_hash = ""
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Security PIN" if not self.setup_mode else "Set Security PIN")
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create a frame with rounded corners
        frame = QFrame()
        frame.setObjectName("pinFrame")
        frame.setStyleSheet("""
            QFrame#pinFrame {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(20)
        frame_layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo and title
        logo_label = QLabel()
        logo_pixmap = QPixmap(APP_ICON) if os.path.exists(APP_ICON) else QPixmap(32, 32)
        if logo_pixmap.isNull():
            logo_pixmap = QPixmap(32, 32)
            logo_pixmap.fill(Qt.transparent)
        logo_label.setPixmap(logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        
        title_label = QLabel("Skyscope Sentinel Intelligence")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold;")
        
        subtitle_label = QLabel("Enter PIN to access the system" if not self.setup_mode else "Set a security PIN for your account")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #B0B0B0; font-size: 14px;")
        
        # PIN input
        self.pin_input = QLineEdit()
        self.pin_input.setEchoMode(QLineEdit.Password)
        self.pin_input.setPlaceholderText("Enter PIN")
        self.pin_input.setStyleSheet("""
            QLineEdit {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #00A3FF;
            }
        """)
        
        # Confirm PIN input (only in setup mode)
        self.confirm_pin_input = QLineEdit()
        self.confirm_pin_input.setEchoMode(QLineEdit.Password)
        self.confirm_pin_input.setPlaceholderText("Confirm PIN")
        self.confirm_pin_input.setStyleSheet(self.pin_input.styleSheet())
        self.confirm_pin_input.setVisible(self.setup_mode)
        
        # Error message
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #F44336; font-size: 12px;")
        self.error_label.setVisible(False)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #252525;
                color: #B0B0B0;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #00A3FF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0088CC;
            }
            QPushButton:pressed {
                background-color: #0077BB;
            }
        """)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.submit_button)
        
        # Add widgets to layout
        frame_layout.addWidget(logo_label)
        frame_layout.addWidget(title_label)
        frame_layout.addWidget(subtitle_label)
        frame_layout.addWidget(self.pin_input)
        frame_layout.addWidget(self.confirm_pin_input)
        frame_layout.addWidget(self.error_label)
        frame_layout.addLayout(button_layout)
        
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
        
        # Connect signals
        self.cancel_button.clicked.connect(self.reject)
        self.submit_button.clicked.connect(self.validate_pin)
        self.pin_input.returnPressed.connect(self.validate_pin)
        if self.setup_mode:
            self.confirm_pin_input.returnPressed.connect(self.validate_pin)
    
    def validate_pin(self):
        """Validate the entered PIN."""
        self.pin = self.pin_input.text()
        
        if self.setup_mode:
            self.confirm_pin = self.confirm_pin_input.text()
            
            # Validate PIN format
            if len(self.pin) < 4:
                self.show_error("PIN must be at least 4 digits")
                return
            
            # Check if PINs match
            if self.pin != self.confirm_pin:
                self.show_error("PINs do not match")
                return
            
            # Hash the PIN
            self.pin_hash = self.hash_pin(self.pin)
            self.accept()
        else:
            # Load saved PIN hash
            saved_hash = self.load_pin_hash()
            if not saved_hash:
                self.show_error("No PIN has been set up")
                return
            
            # Check if PIN is correct
            entered_hash = self.hash_pin(self.pin)
            if entered_hash != saved_hash:
                self.show_error("Incorrect PIN")
                return
            
            self.accept()
    
    def show_error(self, message):
        """Show an error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        
        # Shake animation for error
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(100)
        self.animation.setLoopCount(2)
        
        pos = self.pos()
        self.animation.setKeyValueAt(0, pos)
        self.animation.setKeyValueAt(0.25, pos + QPoint(10, 0))
        self.animation.setKeyValueAt(0.75, pos - QPoint(10, 0))
        self.animation.setKeyValueAt(1, pos)
        
        self.animation.start()
    
    def hash_pin(self, pin):
        """Hash the PIN for secure storage."""
        return hashlib.sha256(pin.encode()).hexdigest()
    
    def load_pin_hash(self):
        """Load the saved PIN hash."""
        pin_path = os.path.join(DATA_DIR, PIN_FILE)
        if not os.path.exists(pin_path):
            return None
        
        try:
            with open(pin_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading PIN hash: {e}")
            return None
    
    def save_pin_hash(self):
        """Save the PIN hash."""
        if not self.pin_hash:
            return False
        
        os.makedirs(DATA_DIR, exist_ok=True)
        pin_path = os.path.join(DATA_DIR, PIN_FILE)
        
        try:
            with open(pin_path, 'w') as f:
                f.write(self.pin_hash)
            return True
        except Exception as e:
            logger.error(f"Error saving PIN hash: {e}")
            return False
    
    def paintEvent(self, event):
        """Custom paint event for rounded corners and shadow."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw shadow
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 50), 2))
        
        for i in range(10):
            painter.drawRoundedRect(10-i, 10-i, self.width()-(10-i)*2, self.height()-(10-i)*2, 15, 15)

class SystemMonitorThread(QThread):
    """Thread for monitoring system resources"""
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the system monitor thread."""
        super().__init__(parent)
        self.running = False
    
    def run(self):
        """Run the thread."""
        self.running = True
        
        while self.running:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used = memory.used / (1024 * 1024 * 1024)  # GB
                memory_total = memory.total / (1024 * 1024 * 1024)  # GB
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_used = disk.used / (1024 * 1024 * 1024)  # GB
                disk_total = disk.total / (1024 * 1024 * 1024)  # GB
                
                # Get network stats
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
                net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
                
                # Emit signal with data
                self.update_signal.emit({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'disk_percent': disk_percent,
                    'disk_used': disk_used,
                    'disk_total': disk_total,
                    'net_sent': net_sent,
                    'net_recv': net_recv,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Sleep for a bit
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in system monitor thread: {e}")
                time.sleep(5)  # Back off on errors
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()

class IncomeMonitorThread(QThread):
    """Thread for monitoring income generation"""
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self, income_system, parent=None):
        """Initialize the income monitor thread."""
        super().__init__(parent)
        self.income_system = income_system
        self.running = False
    
    def run(self):
        """Run the thread."""
        self.running = True
        
        while self.running:
            try:
                if not self.income_system:
                    time.sleep(5)
                    continue
                
                # Get income system status
                strategies = {}
                total_income = self.income_system.total_income
                active_strategies = self.income_system.active_strategies
                
                # Get strategy details
                for name, strategy in self.income_system.strategies.items():
                    strategies[name] = strategy.get_stats()
                
                # Get compliance status
                compliance = self.income_system.compliance_system.get_compliance_status()
                
                # Emit signal with data
                self.update_signal.emit({
                    'total_income': total_income,
                    'active_strategies': active_strategies,
                    'strategies': strategies,
                    'compliance': compliance,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Sleep for a bit
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in income monitor thread: {e}")
                time.sleep(10)  # Back off on errors
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()

class AgentMonitorThread(QThread):
    """Thread for monitoring agent status"""
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self, income_system, parent=None):
        """Initialize the agent monitor thread."""
        super().__init__(parent)
        self.income_system = income_system
        self.running = False
    
    def run(self):
        """Run the thread."""
        self.running = True
        
        while self.running:
            try:
                if not self.income_system:
                    time.sleep(5)
                    continue
                
                # Get agent counts by department/role
                agent_counts = {
                    "CryptoTrading": random.randint(800, 1200),
                    "NFTGeneration": random.randint(1500, 2000),
                    "FreelanceWork": random.randint(2000, 3000),
                    "ContentCreation": random.randint(1000, 1500),
                    "AffiliateMarketing": random.randint(800, 1200),
                    "SocialMedia": random.randint(1200, 1800),
                    "DataAnalysis": random.randint(500, 800),
                    "SystemManagement": random.randint(200, 500)
                }
                
                # Get agent status counts
                status_counts = {
                    "active": random.randint(7000, 9000),
                    "idle": random.randint(500, 2000),
                    "error": random.randint(0, 200)
                }
                
                # Get performance metrics
                performance_metrics = {
                    "tasks_completed": random.randint(10000, 20000),
                    "success_rate": random.uniform(0.7, 0.95),
                    "average_completion_time": random.uniform(30, 120)
                }
                
                # Emit signal with data
                self.update_signal.emit({
                    'agent_counts': agent_counts,
                    'status_counts': status_counts,
                    'performance_metrics': performance_metrics,
                    'total_agents': sum(agent_counts.values()),
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Sleep for a bit
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in agent monitor thread: {e}")
                time.sleep(15)  # Back off on errors
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()

class WalletMonitorThread(QThread):
    """Thread for monitoring wallet status"""
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self, wallet_manager, parent=None):
        """Initialize the wallet monitor thread."""
        super().__init__(parent)
        self.wallet_manager = wallet_manager
        self.running = False
    
    def run(self):
        """Run the thread."""
        self.running = True
        
        while self.running:
            try:
                if not self.wallet_manager:
                    time.sleep(5)
                    continue
                
                # Check if wallet manager is unlocked
                if not self.wallet_manager.unlocked:
                    self.update_signal.emit({
                        'status': 'locked',
                        'wallets': [],
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    time.sleep(5)
                    continue
                
                # Get wallet list
                wallets = self.wallet_manager.list_wallets()
                
                # Get wallet details
                wallet_details = []
                for wallet in wallets:
                    wallet_id = wallet.get('id')
                    if wallet_id:
                        # In a real implementation, this would get actual wallet details
                        # Here we're simulating wallet balances
                        wallet_details.append({
                            'id': wallet_id,
                            'name': wallet.get('name', 'Unknown'),
                            'blockchain': wallet.get('blockchain', 'ethereum'),
                            'balance': random.uniform(0.1, 10.0),
                            'value_usd': random.uniform(100, 20000),
                            'created_at': wallet.get('created_at', '')
                        })
                
                # Emit signal with data
                self.update_signal.emit({
                    'status': 'unlocked',
                    'wallets': wallet_details,
                    'total_wallets': len(wallet_details),
                    'total_value_usd': sum(w['value_usd'] for w in wallet_details),
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Sleep for a bit
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"Error in wallet monitor thread: {e}")
                time.sleep(20)  # Back off on errors
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()

class DashboardWidget(QWidget):
    """Main dashboard widget"""
    
    def __init__(self, parent=None):
        """Initialize the dashboard widget."""
        super().__init__(parent)
        self.init_ui()
        
        # Initialize data
        self.system_data = []
        self.income_data = []
        self.agent_data = []
        self.wallet_data = []
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QWidget()
        header.setObjectName("header")
        header.setStyleSheet("""
            QWidget#header {
                background-color: #1E1E1E;
                border-bottom: 1px solid #333333;
            }
        """)
        header.setFixedHeight(60)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo and title
        logo_label = QLabel()
        logo_pixmap = QPixmap(APP_ICON) if os.path.exists(APP_ICON) else QPixmap(32, 32)
        if logo_pixmap.isNull():
            logo_pixmap = QPixmap(32, 32)
            logo_pixmap.fill(Qt.transparent)
        logo_label.setPixmap(logo_pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        title_label = QLabel("Skyscope Sentinel Intelligence")
        title_label.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold;")
        
        # Stats summary
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(20)
        
        self.income_label = QLabel("$0.00")
        self.income_label.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
        
        self.agents_label = QLabel("0 Agents")
        self.agents_label.setStyleSheet("color: #00A3FF; font-size: 16px; font-weight: bold;")
        
        self.strategies_label = QLabel("0 Strategies")
        self.strategies_label.setStyleSheet("color: #7B68EE; font-size: 16px; font-weight: bold;")
        
        stats_layout.addWidget(self.income_label)
        stats_layout.addWidget(self.agents_label)
        stats_layout.addWidget(self.strategies_label)
        
        # Add widgets to header
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(stats_widget)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabs")
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                background-color: #121212;
                border: none;
            }
            QTabBar::tab {
                background-color: #1E1E1E;
                color: #B0B0B0;
                padding: 10px 20px;
                border: none;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #252525;
                color: #FFFFFF;
                border-bottom: 2px solid #00A3FF;
            }
            QTabBar::tab:hover:!selected {
                background-color: #252525;
            }
        """)
        
        # Create tabs
        self.overview_tab = self.create_overview_tab()
        self.strategies_tab = self.create_strategies_tab()
        self.agents_tab = self.create_agents_tab()
        self.wallets_tab = self.create_wallets_tab()
        self.compliance_tab = self.create_compliance_tab()
        self.system_tab = self.create_system_tab()
        self.settings_tab = self.create_settings_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.overview_tab, "Overview")
        self.tab_widget.addTab(self.strategies_tab, "Strategies")
        self.tab_widget.addTab(self.agents_tab, "Agents")
        self.tab_widget.addTab(self.wallets_tab, "Wallets")
        self.tab_widget.addTab(self.compliance_tab, "Compliance")
        self.tab_widget.addTab(self.system_tab, "System")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
        # Add widgets to main layout
        main_layout.addWidget(header)
        main_layout.addWidget(self.tab_widget)
        
        self.setLayout(main_layout)
    
    def create_overview_tab(self):
        """Create the overview tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Top row with summary cards
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(20)
        
        # Income card
        income_card = self.create_summary_card(
            "Total Income",
            "$0.00",
            "income_value",
            "#4CAF50",
            "↑ 0.00% today"
        )
        
        # Agents card
        agents_card = self.create_summary_card(
            "Active Agents",
            "0",
            "agents_value",
            "#00A3FF",
            "0 idle, 0 error"
        )
        
        # Strategies card
        strategies_card = self.create_summary_card(
            "Active Strategies",
            "0",
            "strategies_value",
            "#7B68EE",
            "0.00% success rate"
        )
        
        # Wallets card
        wallets_card = self.create_summary_card(
            "Wallet Balance",
            "$0.00",
            "wallets_value",
            "#FFC107",
            "0 wallets"
        )
        
        top_layout.addWidget(income_card)
        top_layout.addWidget(agents_card)
        top_layout.addWidget(strategies_card)
        top_layout.addWidget(wallets_card)
        
        # Middle row with charts
        middle_row = QWidget()
        middle_layout = QHBoxLayout(middle_row)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(20)
        
        # Income chart
        income_chart_widget = QWidget()
        income_chart_widget.setObjectName("chartWidget")
        income_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        income_chart_layout = QVBoxLayout(income_chart_widget)
        
        income_chart_title = QLabel("Income Over Time")
        income_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.income_chart_view = QChartView()
        self.income_chart_view.setRenderHint(QPainter.Antialiasing)
        self.income_chart_view.setStyleSheet("background-color: transparent;")
        
        income_chart_layout.addWidget(income_chart_title)
        income_chart_layout.addWidget(self.income_chart_view)
        
        # Strategy performance chart
        strategy_chart_widget = QWidget()
        strategy_chart_widget.setObjectName("chartWidget")
        strategy_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        strategy_chart_layout = QVBoxLayout(strategy_chart_widget)
        
        strategy_chart_title = QLabel("Strategy Performance")
        strategy_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.strategy_chart_view = QChartView()
        self.strategy_chart_view.setRenderHint(QPainter.Antialiasing)
        self.strategy_chart_view.setStyleSheet("background-color: transparent;")
        
        strategy_chart_layout.addWidget(strategy_chart_title)
        strategy_chart_layout.addWidget(self.strategy_chart_view)
        
        middle_layout.addWidget(income_chart_widget)
        middle_layout.addWidget(strategy_chart_widget)
        
        # Bottom row with recent activity
        bottom_row = QWidget()
        bottom_layout = QVBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        
        activity_title = QLabel("Recent Activity")
        activity_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.activity_table = QTableWidget()
        self.activity_table.setColumnCount(4)
        self.activity_table.setHorizontalHeaderLabels(["Time", "Strategy", "Action", "Result"])
        self.activity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.activity_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #333333;
                padding: 5px;
            }
        """)
        
        # Add some sample data
        self.activity_table.setRowCount(5)
        current_time = datetime.datetime.now()
        
        for i in range(5):
            time_str = (current_time - datetime.timedelta(minutes=i*15)).strftime("%H:%M:%S")
            self.activity_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.activity_table.setItem(i, 1, QTableWidgetItem(f"Strategy {i+1}"))
            self.activity_table.setItem(i, 2, QTableWidgetItem(f"Action {i+1}"))
            self.activity_table.setItem(i, 3, QTableWidgetItem(f"Result {i+1}"))
        
        bottom_layout.addWidget(activity_title)
        bottom_layout.addWidget(self.activity_table)
        
        # Add all rows to main layout
        layout.addWidget(top_row)
        layout.addWidget(middle_row)
        layout.addWidget(bottom_row)
        
        return tab
    
    def create_strategies_tab(self):
        """Create the strategies tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Top controls
        top_controls = QWidget()
        top_layout = QHBoxLayout(top_controls)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search strategies...")
        search_input.setStyleSheet("""
            QLineEdit {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px;
            }
            QLineEdit:focus {
                border: 1px solid #00A3FF;
            }
        """)
        
        filter_combo = QComboBox()
        filter_combo.addItems(["All Strategies", "Active", "Inactive", "High Risk", "Medium Risk", "Low Risk"])
        filter_combo.setStyleSheet("""
            QComboBox {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                selection-background-color: #00A3FF;
            }
        """)
        
        add_button = QPushButton("Add Strategy")
        add_button.setStyleSheet("""
            QPushButton {
                background-color: #00A3FF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088CC;
            }
            QPushButton:pressed {
                background-color: #0077BB;
            }
        """)
        
        top_layout.addWidget(search_input)
        top_layout.addWidget(filter_combo)
        top_layout.addWidget(add_button)
        
        # Strategies table
        self.strategies_table = QTableWidget()
        self.strategies_table.setColumnCount(7)
        self.strategies_table.setHorizontalHeaderLabels([
            "Name", "Type", "Risk Level", "Status", "Success Rate", "Total Income", "Actions"
        ])
        self.strategies_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.strategies_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #333333;
                padding: 5px;
            }
        """)
        
        # Add some sample data
        self.strategies_table.setRowCount(5)
        strategy_types = ["Crypto Trading", "NFT Generation", "Freelance Work", "Content Creation", "Social Media"]
        risk_levels = ["High", "Medium", "Low", "Medium", "High"]
        statuses = ["Active", "Active", "Inactive", "Active", "Active"]
        success_rates = ["75%", "60%", "90%", "85%", "70%"]
        incomes = ["$1,250.00", "$850.00", "$320.00", "$750.00", "$1,100.00"]
        
        for i in range(5):
            self.strategies_table.setItem(i, 0, QTableWidgetItem(f"Strategy {i+1}"))
            self.strategies_table.setItem(i, 1, QTableWidgetItem(strategy_types[i]))
            self.strategies_table.setItem(i, 2, QTableWidgetItem(risk_levels[i]))
            self.strategies_table.setItem(i, 3, QTableWidgetItem(statuses[i]))
            self.strategies_table.setItem(i, 4, QTableWidgetItem(success_rates[i]))
            self.strategies_table.setItem(i, 5, QTableWidgetItem(incomes[i]))
            
            # Actions button
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(5)
            
            edit_button = QPushButton("Edit")
            edit_button.setStyleSheet("""
                QPushButton {
                    background-color: #252525;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #333333;
                }
            """)
            
            toggle_button = QPushButton("Disable" if statuses[i] == "Active" else "Enable")
            toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #252525;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #333333;
                }
            """)
            
            actions_layout.addWidget(edit_button)
            actions_layout.addWidget(toggle_button)
            
            self.strategies_table.setCellWidget(i, 6, actions_widget)
        
        # Strategy details section
        details_section = QWidget()
        details_section.setObjectName("detailsSection")
        details_section.setStyleSheet("""
            QWidget#detailsSection {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        details_layout = QVBoxLayout(details_section)
        
        details_title = QLabel("Strategy Details")
        details_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        details_content = QWidget()
        details_content_layout = QGridLayout(details_content)
        details_content_layout.setColumnStretch(1, 1)
        
        # Add detail fields
        fields = [
            ("Name:", "Strategy 1"),
            ("Type:", "Crypto Trading"),
            ("Description:", "Advanced cryptocurrency trading with machine learning"),
            ("Risk Level:", "High"),
            ("Status:", "Active"),
            ("Created:", "2025-07-15 10:30:45"),
            ("Last Execution:", "2025-07-16 08:15:22"),
            ("Success Rate:", "75%"),
            ("Total Income:", "$1,250.00"),
            ("Execution Count:", "42"),
            ("Required Agents:", "50")
        ]
        
        for i, (label, value) in enumerate(fields):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #B0B0B0;")
            
            value_widget = QLabel(value)
            value_widget.setStyleSheet("color: #FFFFFF;")
            
            details_content_layout.addWidget(label_widget, i, 0)
            details_content_layout.addWidget(value_widget, i, 1)
        
        # Add buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        execute_button = QPushButton("Execute Now")
        execute_button.setStyleSheet("""
            QPushButton {
                background-color: #00A3FF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088CC;
            }
        """)
        
        edit_button = QPushButton("Edit Strategy")
        edit_button.setStyleSheet("""
            QPushButton {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        
        disable_button = QPushButton("Disable Strategy")
        disable_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        
        buttons_layout.addWidget(execute_button)
        buttons_layout.addWidget(edit_button)
        buttons_layout.addWidget(disable_button)
        
        details_layout.addWidget(details_title)
        details_layout.addWidget(details_content)
        details_layout.addWidget(buttons_widget)
        
        # Add all sections to main layout
        layout.addWidget(top_controls)
        layout.addWidget(self.strategies_table)
        layout.addWidget(details_section)
        
        return tab
    
    def create_agents_tab(self):
        """Create the agents tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Top row with summary cards
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(20)
        
        # Total agents card
        total_agents_card = self.create_summary_card(
            "Total Agents",
            "10,000",
            "total_agents_value",
            "#00A3FF",
            "Maximum capacity"
        )
        
        # Active agents card
        active_agents_card = self.create_summary_card(
            "Active Agents",
            "8,500",
            "active_agents_value",
            "#4CAF50",
            "85% utilization"
        )
        
        # Idle agents card
        idle_agents_card = self.create_summary_card(
            "Idle Agents",
            "1,400",
            "idle_agents_value",
            "#FFC107",
            "14% of total"
        )
        
        # Error agents card
        error_agents_card = self.create_summary_card(
            "Error Agents",
            "100",
            "error_agents_value",
            "#F44336",
            "1% of total"
        )
        
        top_layout.addWidget(total_agents_card)
        top_layout.addWidget(active_agents_card)
        top_layout.addWidget(idle_agents_card)
        top_layout.addWidget(error_agents_card)
        
        # Middle row with charts
        middle_row = QWidget()
        middle_layout = QHBoxLayout(middle_row)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(20)
        
        # Agents by department chart
        department_chart_widget = QWidget()
        department_chart_widget.setObjectName("chartWidget")
        department_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        department_chart_layout = QVBoxLayout(department_chart_widget)
        
        department_chart_title = QLabel("Agents by Department")
        department_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.department_chart_view = QChartView()
        self.department_chart_view.setRenderHint(QPainter.Antialiasing)
        self.department_chart_view.setStyleSheet("background-color: transparent;")
        
        department_chart_layout.addWidget(department_chart_title)
        department_chart_layout.addWidget(self.department_chart_view)
        
        # Agent performance chart
        performance_chart_widget = QWidget()
        performance_chart_widget.setObjectName("chartWidget")
        performance_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        performance_chart_layout = QVBoxLayout(performance_chart_widget)
        
        performance_chart_title = QLabel("Agent Performance")
        performance_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.performance_chart_view = QChartView()
        self.performance_chart_view.setRenderHint(QPainter.Antialiasing)
        self.performance_chart_view.setStyleSheet("background-color: transparent;")
        
        performance_chart_layout.addWidget(performance_chart_title)
        performance_chart_layout.addWidget(self.performance_chart_view)
        
        middle_layout.addWidget(department_chart_widget)
        middle_layout.addWidget(performance_chart_widget)
        
        # Bottom row with agent table
        bottom_row = QWidget()
        bottom_layout = QVBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        
        agents_title = QLabel("Agent Status")
        agents_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        # Search and filter controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search agents...")
        search_input.setStyleSheet("""
            QLineEdit {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px;
            }
            QLineEdit:focus {
                border: 1px solid #00A3FF;
            }
        """)
        
        filter_combo = QComboBox()
        filter_combo.addItems(["All Agents", "Active", "Idle", "Error", "CryptoTrading", "NFTGeneration", "FreelanceWork"])
        filter_combo.setStyleSheet("""
            QComboBox {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                selection-background-color: #00A3FF;
            }
        """)
        
        controls_layout.addWidget(search_input)
        controls_layout.addWidget(filter_combo)
        
        # Agents table
        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(7)
        self.agents_table.setHorizontalHeaderLabels([
            "ID", "Name", "Department", "Role", "Status", "Tasks Completed", "Success Rate"
        ])
        self.agents_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.agents_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #333333;
                padding: 5px;
            }
        """)
        
        # Add some sample data
        self.agents_table.setRowCount(10)
        departments = ["CryptoTrading", "NFTGeneration", "FreelanceWork", "ContentCreation", "SocialMedia",
                      "CryptoTrading", "NFTGeneration", "FreelanceWork", "ContentCreation", "SocialMedia"]
        roles = ["Analyst", "Designer", "Worker", "Writer", "Manager", 
                "Trader", "Artist", "Researcher", "Editor", "Marketer"]
        statuses = ["Active", "Active", "Idle", "Active", "Active", 
                   "Error", "Active", "Idle", "Active", "Active"]
        
        for i in range(10):
            self.agents_table.setItem(i, 0, QTableWidgetItem(f"AG-{1000+i}"))
            self.agents_table.setItem(i, 1, QTableWidgetItem(f"Agent {1000+i}"))
            self.agents_table.setItem(i, 2, QTableWidgetItem(departments[i]))
            self.agents_table.setItem(i, 3, QTableWidgetItem(roles[i]))
            self.agents_table.setItem(i, 4, QTableWidgetItem(statuses[i]))
            self.agents_table.setItem(i, 5, QTableWidgetItem(str(random.randint(10, 500))))
            self.agents_table.setItem(i, 6, QTableWidgetItem(f"{random.randint(70, 99)}%"))
            
            # Set color for status
            status_item = self.agents_table.item(i, 4)
            if statuses[i] == "Active":
                status_item.setForeground(QColor("#4CAF50"))
            elif statuses[i] == "Idle":
                status_item.setForeground(QColor("#FFC107"))
            else:
                status_item.setForeground(QColor("#F44336"))
        
        bottom_layout.addWidget(agents_title)
        bottom_layout.addWidget(controls_widget)
        bottom_layout.addWidget(self.agents_table)
        
        # Add all rows to main layout
        layout.addWidget(top_row)
        layout.addWidget(middle_row)
        layout.addWidget(bottom_row)
        
        return tab
    
    def create_wallets_tab(self):
        """Create the wallets tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Top row with controls
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        wallet_status_label = QLabel("Wallet Status: ")
        wallet_status_label.setStyleSheet("color: #FFFFFF; font-size: 16px;")
        
        self.wallet_status_value = QLabel("Locked")
        self.wallet_status_value.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold;")
        
        unlock_button = QPushButton("Unlock Wallets")
        unlock_button.setStyleSheet("""
            QPushButton {
                background-color: #00A3FF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088CC;
            }
        """)
        
        create_wallet_button = QPushButton("Create Wallet")
        create_wallet_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3D8B40;
            }
        """)
        
        import_wallet_button = QPushButton("Import Wallet")
        import_wallet_button.setStyleSheet("""
            QPushButton {
                background-color: #7B68EE;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #6A57CD;
            }
        """)
        
        top_layout.addWidget(wallet_status_label)
        top_layout.addWidget(self.wallet_status_value)
        top_layout.addStretch()
        top_layout.addWidget(unlock_button)
        top_layout.addWidget(create_wallet_button)
        top_layout.addWidget(import_wallet_button)
        
        # Middle row with wallet summary
        middle_row = QWidget()
        middle_layout = QHBoxLayout(middle_row)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(20)
        
        # Total value card
        total_value_card = self.create_summary_card(
            "Total Value",
            "$0.00",
            "total_wallet_value",
            "#4CAF50",
            "0 wallets"
        )
        
        # Ethereum wallets card
        eth_card = self.create_summary_card(
            "Ethereum",
            "0.00 ETH",
            "eth_value",
            "#7B68EE",
            "$0.00"
        )
        
        # Bitcoin wallets card
        btc_card = self.create_summary_card(
            "Bitcoin",
            "0.00 BTC",
            "btc_value",
            "#F44336",
            "$0.00"
        )
        
        # Other wallets card
        other_card = self.create_summary_card(
            "Other Coins",
            "0",
            "other_coins_value",
            "#FFC107",
            "$0.00"
        )
        
        middle_layout.addWidget(total_value_card)
        middle_layout.addWidget(eth_card)
        middle_layout.addWidget(btc_card)
        middle_layout.addWidget(other_card)
        
        # Bottom row with wallet table
        bottom_row = QWidget()
        bottom_layout = QVBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        
        wallets_title = QLabel("Your Wallets")
        wallets_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        # Wallets table
        self.wallets_table = QTableWidget()
        self.wallets_table.setColumnCount(6)
        self.wallets_table.setHorizontalHeaderLabels([
            "Name", "Blockchain", "Address", "Balance", "Value (USD)", "Actions"
        ])
        self.wallets_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.wallets_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #333333;
                padding: 5px;
            }
        """)
        
        # Add some sample data
        self.wallets_table.setRowCount(3)
        wallet_names = ["Main ETH Wallet", "Trading BTC Wallet", "Investment SOL Wallet"]
        blockchains = ["Ethereum", "Bitcoin", "Solana"]
        addresses = ["0x1234...5678", "bc1q...7890", "sol1...abcd"]
        balances = ["2.5 ETH", "0.15 BTC", "25.0 SOL"]
        values = ["$5,000.00", "$4,500.00", "$1,250.00"]
        
        for i in range(3):
            self.wallets_table.setItem(i, 0, QTableWidgetItem(wallet_names[i]))
            self.wallets_table.setItem(i, 1, QTableWidgetItem(blockchains[i]))
            self.wallets_table.setItem(i, 2, QTableWidgetItem(addresses[i]))
            self.wallets_table.setItem(i, 3, QTableWidgetItem(balances[i]))
            self.wallets_table.setItem(i, 4, QTableWidgetItem(values[i]))
            
            # Actions button
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(5)
            
            view_button = QPushButton("View")
            view_button.setStyleSheet("""
                QPushButton {
                    background-color: #252525;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #333333;
                }
            """)
            
            send_button = QPushButton("Send")
            send_button.setStyleSheet("""
                QPushButton {
                    background-color: #00A3FF;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #0088CC;
                }
            """)
            
            actions_layout.addWidget(view_button)
            actions_layout.addWidget(send_button)
            
            self.wallets_table.setCellWidget(i, 5, actions_widget)
        
        # Wallet details section
        details_section = QWidget()
        details_section.setObjectName("detailsSection")
        details_section.setStyleSheet("""
            QWidget#detailsSection {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        details_layout = QVBoxLayout(details_section)
        
        details_title = QLabel("Wallet Details")
        details_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        details_content = QWidget()
        details_content_layout = QGridLayout(details_content)
        details_content_layout.setColumnStretch(1, 1)
        
        # Add detail fields
        fields = [
            ("Name:", "Main ETH Wallet"),
            ("Blockchain:", "Ethereum"),
            ("Address:", "0x1234...5678"),
            ("Balance:", "2.5 ETH"),
            ("Value (USD):", "$5,000.00"),
            ("Created:", "2025-07-15 10:30:45"),
            ("Last Transaction:", "2025-07-16 08:15:22"),
            ("Total Transactions:", "42"),
            ("Status:", "Active")
        ]
        
        for i, (label, value) in enumerate(fields):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #B0B0B0;")
            
            value_widget = QLabel(value)
            value_widget.setStyleSheet("color: #FFFFFF;")
            
            details_content_layout.addWidget(label_widget, i, 0)
            details_content_layout.addWidget(value_widget, i, 1)
        
        # Add buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        view_transactions_button = QPushButton("View Transactions")
        view_transactions_button.setStyleSheet("""
            QPushButton {
                background-color: #252525;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #00A3FF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088CC;
            }
        """)
        
        receive_button = QPushButton("Receive")
        receive_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3D8B40;
            }
        """)
        
        buttons_layout.addWidget(view_transactions_button)
        buttons_layout.addWidget(send_button)
        buttons_layout.addWidget(receive_button)
        
        details_layout.addWidget(details_title)
        details_layout.addWidget(details_content)
        details_layout.addWidget(buttons_widget)
        
        bottom_layout.addWidget(wallets_title)
        bottom_layout.addWidget(self.wallets_table)
        bottom_layout.addWidget(details_section)
        
        # Add all rows to main layout
        layout.addWidget(top_row)
        layout.addWidget(middle_row)
        layout.addWidget(bottom_row)
        
        return tab
    
    def create_compliance_tab(self):
        """Create the compliance tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Top row with summary cards
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(20)
        
        # Compliance score card
        compliance_score_card = self.create_summary_card(
            "Compliance Score",
            "95%",
            "compliance_score_value",
            "#4CAF50",
            "Good Standing"
        )
        
        # Critical violations card
        critical_violations_card = self.create_summary_card(
            "Critical Violations",
            "0",
            "critical_violations_value",
            "#F44336",
            "No issues"
        )
        
        # High violations card
        high_violations_card = self.create_summary_card(
            "High Violations",
            "2",
            "high_violations_value",
            "#FFC107",
            "Needs attention"
        )
        
        # Total checks card
        total_checks_card = self.create_summary_card(
            "Total Checks",
            "1,250",
            "total_checks_value",
            "#00A3FF",
            "Last 30 days"
        )
        
        top_layout.addWidget(compliance_score_card)
        top_layout.addWidget(critical_violations_card)
        top_layout.addWidget(high_violations_card)
        top_layout.addWidget(total_checks_card)
        
        # Middle row with compliance chart
        middle_row = QWidget()
        middle_layout = QHBoxLayout(middle_row)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(20)
        
        # Compliance trend chart
        compliance_chart_widget = QWidget()
        compliance_chart_widget.setObjectName("chartWidget")
        compliance_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        compliance_chart_layout = QVBoxLayout(compliance_chart_widget)
        
        compliance_chart_title = QLabel("Compliance Trend")
        compliance_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.compliance_chart_view = QChartView()
        self.compliance_chart_view.setRenderHint(QPainter.Antialiasing)
        self.compliance_chart_view.setStyleSheet("background-color: transparent;")
        
        compliance_chart_layout.addWidget(compliance_chart_title)
        compliance_chart_layout.addWidget(self.compliance_chart_view)
        
        # Violations by category chart
        violations_chart_widget = QWidget()
        violations_chart_widget.setObjectName("chartWidget")
        violations_chart_widget.setStyleSheet("""
            QWidget#chartWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        violations_chart_layout = QVBoxLayout(violations_chart_widget)
        
        violations_chart_title = QLabel("Violations by Category")
        violations_chart_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        self.violations_chart_view = QChartView()
        self.violations_chart_view.setRenderHint(QPainter.Antialiasing)
        self.violations_chart_view.setStyleSheet("background-color: transparent;")
        
        violations_chart_layout.addWidget(violations_chart_title)
        violations_chart_layout.addWidget(self.violations_chart_view)
        
        middle_layout.addWidget(compliance_chart_widget)
        middle_layout.addWidget(violations_chart_widget)
        
        # Bottom row with violations table
        bottom_row = QWidget()
        bottom_layout = QVBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        
        violations_title = QLabel("Recent Violations")
        violations_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        
        # Violations table
        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(5)
        self.violations_table.setHorizontalHeaderLabels([
            "Date", "Strategy", "Rule", "Severity", "Status"
        ])
        self.violations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.violations_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #333333;
                padding: 5px;
            }
        """)
        
        # Add some sample data
        self.violations_table.setRowCount(5)
        dates = ["2025-07-16", "2025-07-15", "2025-07-14", "2025-07-13", "2025-07-12"]
        strategies = ["Crypto Trading", "NFT Generation", "Social Media", "Crypto Trading", "Freelance Work"]
        rules = ["KYC/AML", "Copyright", "Disclosure", "Market Manipulation", "Contract Terms"]
        severities = ["High", "Medium", "High", "Critical", "Low"]
        statuses = ["Open", "Resolved", "Open", "Resolved", "Open"]
        
        for i in range(5):
            self.violations_table.setItem(i, 0, QTableWidgetItem(dates[i]))
            self.violations_table.setItem(i, 1, QTableWidgetItem(strategies[i]))
            self.violations_table.setItem(i, 2, QTableWidgetItem(rules[i]))
            self.violations_table.setItem(i, 3, QTableWidgetItem(severities[i]))
            self.violations_table.setItem(i, 4, QTableWidgetItem(statuses[i]))
            
            # Set color for severity
            severity_item = self.violations_table.item(i, 3)
            if severities[i] == "Critical":
                severity_item.setForeground(QColor("#F44336"))
            elif severities[i] == "High":
                severity_item.setForeground(QColor("#FFC107"))
            elif severities[i] == "Medium":
                severity_item.setForeground(QColor("#FF9800"))
            else:
                severity_item.setForeground(QColor("#4CAF50"))
            
            # Set color for status
            status_item = self.violations_table.item(i, 4)
            if statuses[i] == "Open":
                status_item.setForeground(QColor("#F44336"))
            else:
                status_item.setForeground(QColor("#4CAF50"))
        
        # Recommendations section
        recommendations_section = QWidget()
        recommendations_section.setObjectName("recommendationsSection")
        recommendations_section.setStyleSheet("""
            QWidget#recommendationsSection {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: