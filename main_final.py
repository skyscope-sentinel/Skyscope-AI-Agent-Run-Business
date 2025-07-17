
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Enterprise Suite - Complete Main Application
====================================================
This is the main entry point for the Skyscope Enterprise Suite.
It integrates openai-unofficial for GPT-4 access and provides a complete GUI.
"""

import os
import sys
import json
import time
import random
import logging
import threading
import base64
import keyring
import datetime
import webbrowser
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# GUI imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QProgressBar, QScrollArea, QGridLayout, QFrame, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QCheckBox, QGroupBox, QFormLayout, QSpinBox,
    QStackedWidget, QToolButton, QMenu, QDialog, QListWidget,
    QListWidgetItem, QInputDialog, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QSize, QThread, pyqtSignal, QTimer, QUrl, QSettings, 
    QByteArray, QBuffer, QIODevice, QRect, QPoint, QModelIndex
)
from PyQt6.QtGui import (
    QPalette, QColor, QFont, QIcon, QPixmap, QPainter, QBrush, 
    QPen, QLinearGradient, QTextCursor, QAction, QKeySequence
)
# NOTE: Module is 'QtCharts' (plural), not 'QtChart'
from PyQt6.QtCharts import (
    QChart,
    QChartView,
    QPieSeries,
    QLineSeries,
    QBarSeries,
    QBarSet,
    QBarCategoryAxis,
    QValueAxis,
    QPieSlice,  # added missing import
)

# Import new unofficial OpenAI provider (no auth required)
try:
    # Local import – path already in PYTHONPATH because we're inside src/
    from unofficial_openai_provider import UnofficialOpenAIProvider
    OPENAI_PROVIDER = UnofficialOpenAIProvider()
    OPENAI_PROVIDER_AVAILABLE = OPENAI_PROVIDER.is_available()
except Exception as _e:  # noqa: N818
    OPENAI_PROVIDER = None
    OPENAI_PROVIDER_AVAILABLE = False
    print("Warning: unofficial_openai_provider not available – AI features disabled:", _e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/SkyscopeEnterprise/logs/app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
APP_NAME = "Skyscope Enterprise Suite"
APP_VERSION = "1.0.0"
CONFIG_DIR = os.path.expanduser("~/SkyscopeEnterprise/config")
DATA_DIR = os.path.expanduser("~/SkyscopeEnterprise/data")
AGENTS_DIR = os.path.expanduser("~/SkyscopeEnterprise/agents")
STRATEGIES_DIR = os.path.expanduser("~/SkyscopeEnterprise/strategies")
LOGS_DIR = os.path.expanduser("~/SkyscopeEnterprise/logs")

# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, AGENTS_DIR, STRATEGIES_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Agent types and strategies
AGENT_TYPES = [
    "Content Creator", "Crypto Trader", "Social Media Manager", 
    "SEO Specialist", "Data Analyst", "Web Researcher",
    "Customer Support", "Product Developer", "Market Researcher",
    "Copywriter"
]

INCOME_STRATEGIES = [
    "Crypto Trading", "Content Creation", "Social Media Management",
    "SEO Optimization", "Data Analysis", "Web Research",
    "Customer Support", "Product Development", "Market Research",
    "Copywriting"
]

# Secure token storage
def save_session_token(token: str) -> bool:
    """Save the OpenAI session token securely."""
    try:
        keyring.set_password(APP_NAME, "openai_session_token", token)
        return True
    except Exception as e:
        logger.error(f"Failed to save session token: {e}")
        return False

def get_session_token() -> Optional[str]:
    """Get the OpenAI session token."""
    try:
        token = keyring.get_password(APP_NAME, "openai_session_token")
        return token
    except Exception as e:
        logger.error(f"Failed to get session token: {e}")
        return None

# Configuration management
def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = os.path.join(CONFIG_DIR, "config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "app_name": APP_NAME,
                "version": APP_VERSION,
                "theme": "dark",
                "agents": {
                    "count": 10000,
                    "active": 0
                },
                "ai_providers": {
                    "primary": "openai-unofficial",
                    "model": "gpt-4"
                },
                "income_strategies": {
                    "active": []
                },
                "analytics": {
                    "refresh_interval": 60
                }
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    config_path = os.path.join(CONFIG_DIR, "config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False

# Thread for running GPT-4 queries via the new provider
class GPT4Thread(QThread):
    """Thread for running GPT-4 queries without blocking the UI (uses free provider)."""
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, prompt: str, model: str = "gpt-4o-mini"):
        super().__init__()
        self.prompt = prompt
        self.model = model
    
    def run(self):
        try:
            if not OPENAI_PROVIDER_AVAILABLE or OPENAI_PROVIDER is None:
                self.error_signal.emit("Unofficial OpenAI API is not available.")
                return
            self.status_signal.emit("Contacting GPT-4...")

            # Build messages list identical to OpenAI chat format
            messages = [{"role": "user", "content": self.prompt}]
            result = OPENAI_PROVIDER.chat_completion(
                messages=messages,
                model=self.model,
                stream=False,
            )

            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                self.response_signal.emit(content)
            else:
                self.error_signal.emit("No response from provider.")
        except Exception as e:
            logger.error(f"GPT-4 query failed: {e}")
            self.error_signal.emit(f"GPT-4 query failed: {str(e)}")

# Thread for agent management
class AgentManagerThread(QThread):
    """Thread for managing agents without blocking the UI."""
    status_signal = pyqtSignal(str)
    agent_created_signal = pyqtSignal(dict)
    agent_updated_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, action: str, agent_data: Dict[str, Any] = None):
        super().__init__()
        self.action = action
        self.agent_data = agent_data or {}
    
    def run(self):
        try:
            if self.action == "create":
                self.status_signal.emit(f"Creating agent: {self.agent_data.get('name', 'Unknown')}")
                # Simulate agent creation
                time.sleep(1)
                
                # Generate unique ID
                agent_id = f"agent_{int(time.time())}_{random.randint(1000, 9999)}"
                self.agent_data["id"] = agent_id
                self.agent_data["created_at"] = datetime.datetime.now().isoformat()
                self.agent_data["status"] = "idle"
                
                # Save agent data
                agent_file = os.path.join(AGENTS_DIR, f"{agent_id}.json")
                with open(agent_file, 'w') as f:
                    json.dump(self.agent_data, f, indent=2)
                
                self.agent_created_signal.emit(self.agent_data)
                self.status_signal.emit(f"Agent {self.agent_data.get('name')} created successfully")
            
            elif self.action == "update":
                self.status_signal.emit(f"Updating agent: {self.agent_data.get('name', 'Unknown')}")
                # Simulate agent update
                time.sleep(0.5)
                
                # Save agent data
                agent_id = self.agent_data.get("id")
                if not agent_id:
                    raise ValueError("Agent ID is required for update")
                
                agent_file = os.path.join(AGENTS_DIR, f"{agent_id}.json")
                with open(agent_file, 'w') as f:
                    json.dump(self.agent_data, f, indent=2)
                
                self.agent_updated_signal.emit(self.agent_data)
                self.status_signal.emit(f"Agent {self.agent_data.get('name')} updated successfully")
            
            elif self.action == "delete":
                self.status_signal.emit(f"Deleting agent: {self.agent_data.get('name', 'Unknown')}")
                # Simulate agent deletion
                time.sleep(0.5)
                
                # Delete agent file
                agent_id = self.agent_data.get("id")
                if not agent_id:
                    raise ValueError("Agent ID is required for deletion")
                
                agent_file = os.path.join(AGENTS_DIR, f"{agent_id}.json")
                if os.path.exists(agent_file):
                    os.remove(agent_file)
                
                self.status_signal.emit(f"Agent {self.agent_data.get('name')} deleted successfully")
            
        except Exception as e:
            logger.error(f"Agent action failed: {e}")
            self.error_signal.emit(f"Agent action failed: {str(e)}")

# Custom styled components
class StyledButton(QPushButton):
    """Custom styled button."""
    def __init__(self, text="", parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setMinimumHeight(36)
        self.setFont(QFont("Arial", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                background-color: #1d1d1d;
                color: #5d5d5d;
            }
        """)
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2c7dfa;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #4c8dfa;
                }
                QPushButton:pressed {
                    background-color: #1c6dfa;
                }
                QPushButton:disabled {
                    background-color: #1d1d1d;
                    color: #5d5d5d;
                }
            """)

class StyledLineEdit(QLineEdit):
    """Custom styled line edit."""
    def __init__(self, parent=None, placeholder=""):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setFont(QFont("Arial", 10))
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QLineEdit:focus {
                border: 1px solid #5d5d5d;
            }
            QLineEdit:disabled {
                background-color: #1a1a1a;
                color: #5d5d5d;
            }
        """)

class StyledTextEdit(QTextEdit):
    """Custom styled text edit."""
    def __init__(self, parent=None, placeholder=""):
        super().__init__(parent)
        self.setFont(QFont("Arial", 10))
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
            QTextEdit:focus {
                border: 1px solid #5d5d5d;
            }
            QTextEdit:disabled {
                background-color: #1a1a1a;
                color: #5d5d5d;
            }
        """)

class StyledComboBox(QComboBox):
    """Custom styled combo box."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setFont(QFont("Arial", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QComboBox {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3d3d3d;
            }
            QComboBox QAbstractItemView {
                background-color: #1d1d1d;
                color: white;
                selection-background-color: #3d3d3d;
                selection-color: white;
            }
        """)

class StyledProgressBar(QProgressBar):
    """Custom styled progress bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(10)
        self.setMaximumHeight(10)
        self.setTextVisible(False)
        self.setStyleSheet("""
            QProgressBar {
                background-color: #1d1d1d;
                border: none;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #2c7dfa;
                border-radius: 5px;
            }
        """)

class StatusBar(QWidget):
    """Custom status bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.connection_label = QLabel("Not connected")
        self.connection_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.connection_label)
        
        self.setStyleSheet("""
            StatusBar {
                background-color: #1a1a1a;
                border-top: 1px solid #3d3d3d;
            }
            QLabel {
                color: #9d9d9d;
            }
        """)
    
    def set_status(self, text):
        """Set status text."""
        self.status_label.setText(text)
    
    def set_connection(self, connected):
        """Set connection status."""
        if connected:
            self.connection_label.setText("Connected to GPT-4")
            self.connection_label.setStyleSheet("color: #2ecc71;")
        else:
            self.connection_label.setText("Not connected")
            self.connection_label.setStyleSheet("color: #e74c3c;")

# Tab widgets
class DashboardTab(QWidget):
    """Dashboard tab with AI chat interface."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Welcome message
        welcome_label = QLabel("Welcome to Skyscope Enterprise Suite")
        welcome_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        
        # Status overview
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        status_layout = QHBoxLayout(status_frame)
        
        # Active agents
        agents_widget = QWidget()
        agents_layout = QVBoxLayout(agents_widget)
        agents_label = QLabel("Active Agents")
        agents_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        agents_count = QLabel("0 / 10,000")
        agents_count.setFont(QFont("Arial", 16))
        agents_layout.addWidget(agents_label)
        agents_layout.addWidget(agents_count)
        status_layout.addWidget(agents_widget)
        
        # Active strategies
        strategies_widget = QWidget()
        strategies_layout = QVBoxLayout(strategies_widget)
        strategies_label = QLabel("Active Strategies")
        strategies_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        strategies_count = QLabel("0 / 10")
        strategies_count.setFont(QFont("Arial", 16))
        strategies_layout.addWidget(strategies_label)
        strategies_layout.addWidget(strategies_count)
        status_layout.addWidget(strategies_widget)
        
        # Income generated
        income_widget = QWidget()
        income_layout = QVBoxLayout(income_widget)
        income_label = QLabel("Income Generated")
        income_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        income_value = QLabel("$0.00")
        income_value.setFont(QFont("Arial", 16))
        income_layout.addWidget(income_label)
        income_layout.addWidget(income_value)
        status_layout.addWidget(income_widget)
        
        # System status
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)
        system_label = QLabel("System Status")
        system_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        system_status = QLabel("Online")
        system_status.setFont(QFont("Arial", 16))
        system_status.setStyleSheet("color: #2ecc71;")
        system_layout.addWidget(system_label)
        system_layout.addWidget(system_status)
        status_layout.addWidget(system_widget)
        
        layout.addWidget(status_frame)
        
        # Chat interface
        chat_label = QLabel("AI Assistant")
        chat_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(chat_label)
        
        chat_frame = QFrame()
        chat_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chat_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        chat_layout = QVBoxLayout(chat_frame)
        
        self.chat_history = StyledTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Chat history will appear here...")
        chat_layout.addWidget(self.chat_history)
        
        input_layout = QHBoxLayout()
        self.chat_input = StyledLineEdit(placeholder="Type your message here...")
        input_layout.addWidget(self.chat_input)
        
        self.send_button = StyledButton("Send", primary=True)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        chat_layout.addLayout(input_layout)
        layout.addWidget(chat_frame)
        
        # Quick actions
        actions_label = QLabel("Quick Actions")
        actions_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(actions_label)
        
        actions_layout = QHBoxLayout()
        
        create_agent_btn = StyledButton("Create Agent")
        create_agent_btn.clicked.connect(self.create_agent)
        actions_layout.addWidget(create_agent_btn)
        
        add_strategy_btn = StyledButton("Add Strategy")
        add_strategy_btn.clicked.connect(self.add_strategy)
        actions_layout.addWidget(add_strategy_btn)
        
        view_analytics_btn = StyledButton("View Analytics")
        view_analytics_btn.clicked.connect(self.view_analytics)
        actions_layout.addWidget(view_analytics_btn)
        
        settings_btn = StyledButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        actions_layout.addWidget(settings_btn)
        
        layout.addLayout(actions_layout)
        
        # Store references
        self.agents_count = agents_count
        self.strategies_count = strategies_count
        self.income_value = income_value
        self.system_status = system_status
        
        # Update dashboard
        self.update_dashboard()
    
    def update_dashboard(self):
        """Update dashboard with current data."""
        # Load agents
        try:
            agent_files = [f for f in os.listdir(AGENTS_DIR) if f.endswith('.json')]
            active_agents = 0
            for agent_file in agent_files:
                with open(os.path.join(AGENTS_DIR, agent_file), 'r') as f:
                    agent_data = json.load(f)
                    if agent_data.get('status') == 'active':
                        active_agents += 1
            
            self.agents_count.setText(f"{active_agents} / 10,000")
        except Exception as e:
            logger.error(f"Failed to load agents: {e}")
        
        # Load strategies
        try:
            strategy_files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.json')]
            active_strategies = 0
            for strategy_file in strategy_files:
                with open(os.path.join(STRATEGIES_DIR, strategy_file), 'r') as f:
                    strategy_data = json.load(f)
                    if strategy_data.get('status') == 'active':
                        active_strategies += 1
            
            self.strategies_count.setText(f"{active_strategies} / 10")
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
        
        # Load income
        try:
            income_file = os.path.join(DATA_DIR, "income.json")
            if os.path.exists(income_file):
                with open(income_file, 'r') as f:
                    income_data = json.load(f)
                    total_income = income_data.get('total', 0)
                    self.income_value.setText(f"${total_income:.2f}")
            else:
                self.income_value.setText("$0.00")
        except Exception as e:
            logger.error(f"Failed to load income data: {e}")
    
    def send_message(self):
        """Send message to GPT-4."""
        message = self.chat_input.text().strip()
        if not message:
            return
        
        # Add user message to chat history
        self.chat_history.append(f"<b>You:</b> {message}")
        self.chat_input.clear()
        
        # Get session token
        token = get_session_token()
        if not token:
            self.chat_history.append("<i>Error: No session token found. Please configure in Settings.</i>")
            return
        
        # Start GPT-4 thread
        self.gpt4_thread = GPT4Thread(token, message)
        self.gpt4_thread.response_signal.connect(self.handle_response)
        self.gpt4_thread.error_signal.connect(self.handle_error)
        self.gpt4_thread.status_signal.connect(self.handle_status)
        self.gpt4_thread.start()
    
    def handle_response(self, response):
        """Handle GPT-4 response."""
        self.chat_history.append(f"<b>AI:</b> {response}")
    
    def handle_error(self, error):
        """Handle GPT-4 error."""
        self.chat_history.append(f"<i>Error: {error}</i>")
    
    def handle_status(self, status):
        """Handle GPT-4 status update."""
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(status)
    
    def create_agent(self):
        """Switch to Agents tab and trigger agent creation."""
        if hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(1)  # Switch to Agents tab
            # Trigger agent creation in Agents tab
            agents_tab = self.parent.tabs.widget(1)
            if hasattr(agents_tab, 'create_agent'):
                agents_tab.create_agent()
    
    def add_strategy(self):
        """Switch to Strategies tab."""
        if hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(2)  # Switch to Strategies tab
    
    def view_analytics(self):
        """Switch to Analytics tab."""
        if hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(3)  # Switch to Analytics tab
    
    def open_settings(self):
        """Switch to Settings tab."""
        if hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(4)  # Switch to Settings tab

class AgentsTab(QWidget):
    """Agents tab for managing AI agents."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.agents = []
        self.init_ui()
        self.load_agents()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Agent Management")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.create_agent_btn = StyledButton("Create Agent", primary=True)
        self.create_agent_btn.clicked.connect(self.create_agent)
        header_layout.addWidget(self.create_agent_btn)
        
        layout.addLayout(header_layout)
        
        # Agent list
        self.agent_table = QTableWidget()
        self.agent_table.setColumnCount(6)
        self.agent_table.setHorizontalHeaderLabels(["Name", "Type", "Status", "Created", "Tasks", "Actions"])
        self.agent_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.agent_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.agent_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.agent_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.agent_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.agent_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.agent_table.setStyleSheet("""
            QTableWidget {
                background-color: #1d1d1d;
                color: white;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: white;
                padding: 5px;
                border: 1px solid #3d3d3d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3d3d3d;
            }
        """)
        layout.addWidget(self.agent_table)
        
        # Agent details
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        details_layout = QVBoxLayout(details_frame)
        
        details_label = QLabel("Agent Details")
        details_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        details_layout.addWidget(details_label)
        
        form_layout = QFormLayout()
        
        self.detail_name = QLabel("Select an agent to view details")
        self.detail_type = QLabel("")
        self.detail_status = QLabel("")
        self.detail_created = QLabel("")
        self.detail_tasks = QLabel("")
        
        form_layout.addRow("Name:", self.detail_name)
        form_layout.addRow("Type:", self.detail_type)
        form_layout.addRow("Status:", self.detail_status)
        form_layout.addRow("Created:", self.detail_created)
        form_layout.addRow("Tasks Completed:", self.detail_tasks)
        
        details_layout.addLayout(form_layout)
        
        # Agent actions
        actions_layout = QHBoxLayout()
        
        self.start_agent_btn = StyledButton("Start Agent")
        self.start_agent_btn.clicked.connect(self.start_agent)
        self.start_agent_btn.setEnabled(False)
        actions_layout.addWidget(self.start_agent_btn)
        
        self.stop_agent_btn = StyledButton("Stop Agent")
        self.stop_agent_btn.clicked.connect(self.stop_agent)
        self.stop_agent_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_agent_btn)
        
        self.edit_agent_btn = StyledButton("Edit Agent")
        self.edit_agent_btn.clicked.connect(self.edit_agent)
        self.edit_agent_btn.setEnabled(False)
        actions_layout.addWidget(self.edit_agent_btn)
        
        self.delete_agent_btn = StyledButton("Delete Agent")
        self.delete_agent_btn.clicked.connect(self.delete_agent)
        self.delete_agent_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_agent_btn)
        
        details_layout.addLayout(actions_layout)
        
        layout.addWidget(details_frame)
        
        # Connect signals
        self.agent_table.itemSelectionChanged.connect(self.update_agent_details)
    
    def load_agents(self):
        """Load agents from files."""
        try:
            self.agents = []
            agent_files = [f for f in os.listdir(AGENTS_DIR) if f.endswith('.json')]
            
            for agent_file in agent_files:
                with open(os.path.join(AGENTS_DIR, agent_file), 'r') as f:
                    agent_data = json.load(f)
                    self.agents.append(agent_data)
            
            self.update_agent_table()
        except Exception as e:
            logger.error(f"Failed to load agents: {e}")
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Failed to load agents: {str(e)}")
    
    def update_agent_table(self):
        """Update agent table with loaded agents."""
        self.agent_table.setRowCount(0)
        
        for agent in self.agents:
            row = self.agent_table.rowCount()
            self.agent_table.insertRow(row)
            
            # Name
            name_item = QTableWidgetItem(agent.get('name', 'Unknown'))
            self.agent_table.setItem(row, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(agent.get('type', 'Unknown'))
            self.agent_table.setItem(row, 1, type_item)
            
            # Status
            status = agent.get('status', 'idle')
            status_item = QTableWidgetItem(status)
            if status == 'active':
                status_item.setForeground(QColor('#2ecc71'))
            elif status == 'idle':
                status_item.setForeground(QColor('#f39c12'))
            else:
                status_item.setForeground(QColor('#e74c3c'))
            self.agent_table.setItem(row, 2, status_item)
            
            # Created
            created_at = agent.get('created_at', '')
            if created_at:
                try:
                    created_dt = datetime.datetime.fromisoformat(created_at)
                    created_str = created_dt.strftime('%Y-%m-%d %H:%M')
                except:
                    created_str = created_at
            else:
                created_str = 'Unknown'
            created_item = QTableWidgetItem(created_str)
            self.agent_table.setItem(row, 3, created_item)
            
            # Tasks
            tasks_completed = agent.get('tasks_completed', 0)
            tasks_item = QTableWidgetItem(str(tasks_completed))
            self.agent_table.setItem(row, 4, tasks_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            view_btn = QPushButton("View")
            view_btn.setStyleSheet("background-color: #2d2d2d; color: white; border: none;")
            view_btn.clicked.connect(lambda checked, a=agent: self.view_agent(a))
            actions_layout.addWidget(view_btn)
            
            self.agent_table.setCellWidget(row, 5, actions_widget)
    
    def update_agent_details(self):
        """Update agent details panel based on selection."""
        selected_items = self.agent_table.selectedItems()
        if not selected_items:
            self.clear_agent_details()
            return
        
        row = selected_items[0].row()
        name = self.agent_table.item(row, 0).text()
        
        # Find agent data
        agent = next((a for a in self.agents if a.get('name') == name), None)
        if not agent:
            self.clear_agent_details()
            return
        
        # Update details
        self.detail_name.setText(agent.get('name', 'Unknown'))
        self.detail_type.setText(agent.get('type', 'Unknown'))
        self.detail_status.setText(agent.get('status', 'idle'))
        
        created_at = agent.get('created_at', '')
        if created_at:
            try:
                created_dt = datetime.datetime.fromisoformat(created_at)
                created_str = created_dt.strftime('%Y-%m-%d %H:%M')
            except:
                created_str = created_at
        else:
            created_str = 'Unknown'
        self.detail_created.setText(created_str)
        
        self.detail_tasks.setText(str(agent.get('tasks_completed', 0)))
        
        # Enable/disable buttons based on status
        status = agent.get('status', 'idle')
        self.start_agent_btn.setEnabled(status != 'active')
        self.stop_agent_btn.setEnabled(status == 'active')
        self.edit_agent_btn.setEnabled(True)
        self.delete_agent_btn.setEnabled(True)
        
        # Store current agent
        self.current_agent = agent
    
    def clear_agent_details(self):
        """Clear agent details panel."""
        self.detail_name.setText("Select an agent to view details")
        self.detail_type.setText("")
        self.detail_status.setText("")
        self.detail_created.setText("")
        self.detail_tasks.setText("")
        
        self.start_agent_btn.setEnabled(False)
        self.stop_agent_btn.setEnabled(False)
        self.edit_agent_btn.setEnabled(False)
        self.delete_agent_btn.setEnabled(False)
        
        self.current_agent = None
    
    def create_agent(self):
        """Create a new agent."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Agent")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        name_input = StyledLineEdit(placeholder="Enter agent name")
        form_layout.addRow("Name:", name_input)
        
        type_combo = StyledComboBox()
        for agent_type in AGENT_TYPES:
            type_combo.addItem(agent_type)
        form_layout.addRow("Type:", type_combo)
        
        description_input = StyledTextEdit()
        description_input.setPlaceholderText("Enter agent description...")
        description_input.setMaximumHeight(100)
        form_layout.addRow("Description:", description_input)
        
        layout.addLayout(form_layout)
        
        buttons_layout = QHBoxLayout()
        cancel_btn = StyledButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        
        create_btn = StyledButton("Create", primary=True)
        create_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(create_btn)
        
        layout.addLayout(buttons_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Agent name is required.")
                return
            
            agent_data = {
                "name": name,
                "type": type_combo.currentText(),
                "description": description_input.toPlainText().strip(),
                "tasks_completed": 0
            }
            
            # Start agent creation thread
            self.agent_thread = AgentManagerThread("create", agent_data)
            self.agent_thread.status_signal.connect(self.handle_agent_status)
            self.agent_thread.agent_created_signal.connect(self.handle_agent_created)
            self.agent_thread.error_signal.connect(self.handle_agent_error)
            self.agent_thread.start()
    
    def handle_agent_status(self, status):
        """Handle agent status update."""
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(status)
    
    def handle_agent_created(self, agent_data):
        """Handle agent created."""
        self.agents.append(agent_data)
        self.update_agent_table()
    
    def handle_agent_error(self, error):
        """Handle agent error."""
        QMessageBox.warning(self, "Error", error)
    
    def view_agent(self, agent):
        """View agent details."""
        # Find the row with this agent
        for row in range(self.agent_table.rowCount()):
            if self.agent_table.item(row, 0).text() == agent.get('name'):
                self.agent_table.selectRow(row)
                break
    
    def start_agent(self):
        """Start the selected agent."""
        if not hasattr(self, 'current_agent') or not self.current_agent:
            return
        
        # Update agent status
        self.current_agent['status'] = 'active'
        
        # Start agent update thread
        self.agent_thread = AgentManagerThread("update", self.current_agent)
        self.agent_thread.status_signal.connect(self.handle_agent_status)
        self.agent_thread.agent_updated_signal.connect(self.handle_agent_updated)
        self.agent_thread.error_signal.connect(self.handle_agent_error)
        self.agent_thread.start()
    
    def handle_agent_updated(self, agent_data):
        """Handle agent updated."""
        # Update agent in list
        for i, agent in enumerate(self.agents):
            if agent.get('id') == agent_data.get('id'):
                self.agents[i] = agent_data
                break
        
        self.update_agent_table()
        self.update_agent_details()
    
    def stop_agent(self):
        """Stop the selected agent."""
        if not hasattr(self, 'current_agent') or not self.current_agent:
            return
        
        # Update agent status
        self.current_agent['status'] = 'idle'
        
        # Start agent update thread
        self.agent_thread = AgentManagerThread("update", self.current_agent)
        self.agent_thread.status_signal.connect(self.handle_agent_status)
        self.agent_thread.agent_updated_signal.connect(self.handle_agent_updated)
        self.agent_thread.error_signal.connect(self.handle_agent_error)
        self.agent_thread.start()
    
    def edit_agent(self):
        """Edit the selected agent."""
        if not hasattr(self, 'current_agent') or not self.current_agent:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Agent")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        name_input = StyledLineEdit()
        name_input.setText(self.current_agent.get('name', ''))
        form_layout.addRow("Name:", name_input)
        
        type_combo = StyledComboBox()
        current_type = self.current_agent.get('type', '')
        for agent_type in AGENT_TYPES:
            type_combo.addItem(agent_type)
        if current_type in AGENT_TYPES:
            type_combo.setCurrentText(current_type)
        form_layout.addRow("Type:", type_combo)
        
        description_input = StyledTextEdit()
        description_input.setText(self.current_agent.get('description', ''))
        description_input.setMaximumHeight(100)
        form_layout.addRow("Description:", description_input)
        
        layout.addLayout(form_layout)
        
        buttons_layout = QHBoxLayout()
        cancel_btn = StyledButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        
        save_btn = StyledButton("Save", primary=True)
        save_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(save_btn)
        
        layout.addLayout(buttons_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Agent name is required.")
                return
            
            # Update agent data
            self.current_agent['name'] = name
            self.current_agent['type'] = type_combo.currentText()
            self.current_agent['description'] = description_input.toPlainText().strip()
            
            # Start agent update thread
            self.agent_thread = AgentManagerThread("update", self.current_agent)
            self.agent_thread.status_signal.connect(self.handle_agent_status)
            self.agent_thread.agent_updated_signal.connect(self.handle_agent_updated)
            self.agent_thread.error_signal.connect(self.handle_agent_error)
            self.agent_thread.start()
    
    def delete_agent(self):
        """Delete the selected agent."""
        if not hasattr(self, 'current_agent') or not self.current_agent:
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete the agent '{self.current_agent.get('name')}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Start agent deletion thread
            self.agent_thread = AgentManagerThread("delete", self.current_agent)
            self.agent_thread.status_signal.connect(self.handle_agent_status)
            self.agent_thread.error_signal.connect(self.handle_agent_error)
            self.agent_thread.start()
            
            # Remove agent from list
            self.agents = [a for a in self.agents if a.get('id') != self.current_agent.get('id')]
            self.update_agent_table()
            self.clear_agent_details()

class StrategiesTab(QWidget):
    """Strategies tab for managing income strategies."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.strategies = []
        self.init_ui()
        self.load_strategies()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Income Strategies")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.create_strategy_btn = StyledButton("Create Strategy", primary=True)
        self.create_strategy_btn.clicked.connect(self.create_strategy)
        header_layout.addWidget(self.create_strategy_btn)
        
        layout.addLayout(header_layout)
        
        # Strategy grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1d1d1d;
                border: none;
            }
        """)
        
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.grid_layout.setSpacing(20)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Strategy details
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        details_layout = QVBoxLayout(details_frame)
        
        details_label = QLabel("Strategy Details")
        details_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        details_layout.addWidget(details_label)
        
        form_layout = QFormLayout()
        
        self.detail_name = QLabel("Select a strategy to view details")
        self.detail_type = QLabel("")
        self.detail_status = QLabel("")
        self.detail_income = QLabel("")
        self.detail_agents = QLabel("")
        
        form_layout.addRow("Name:", self.detail_name)
        form_layout.addRow("Type:", self.detail_type)
        form_layout.addRow("Status:", self.detail_status)
        form_layout.addRow("Income Generated:", self.detail_income)
        form_layout.addRow("Assigned Agents:", self.detail_agents)
        
        details_layout.addLayout(form_layout)
        
        # Strategy actions
        actions_layout = QHBoxLayout()
        
        self.start_strategy_btn = StyledButton("Start Strategy")
        self.start_strategy_btn.clicked.connect(self.start_strategy)
        self.start_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.start_strategy_btn)
        
        self.stop_strategy_btn = StyledButton("Stop Strategy")
        self.stop_strategy_btn.clicked.connect(self.stop_strategy)
        self.stop_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_strategy_btn)
        
        self.edit_strategy_btn = StyledButton("Edit Strategy")
        self.edit_strategy_btn.clicked.connect(self.edit_strategy)
        self.edit_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.edit_strategy_btn)
        
        self.delete_strategy_btn = StyledButton("Delete Strategy")
        self.delete_strategy_btn.clicked.connect(self.delete_strategy)
        self.delete_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_strategy_btn)
        
        details_layout.addLayout(actions_layout)
        
        layout.addWidget(details_frame)
    
    def load_strategies(self):
        """Load strategies from files."""
        try:
            self.strategies = []
            
            # Check if strategies directory exists
            if not os.path.exists(STRATEGIES_DIR):
                os.makedirs(STRATEGIES_DIR)
            
            # Load strategy files
            strategy_files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.json')]
            
            if not strategy_files:
                # Create sample strategies
                self.create_sample_strategies()
                strategy_files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.json')]
            
            for strategy_file in strategy_files:
                with open(os.path.join(STRATEGIES_DIR, strategy_file), 'r') as f:
                    strategy_data = json.load(f)
                    self.strategies.append(strategy_data)
            
            self.update_strategy_grid()
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Failed to load strategies: {str(e)}")
    
    def create_sample_strategies(self):
        """Create sample strategies."""
        for i, strategy_type in enumerate(INCOME_STRATEGIES[:5]):
            strategy_id = f"strategy_{int(time.time())}_{i}"
            strategy_data = {
                "id": strategy_id,
                "name": f"{strategy_type} Strategy",
                "type": strategy_type,
                "description": f"Sample strategy for {strategy_type}",
                "status": "idle",
                "income_generated": 0.0,
                "assigned_agents": 0,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=2)
    
    def update_strategy_grid(self):
        """Update strategy grid with loaded strategies."""
        # Clear existing widgets
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add strategy cards
        row, col = 0, 0
        max_cols = 3
        
        for strategy in self.strategies:
            card = self.create_strategy_card(strategy)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def create_strategy_card(self, strategy):
        """Create a strategy card widget."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 8px;
                min-height: 200px;
                max-height: 200px;
            }
        """)
        
        layout = QVBoxLayout(card)
        
        # Header with name and status
        header_layout = QHBoxLayout()
        
        name_label = QLabel(strategy.get('name', 'Unknown'))
        name_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        status = strategy.get('status', 'idle')
        status_label = QLabel(status)
        if status == 'active':
            status_label.setStyleSheet("color: #2ecc71;")
        elif status == 'idle':
            status_label.setStyleSheet("color: #f39c12;")
        else:
            status_label.setStyleSheet("color: #e74c3c;")
        header_layout.addWidget(status_label)
        
        layout.addLayout(header_layout)
        
        # Type
        type_label = QLabel(strategy.get('type', 'Unknown'))
        type_label.setStyleSheet("color: #9d9d9d;")
        layout.addWidget(type_label)
        
        # Description
        description = strategy.get('description', '')
        if len(description) > 100:
            description = description[:97] + '...'
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        
        layout.addStretch()
        
        # Income and agents
        info_layout = QHBoxLayout()
        
        income_label = QLabel(f"${strategy.get('income_generated', 0):.2f}")
        income_label.setStyleSheet("color: #2ecc71;")
        info_layout.addWidget(income_label)
        
        info_layout.addStretch()
        
        agents_label = QLabel(f"{strategy.get('assigned_agents', 0)} agents")
        info_layout.addWidget(agents_label)
        
        layout.addLayout(info_layout)
        
        # View button
        view_btn = StyledButton("View Details")
        view_btn.clicked.connect(lambda: self.view_strategy(strategy))
        layout.addWidget(view_btn)
        
        return card
    
    def view_strategy(self, strategy):
        """View strategy details."""
        # Update details panel
        self.detail_name.setText(strategy.get('name', 'Unknown'))
        self.detail_type.setText(strategy.get('type', 'Unknown'))
        self.detail_status.setText(strategy.get('status', 'idle'))
        self.detail_income.setText(f"${strategy.get('income_generated', 0):.2f}")
        self.detail_agents.setText(f"{strategy.get('assigned_agents', 0)}")
        
        # Enable/disable buttons based on status
        status = strategy.get('status', 'idle')
        self.start_strategy_btn.setEnabled(status != 'active')
        self.stop_strategy_btn.setEnabled(status == 'active')
        self.edit_strategy_btn.setEnabled(True)
        self.delete_strategy_btn.setEnabled(True)
        
        # Store current strategy
        self.current_strategy = strategy
    
    def create_strategy(self):
        """Create a new strategy."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Strategy")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        name_input = StyledLineEdit(placeholder="Enter strategy name")
        form_layout.addRow("Name:", name_input)
        
        type_combo = StyledComboBox()
        for strategy_type in INCOME_STRATEGIES:
            type_combo.addItem(strategy_type)
        form_layout.addRow("Type:", type_combo)
        
        description_input = StyledTextEdit()
        description_input.setPlaceholderText("Enter strategy description...")
        description_input.setMaximumHeight(100)
        form_layout.addRow("Description:", description_input)
        
        agents_spin = QSpinBox()
        agents_spin.setMinimum(0)
        agents_spin.setMaximum(100)
        agents_spin.setValue(1)
        agents_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                min-height: 36px;
            }
        """)
        form_layout.addRow("Assigned Agents:", agents_spin)
        
        layout.addLayout(form_layout)
        
        buttons_layout = QHBoxLayout()
        cancel_btn = StyledButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        
        create_btn = StyledButton("Create", primary=True)
        create_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(create_btn)
        
        layout.addLayout(buttons_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Strategy name is required.")
                return
            
            # Create strategy data
            strategy_id = f"strategy_{int(time.time())}_{random.randint(1000, 9999)}"
            strategy_data = {
                "id": strategy_id,
                "name": name,
                "type": type_combo.currentText(),
                "description": description_input.toPlainText().strip(),
                "status": "idle",
                "income_generated": 0.0,
                "assigned_agents": agents_spin.value(),
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Save strategy
            strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=2)
            
            # Add to list and update grid
            self.strategies.append(strategy_data)
            self.update_strategy_grid()
            
            # Show success message
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Strategy '{name}' created successfully")
    
    def start_strategy(self):
        """Start the selected strategy."""
        if not hasattr(self, 'current_strategy') or not self.current_strategy:
            return
        
        # Update strategy status
        self.current_strategy['status'] = 'active'
        
        # Save strategy
        strategy_id = self.current_strategy.get('id')
        strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
        with open(strategy_file, 'w') as f:
            json.dump(self.current_strategy, f, indent=2)
        
        # Update UI
        self.update_strategy_grid()
        self.view_strategy(self.current_strategy)
        
        # Show success message
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(f"Strategy '{self.current_strategy.get('name')}' started successfully")
    
    def stop_strategy(self):
        """Stop the selected strategy."""
        if not hasattr(self, 'current_strategy') or not self.current_strategy:
            return
        
        # Update strategy status
        self.current_strategy['status'] = 'idle'
        
        # Save strategy
        strategy_id = self.current_strategy.get('id')
        strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
        with open(strategy_file, 'w') as f:
            json.dump(self.current_strategy, f, indent=2)
        
        # Update UI
        self.update_strategy_grid()
        self.view_strategy(self.current_strategy)
        
        # Show success message
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(f"Strategy '{self.current_strategy.get('name')}' stopped successfully")
    
    def edit_strategy(self):
        """Edit the selected strategy."""
        if not hasattr(self, 'current_strategy') or not self.current_strategy:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Strategy")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        name_input = StyledLineEdit()
        name_input.setText(self.current_strategy.get('name', ''))
        form_layout.addRow("Name:", name_input)
        
        type_combo = StyledComboBox()
        current_type = self.current_strategy.get('type', '')
        for strategy_type in INCOME_STRATEGIES:
            type_combo.addItem(strategy_type)
        if current_type in INCOME_STRATEGIES:
            type_combo.setCurrentText(current_type)
        form_layout.addRow("Type:", type_combo)
        
        description_input = StyledTextEdit()
        description_input.setText(self.current_strategy.get('description', ''))
        description_input.setMaximumHeight(100)
        form_layout.addRow("Description:", description_input)
        
        agents_spin = QSpinBox()
        agents_spin.setMinimum(0)
        agents_spin.setMaximum(100)
        agents_spin.setValue(self.current_strategy.get('assigned_agents', 1))
        agents_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                min-height: 36px;
            }
        """)
        form_layout.addRow("Assigned Agents:", agents_spin)
        
        layout.addLayout(form_layout)
        
        buttons_layout = QHBoxLayout()
        cancel_btn = StyledButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        
        save_btn = StyledButton("Save", primary=True)
        save_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(save_btn)
        
        layout.addLayout(buttons_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Strategy name is required.")
                return
            
            # Update strategy data
            self.current_strategy['name'] = name
            self.current_strategy['type'] = type_combo.currentText()
            self.current_strategy['description'] = description_input.toPlainText().strip()
            self.current_strategy['assigned_agents'] = agents_spin.value()
            
            # Save strategy
            strategy_id = self.current_strategy.get('id')
            strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
            with open(strategy_file, 'w') as f:
                json.dump(self.current_strategy, f, indent=2)
            
            # Update UI
            self.update_strategy_grid()
            self.view_strategy(self.current_strategy)
            
            # Show success message
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Strategy '{name}' updated successfully")
    
    def delete_strategy(self):
        """Delete the selected strategy."""
        if not hasattr(self, 'current_strategy') or not self.current_strategy:
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete the strategy '{self.current_strategy.get('name')}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Delete strategy file
            strategy_id = self.current_strategy.get('id')
            strategy_file = os.path.join(STRATEGIES_DIR, f"{strategy_id}.json")
            if os.path.exists(strategy_file):
                os.remove(strategy_file)
            
            # Remove from list and update grid
            self.strategies = [s for s in self.strategies if s.get('id') != strategy_id]
            self.update_strategy_grid()
            
            # Clear details
            self.detail_name.setText("Select a strategy to view details")
            self.detail_type.setText("")
            self.detail_status.setText("")
            self.detail_income.setText("")
            self.detail_agents.setText("")
            
            self.start_strategy_btn.setEnabled(False)
            self.stop_strategy_btn.setEnabled(False)
            self.edit_strategy_btn.setEnabled(False)
            self.delete_strategy_btn.setEnabled(False)
            
            self.current_strategy = None
            
            # Show success message
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status("Strategy deleted successfully")

class AnalyticsTab(QWidget):
    """Analytics tab for displaying metrics and charts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.load_data()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Analytics Dashboard")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        refresh_btn = StyledButton("Refresh")
        refresh_btn.clicked.connect(self.load_data)
        header_layout.addWidget(refresh_btn)
        
        export_btn = StyledButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        header_layout.addWidget(export_btn)
        
        layout.addLayout(header_layout)
        
        # Key metrics
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.Shape.StyledPanel)
        metrics_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        metrics_layout = QHBoxLayout(metrics_frame)
        
        # Total income
        income_widget = QWidget()
        income_layout = QVBoxLayout(income_widget)
        income_label = QLabel("Total Income")
        income_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.income_value = QLabel("$0.00")
        self.income_value.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.income_value.setStyleSheet("color: #2ecc71;")
        income_layout.addWidget(income_label)
        income_layout.addWidget(self.income_value)
        metrics_layout.addWidget(income_widget)
        
        # Active agents
        agents_widget = QWidget()
        agents_layout = QVBoxLayout(agents_widget)
        agents_label = QLabel("Active Agents")
        agents_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.agents_value = QLabel("0")
        self.agents_value.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        agents_layout.addWidget(agents_label)
        agents_layout.addWidget(self.agents_value)
        metrics_layout.addWidget(agents_widget)
        
        # Active strategies
        strategies_widget = QWidget()
        strategies_layout = QVBoxLayout(strategies_widget)
        strategies_label = QLabel("Active Strategies")
        strategies_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.strategies_value = QLabel("0")
        self.strategies_value.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        strategies_layout.addWidget(strategies_label)
        strategies_layout.addWidget(self.strategies_value)
        metrics_layout.addWidget(strategies_widget)
        
        # Tasks completed
        tasks_widget = QWidget()
        tasks_layout = QVBoxLayout(tasks_widget)
        tasks_label = QLabel("Tasks Completed")
        tasks_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.tasks_value = QLabel("0")
        self.tasks_value.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        tasks_layout.addWidget(tasks_label)
        tasks_layout.addWidget(self.tasks_value)
        metrics_layout.addWidget(tasks_widget)
        
        layout.addWidget(metrics_frame)
        
        # Charts
        charts_layout = QHBoxLayout()
        
        # Income chart
        income_chart_frame = QFrame()
        income_chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
        income_chart_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        income_chart_layout = QVBoxLayout(income_chart_frame)
        
        income_chart_label = QLabel("Income Over Time")
        income_chart_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        income_chart_layout.addWidget(income_chart_label)
        
        self.income_chart_view = QChartView()
        self.income_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.income_chart_view.setStyleSheet("background-color: transparent;")
        income_chart_layout.addWidget(self.income_chart_view)
        
        charts_layout.addWidget(income_chart_frame)
        
        # Strategy distribution chart
        strategy_chart_frame = QFrame()
        strategy_chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
        strategy_chart_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        strategy_chart_layout = QVBoxLayout(strategy_chart_frame)
        
        strategy_chart_label = QLabel("Income by Strategy")
        strategy_chart_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        strategy_chart_layout.addWidget(strategy_chart_label)
        
        self.strategy_chart_view = QChartView()
        self.strategy_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.strategy_chart_view.setStyleSheet("background-color: transparent;")
        strategy_chart_layout.addWidget(self.strategy_chart_view)
        
        charts_layout.addWidget(strategy_chart_frame)
        
        layout.addLayout(charts_layout)
        
        # Agent performance
        performance_frame = QFrame()
        performance_frame.setFrameShape(QFrame.Shape.StyledPanel)
        performance_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        performance_layout = QVBoxLayout(performance_frame)
        
        performance_label = QLabel("Agent Performance")
        performance_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        performance_layout.addWidget(performance_label)
        
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(5)
        self.performance_table.setHorizontalHeaderLabels(["Agent", "Type", "Tasks", "Income", "Efficiency"])
        self.performance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.performance_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.performance_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.performance_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.performance_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.performance_table.setStyleSheet("""
            QTableWidget {
                background-color: #1d1d1d;
                color: white;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: white;
                padding: 5px;
                border: 1px solid #3d3d3d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3d3d3d;
            }
        """)
        performance_layout.addWidget(self.performance_table)
        
        layout.addWidget(performance_frame)
    
    def load_data(self):
        """Load analytics data."""
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status("Loading analytics data...")
        
        try:
            # Load income data
            total_income = 0.0
            income_data = []
            income_file = os.path.join(DATA_DIR, "income.json")
            if os.path.exists(income_file):
                with open(income_file, 'r') as f:
                    income_json = json.load(f)
                    total_income = income_json.get('total', 0.0)
                    income_data = income_json.get('history', [])
            
            # If no income data, create sample data
            if not income_data:
                income_data = self.generate_sample_income_data()
                # Save sample data
                with open(income_file, 'w') as f:
                    json.dump({
                        'total': sum(item['amount'] for item in income_data),
                        'history': income_data
                    }, f, indent=2)
                total_income = sum(item['amount'] for item in income_data)
            
            # Update income value
            self.income_value.setText(f"${total_income:.2f}")
            
            # Load agents data
            active_agents = 0
            agents_data = []
            agent_files = [f for f in os.listdir(AGENTS_DIR) if f.endswith('.json')]
            
            for agent_file in agent_files:
                with open(os.path.join(AGENTS_DIR, agent_file), 'r') as f:
                    agent_data = json.load(f)
                    agents_data.append(agent_data)
                    if agent_data.get('status') == 'active':
                        active_agents += 1
            
            # Update agents value
            self.agents_value.setText(str(active_agents))
            
            # Load strategies data
            active_strategies = 0
            strategies_data = []
            strategy_files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.json')]
            
            for strategy_file in strategy_files:
                with open(os.path.join(STRATEGIES_DIR, strategy_file), 'r') as f:
                    strategy_data = json.load(f)
                    strategies_data.append(strategy_data)
                    if strategy_data.get('status') == 'active':
                        active_strategies += 1
            
            # Update strategies value
            self.strategies_value.setText(str(active_strategies))
            
            # Calculate total tasks
            total_tasks = sum(agent.get('tasks_completed', 0) for agent in agents_data)
            self.tasks_value.setText(str(total_tasks))
            
            # Create charts
            self.create_income_chart(income_data)
            self.create_strategy_chart(strategies_data)
            
            # Update performance table
            self.update_performance_table(agents_data)
            
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status("Analytics data loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Failed to load analytics data: {str(e)}")
    
    def generate_sample_income_data(self):
        """Generate sample income data for demonstration."""
        income_data = []
        
        # Generate data for the last 30 days
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
        current_date = start_date
        
        total = 0.0
        while current_date <= end_date:
            # Generate random amount between $0 and $100
            amount = round(random.uniform(0, 100), 2)
            total += amount
            
            income_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'amount': amount,
                'cumulative': total,
                'source': random.choice(INCOME_STRATEGIES)
            })
            
            current_date += datetime.timedelta(days=1)
        
        return income_data
    
    def create_income_chart(self, income_data):
        """Create income over time chart."""
        # Create line series for income
        series = QLineSeries()
        
        # Add data points
        for i, item in enumerate(income_data):
            date_obj = datetime.datetime.strptime(item['date'], '%Y-%m-%d')
            x = i  # Use index as x-coordinate
            y = item['cumulative']  # Use cumulative income as y-coordinate
            series.append(x, y)
        
        # Create chart
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Income Over Time")
        chart.setTitleBrush(QColor("white"))
        chart.setBackgroundBrush(QBrush(QColor("#2d2d2d")))
        
        # Create axes
        axis_x = QValueAxis()
        axis_x.setRange(0, len(income_data) - 1)
        axis_x.setLabelFormat("%d")
        axis_x.setLabelsColor(QColor("white"))
        axis_x.setTitleText("Days")
        axis_x.setTitleBrush(QColor("white"))
        
        axis_y = QValueAxis()
        max_income = max(item['cumulative'] for item in income_data) if income_data else 100
        axis_y.setRange(0, max_income * 1.1)  # Add 10% margin
        axis_y.setLabelFormat("$%.2f")
        axis_y.setLabelsColor(QColor("white"))
        axis_y.setTitleText("Cumulative Income")
        axis_y.setTitleBrush(QColor("white"))
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
        # Set pen for series
        pen = QPen(QColor("#2ecc71"))
        pen.setWidth(2)
        series.setPen(pen)
        
        # Set chart in view
        self.income_chart_view.setChart(chart)
    
    def create_strategy_chart(self, strategies_data):
        """Create income by strategy chart."""
        # Create pie series
        series = QPieSeries()
        
        # Group by strategy type
        strategy_income = {}
        for strategy in strategies_data:
            strategy_type = strategy.get('type', 'Unknown')
            income = strategy.get('income_generated', 0.0)
            
            if strategy_type in strategy_income:
                strategy_income[strategy_type] += income
            else:
                strategy_income[strategy_type] = income
        
        # If no data, create sample data
        if not strategy_income:
            for strategy_type in INCOME_STRATEGIES[:5]:
                strategy_income[strategy_type] = random.uniform(50, 500)
        
        # Add slices to series
        for strategy_type, income in strategy_income.items():
            slice = series.append(strategy_type, income)
            slice.setLabelVisible(True)
            slice.setLabelColor(QColor("white"))
            slice.setLabelPosition(QPieSlice.LabelPosition.LabelOutside)
            
            # Set random color for slice
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            slice.setBrush(QColor(r, g, b))
        
        # Create chart
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Income by Strategy")
        chart.setTitleBrush(QColor("white"))
        chart.setBackgroundBrush(QBrush(QColor("#2d2d2d")))
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        chart.legend().setLabelColor(QColor("white"))
        
        # Set chart in view
        self.strategy_chart_view.setChart(chart)
    
    def update_performance_table(self, agents_data):
        """Update agent performance table."""
        self.performance_table.setRowCount(0)
        
        for agent in agents_data:
            row = self.performance_table.rowCount()
            self.performance_table.insertRow(row)
            
            # Name
            name_item = QTableWidgetItem(agent.get('name', 'Unknown'))
            self.performance_table.setItem(row, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(agent.get('type', 'Unknown'))
            self.performance_table.setItem(row, 1, type_item)
            
            # Tasks
            tasks = agent.get('tasks_completed', 0)
            tasks_item = QTableWidgetItem(str(tasks))
            self.performance_table.setItem(row, 2, tasks_item)
            
            # Income
            income = agent.get('income_generated', 0.0)
            income_item = QTableWidgetItem(f"${income:.2f}")
            self.performance_table.setItem(row, 3, income_item)
            
            # Efficiency
            if tasks > 0:
                efficiency = income / tasks
            else:
                efficiency = 0.0
            efficiency_item = QTableWidgetItem(f"${efficiency:.2f}/task")
            self.performance_table.setItem(row, 4, efficiency_item)
    
    def export_data(self):
        """Export analytics data to CSV."""
        try:
            # Choose export directory
            export_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", os.path.expanduser("~"),
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not export_dir:
                return
            
            # Export income data
            income_file = os.path.join(DATA_DIR, "income.json")
            if os.path.exists(income_file):
                with open(income_file, 'r') as f:
                    income_json = json.load(f)
                    income_data = income_json.get('history', [])
                
                if income_data:
                    csv_path = os.path.join(export_dir, "income_data.csv")
                    with open(csv_path, 'w') as f:
                        f.write("Date,Amount,Cumulative,Source\n")
                        for item in income_data:
                            f.write(f"{item.get('date', '')},{item.get('amount', 0)},{item.get('cumulative', 0)},{item.get('source', '')}\n")
            
            # Export agent data
            agents_data = []
            agent_files = [f for f in os.listdir(AGENTS_DIR) if f.endswith('.json')]
            
            for agent_file in agent_files:
                with open(os.path.join(AGENTS_DIR, agent_file), 'r') as f:
                    agent_data = json.load(f)
                    agents_data.append(agent_data)
            
            if agents_data:
                csv_path = os.path.join(export_dir, "agents_data.csv")
                with open(csv_path, 'w') as f:
                    f.write("Name,Type,Status,Tasks,Income\n")
                    for agent in agents_data:
                        f.write(f"{agent.get('name', '')},{agent.get('type', '')},{agent.get('status', '')},{agent.get('tasks_completed', 0)},{agent.get('income_generated', 0)}\n")
            
            # Export strategy data
            strategies_data = []
            strategy_files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.json')]
            
            for strategy_file in strategy_files:
                with open(os.path.join(STRATEGIES_DIR, strategy_file), 'r') as f:
                    strategy_data = json.load(f)
                    strategies_data.append(strategy_data)
            
            if strategies_data:
                csv_path = os.path.join(export_dir, "strategies_data.csv")
                with open(csv_path, 'w') as f:
                    f.write("Name,Type,Status,Income,Agents\n")
                    for strategy in strategies_data:
                        f.write(f"{strategy.get('name', '')},{strategy.get('type', '')},{strategy.get('status', '')},{strategy.get('income_generated', 0)},{strategy.get('assigned_agents', 0)}\n")
            
            # Show success message
            QMessageBox.information(self, "Export Complete", f"Analytics data exported to {export_dir}")
            
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status("Analytics data exported successfully")
        
        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")
            QMessageBox.warning(self, "Export Failed", f"Failed to export analytics data: {str(e)}")
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status(f"Failed to export analytics data: {str(e)}")

class SettingsTab(QWidget):
    """Settings tab for configuring application settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.config = load_config()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        title_label = QLabel("Settings")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Settings tabs
        settings_tabs = QTabWidget()
        settings_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #1d1d1d;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3d3d3d;
            }
        """)
        
        # API Settings
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        
        api_group = QGroupBox("OpenAI API Settings")
        api_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        api_group_layout = QVBoxLayout(api_group)
        
        # Session token instructions
        token_instructions = QLabel(
            "To use GPT-4 with openai-unofficial, you need to provide your OpenAI session token.\n\n"
            "How to get your session token:\n"
            "1. Go to https://chat.openai.com and log in\n"
            "2. Open developer tools by pressing F12 or right-click > Inspect\n"
            "3. Go to Application > Storage > Cookies\n"
            "4. Find the cookie named '__Secure-next-auth.session-token'\n"
            "5. Copy the value and paste it below\n\n"
            "Note: This token is sensitive and should be kept secure. It will be stored securely in your system keyring."
        )
        token_instructions.setWordWrap(True)
        token_instructions.setStyleSheet("color: #9d9d9d;")
        api_group_layout.addWidget(token_instructions)
        
        # Session token input
        token_layout = QHBoxLayout()
        token_label = QLabel("Session Token:")
        token_layout.addWidget(token_label)
        
        self.token_input = StyledLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setPlaceholderText("Paste your OpenAI session token here")
        
        # Load token if exists
        token = get_session_token()
        if token:
            self.token_input.setText(token)
            self.token_input.setPlaceholderText("Token is set (hidden for security)")
        
        token_layout.addWidget(self.token_input)
        
        self.save_token_btn = StyledButton("Save Token", primary=True)
        self.save_token_btn.clicked.connect(self.save_token)
        token_layout.addWidget(self.save_token_btn)
        
        api_group_layout.addLayout(token_layout)
        
        # Test connection button
        self.test_connection_btn = StyledButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_connection)
        api_group_layout.addWidget(self.test_connection_btn)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("GPT Model:")
        model_layout.addWidget(model_label)
        
        self.model_combo = StyledComboBox()
        self.model_combo.addItem("gpt-4")
        self.model_combo.addItem("gpt-4-browsing")
        self.model_combo.addItem("gpt-4-plugins")
        
        # Set current model from config
        current_model = self.config.get("ai_providers", {}).get("model", "gpt-4")
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        model_layout.addWidget(self.model_combo)
        
        api_group_layout.addLayout(model_layout)
        
        api_layout.addWidget(api_group)
        
        # Alternative AI providers
        alt_providers_group = QGroupBox("Alternative AI Providers")
        alt_providers_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        alt_providers_layout = QVBoxLayout(alt_providers_group)
        
        alt_providers_label = QLabel(
            "Configure alternative AI providers that can be used if the primary provider is unavailable."
        )
        alt_providers_label.setWordWrap(True)
        alt_providers_layout.addWidget(alt_providers_label)
        
        # Google AI
        google_check = QCheckBox("Enable Google Gemini")
        google_check.setChecked(
            "google" in self.config.get("ai_providers", {}).get("fallbacks", [])
        )
        alt_providers_layout.addWidget(google_check)
        
        # Anthropic
        anthropic_check = QCheckBox("Enable Anthropic Claude")
        anthropic_check.setChecked(
            "anthropic" in self.config.get("ai_providers", {}).get("fallbacks", [])
        )
        alt_providers_layout.addWidget(anthropic_check)
        
        # HuggingFace
        huggingface_check = QCheckBox("Enable HuggingFace")
        huggingface_check.setChecked(
            "huggingface" in self.config.get("ai_providers", {}).get("fallbacks", [])
        )
        alt_providers_layout.addWidget(huggingface_check)
        
        api_layout.addWidget(alt_providers_group)
        api_layout.addStretch()
        
        # Application Settings
        app_tab = QWidget()
        app_layout = QVBoxLayout(app_tab)
        
        # Theme settings
        theme_group = QGroupBox("Theme Settings")
        theme_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        theme_layout = QVBoxLayout(theme_group)
        
        theme_label = QLabel("Select application theme:")
        theme_layout.addWidget(theme_label)
        
        self.theme_combo = StyledComboBox()
        self.theme_combo.addItem("Dark")
        self.theme_combo.addItem("Light")
        
        # Set current theme from config
        current_theme = self.config.get("theme", "dark")
        index = self.theme_combo.findText(current_theme.capitalize())
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        theme_layout.addWidget(self.theme_combo)
        
        app_layout.addWidget(theme_group)
        
        # Agent settings
        agent_group = QGroupBox("Agent Settings")
        agent_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        agent_layout = QVBoxLayout(agent_group)
        
        agent_count_layout = QHBoxLayout()
        agent_count_label = QLabel("Maximum Agent Count:")
        agent_count_layout.addWidget(agent_count_label)
        
        self.agent_count_spin = QSpinBox()
        self.agent_count_spin.setMinimum(1)
        self.agent_count_spin.setMaximum(10000)
        self.agent_count_spin.setValue(
            self.config.get("agents", {}).get("count", 10000)
        )
        self.agent_count_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                min-height: 36px;
            }
        """)
        agent_count_layout.addWidget(self.agent_count_spin)
        
        agent_layout.addLayout(agent_count_layout)
        
        app_layout.addWidget(agent_group)
        app_layout.addStretch()
        
        # Advanced Settings
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # Analytics settings
        analytics_group = QGroupBox("Analytics Settings")
        analytics_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        analytics_layout = QVBoxLayout(analytics_group)
        
        refresh_layout = QHBoxLayout()
        refresh_label = QLabel("Analytics Refresh Interval (seconds):")
        refresh_layout.addWidget(refresh_label)
        
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setMinimum(10)
        self.refresh_spin.setMaximum(3600)
        self.refresh_spin.setValue(
            self.config.get("analytics", {}).get("refresh_interval", 60)
        )
        self.refresh_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1d1d1d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                min-height: 36px;
            }
        """)
        refresh_layout.addWidget(self.refresh_spin)
        
        analytics_layout.addLayout(refresh_layout)
        
        advanced_layout.addWidget(analytics_group)
        
        # Logging settings
        logging_group = QGroupBox("Logging Settings")
        logging_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
        """)
        logging_layout = QVBoxLayout(logging_group)
        
        log_level_layout = QHBoxLayout()
        log_level_label = QLabel("Log Level:")
        log_level_layout.addWidget(log_level_label)
        
        self.log_level_combo = StyledComboBox()
        self.log_level_combo.addItem("DEBUG")
        self.log_level_combo.addItem("INFO")
        self.log_level_combo.addItem("WARNING")
        self.log_level_combo.addItem("ERROR")
        
        # Set current log level from config
        current_log_level = self.config.get("logging", {}).get("level", "info").upper()
        index = self.log_level_combo.findText(current_log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        
        log_level_layout.addWidget(self.log_level_combo)
        
        logging_layout.addLayout(log_level_layout)
        
        # View logs button
        view_logs_btn = StyledButton("View Logs")
        view_logs_btn.clicked.connect(self.view_logs)
        logging_layout.addWidget(view_logs_btn)
        
        advanced_layout.addWidget(logging_group)
        advanced_layout.addStretch()
        
        # Add tabs
        settings_tabs.addTab(api_tab, "API Settings")
        settings_tabs.addTab(app_tab, "Application Settings")
        settings_tabs.addTab(advanced_tab, "Advanced Settings")
        
        layout.addWidget(settings_tabs)
        
        # Save settings button
        save_btn = StyledButton("Save All Settings", primary=True)
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        # Store references
        self.google_check = google_check
        self.anthropic_check = anthropic_check
        self.huggingface_check = huggingface_check
    
    def save_token(self):
        """Save session token."""
        token = self.token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "Error", "Please enter a session token.")
            return
        
        if save_session_token(token):
            QMessageBox.information(self, "Success", "Session token saved successfully.")
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.set_status("Session token saved successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to save session token.")
    
    def test_connection(self):
        """Test connection to OpenAI."""
        token = get_session_token()
        if not token:
            QMessageBox.warning(self, "Error", "No session token found. Please save a token first.")
            return
        
        model = self.model_combo.currentText()
        
        # Show testing message
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status("Testing connection to OpenAI...")
        
        # Create thread for testing
        self.test_thread = GPT4Thread(token, "Hello, are you working?", model)
        self.test_thread.response_signal.connect(self.handle_test_response)
        self.test_thread.error_signal.connect(self.handle_test_error)
        self.test_thread.status_signal.connect(self.handle_test_status)
        self.test_thread.start()
    
    def handle_test_response(self, response):
        """Handle successful test response."""
        QMessageBox.information(self, "Connection Successful", 
                               f"Successfully connected to OpenAI!\n\nResponse: {response[:100]}...")
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status("Connection to OpenAI successful")
            self.parent.status_bar.set_connection(True)
    
    def handle_test_error(self, error):
        """Handle test error."""
        QMessageBox.warning(self, "Connection Failed", f"Failed to connect to OpenAI: {error}")
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(f"Connection to OpenAI failed: {error}")
            self.parent.status_bar.set_connection(False)
    
    def handle_test_status(self, status):
        """Handle test status update."""
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.set_status(status)
    
    def save_settings(self):
        """Save all settings."""
        try:
            # Update config
            self.config["ai_providers"] = {
                "primary": "openai-unofficial",
                "model": self.model_combo.currentText(),
                "fallbacks": []
            }
            
            # Add fallbacks
            if self.google_check.isChecked():
                self.config["ai_providers"]["fallbacks"].append("google")
            if self.anthropic_check.isChecked():
                self.config["ai_providers"]["fallbacks"].append("anthropic")
            if self.huggingface_check.isChecked():
                self.config["ai_providers"]["fallbacks"].append("huggingface")
            
            # Update theme
            self.config["theme"] = self.theme_combo.currentText().lower()
            
            # Update agent settings
            if "agents" not in self.config:
                self.config["agents"] = {}
            self.config["agents"]["count"] = self.agent_count_spin.value()
            
            # Update analytics settings
            if "analytics" not in self.config:
                self.config["analytics"] = {}
            self.config["analytics"]["refresh_interval"] = self.refresh_spin.value()
            
            # Update logging settings
            if "logging" not in self.config:
                self.config["logging"] = {}
            self.config["logging"]["level"] = self.log_level_combo.currentText().lower()
            
            # Save config
            if save_config(self.config):
                QMessageBox.information(self, "Success", "Settings saved successfully.")
                if hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.set_status("Settings saved successfully")
            else:
                QMessageBox.warning(self, "Error", "Failed to save settings.")
        
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save settings: {str(e)}")
    
    def view_logs(self):
        """View application logs."""
        log_file = os.path.join(LOGS_DIR, "app.log")
        
        if not os.path.exists(log_file):
            QMessageBox.information(self, "No Logs", "No log file found.")
            return
        
        try:
            # Open log file with default text editor
            if sys.platform == 'darwin':  # macOS
                os.system(f"open {log_file}")
            elif sys.platform == 'win32':  # Windows
                os.startfile(log_file)
            else:  # Linux
                os.system(f"xdg-open {log_file}")
        except Exception as e:
            logger.error(f"Failed to open log file: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open log file: {str(e)}")

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        # Set window properties
        self.setWindowTitle(f"{APP_NAME} - v{APP_VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #1d1d1d;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3d3d3d;
            }
        """)
        
        # Create tabs
        dashboard_tab = DashboardTab(self)
        agents_tab = AgentsTab(self)
        strategies_tab = StrategiesTab(self)
        analytics_tab = AnalyticsTab(self)
        settings_tab = SettingsTab(self)
        
        # Add tabs
        self.tabs.addTab(dashboard_tab, "Dashboard")
        self.tabs.addTab(agents_tab, "Agents")
        self.tabs.addTab(strategies_tab, "Strategies")
        self.tabs.addTab(analytics_tab, "Analytics")
        self.tabs.addTab(settings_tab, "Settings")
        
        main_layout.addWidget(self.tabs)
        
        # Create status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)
        
        # Check session token
        self.check_session_token()
        
        # Show ready message
        self.status_bar.set_status("Ready")
    
    def set_dark_theme(self):
        """Set dark theme for the application."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)
    
    def check_session_token(self):
        """Check if session token is set."""
        token = get_session_token()
        if not token:
            # Show warning
            QMessageBox.warning(
                self, "Session Token Required",
                "No OpenAI session token found. Please configure it in Settings tab.\n\n"
                "You need a session token to use GPT-4 with openai-unofficial."
            )
            
            # Switch to Settings tab
            self.tabs.setCurrentIndex(4)
            
            # Update connection status
            self.status_bar.set_connection(False)
        else:
            # Update connection status
            self.status_bar.set_connection(True)

def main():
    """Main application entry point."""
    try:
        # Create application
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Use Fusion style for better dark theme support
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start application event loop
        sys.exit(app.exec())
    except Exception as e:
        logger.exception(f"Application failed to start: {e}")
        QMessageBox.critical(None, "Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()
