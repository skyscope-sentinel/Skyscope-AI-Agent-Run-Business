import os
import sys
import json
import time
import logging
import threading
import subprocess
import webbrowser
import platform
import signal
import socket
import shutil
import tempfile
import uuid
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import multiprocessing as mp

# GUI frameworks
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webview
import eel
from PIL import Image, ImageTk

# For packaging
import pkg_resources
import importlib.metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gui_application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gui_application")

# Constants
APP_NAME = "Skyscope Sentinel Intelligence"
APP_VERSION = "1.0.0"
APP_ICON = "static/images/icon.png"
APP_LOGO = "static/images/logo.png"
APP_THEME = "dark_glass"  # Default theme
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"
WEB_PORT = 8502
CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "gui_config.json"
TEMP_DIR = Path(tempfile.gettempdir()) / "skyscope_sentinel"
STATIC_DIR = Path("static")
UI_DIR = STATIC_DIR / "ui"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

class GUIType(Enum):
    """Types of GUI interfaces."""
    NATIVE = "native"  # Tkinter
    WEBVIEW = "webview"  # pywebview
    WEB = "web"  # Eel or direct browser
    HEADLESS = "headless"  # No GUI, API only

class AppMode(Enum):
    """Application operating modes."""
    NORMAL = "normal"  # Full functionality
    LITE = "lite"  # Reduced functionality for low-resource systems
    OFFLINE = "offline"  # No internet connectivity required
    SERVER = "server"  # Run as server only
    CLIENT = "client"  # Connect to remote server

@dataclass
class GUIConfig:
    """Configuration for the GUI application."""
    gui_type: GUIType = GUIType.WEBVIEW
    app_mode: AppMode = AppMode.NORMAL
    theme: str = APP_THEME
    streamlit_port: int = STREAMLIT_PORT
    web_port: int = WEB_PORT
    window_width: int = 1280
    window_height: int = 800
    window_resizable: bool = True
    window_fullscreen: bool = False
    window_position: Tuple[int, int] = (-1, -1)  # -1, -1 means center
    auto_start_browser: bool = True
    auto_update_check: bool = True
    show_splash_screen: bool = True
    splash_duration: float = 2.0  # seconds
    enable_tray_icon: bool = True
    minimize_to_tray: bool = True
    enable_notifications: bool = True
    enable_animations: bool = True
    log_level: str = "INFO"
    custom_css_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gui_type": self.gui_type.value,
            "app_mode": self.app_mode.value,
            "theme": self.theme,
            "streamlit_port": self.streamlit_port,
            "web_port": self.web_port,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "window_resizable": self.window_resizable,
            "window_fullscreen": self.window_fullscreen,
            "window_position": self.window_position,
            "auto_start_browser": self.auto_start_browser,
            "auto_update_check": self.auto_update_check,
            "show_splash_screen": self.show_splash_screen,
            "splash_duration": self.splash_duration,
            "enable_tray_icon": self.enable_tray_icon,
            "minimize_to_tray": self.minimize_to_tray,
            "enable_notifications": self.enable_notifications,
            "enable_animations": self.enable_animations,
            "log_level": self.log_level,
            "custom_css_path": self.custom_css_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GUIConfig':
        """Create from dictionary."""
        return cls(
            gui_type=GUIType(data.get("gui_type", GUIType.WEBVIEW.value)),
            app_mode=AppMode(data.get("app_mode", AppMode.NORMAL.value)),
            theme=data.get("theme", APP_THEME),
            streamlit_port=data.get("streamlit_port", STREAMLIT_PORT),
            web_port=data.get("web_port", WEB_PORT),
            window_width=data.get("window_width", 1280),
            window_height=data.get("window_height", 800),
            window_resizable=data.get("window_resizable", True),
            window_fullscreen=data.get("window_fullscreen", False),
            window_position=data.get("window_position", (-1, -1)),
            auto_start_browser=data.get("auto_start_browser", True),
            auto_update_check=data.get("auto_update_check", True),
            show_splash_screen=data.get("show_splash_screen", True),
            splash_duration=data.get("splash_duration", 2.0),
            enable_tray_icon=data.get("enable_tray_icon", True),
            minimize_to_tray=data.get("minimize_to_tray", True),
            enable_notifications=data.get("enable_notifications", True),
            enable_animations=data.get("enable_animations", True),
            log_level=data.get("log_level", "INFO"),
            custom_css_path=data.get("custom_css_path")
        )

class StreamlitManager:
    """Manager for Streamlit application."""
    
    def __init__(self, config: GUIConfig):
        self.config = config
        self.process = None
        self.is_running = False
        self.url = f"http://{STREAMLIT_HOST}:{config.streamlit_port}"
        self.lock = threading.RLock()
    
    def start(self) -> bool:
        """Start the Streamlit application."""
        with self.lock:
            if self.is_running:
                return True
            
            try:
                logger.info("Starting Streamlit application...")
                
                # Create command
                cmd = [
                    sys.executable,
                    "-m", "streamlit", "run",
                    "app.py",
                    "--server.port", str(self.config.streamlit_port),
                    "--server.headless", "true",
                    "--server.enableCORS", "false",
                    "--server.enableXsrfProtection", "false",
                    "--server.maxUploadSize", "1000",
                    "--browser.gatherUsageStats", "false"
                ]
                
                # Set environment variables
                env = os.environ.copy()
                env["STREAMLIT_THEME"] = self.config.theme
                
                # Start process
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Start output monitoring
                threading.Thread(target=self._monitor_output, daemon=True).start()
                
                # Wait for server to start
                start_time = time.time()
                while time.time() - start_time < 30:  # 30 second timeout
                    if self._check_server():
                        self.is_running = True
                        logger.info(f"Streamlit application started at {self.url}")
                        return True
                    time.sleep(0.5)
                
                logger.error("Timeout waiting for Streamlit to start")
                return False
            except Exception as e:
                logger.error(f"Error starting Streamlit application: {e}")
                return False
    
    def stop(self) -> bool:
        """Stop the Streamlit application."""
        with self.lock:
            if not self.is_running or not self.process:
                return True
            
            try:
                logger.info("Stopping Streamlit application...")
                
                # Send termination signal
                if platform.system() == "Windows":
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                
                # Wait for process to terminate
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if timeout
                    self.process.kill()
                
                self.process = None
                self.is_running = False
                logger.info("Streamlit application stopped")
                return True
            except Exception as e:
                logger.error(f"Error stopping Streamlit application: {e}")
                return False
    
    def restart(self) -> bool:
        """Restart the Streamlit application."""
        with self.lock:
            self.stop()
            time.sleep(1)
            return self.start()
    
    def _check_server(self) -> bool:
        """Check if Streamlit server is running."""
        try:
            with socket.create_connection((STREAMLIT_HOST, self.config.streamlit_port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
    
    def _monitor_output(self) -> None:
        """Monitor Streamlit process output."""
        if not self.process:
            return
        
        for line in self.process.stdout:
            logger.debug(f"Streamlit: {line.strip()}")
        
        for line in self.process.stderr:
            logger.error(f"Streamlit error: {line.strip()}")

class SplashScreen:
    """Splash screen for application startup."""
    
    def __init__(self, config: GUIConfig):
        self.config = config
        self.window = None
    
    def show(self) -> None:
        """Show the splash screen."""
        if not self.config.show_splash_screen:
            return
        
        # Create root window
        self.window = tk.Tk()
        self.window.overrideredirect(True)  # No window decorations
        self.window.attributes('-topmost', True)
        self.window.configure(bg='black')
        
        # Calculate position
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        width = 600
        height = 400
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Load logo image
        try:
            logo_path = Path(APP_LOGO)
            if logo_path.exists():
                img = Image.open(logo_path)
                img = img.resize((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Create label with image
                label = tk.Label(self.window, image=photo, bg='black')
                label.image = photo  # Keep a reference
                label.pack(pady=20)
            else:
                # Fallback to text
                label = tk.Label(
                    self.window, 
                    text=APP_NAME, 
                    font=("Helvetica", 24, "bold"),
                    fg="#00FFFF",
                    bg="black"
                )
                label.pack(pady=40)
        except Exception as e:
            logger.error(f"Error loading splash image: {e}")
            # Fallback to text
            label = tk.Label(
                self.window, 
                text=APP_NAME, 
                font=("Helvetica", 24, "bold"),
                fg="#00FFFF",
                bg="black"
            )
            label.pack(pady=40)
        
        # Version label
        version_label = tk.Label(
            self.window,
            text=f"Version {APP_VERSION}",
            font=("Helvetica", 12),
            fg="#AAAAAA",
            bg="black"
        )
        version_label.pack()
        
        # Loading bar
        progress_frame = tk.Frame(self.window, bg='black')
        progress_frame.pack(fill='x', padx=50, pady=20)
        
        progress_bar = ttk.Progressbar(
            progress_frame,
            orient='horizontal',
            length=500,
            mode='indeterminate'
        )
        progress_bar.pack(fill='x')
        progress_bar.start(10)
        
        # Loading label
        loading_label = tk.Label(
            self.window,
            text="Loading...",
            font=("Helvetica", 10),
            fg="#FFFFFF",
            bg="black"
        )
        loading_label.pack(pady=10)
        
        # Update UI
        self.window.update()
        
        # Schedule closing
        self.window.after(int(self.config.splash_duration * 1000), self.close)
    
    def close(self) -> None:
        """Close the splash screen."""
        if self.window:
            self.window.destroy()
            self.window = None

class TrayIcon:
    """System tray icon for the application."""
    
    def __init__(self, app: 'GUIApplication'):
        self.app = app
        self.tray = None
        self.icon_path = None
    
    def setup(self) -> bool:
        """Set up the tray icon."""
        if not self.app.config.enable_tray_icon:
            return False
        
        try:
            # Import pystray (optional dependency)
            import pystray
            from PIL import Image
            
            # Load icon
            icon_path = Path(APP_ICON)
            if not icon_path.exists():
                logger.error(f"Tray icon image not found: {icon_path}")
                return False
            
            self.icon_path = icon_path
            icon_image = Image.open(icon_path)
            
            # Create menu
            menu = pystray.Menu(
                pystray.MenuItem("Show", self._on_show),
                pystray.MenuItem("Restart", self._on_restart),
                pystray.MenuItem("Exit", self._on_exit)
            )
            
            # Create tray icon
            self.tray = pystray.Icon(
                name=APP_NAME,
                icon=icon_image,
                title=APP_NAME,
                menu=menu
            )
            
            # Start in a separate thread
            threading.Thread(target=self.tray.run, daemon=True).start()
            
            logger.info("Tray icon initialized")
            return True
        except ImportError:
            logger.warning("pystray module not found, tray icon disabled")
            return False
        except Exception as e:
            logger.error(f"Error setting up tray icon: {e}")
            return False
    
    def _on_show(self, icon, item) -> None:
        """Show the main window."""
        self.app.show()
    
    def _on_restart(self, icon, item) -> None:
        """Restart the application."""
        self.app.restart()
    
    def _on_exit(self, icon, item) -> None:
        """Exit the application."""
        self.app.exit()
    
    def update_tooltip(self, text: str) -> None:
        """Update the tray icon tooltip."""
        if self.tray:
            self.tray.title = text
    
    def cleanup(self) -> None:
        """Clean up tray icon resources."""
        if self.tray:
            self.tray.stop()
            self.tray = None

class NativeGUI:
    """Native GUI implementation using Tkinter."""
    
    def __init__(self, app
