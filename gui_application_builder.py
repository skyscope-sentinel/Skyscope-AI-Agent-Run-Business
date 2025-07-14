import os
import sys
import json
import time
import logging
import threading
import subprocess
import shutil
import platform
import tempfile
import uuid
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import importlib.metadata

# GUI frameworks
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Check for optional dependencies
PYQT_AVAILABLE = False
WXPYTHON_AVAILABLE = False
PYSIDE_AVAILABLE = False
KIVY_AVAILABLE = False
CUSTOMTKINTER_AVAILABLE = False

try:
    import PyQt5
    from PyQt5 import QtCore, QtGui, QtWidgets
    PYQT_AVAILABLE = True
except ImportError:
    pass

try:
    import wx
    WXPYTHON_AVAILABLE = True
except ImportError:
    pass

try:
    import PySide2
    from PySide2 import QtCore as PySideQtCore, QtGui as PySideQtGui, QtWidgets as PySideQtWidgets
    PYSIDE_AVAILABLE = True
except ImportError:
    pass

try:
    import kivy
    from kivy.app import App
    from kivy.uix.widget import Widget
    KIVY_AVAILABLE = True
except ImportError:
    pass

try:
    import customtkinter
    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    pass

# Packaging tools
PYINSTALLER_AVAILABLE = False
CX_FREEZE_AVAILABLE = False
PY2APP_AVAILABLE = False
PY2EXE_AVAILABLE = False

try:
    import PyInstaller
    PYINSTALLER_AVAILABLE = True
except ImportError:
    pass

try:
    import cx_Freeze
    CX_FREEZE_AVAILABLE = True
except ImportError:
    pass

try:
    import py2app
    PY2APP_AVAILABLE = True
except ImportError:
    pass

try:
    import py2exe
    PY2EXE_AVAILABLE = True
except ImportError:
    pass

# System tray and notifications
PYSTRAY_AVAILABLE = False
PLYER_AVAILABLE = False

try:
    import pystray
    from PIL import Image as PilImage
    PYSTRAY_AVAILABLE = True
except ImportError:
    pass

try:
    import plyer
    PLYER_AVAILABLE = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gui_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gui_application_builder")

# Constants
APP_BUILDER_VERSION = "1.0.0"
CONFIG_DIR = Path("config")
TEMPLATES_DIR = Path("templates")
ASSETS_DIR = Path("assets")
BUILDS_DIR = Path("builds")
ICONS_DIR = ASSETS_DIR / "icons"
THEMES_DIR = ASSETS_DIR / "themes"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "gui_builder_config.json"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
BUILDS_DIR.mkdir(parents=True, exist_ok=True)
ICONS_DIR.mkdir(parents=True, exist_ok=True)
THEMES_DIR.mkdir(parents=True, exist_ok=True)

class GuiFramework(Enum):
    """Supported GUI frameworks."""
    TKINTER = "tkinter"
    PYQT5 = "pyqt5"
    WXPYTHON = "wxpython"
    PYSIDE2 = "pyside2"
    KIVY = "kivy"
    CUSTOMTKINTER = "customtkinter"
    STREAMLIT = "streamlit"
    ELECTRON = "electron"

class PackagingTool(Enum):
    """Supported packaging tools."""
    PYINSTALLER = "pyinstaller"
    CX_FREEZE = "cx_freeze"
    PY2APP = "py2app"
    PY2EXE = "py2exe"
    SETUPTOOLS = "setuptools"
    ELECTRON_BUILDER = "electron-builder"

class ApplicationType(Enum):
    """Types of applications."""
    DESKTOP = "desktop"
    WEB = "web"
    HYBRID = "hybrid"
    SYSTRAY = "systray"
    SERVICE = "service"

class TargetPlatform(Enum):
    """Target platforms."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ALL = "all"

class ThemeType(Enum):
    """UI theme types."""
    LIGHT = "light"
    DARK = "dark"
    GLASS = "glass"
    FUTURISTIC = "futuristic"
    MINIMAL = "minimal"
    CUSTOM = "custom"

@dataclass
class AppIcon:
    """Application icon in various formats."""
    path: Path
    ico: Optional[Path] = None  # Windows
    icns: Optional[Path] = None  # macOS
    png: Optional[Path] = None  # Linux/General
    svg: Optional[Path] = None  # Vector
    
    def __post_init__(self):
        """Initialize icon paths."""
        if not self.ico and platform.system() == "Windows":
            # Look for .ico file
            ico_path = self.path.with_suffix(".ico")
            if ico_path.exists():
                self.ico = ico_path
        
        if not self.icns and platform.system() == "Darwin":
            # Look for .icns file
            icns_path = self.path.with_suffix(".icns")
            if icns_path.exists():
                self.icns = icns_path
        
        if not self.png:
            # Look for .png file
            png_path = self.path.with_suffix(".png")
            if png_path.exists():
                self.png = png_path
        
        if not self.svg:
            # Look for .svg file
            svg_path = self.path.with_suffix(".svg")
            if svg_path.exists():
                self.svg = svg_path
    
    def get_platform_icon(self) -> Optional[Path]:
        """Get the appropriate icon for the current platform."""
        system = platform.system()
        if system == "Windows" and self.ico:
            return self.ico
        elif system == "Darwin" and self.icns:
            return self.icns
        elif self.png:
            return self.png
        elif self.svg:
            return self.svg
        return None
    
    def convert_to_all_formats(self) -> bool:
        """Convert the icon to all required formats."""
        try:
            # Check if we have at least one format
            if not any([self.ico, self.icns, self.png, self.svg]):
                logger.error("No icon format available for conversion")
                return False
            
            # Use the first available format as source
            source = next(p for p in [self.png, self.svg, self.ico, self.icns] if p is not None)
            
            # Try to import PIL for image conversion
            try:
                from PIL import Image
                has_pil = True
            except ImportError:
                has_pil = False
                logger.warning("PIL not available, icon conversion may be limited")
            
            # Convert to PNG if needed
            if not self.png and has_pil:
                png_path = self.path.with_suffix(".png")
                Image.open(source).save(png_path)
                self.png = png_path
                logger.info(f"Converted icon to PNG: {png_path}")
            
            # Convert to ICO if needed (Windows)
            if not self.ico and has_pil:
                ico_path = self.path.with_suffix(".ico")
                img = Image.open(source)
                # Ensure image is in RGBA mode
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                # Windows icons typically include multiple sizes
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)]
                imgs = [img.resize(size, Image.LANCZOS) for size in sizes]
                imgs[0].save(ico_path, format="ICO", sizes=[(img.width, img.height) for img in imgs])
                self.ico = ico_path
                logger.info(f"Converted icon to ICO: {ico_path}")
            
            # Convert to ICNS if needed (macOS)
            if not self.icns and platform.system() == "Darwin":
                icns_path = self.path.with_suffix(".icns")
                
                # Use iconutil on macOS
                temp_dir = Path(tempfile.mkdtemp())
                iconset_dir = temp_dir / "icon.iconset"
                iconset_dir.mkdir(exist_ok=True)
                
                if has_pil:
                    img = Image.open(source)
                    # Create iconset with various sizes
                    sizes = [16, 32, 64, 128, 256, 512, 1024]
                    for size in sizes:
                        resized = img.resize((size, size), Image.LANCZOS)
                        resized.save(iconset_dir / f"icon_{size}x{size}.png")
                        # Also create @2x versions
                        if size <= 512:
                            resized = img.resize((size*2, size*2), Image.LANCZOS)
                            resized.save(iconset_dir / f"icon_{size}x{size}@2x.png")
                
                # Convert iconset to icns
                subprocess.run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)], check=True)
                self.icns = icns_path
                logger.info(f"Converted icon to ICNS: {icns_path}")
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            
            return True
        except Exception as e:
            logger.error(f"Error converting icon formats: {e}")
            return False

@dataclass
class AppMetadata:
    """Application metadata."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    author_email: str = ""
    url: str = ""
    license: str = "MIT"
    copyright: str = ""
    identifier: str = ""  # Bundle identifier (e.g., com.company.app)
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.copyright:
            import datetime
            year = datetime.datetime.now().year
            self.copyright = f"Copyright © {year} {self.author}" if self.author else f"Copyright © {year}"
        
        if not self.identifier:
            # Generate a bundle identifier from the name
            if self.author:
                company = re.sub(r'[^a-zA-Z0-9]', '', self.author.lower())
                app_name = re.sub(r'[^a-zA-Z0-9]', '', self.name.lower())
                self.identifier = f"com.{company}.{app_name}"
            else:
                app_name = re.sub(r'[^a-zA-Z0-9]', '', self.name.lower())
                self.identifier = f"com.example.{app_name}"

@dataclass
class BuildConfig:
    """Configuration for building the application."""
    app_type: ApplicationType = ApplicationType.DESKTOP
    framework: GuiFramework = GuiFramework.TKINTER
    packaging_tool: PackagingTool = PackagingTool.PYINSTALLER
    target_platforms: List[TargetPlatform] = field(default_factory=lambda: [TargetPlatform.ALL])
    theme: ThemeType = ThemeType.DARK
    custom_theme_path: Optional[Path] = None
    icon: Optional[AppIcon] = None
    splash_screen: Optional[Path] = None
    main_script: Path = Path("app.py")
    output_dir: Path = BUILDS_DIR
    include_files: List[Path] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)
    hidden_imports: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    build_options: Dict[str, Any] = field(default_factory=dict)
    create_installer: bool = True
    sign_app: bool = False
    signing_identity: Optional[str] = None
    notarize_app: bool = False  # macOS only
    apple_id: Optional[str] = None
    apple_password: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "app_type": self.app_type.value,
            "framework": self.framework.value,
            "packaging_tool": self.packaging_tool.value,
            "target_platforms": [p.value for p in self.target_platforms],
            "theme": self.theme.value,
            "custom_theme_path": str(self.custom_theme_path) if self.custom_theme_path else None,
            "icon": str(self.icon.path) if self.icon else None,
            "splash_screen": str(self.splash_screen) if self.splash_screen else None,
            "main_script": str(self.main_script),
            "output_dir": str(self.output_dir),
            "include_files": [str(f) for f in self.include_files],
            "exclude_modules": self.exclude_modules,
            "hidden_imports": self.hidden_imports,
            "requirements": self.requirements,
            "environment_variables": self.environment_variables,
            "build_options": self.build_options,
            "create_installer": self.create_installer,
            "sign_app": self.sign_app,
            "signing_identity": self.signing_identity,
            "notarize_app": self.notarize_app,
            "apple_id": self.apple_id,
            "apple_password": self.apple_password
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BuildConfig':
        """Create from dictionary."""
        icon_path = data.get("icon")
        icon = AppIcon(Path(icon_path)) if icon_path else None
        
        custom_theme_path = data.get("custom_theme_path")
        custom_theme = Path(custom_theme_path) if custom_theme_path else None
        
        splash_screen = data.get("splash_screen")
        splash = Path(splash_screen) if splash_screen else None
        
        return cls(
            app_type=ApplicationType(data.get("app_type", ApplicationType.DESKTOP.value)),
            framework=GuiFramework(data.get("framework", GuiFramework.TKINTER.value)),
            packaging_tool=PackagingTool(data.get("packaging_tool", PackagingTool.PYINSTALLER.value)),
            target_platforms=[TargetPlatform(p) for p in data.get("target_platforms", [TargetPlatform.ALL.value])],
            theme=ThemeType(data.get("theme", ThemeType.DARK.value)),
            custom_theme_path=custom_theme,
            icon=icon,
            splash_screen=splash,
            main_script=Path(data.get("main_script", "app.py")),
            output_dir=Path(data.get("output_dir", str(BUILDS_DIR))),
            include_files=[Path(f) for f in data.get("include_files", [])],
            exclude_modules=data.get("exclude_modules", []),
            hidden_imports=data.get("hidden_imports", []),
            requirements=data.get("requirements", []),
            environment_variables=data.get("environment_variables", {}),
            build_options=data.get("build_options", {}),
            create_installer=data.get("create_installer", True),
            sign_app=data.get("sign_app", False),
            signing_identity=data.get("signing_identity"),
            notarize_app=data.get("notarize_app", False),
            apple_id=data.get("apple_id"),
            apple_password=data.get("apple_password")
        )
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Build configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving build configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'BuildConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Build configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading build configuration: {e}")
            return cls()

class TemplateManager:
    """Manager for application templates."""
    
    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.templates = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load available templates."""
        try:
            if not self.templates_dir.exists():
                logger.warning(f"Templates directory {self.templates_dir} does not exist")
                return
            
            # Scan for template directories
            for template_dir in self.templates_dir.iterdir():
                if template_dir.is_dir():
                    # Check for template.json
                    template_json = template_dir / "template.json"
                    if template_json.exists():
                        try:
                            with open(template_json, 'r') as f:
                                template_data = json.load(f)
                            
                            # Add template to dictionary
                            self.templates[template_dir.name] = {
                                "name": template_data.get("name", template_dir.name),
                                "description": template_data.get("description", ""),
                                "framework": template_data.get("framework", GuiFramework.TKINTER.value),
                                "app_type": template_data.get("app_type", ApplicationType.DESKTOP.value),
                                "preview": template_dir / template_data.get("preview", "preview.png"),
                                "files": template_data.get("files", []),
                                "requirements": template_data.get("requirements", []),
                                "path": template_dir
                            }
                        except Exception as e:
                            logger.error(f"Error loading template {template_dir.name}: {e}")
            
            logger.info(f"Loaded {len(self.templates)} templates")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_for_framework(self, framework: GuiFramework) -> List[Dict[str, Any]]:
        """Get templates for a specific framework."""
        return [t for t in self.templates.values() if t.get("framework") == framework.value]
    
    def get_templates_for_app_type(self, app_type: ApplicationType) -> List[Dict[str, Any]]:
        """Get templates for a specific application type."""
        return [t for t in self.templates.values() if t.get("app_type") == app_type.value]
    
    def create_from_template(self, template_id: str, output_dir: Path, app_metadata: AppMetadata) -> bool:
        """Create a new application from a template."""
        template = self.get_template(template_id)
        if not template:
            logger.error(f"Template {template_id} not found")
            return False
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy template files
            template_path = template["path"]
            for file_entry in template["files"]:
                src_path = template_path / file_entry["source"]
                dst_path = output_dir / file_entry["destination"]
                
                # Create parent directories
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if this is a template file that needs variable substitution
                if file_entry.get("template", False):
                    with open(src_path, 'r') as f:
                        content = f.read()
                    
                    # Replace variables
                    content = content.replace("{{APP_NAME}}", app_metadata.name)
                    content = content.replace("{{APP_VERSION}}", app_metadata.version)
                    content = content.replace("{{APP_DESCRIPTION}}", app_metadata.description)
                    content = content.replace("{{APP_AUTHOR}}", app_metadata.author)
                    content = content.replace("{{APP_AUTHOR_EMAIL}}", app_metadata.author_email)
                    content = content.replace("{{APP_URL}}", app_metadata.url)
                    content = content.replace("{{APP_LICENSE}}", app_metadata.license)
                    content = content.replace("{{APP_COPYRIGHT}}", app_metadata.copyright)
                    content = content.replace("{{APP_IDENTIFIER}}", app_metadata.identifier)
                    
                    with open(dst_path, 'w') as f:
                        f.write(content)
                else:
                    # Simple file copy
                    shutil.copy2(src_path, dst_path)
            
            # Create requirements.txt if not already in template
            if template["requirements"] and not (output_dir / "requirements.txt").exists():
                with open(output_dir / "requirements.txt", 'w') as f:
                    for req in template["requirements"]:
                        f.write(f"{req}\n")
            
            logger.info(f"Created application from template {template_id} in {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating application from template: {e}")
            return False
    
    def create_template(self, name: str, description: str, framework: GuiFramework, 
                       app_type: ApplicationType, source_dir: Path) -> Optional[str]:
        """Create a new template from an existing application."""
        try:
            # Generate template ID
            template_id = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
            template_dir = self.templates_dir / template_id
            
            # Check if template already exists
            if template_dir.exists():
                logger.error(f"Template {template_id} already exists")
                return None
            
            # Create template directory
            template_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect files
            files = []
            for root, _, filenames in os.walk(source_dir):
                rel_root = Path(root).relative_to(source_dir)
                for filename in filenames:
                    src_path = Path(root) / filename
                    rel_path = rel_root / filename
                    
                    # Skip __pycache__, .git, etc.
                    if any(part.startswith(".") or part == "__pycache__" for part in rel_path.parts):
                        continue
                    
                    # Determine if this is a template file
                    is_template = False
                    if filename.endswith((".py", ".html", ".css", ".js", ".json", ".xml", ".md")):
                        # Check file content for potential template variables
                        with open(src_path, 'r', errors='ignore') as f:
                            content = f.read()
                            if "{{" in content and "}}" in content:
                                is_template = True
                    
                    # Copy file to template directory
                    dst_path = template_dir / rel_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    
                    # Add to files list
                    files.append({
                        "source": str(rel_path),
                        "destination": str(rel_path),
                        "template": is_template
                    })
            
            # Extract requirements
            requirements = []
            req_file = source_dir / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            requirements.append(line)
            
            # Create template.json
            template_data = {
                "name": name,
                "description": description,
                "framework": framework.value,
                "app_type": app_type.value,
                "preview": "preview.png",
                "files": files,
                "requirements": requirements
            }
            
            with open(template_dir / "template.json", 'w') as f:
                json.dump(template_data, f, indent=2)
            
            # Create a placeholder preview image
            shutil.copy(ASSETS_DIR / "placeholder.png", template_dir / "preview.png")
            
            # Reload templates
            self.load_templates()
            
            logger.info(f"Created template {template_id} from {source_dir}")
            return template_id
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return None

class DependencyManager:
    """Manager for application dependencies."""
    
    def __init__(self):
        self.installed_packages = self._get_installed_packages()
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed Python packages."""
        try:
            packages = {}
            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"].lower()] = dist.version
            return packages
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return {}
    
    def check_requirements(self, requirements: List[str]) -> Dict[str, bool]:
        """Check if requirements are installed."""
        results = {}
        for req in requirements:
            # Parse requirement
            req_name = req.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
            req_version = None
            
            if "==" in req:
                req_version = req.split("==")[1].strip()
            elif ">=" in req:
                req_version = req.split(">=")[1].strip()
            elif "<=" in req:
                req_version = req.split("<=")[1].strip()
            
            # Check if installed
            if req_name in self.installed_packages:
                if req_version:
                    # Check version
                    installed_version = self.installed_packages[req_name]
                    if "==" in req:
                        results[req] = installed_version == req_version
                    elif ">=" in req:
                        results[req] = installed_version >= req_version
                    elif "<=" in req:
                        results[req] = installed_version <= req_version
                    else:
                        results[req] = True
                else:
                    results[req] = True
            else:
                results[req] = False
        
        return results
    
    def install_requirements(self, requirements: List[str]) -> Dict[str, bool]:
        """Install requirements using pip."""
        results = {}
        for req in requirements:
            try:
                logger.info(f"Installing {req}")
                subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
                results[req] = True
            except Exception as e:
                logger.error(f"Error installing {req}: {e}")
                results[req] = False
        
        # Update installed packages
        self.installed_packages = self._get_installed_packages()
        
        return results
    
    def get_framework_requirements(self, framework: GuiFramework) -> List[str]:
        """Get requirements for a specific GUI framework."""
        if framework == GuiFramework.TKINTER:
            return []  # Built-in
        elif framework == GuiFramework.PYQT5:
            return ["PyQt5"]
        elif framework == GuiFramework.WXPYTHON:
            return ["wxPython"]
        elif framework == GuiFramework.PYSIDE2:
            return ["PySide2"]
        elif framework == GuiFramework.KIVY:
            return ["kivy"]
        elif framework == GuiFramework.CUSTOMTKINTER:
            return ["customtkinter"]
        elif framework == GuiFramework.STREAMLIT:
            return ["streamlit"]
        elif framework == GuiFramework.ELECTRON:
            return ["pywebview", "flask"]
        else:
            return []
    
    def get_packaging_requirements(self, tool: PackagingTool) -> List[str]:
        """Get requirements for a specific packaging tool."""
        if tool == PackagingTool.PYINSTALLER:
            return ["pyinstaller"]
        elif tool == PackagingTool.CX_FREEZE:
            return ["cx_Freeze"]
        elif tool == PackagingTool.PY2APP:
            return ["py2app"]
        elif tool == PackagingTool.PY2EXE:
            return ["py2exe"]
        elif tool == PackagingTool.SETUPTOOLS:
            return ["setuptools"]
        elif tool == PackagingTool.ELECTRON_BUILDER:
            return ["pywebview", "flask", "electron-builder"]
        else:
            return []

class ApplicationBuilder:
    """Builder for GUI applications."""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.dependency_manager = DependencyManager()
        self.build_config = BuildConfig()
        self.app_metadata = None
        self.build_status = {}
        self.build_log = []
    
    def set_metadata(self, metadata: AppMetadata) -> None:
        """Set application metadata."""
        self.app_metadata = metadata
    
    def set_config(self, config: BuildConfig) -> None:
        """Set build configuration."""
        self.build_config = config
    
    def load_config(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load build configuration from file."""
        self.build_config = BuildConfig.load(filepath)
    
    def save_config(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save build configuration to file."""
        self.build_config.save(filepath)
    
    def create_from_template(self, template_id: str, output_dir: Path) -> bool:
        """Create a new application from a template."""
        if not self.app_metadata:
            logger.error("Application metadata not set")
            return False
        
        return self.template_manager.create_from_template(template_id, output_dir, self.app_metadata)
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed."""
        # Collect all requirements
        requirements = []
        
        # Framework requirements
        framework_reqs = self.dependency_manager.get_framework_requirements(self.build_config.framework)
        requirements.extend(framework_reqs)
        
        # Packaging requirements
        packaging_reqs = self.dependency_manager.get_packaging_requirements(self.build_config.packaging_tool)
        requirements.extend(packaging_reqs)
        
        # Additional requirements
        requirements.extend(self.build_config.requirements)
        
        # Check requirements
        return self.dependency_manager.check_requirements(requirements)
    
    def install_dependencies(self) -> Dict[str, bool]:
        """Install all required dependencies."""
        # Collect all requirements
        requirements = []
        
        # Framework requirements
        framework_reqs = self.dependency_manager.get_framework_requirements(self.build_config.framework)
        requirements.extend(framework_reqs)
        
        # Packaging requirements
        packaging_reqs = self.dependency_manager.get_packaging_requirements(self.build_config.packaging_tool)
        requirements.extend(packaging_reqs)
        
        # Additional requirements
        requirements.extend(self.build_config.requirements)
        
        # Install requirements
        return self.dependency_manager.install_requirements(requirements)
    
    def prepare_build_environment(self) -> bool:
        """Prepare the build environment."""
        try:
            # Check dependencies
            dep_status = self.check_dependencies()
            missing_deps = [dep for dep, installed in dep_status.items() if not installed]
            
            if missing_deps:
                logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                # Install missing dependencies
                install_status = self.dependency_manager.install_requirements(missing_deps)
                failed_installs = [dep for dep, success in install_status.items() if not success]
                
                if failed_installs:
                    logger.error(f"Failed to install dependencies: {', '.join(failed_installs)}")
                    return False
            
            # Create output directory
            self.build_config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare icon if needed
            if self.build_config.icon:
                self.build_config.icon.convert_to_all_formats()
            
            return True
        except Exception as e:
            logger.error(f"Error preparing build environment: {e}")
            return False
    
    def build_with_pyinstaller(self) -> bool:
        """Build the application using PyInstaller."""
        try:
            if not PYINSTALLER_AVAILABLE:
                logger.error("PyInstaller is not installed")
                return False
            
            # Create a temporary spec file
            spec_content = self._generate_pyinstaller_spec()
            spec_file = self.build_config.output_dir / f"{self.app_metadata.name.lower().replace(' ', '_')}.spec"
            
            with open(spec_file, 'w') as f:
                f.write(spec_content)
            
            # Build command
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ]
            
            # Add log file
            log_file = self.build_config.output_dir / "build.log"
            
            # Run PyInstaller
            self._log_build_step("Starting PyInstaller build")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Process output
                for line in process.stdout:
                    f.write(line)
                    self._log_build_step(line.strip())
                
                process.wait()
            
            if process.returncode != 0:
                self._log_build_step(f"PyInstaller build failed with code {process.returncode}")
                return False
            
            self._log_build_step("PyInstaller build completed successfully")
            
            # Check if we need to create an installer
            if self.build_config.create_installer:
                return self._create_installer()
            
            return True
        except Exception as e:
            self._log_build_step(f"Error building with PyInstaller: {e}")
            return False
    
    def _generate_pyinstaller_spec(self) -> str:
        """Generate a PyInstaller spec file."""
        app_name = self.app_metadata.name.lower().replace(' ', '_')
        main_script = self.build_config.main_script
        
        # Icon path
        icon_path = ""
        if self.build_config.icon:
            platform_icon = self.build_config.icon.get_platform_icon()
            if platform_icon:
                icon_path = f", icon='{platform_icon}'"
        
        # Hidden imports
        hidden_imports = []
        hidden_imports.extend(self.build_config.hidden_imports)
        
        # Add framework-specific imports
        if self.build_config.framework == GuiFramework.TKINTER:
            hidden_imports.extend(["tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.filedialog"])
        elif self.build_config.framework == GuiFramework.PYQT5:
            hidden_imports.extend(["PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets"])
        elif self.build_config.framework == GuiFramework.WXPYTHON:
            hidden_imports.extend(["wx"])
        elif self.build_config.framework == GuiFramework.PYSIDE2:
            hidden_imports.extend(["PySide2", "PySide2.QtCore", "PySide2.QtGui", "PySide2.QtWidgets"])
        elif self.build_config.framework == GuiFramework.KIVY:
            hidden_imports.extend(["kivy"])
        elif self.build_config.framework == GuiFramework.CUSTOMTKINTER:
            hidden_imports.extend(["customtkinter", "tkinter"])
        elif self.build_config.framework == GuiFramework.STREAMLIT:
            hidden_imports.extend(["streamlit"])
        
        hidden_imports_str = ", ".join([f"'{imp}'" for imp in hidden_imports])
        
        # Include files
        include_files = []
        for file_path in self.build_config.include_files:
            include_files.append((str(file_path), str(file_path.name)))
        
        # Data files for PyInstaller
        datas = []
        for src, dst in include_files:
            datas.append(f"('{src}', '{dst}')")
        
        datas_str = ", ".join(datas)
        
        # Exclude modules
        exclude_modules = []
        exclude_modules.extend(self.build_config.exclude_modules)
        exclude_modules_str = ", ".join([f"'{mod}'" for mod in exclude_modules])
        
        # Build options
        one_file = self.build_config.build_options.get("one_file", True)
        console = self.build_config.build_options.get("console", False)
        
        # Generate spec content
        spec = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{main_script}'],
    pathex=[],
    binaries=[],
    datas=[{datas_str}],
    hiddenimports=[{hidden_imports_str}],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[{exclude_modules_str}],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

"""
        
        if one_file:
            spec += f"""
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console={str(console).lower()}{icon_path},
)
"""
        else:
            spec += f"""
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={str(console).lower()}{icon_path},
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{app_name}',
)
"""
        
        # Add macOS bundle for desktop apps
        if platform.system() == "Darwin" and self.build_config.app_type == ApplicationType.DESKTOP:
            spec += f"""
app = BUNDLE(
    coll,
    name='{app_name}.app',
    icon='{self.build_config.icon.icns if self.build_config.icon and self.build_config.icon.icns else None}',
    bundle_identifier='{self.app_metadata.identifier}',
    info_plist={{
        'CFBundleShortVersionString': '{self.app_metadata.version}',
        'CFBundleVersion': '{self.app_metadata.version}',
        'NSHighResolutionCapable': 'True'
    }},
)
"""
        
        return spec
    
    def build_with_cx_freeze(self) -> bool:
        """Build the application using cx_Freeze."""
        try:
            if not CX_FREEZE_AVAILABLE:
                logger.error("cx_Freeze is not installed")
                return False
            
            # Create a temporary setup.py file
            setup_content = self._generate_cx_freeze_setup()
            setup_file = self.build_config.output_dir / "setup.py"
            
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            
            # Build command
            cmd = [
                sys.executable, str(setup_file),
                "build_exe"
            ]
            
            # Add log file
            log_file = self.build_config.output_dir / "build.log"
            
            # Run cx_Freeze
            self._log_build_step("Starting cx_Freeze build")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=self.build_config.output_dir
                )
                
                # Process output
                for line in process.stdout:
                    f.write(line)
                    self._log_build_step(line.strip())
                
                process.wait()
            
            if process.returncode != 0:
                self._log_build_step(f"cx_Freeze build failed with code {process.returncode}")
                return False
            
            self._log_build_step("cx_Freeze build completed successfully")
            
            # Check if we need to create an installer
            if self.build_config.create_installer:
                return self._create_installer()
            
            return True
        except Exception as e:
            self._log_build_step(f"Error building with cx_Freeze: {e}")
            return False
    
    def _generate_cx_freeze_setup(self) -> str:
        """Generate a cx_Freeze setup.py file."""
        app_name = self.app_metadata.name
        app_version = self.app_metadata.version
        main_script = self.build_config.main_script
        
        # Icon path
        icon_path = "None"
        if self.build_config.icon:
            platform_icon = self.build_config.icon.get_platform_icon()
            if platform_icon:
                icon_path = f"r'{platform_icon}'"
        
        # Include files
        include_files = []
        for file_path in self.build_config.include_files:
            include_files.append(f"(r'{file_path}', r'{file_path.name}')")
        
        include_files_str = ", ".join(include_files)
        
        # Packages
        packages = []
        
        # Add framework-specific packages
        if self.build_config.framework == GuiFramework.TKINTER:
            packages.extend(["tkinter"])
        elif self.build_config.framework == GuiFramework.PYQT5:
            packages.extend(["PyQt5"])
        elif self.build_config.framework == GuiFramework.WXPYTHON:
            packages.extend(["wx"])
        elif self.build_config.framework == GuiFramework.PYSIDE2:
            packages.extend(["PySide2"])
        elif self.build_config.framework == GuiFramework.KIVY:
            packages.extend(["kivy"])
        elif self.build_config.framework == GuiFramework.CUSTOMTKINTER:
            packages.extend(["customtkinter", "tkinter"])
        elif self.build_config.framework == GuiFramework.STREAMLIT:
            packages.extend(["streamlit"])
        
        # Add hidden imports
        packages.extend(self.build_config.hidden_imports)
        
        packages_str = ", ".join([f"'{pkg}'" for pkg in packages])
        
        # Excludes
        excludes = []
        excludes.extend(self.build_config.exclude_modules)
        excludes_str = ", ".join([f"'{mod}'" for mod in excludes])
        
        # Build options
        build_exe_options = {
            "packages": packages,
            "excludes": self.build_config.exclude_modules,
            "include_files": self.build_config.include_files
        }
        
        # Generate setup.py content
        setup = f"""import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {{
    "packages": [{packages_str}],
    "excludes": [{excludes_str}],
    "include_files": [{include_files_str}]
}}

# Base for GUI applications
base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [
    Executable(
        r"{main_script}",
        base=base,
        target_name="{app_name}",
        icon={icon_path}
    )
]

setup(
    name="{app_name}",
    version="{app_version}",
    description="{self.app_metadata.description}",
    author="{self.app_metadata.author}",
    author_email="{self.app_metadata.author_email}",
    options={{"build_exe": build_exe_options}},
    executables=executables
)
"""
        
        return setup
    
    def build_with_py2app(self) -> bool:
        """Build the application using py2app (macOS only)."""
        if platform.system() != "Darwin":
            self._log_build_step("py2app is only supported on macOS")
            return False
        
        try:
            if not PY2APP_AVAILABLE:
                logger.error("py2app is not installed")
                return False
            
            # Create a temporary setup.py file
            setup_content = self._generate_py2app_setup()
            setup_file = self.build_config.output_dir / "setup.py"
            
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            
            # Build command
            cmd = [
                sys.executable, str(setup_file),
                "py2app"
            ]
            
            # Add log file
            log_file = self.build_config.output_dir / "build.log"
            
            # Run py2app
            self._log_build_step("Starting py2app build")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=self.build_config.output_dir
                )
                
                # Process output
                for line in process.stdout:
                    f.write(line)
                    self._log_build_step(line.strip())
                
                process.wait()
            
            if process.returncode != 0:
                self._log_build_step(f"py2app build failed with code {process.returncode}")
                return False
            
            self._log_build_step("py2app build completed successfully")
            
            # Sign the app if requested
            if self.build_config.sign_app and self.build_config.signing_identity:
                return self._sign_macos_app()
            
            return True
        except Exception as e:
            self._log_build_step(f"Error building with py2app: {e}")
            return False
    
    def _generate_py2app_setup(self) -> str:
        """Generate a py2app setup.py file."""
        app_name = self.app_metadata.name
        app_version = self.app_metadata.version
        main_script = self.build_config.main_script
        
        # Icon path
        icon_path = "None"
        if self.build_config.icon and self.build_config.icon.icns:
            icon_path = f"r'{self.build_config.icon.icns}'"
        
        # Include files
        include_files = []
        for file_path in self.build_config.include_files:
            include_files.append(f"'{file_path}'")
        
        include_files_str = ", ".join(include_files)
        
        # Packages
        packages = []
        
        # Add framework-specific packages
        if self.build_config.framework == GuiFramework.TKINTER:
            packages.extend(["tkinter"])
        elif self.build_config.framework == GuiFramework.PYQT5:
            packages.extend(["PyQt5"])
        elif self.build_config.framework == GuiFramework.WXPYTHON:
            packages.extend(["wx"])
        elif self.build_config.framework == GuiFramework.PYSIDE2:
            packages.extend(["PySide2"])
        elif self.build_config.framework == GuiFramework.KIVY:
            packages.extend(["kivy"])
        elif self.build_config.framework == GuiFramework.CUSTOMTKINTER:
            packages.extend(["customtkinter", "tkinter"])
        elif self.build_config.framework == GuiFramework.STREAMLIT:
            packages.extend(["streamlit"])
        
        # Add hidden imports
        packages.extend(self.build_config.hidden_imports)
        
        packages_str = ", ".join([f"'{pkg}'" for pkg in packages])
        
        # Generate setup.py content
        setup = f"""from setuptools import setup

APP = ['{main_script}']
DATA_FILES = [{include_files_str}]
OPTIONS = {{
    'argv_emulation': True,
    'packages': [{packages_str}],
    'iconfile': {icon_path},
    'plist': {{
        'CFBundleName': '{app_name}',
        'CFBundleDisplayName': '{app_name}',
        'CFBundleIdentifier': '{self.app_metadata.identifier}',
        'CFBundleVersion': '{app_version}',
        'CFBundleShortVersionString': '{app_version}',
        'NSHumanReadableCopyright': '{self.app_metadata.copyright}'
    }}
}}

setup(
    name="{app_name}",
    app=APP,
    data_files=DATA_FILES,
    options={{'py2app': OPTIONS}},
    setup_requires=['py2app'],
)
"""
        
        return setup
    
    def _sign_macos_app(self) -> bool:
        """Sign a macOS application."""
        try:
            app_name = self.app_metadata.name
            app_path = self.build_config.output_dir / "dist" / f"{app_name}.app"
            
            if not app_path.exists():
                self._log_build_step(f"App bundle not found at {app_path}")
                return False
            
            # Sign the app
            self._log_build_step(f"Signing app with identity: {self.build_config.signing_identity}")
            cmd = [
                "codesign",
                "--force",
                "--sign", self.build_config.signing_identity,
                "--deep",
                "--timestamp",
                str(app_path)
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"Signing failed: {process.stderr}")
                return False
            
            self._log_build_step("App signed successfully")
            
            # Notarize if requested
            if self.build_config.notarize_app and self.build_config.apple_id and self.build_config.apple_password:
                return self._notarize_macos_app(app_path)
            
            return True
        except Exception as e:
            self._log_build_step(f"Error signing macOS app: {e}")
            return False
    
    def _notarize_macos_app(self, app_path: Path) -> bool:
        """Notarize a macOS application."""
        try:
            # Create a ZIP archive of the app
            zip_path = self.build_config.output_dir / f"{self.app_metadata.name}.zip"
            self._log_build_step(f"Creating ZIP archive for notarization: {zip_path}")
            
            subprocess.run(["ditto", "-c", "-k", "--keepParent", str(app_path), str(zip_path)], check=True)
            
            # Submit for notarization
            self._log_build_step("Submitting app for notarization")
            cmd = [
                "xcrun", "altool", "--notarize-app",
                "--primary-bundle-id", self.app_metadata.identifier,
                "--username", self.build_config.apple_id,
                "--password", self.build_config.apple_password,
                "--file", str(zip_path)
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"Notarization submission failed: {process.stderr}")
                return False
            
            # Extract request UUID
            output = process.stdout
            match = re.search(r'RequestUUID = (\S+)', output)
            if not match:
                self._log_build_step("Could not find RequestUUID in notarization response")
                return False
            
            request_uuid = match.group(1)
            self._log_build_step(f"Notarization request submitted with UUID: {request_uuid}")
            
            # Wait for notarization to complete
            self._log_build_step("Waiting for notarization to complete (this may take several minutes)")
            notarization_complete = False
            attempts = 0
            max_attempts = 30
            
            while not notarization_complete and attempts < max_attempts:
                time.sleep(30)  # Wait 30 seconds between checks
                attempts += 1
                
                # Check notarization status
                cmd = [
                    "xcrun", "altool", "--notarization-info", request_uuid,
                    "--username", self.build_config.apple_id,
                    "--password", self.build_config.apple_password
                ]
                
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    self._log_build_step(f"Error checking notarization status: {process.stderr}")
                    continue
                
                # Check for success
                output = process.stdout
                if "Status: success" in output:
                    notarization_complete = True
                    self._log_build_step("Notarization completed successfully")
                elif "Status: in progress" in output:
                    self._log_build_step(f"Notarization in progress (attempt {attempts}/{max_attempts})")
                else:
                    self._log_build_step(f"Notarization failed: {output}")
                    return False
            
            if not notarization_complete:
                self._log_build_step("Notarization timed out")
                return False
            
            # Staple the notarization to the app
            self._log_build_step("Stapling notarization ticket to app")
            cmd = ["xcrun", "stapler", "staple", str(app_path)]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"Stapling failed: {process.stderr}")
                return False
            
            self._log_build_step("App notarized and stapled successfully")
            return True
        except Exception as e:
            self._log_build_step(f"Error notarizing macOS app: {e}")
            return False
    
    def build_with_setuptools(self) -> bool:
        """Build the application using setuptools."""
        try:
            # Create a temporary setup.py file
            setup_content = self._generate_setuptools_setup()
            setup_file = self.build_config.output_dir / "setup.py"
            
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            
            # Build command
            cmd = [
                sys.executable, str(setup_file),
                "bdist_wheel"
            ]
            
            # Add log file
            log_file = self.build_config.output_dir / "build.log"
            
            # Run setuptools
            self._log_build_step("Starting setuptools build")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=self.build_config.output_dir
                )
                
                # Process output
                for line in process.stdout:
                    f.write(line)
                    self._log_build_step(line.strip())
                
                process.wait()
            
            if process.returncode != 0:
                self._log_build_step(f"setuptools build failed with code {process.returncode}")
                return False
            
            self._log_build_step("setuptools build completed successfully")
            return True
        except Exception as e:
            self._log_build_step(f"Error building with setuptools: {e}")
            return False
    
    def _generate_setuptools_setup(self) -> str:
        """Generate a setuptools setup.py file."""
        app_name = self.app_metadata.name.lower().replace(' ', '_')
        app_version = self.app_metadata.version
        main_script = self.build_config.main_script
        
        # Generate entry points
        entry_points = f"'{app_name}=={main_script.stem}:main'"
        
        # Generate setup.py content
        setup = f"""from setuptools import setup, find_packages

setup(
    name="{app_name}",
    version="{app_version}",
    description="{self.app_metadata.description}",
    author="{self.app_metadata.author}",
    author_email="{self.app_metadata.author_email}",
    url="{self.app_metadata.url}",
    packages=find_packages(),
    install_requires={self.build_config.requirements},
    entry_points={{
        'console_scripts': [
            {entry_points}
        ],
    }},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: {self.app_metadata.license} License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
"""
        
        return setup
    
    def _create_installer(self) -> bool:
        """Create an installer for the application."""
        system = platform.system()
        
        if system == "Windows":
            return self._create_windows_installer()
        elif system == "Darwin":
            return self._create_macos_installer()
        elif system == "Linux":
            return self._create_linux_installer()
        else:
            self._log_build_step(f"Installer creation not supported on {system}")
            return False
    
    def _create_windows_installer(self) -> bool:
        """Create a Windows installer using NSIS."""
        try:
            # Check if NSIS is installed
            nsis_path = shutil.which("makensis")
            if not nsis_path:
                self._log_build_step("NSIS not found, installer creation skipped")
                return False
            
            app_name = self.app_metadata.name
            app_version = self.app_metadata.version
            
            # Determine build directory
            if self.build_config.packaging_tool == PackagingTool.PYINSTALLER:
                if self.build_config.build_options.get("one_file", True):
                    build_dir = self.build_config.output_dir / "dist"
                else:
                    build_dir = self.build_config.output_dir / "dist" / app_name.lower().replace(' ', '_')
            elif self.build_config.packaging_tool == PackagingTool.CX_FREEZE:
                build_dir = self.build_config.output_dir / "build" / "exe.win-amd64-3.10"  # May need adjustment
            else:
                self._log_build_step("Unsupported packaging tool for Windows installer")
                return False
            
            # Create NSIS script
            nsis_script = f"""
            !define APPNAME "{app_name}"
            !define APPVERSION "{app_version}"
            !define COMPANYNAME "{self.app_metadata.author}"
            !define DESCRIPTION "{self.app_metadata.description}"
            !define VERSIONMAJOR {app_version.split('.')[0]}
            !define VERSIONMINOR {app_version.split('.')[1] if len(app_version.split('.')) > 1 else 0}
            !define VERSIONBUILD {app_version.split('.')[2] if len(app_version.split('.')) > 2 else 0}
            
            # General settings
            Name "${{APPNAME}} ${{APPVERSION}}"
            OutFile "{self.build_config.output_dir / 'dist' / f'{app_name}_setup.exe'}"
            InstallDir "$PROGRAMFILES\\${{APPNAME}}"
            InstallDirRegKey HKLM "Software\\${{APPNAME}}" "Install_Dir"
            RequestExecutionLevel admin
            
            # Pages
            !include "MUI2.nsh"
            !insertmacro MUI_PAGE_WELCOME
            !insertmacro MUI_PAGE_LICENSE "license.txt"
            !insertmacro MUI_PAGE_DIRECTORY
            !insertmacro MUI_PAGE_INSTFILES
            !insertmacro MUI_PAGE_FINISH
            
            !insertmacro MUI_UNPAGE_WELCOME
            !insertmacro MUI_UNPAGE_CONFIRM
            !insertmacro MUI_UNPAGE_INSTFILES
            !insertmacro MUI_UNPAGE_FINISH
            
            !insertmacro MUI_LANGUAGE "English"
            
            # Default section
            Section "Install"
                SetOutPath $INSTDIR
                
                # Copy files
                File /r "{build_dir}\\*.*"
                
                # Create uninstaller
                WriteUninstaller "$INSTDIR\\uninstall.exe"
                
                # Create shortcuts
                CreateDirectory "$SMPROGRAMS\\${{APPNAME}}"
                CreateShortcut "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\{app_name.lower().replace(' ', '_')}.exe"
                CreateShortcut "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
                CreateShortcut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\{app_name.lower().replace(' ', '_')}.exe"
                
                # Registry entries
                WriteRegStr HKLM "Software\\${{APPNAME}}" "Install_Dir" "$INSTDIR"
                WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayName" "${{APPNAME}}"
                WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "UninstallString" '"$INSTDIR\\uninstall.exe"'
                WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayIcon" '"$INSTDIR\\{app_name.lower().replace(' ', '_')}.exe"'
                WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayVersion" "${{APPVERSION}}"
                WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "Publisher" "${{COMPANYNAME}}"
                WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "NoModify" 1
                WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "NoRepair" 1
            SectionEnd
            
            # Uninstaller section
            Section "Uninstall"
                # Remove files and directories
                Delete "$INSTDIR\\uninstall.exe"
                RMDir /r "$INSTDIR"
                
                # Remove shortcuts
                Delete "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk"
                Delete "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk"
                RMDir "$SMPROGRAMS\\${{APPNAME}}"
                Delete "$DESKTOP\\${{APPNAME}}.lnk"
                
                # Remove registry entries
                DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}"
                DeleteRegKey HKLM "Software\\${{APPNAME}}"
            SectionEnd
            """
            
            # Create a license.txt file if it doesn't exist
            license_file = self.build_config.output_dir / "license.txt"
            if not license_file.exists():
                with open(license_file, 'w') as f:
                    f.write(f"{self.app_metadata.name} {self.app_metadata.version}\n")
                    f.write(f"{self.app_metadata.copyright}\n\n")
                    f.write("This software is provided 'as-is', without any express or implied warranty.\n")
            
            # Write NSIS script to file
            nsis_file = self.build_config.output_dir / "installer.nsi"
            with open(nsis_file, 'w') as f:
                f.write(nsis_script)
            
            # Run NSIS
            self._log_build_step("Creating Windows installer with NSIS")
            process = subprocess.run([nsis_path, str(nsis_file)], capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"NSIS failed: {process.stderr}")
                return False
            
            self._log_build_step("Windows installer created successfully")
            return True
        except Exception as e:
            self._log_build_step(f"Error creating Windows installer: {e}")
            return False
    
    def _create_macos_installer(self) -> bool:
        """Create a macOS DMG installer."""
        try:
            # Check if create-dmg is installed
            create_dmg_path = shutil.which("create-dmg")
            if not create_dmg_path:
                self._log_build_step("create-dmg not found, installer creation skipped")
                return False
            
            app_name = self.app_metadata.name
            app_version = self.app_metadata.version
            
            # Determine app bundle path
            if self.build_config.packaging_tool == PackagingTool.PYINSTALLER:
                app_bundle = self.build_config.output_dir / "dist" / f"{app_name}.app"
            elif self.build_config.packaging_tool == PackagingTool.PY2APP:
                app_bundle = self.build_config.output_dir / "dist" / f"{app_name}.app"
            else:
                self._log_build_step("Unsupported packaging tool for macOS installer")
                return False
            
            if not app_bundle.exists():
                self._log_build_step(f"App bundle not found at {app_bundle}")
                return False
            
            # Create DMG
            self._log_build_step("Creating macOS DMG installer")
            dmg_path = self.build_config.output_dir / "dist" / f"{app_name}_{app_version}.dmg"
            
            cmd = [
                create_dmg_path,
                "--volname", f"{app_name} {app_version}",
                "--volicon", str(self.build_config.icon.icns) if self.build_config.icon and self.build_config.icon.icns else "",
                "--window-pos", "200", "120",
                "--window-size", "800", "400",
                "--icon-size", "100",
                "--icon", f"{app_name}.app", "200", "200",
                "--hide-extension", f"{app_name}.app",
                "--app-drop-link", "600", "200",
                str(dmg_path),
                str(app_bundle)
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"create-dmg failed: {process.stderr}")
                return False
            
            self._log_build_step("macOS DMG installer created successfully")
            return True
        except Exception as e:
            self._log_build_step(f"Error creating macOS installer: {e}")
            return False
    
    def _create_linux_installer(self) -> bool:
        """Create a Linux installer (DEB or RPM)."""
        try:
            app_name = self.app_metadata.name.lower().replace(' ', '-')
            app_version = self.app_metadata.version
            
            # Determine build directory
            if self.build_config.packaging_tool == PackagingTool.PYINSTALLER:
                if self.build_config.build_options.get("one_file", True):
                    build_dir = self.build_config.output_dir / "dist"
                else:
                    build_dir = self.build_config.output_dir / "dist" / app_name.lower().replace(' ', '_')
            else:
                self._log_build_step("Unsupported packaging tool for Linux installer")
                return False
            
            # Check for FPM (Effing Package Management)
            fpm_path = shutil.which("fpm")
            if not fpm_path:
                self._log_build_step("FPM not found, installer creation skipped")
                return False
            
            # Create DEB package
            self._log_build_step("Creating Linux DEB package")
            
            # Create temporary directory structure
            temp_dir = Path(tempfile.mkdtemp())
            app_dir = temp_dir / "usr" / "local" / "bin" / app_name
            app_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy application files
            for item in build_dir.glob("*"):
                if item.is_file():
                    shutil.copy2(item, app_dir)
                else:
                    shutil.copytree(item, app_dir / item.name)
            
            # Create desktop entry
            desktop_dir = temp_dir / "usr" / "share" / "applications"
            desktop_dir.mkdir(parents=True, exist_ok=True)
            
            desktop_file = desktop_dir / f"{app_name}.desktop"
            with open(desktop_file, 'w') as f:
                f.write(f"""[Desktop Entry]
Name={self.app_metadata.name}
Comment={self.app_metadata.description}
Exec=/usr/local/bin/{app_name}/{app_name}
Icon=/usr/share/pixmaps/{app_name}.png
Terminal=false
Type=Application
Categories=Utility;Application;
""")
            
            # Copy icon
            if self.build_config.icon and self.build_config.icon.png:
                icons_dir = temp_dir / "usr" / "share" / "pixmaps"
                icons_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.build_config.icon.png, icons_dir / f"{app_name}.png")
            
            # Build DEB package
            cmd = [
                fpm_path,
                "-s", "dir",
                "-t", "deb",
                "-n", app_name,
                "-v", app_version,
                "--description", self.app_metadata.description,
                "--url", self.app_metadata.url,
                "--license", self.app_metadata.license,
                "--vendor", self.app_metadata.author,
                "-C", str(temp_dir),
                "-p", str(self.build_config.output_dir / "dist" / f"{app_name}_{app_version}.deb")
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                self._log_build_step(f"FPM failed: {process.stderr}")
                shutil.rmtree(temp_dir)
                return False
            
            # Build RPM package
            self._log_build_step("Creating Linux RPM package")
            cmd = [
                fpm_path,
                "-s", "dir",
                "-t", "rpm",
                "-n", app_name,
                "-v", app_version,
                "--description", self.app_metadata.description,
                "--url", self.app_metadata.url,
                "--license", self.app_metadata.license,
                "--vendor", self.app_metadata.author,
                "-C", str(temp_dir),
                "-p", str(self.build_config.output_dir / "dist" / f"{app_name}_{app_version}.rpm")
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            if process.returncode != 0:
                self._log_build_step(f"FPM failed for RPM: {process.stderr}")
                return False
            
            self._log_build_step("Linux installers created successfully")
            return True
        except Exception as e:
            self._log_build_step(f"Error creating Linux installer: {e}")
            return False
    
    def build(self) -> bool:
        """Build the application using the configured packaging tool."""
        try:
            # Prepare build environment
            if not self.prepare_build_environment():
                return False
            
            # Build based on packaging tool
            if self.build_config.packaging_
