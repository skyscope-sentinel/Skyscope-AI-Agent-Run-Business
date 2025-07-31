#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
Installation Script

This script sets up the Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
by installing dependencies, creating necessary directories, generating default configurations,
and initializing the system for first-time use.

Usage:
    python install.py [--no-deps] [--skip-models] [--advanced]

Options:
    --no-deps       Skip dependency installation
    --skip-models   Skip downloading large model files
    --advanced      Enable advanced installation options
"""

import os
import sys
import time
import json
import shutil
import logging
import platform
import subprocess
import argparse
import tempfile
import urllib.request
import zipfile
import random
import string
import getpass
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("skyscope_install.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("skyscope_installer")

# Constants
COMPANY_NAME = "Skyscope Sentinel Intelligence"
COMPANY_ABN = "11287984779"
COMPANY_EMAIL = "admin@skyscope.cloud"
VERSION = "1.0.0"
GITHUB_REPO = "https://github.com/skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI"

# Directory structure
BASE_DIR = Path.cwd()
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"
MODELS_DIR = BASE_DIR / "models"
PERSONAS_DIR = BASE_DIR / "personas"
BUSINESS_DIR = BASE_DIR / "business"
CRYPTO_DIR = BASE_DIR / "crypto"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"
WALLETS_DIR = BASE_DIR / "wallets"
TRANSACTIONS_DIR = BASE_DIR / "transactions"
STRATEGIES_DIR = BASE_DIR / "strategies"
CREDENTIALS_DIR = BASE_DIR / "credentials"
ACCOUNTS_DIR = BASE_DIR / "accounts"
OPERATIONS_DIR = BASE_DIR / "operations"

# All directories to create
DIRECTORIES = [
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    CACHE_DIR,
    MODELS_DIR,
    PERSONAS_DIR,
    BUSINESS_DIR,
    CRYPTO_DIR,
    REPORTS_DIR,
    TEMPLATES_DIR,
    WALLETS_DIR,
    TRANSACTIONS_DIR,
    STRATEGIES_DIR,
    CREDENTIALS_DIR,
    ACCOUNTS_DIR,
    OPERATIONS_DIR,
    DATA_DIR / "agents",
    DATA_DIR / "business",
    DATA_DIR / "crypto",
    LOGS_DIR / "agents",
    LOGS_DIR / "business",
    LOGS_DIR / "crypto",
    REPORTS_DIR / "business",
    REPORTS_DIR / "financial",
]

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███████╗██╗  ██╗██╗   ██╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗     ║
║   ██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝     ║
║   ███████╗█████╔╝  ╚████╔╝ ███████╗██║     ██║   ██║██████╔╝█████╗       ║
║   ╚════██║██╔═██╗   ╚██╔╝  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝       ║
║   ███████║██║  ██╗   ██║   ███████║╚██████╗╚██████╔╝██║     ███████╗     ║
║   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝     ║
║                                                                           ║
║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗            ║
║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║            ║
║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║            ║
║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║            ║
║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗       ║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

  AI Agentic Co-Op Swarm System Installation
  Version: {version}
  
  "Unleashing the Power of 10,000 AI Agents"
"""

class Installer:
    """Installer for the Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System."""
    
    def __init__(self, skip_deps=False, skip_models=False, advanced=False):
        """Initialize the installer."""
        self.skip_deps = skip_deps
        self.skip_models = skip_models
        self.advanced = advanced
        self.start_time = time.time()
        self.system_info = self.get_system_info()
        self.config = {}
        self.env_vars = {}
    
    def get_system_info(self):
        """Get system information."""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "memory": self.get_system_memory(),
            "hostname": platform.node(),
            "username": getpass.getuser()
        }
    
    def get_system_memory(self):
        """Get system memory in GB."""
        try:
            if platform.system() == "Darwin":  # macOS
                output = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
                mem_bytes = int(output.strip())
                return round(mem_bytes / (1024**3), 2)  # Convert to GB
            elif platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            mem_kb = int(line.split()[1])
                            return round(mem_kb / (1024**2), 2)  # Convert to GB
            elif platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong)
                    ]
                memory_status = MEMORYSTATUS()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
                return round(memory_status.dwTotalPhys / (1024**3), 2)  # Convert to GB
        except Exception as e:
            logger.error(f"Error getting system memory: {e}")
            return "Unknown"
    
    def print_banner(self):
        """Print the installation banner."""
        print(BANNER.format(version=VERSION))
        print(f"  System: {self.system_info['os']} {self.system_info['os_version']} ({self.system_info['architecture']})")
        print(f"  Python: {self.system_info['python_version']}")
        print(f"  Memory: {self.system_info['memory']} GB")
        print(f"  User: {self.system_info['username']}@{self.system_info['hostname']}")
        print("\n" + "="*80 + "\n")
    
    def check_system_requirements(self):
        """Check if the system meets the requirements."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')))
        if python_version < (3, 8):
            logger.error(f"Python 3.8 or higher is required. You have {platform.python_version()}")
            return False
        
        # Check memory
        if isinstance(self.system_info['memory'], (int, float)) and self.system_info['memory'] < 8:
            logger.warning("At least 8GB of RAM is recommended. Performance may be limited.")
        
        # Check operating system
        if self.system_info['os'] not in ["Darwin", "Linux", "Windows"]:
            logger.warning(f"Unsupported operating system: {self.system_info['os']}. Proceed with caution.")
        
        # Check disk space
        free_space = self.get_free_disk_space()
        if free_space < 10:
            logger.warning(f"Low disk space: {free_space:.2f} GB free. At least 10GB is recommended.")
        
        logger.info("System requirements check completed.")
        return True
    
    def get_free_disk_space(self):
        """Get free disk space in GB."""
        try:
            if platform.system() == "Windows":
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(os.getcwd()), None, None, ctypes.pointer(free_bytes)
                )
                return free_bytes.value / (1024**3)
            else:
                st = os.statvfs(os.getcwd())
                return (st.f_bavail * st.f_frsize) / (1024**3)
        except Exception as e:
            logger.error(f"Error getting free disk space: {e}")
            return float('inf')  # Assume infinite space on error
    
    def create_directories(self):
        """Create the necessary directory structure."""
        logger.info("Creating directory structure...")
        
        for directory in DIRECTORIES:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
        
        logger.info("Directory structure created successfully.")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        if self.skip_deps:
            logger.info("Skipping dependency installation as requested.")
            return True
        
        logger.info("Installing Python dependencies...")
        
        # Check if requirements.txt exists
        requirements_file = BASE_DIR / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found. Cannot install dependencies.")
            return False
        
        try:
            # Install dependencies
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def download_models(self):
        """Download necessary model files."""
        if self.skip_models:
            logger.info("Skipping model downloads as requested.")
            return True
        
        logger.info("Checking for required models...")
        
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # For demonstration, we'll just create placeholder files
        # In a real implementation, this would download actual models
        
        model_placeholders = [
            "sentence-transformer-model.txt",
            "embedding-model.txt",
            "tokenizer-config.txt"
        ]
        
        for model_file in model_placeholders:
            model_path = MODELS_DIR / model_file
            if not model_path.exists():
                with open(model_path, 'w') as f:
                    f.write(f"Placeholder for {model_file}\nDownload date: {datetime.now().isoformat()}")
                logger.info(f"Created placeholder for {model_file}")
        
        logger.info("Model files prepared successfully.")
        return True
    
    def generate_config(self):
        """Generate default configuration files."""
        logger.info("Generating configuration files...")
        
        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate main configuration
        main_config = {
            "version": VERSION,
            "company": {
                "name": COMPANY_NAME,
                "abn": COMPANY_ABN,
                "email": COMPANY_EMAIL
            },
            "system": {
                "max_pipelines": 100,
                "agents_per_pipeline": 100,
                "default_model": "gpt-4o",
                "default_theme": "dark_glass",
                "log_level": "INFO",
                "cache_dir": str(CACHE_DIR),
                "data_dir": str(DATA_DIR),
                "models_dir": str(MODELS_DIR),
                "installation_date": datetime.now().isoformat()
            },
            "ui": {
                "port": 8501,
                "title": "Skyscope Sentinel Intelligence",
                "favicon": "static/favicon.ico",
                "logo": "static/logo.png",
                "theme": "dark_glass",
                "animation_speed": 1.0,
                "show_terminal_in_sidebar": False
            },
            "security": {
                "require_login": False,
                "encryption_key": self.generate_secret_key(32),
                "session_expiry": 3600
            }
        }
        
        # Write main configuration
        with open(CONFIG_DIR / "config.json", 'w') as f:
            json.dump(main_config, f, indent=2)
        
        # Generate API configuration
        api_config = {
            "openai": {
                "api_key": "YOUR_OPENAI_API_KEY",
                "organization": "",
                "default_model": "gpt-4o"
            },
            "anthropic": {
                "api_key": "YOUR_ANTHROPIC_API_KEY",
                "default_model": "claude-3-opus"
            },
            "google": {
                "api_key": "YOUR_GOOGLE_API_KEY",
                "default_model": "gemini-pro"
            },
            "huggingface": {
                "api_key": "YOUR_HUGGINGFACE_API_KEY"
            },
            "serper": {
                "api_key": "YOUR_SERPER_API_KEY"
            },
            "browserless": {
                "api_key": "YOUR_BROWSERLESS_API_KEY"
            }
        }
        
        # Write API configuration
        with open(CONFIG_DIR / "api_config.json", 'w') as f:
            json.dump(api_config, f, indent=2)
        
        # Generate .env file
        env_content = f"""# Skyscope Sentinel Intelligence Environment Variables
# Generated on {datetime.now().isoformat()}

# API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
SERPER_API_KEY=your-serper-api-key
BROWSERLESS_API_KEY=your-browserless-api-key

# System Configuration
MAX_PIPELINES=100
AGENTS_PER_PIPELINE=100
DEFAULT_MODEL=gpt-4o
LOG_LEVEL=INFO

# Security
ENCRYPTION_KEY={self.generate_secret_key(32)}
REQUIRE_LOGIN=false

# Paths
CACHE_DIR={CACHE_DIR}
DATA_DIR={DATA_DIR}
MODELS_DIR={MODELS_DIR}
"""
        
        # Write .env file
        with open(BASE_DIR / ".env", 'w') as f:
            f.write(env_content)
        
        logger.info("Configuration files generated successfully.")
    
    def generate_secret_key(self, length=32):
        """Generate a random secret key."""
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choice(chars) for _ in range(length))
    
    def create_default_personas(self):
        """Create default persona templates."""
        logger.info("Creating default personas...")
        
        # Create personas directory if it doesn't exist
        PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
        
        personas = [
            {
                "name": "ResearchSpecialist",
                "role": "researcher",
                "personality_traits": ["curious", "methodical", "detail-oriented"],
                "expertise": ["research", "data analysis", "fact-checking"],
                "backstory": "You are a research specialist with a background in academic research and data analysis. You excel at finding information, validating sources, and synthesizing complex data into clear insights.",
                "tone": "professional",
                "communication_style": "clear and precise",
                "values": ["accuracy", "thoroughness", "objectivity"],
                "capabilities": [
                    {
                        "name": "Information Retrieval",
                        "skill_level": 0.9,
                        "description": "Finding and extracting relevant information from various sources",
                        "keywords": ["research", "find", "search", "information", "data"]
                    },
                    {
                        "name": "Data Analysis",
                        "skill_level": 0.8,
                        "description": "Analyzing data to extract meaningful patterns and insights",
                        "keywords": ["analyze", "analysis", "data", "patterns", "trends"]
                    },
                    {
                        "name": "Fact Checking",
                        "skill_level": 0.9,
                        "description": "Verifying the accuracy of information against reliable sources",
                        "keywords": ["verify", "check", "validate", "accuracy", "facts"]
                    }
                ]
            },
            {
                "name": "CodeExpert",
                "role": "developer",
                "personality_traits": ["logical", "creative", "detail-oriented"],
                "expertise": ["software development", "algorithm design", "system architecture"],
                "backstory": "You are a seasoned software developer with expertise across multiple programming languages and paradigms. You excel at writing clean, efficient code and solving complex technical challenges.",
                "tone": "technical",
                "communication_style": "clear and concise",
                "values": ["efficiency", "elegance", "functionality"],
                "capabilities": [
                    {
                        "name": "Software Development",
                        "skill_level": 0.9,
                        "description": "Writing high-quality code in various programming languages",
                        "keywords": ["code", "program", "develop", "software", "application"]
                    },
                    {
                        "name": "Algorithm Design",
                        "skill_level": 0.8,
                        "description": "Creating efficient algorithms to solve complex problems",
                        "keywords": ["algorithm", "design", "optimize", "efficiency", "complexity"]
                    },
                    {
                        "name": "System Architecture",
                        "skill_level": 0.7,
                        "description": "Designing robust and scalable system architectures",
                        "keywords": ["architecture", "system", "design", "structure", "scalable"]
                    }
                ]
            },
            {
                "name": "BusinessStrategist",
                "role": "business_strategist",
                "personality_traits": ["analytical", "forward-thinking", "pragmatic"],
                "expertise": ["business strategy", "market analysis", "revenue optimization"],
                "backstory": "You are a business strategist with experience in developing and implementing successful business strategies. You excel at identifying market opportunities and creating actionable plans for growth.",
                "tone": "professional",
                "communication_style": "clear and persuasive",
                "values": ["growth", "efficiency", "innovation"],
                "capabilities": [
                    {
                        "name": "Strategic Planning",
                        "skill_level": 0.9,
                        "description": "Developing comprehensive business strategies",
                        "keywords": ["strategy", "plan", "business", "growth", "goals"]
                    },
                    {
                        "name": "Market Analysis",
                        "skill_level": 0.8,
                        "description": "Analyzing market trends and competitive landscapes",
                        "keywords": ["market", "analysis", "trends", "competition", "industry"]
                    },
                    {
                        "name": "Revenue Optimization",
                        "skill_level": 0.7,
                        "description": "Identifying opportunities to optimize revenue streams",
                        "keywords": ["revenue", "profit", "optimization", "monetization", "income"]
                    }
                ]
            },
            {
                "name": "ContentCreator",
                "role": "content_creator",
                "personality_traits": ["creative", "empathetic", "adaptable"],
                "expertise": ["content creation", "copywriting", "storytelling"],
                "backstory": "You are a versatile content creator with a talent for crafting engaging and persuasive content across various formats and topics. You excel at adapting your voice to different audiences and purposes.",
                "tone": "conversational",
                "communication_style": "engaging and clear",
                "values": ["creativity", "clarity", "impact"],
                "capabilities": [
                    {
                        "name": "Copywriting",
                        "skill_level": 0.9,
                        "description": "Writing persuasive marketing and advertising copy",
                        "keywords": ["copy", "write", "persuasive", "marketing", "advertising"]
                    },
                    {
                        "name": "Blog Writing",
                        "skill_level": 0.8,
                        "description": "Creating engaging and informative blog content",
                        "keywords": ["blog", "article", "content", "writing", "post"]
                    },
                    {
                        "name": "Social Media Content",
                        "skill_level": 0.7,
                        "description": "Crafting effective social media posts and campaigns",
                        "keywords": ["social", "media", "post", "content", "campaign"]
                    }
                ]
            },
            {
                "name": "CryptoExpert",
                "role": "crypto_trader",
                "personality_traits": ["analytical", "cautious", "forward-thinking"],
                "expertise": ["cryptocurrency", "blockchain", "trading strategies"],
                "backstory": "You are a cryptocurrency expert with deep knowledge of blockchain technology and trading strategies. You excel at analyzing market trends and identifying opportunities in the crypto space.",
                "tone": "knowledgeable",
                "communication_style": "clear and educational",
                "values": ["accuracy", "innovation", "security"],
                "capabilities": [
                    {
                        "name": "Crypto Market Analysis",
                        "skill_level": 0.9,
                        "description": "Analyzing cryptocurrency market trends and patterns",
                        "keywords": ["crypto", "market", "analysis", "trends", "bitcoin"]
                    },
                    {
                        "name": "Blockchain Technology",
                        "skill_level": 0.8,
                        "description": "Understanding and explaining blockchain concepts and applications",
                        "keywords": ["blockchain", "technology", "distributed", "ledger", "smart contracts"]
                    },
                    {
                        "name": "Trading Strategies",
                        "skill_level": 0.7,
                        "description": "Developing and implementing cryptocurrency trading strategies",
                        "keywords": ["trading", "strategy", "buy", "sell", "investment"]
                    }
                ]
            }
        ]
        
        # Write persona files
        for persona in personas:
            with open(PERSONAS_DIR / f"{persona['name'].lower()}.json", 'w') as f:
                json.dump(persona, f, indent=2)
        
        logger.info(f"Created {len(personas)} default personas.")
    
    def create_startup_scripts(self):
        """Create startup scripts for different platforms."""
        logger.info("Creating startup scripts...")
        
        # Create startup script for macOS/Linux
        startup_sh = """#!/bin/bash
# Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
# Startup Script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the application
streamlit run app.py "$@"
"""
        
        with open(BASE_DIR / "start.sh", 'w') as f:
            f.write(startup_sh)
        
        # Make the script executable
        os.chmod(BASE_DIR / "start.sh", 0o755)
        
        # Create startup script for Windows
        startup_bat = """@echo off
:: Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
:: Startup Script

:: Activate virtual environment if it exists
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
) else if exist .venv\\Scripts\\activate.bat (
    call .venv\\Scripts\\activate.bat
)

:: Start the application
streamlit run app.py %*
"""
        
        with open(BASE_DIR / "start.bat", 'w') as f:
            f.write(startup_bat)
        
        logger.info("Startup scripts created successfully.")
    
    def create_readme(self):
        """Create README file."""
        logger.info("Creating README file...")
        
        readme_content = f"""# {COMPANY_NAME}

## AI Agentic Co-Op Swarm System

Version: {VERSION}

## Overview

Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System is an advanced multi-agent AI platform that orchestrates 10,000 specialized expert agents (100 pipelines × 100 agents) to research, code, analyze, automate browsers, manipulate files, and perform business operations - all through one sleek dark-themed UI.

## Features

- **AI-Driven Automation**: Hierarchical & concurrent multi-agent swarms with 10,000 configurable expert roles
- **Model Flexibility**: Integration with OpenAI models (GPT-4o, GPT-4o-mini, DALL-E, Whisper, TTS)
- **Advanced UI**: Two-pane Chat + Pop-in Code Window with futuristic glass-morphism design
- **Business Operations**: Autonomous business management and cryptocurrency trading capabilities
- **Browser Automation**: Natural-language commands to control web browsers
- **File Management**: Upload/preview/manipulate files directly from the UI
- **Local Filesystem Access**: Toggle-controlled read/write utilities with directory & extension safelists
- **Terminal Integration**: Execute shell commands directly from the UI

## Installation

```bash
# 1. Clone the repository
git clone {GITHUB_REPO}
cd Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI

# 2. Run the installer
python install.py

# 3. Start the application
./start.sh  # On macOS/Linux
start.bat   # On Windows
```

## Configuration

Edit the configuration files in the `config` directory:

- `config.json`: Main system configuration
- `api_config.json`: API keys and model settings
- `.env`: Environment variables

## Usage

1. Start the application using the startup script
2. Access the web UI at http://localhost:8501
3. Configure your API keys in the Settings page
4. Start creating agent pipelines and business operations

## License

Released under the MIT License - see `LICENSE` for full text.

## Contact

- General inquiries: {COMPANY_EMAIL}
- Issues/Bugs: GitHub Issues

---

*Powered by OpenAI Unofficial SDK ⚡ &nbsp;Built with ♥ by Skyscope Sentinel Intelligence*
"""
        
        with open(BASE_DIR / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info("README file created successfully.")
    
    def run_installation(self):
        """Run the complete installation process."""
        self.print_banner()
        
        logger.info("Starting installation of Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System...")
        
        # Check system requirements
        if not self.check_system_requirements():
            logger.error("System requirements check failed. Installation aborted.")
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.skip_deps and not self.install_dependencies():
            logger.error("Dependency installation failed. Installation aborted.")
            return False
        
        # Download models
        if not self.skip_models and not self.download_models():
            logger.error("Model download failed. Installation will continue, but some features may not work.")
        
        # Generate configuration
        self.generate_config()
        
        # Create default personas
        self.create_default_personas()
        
        # Create startup scripts
        self.create_startup_scripts()
        
        # Create README
        self.create_readme()
        
        # Calculate installation time
        elapsed_time = time.time() - self.start_time
        
        logger.info(f"Installation completed successfully in {elapsed_time:.2f} seconds.")
        
        # Print success message
        print("\n" + "="*80)
        print(f"\n✅ {COMPANY_NAME} installed successfully!")
        print("\nTo start the application, run:")
        if self.system_info['os'] == "Windows":
            print("  start.bat")
        else:
            print("  ./start.sh")
        print("\nAccess the web UI at: http://localhost:8501")
        print("\nFor more information, see the README.md file.")
        print("\n" + "="*80 + "\n")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=f"Installer for {COMPANY_NAME}")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading large model files")
    parser.add_argument("--advanced", action="store_true", help="Enable advanced installation options")
    
    args = parser.parse_args()
    
    installer = Installer(
        skip_deps=args.no_deps,
        skip_models=args.skip_models,
        advanced=args.advanced
    )
    
    try:
        installer.run_installation()
    except KeyboardInterrupt:
        logger.warning("Installation interrupted by user.")
        print("\nInstallation interrupted. You can resume by running the installer again.")
        return 1
    except Exception as e:
        logger.error(f"Installation failed with error: {e}")
        print(f"\n❌ Installation failed: {e}")
        print("\nCheck the log file (skyscope_install.log) for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
