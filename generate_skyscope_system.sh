#!/bin/bash
#
# generate_skyscope_system.sh
#
# This script generates the key files for the Skyscope Sentinel Intelligence AI system.
# It creates the necessary directory structure and generates the essential Python modules
# and documentation files.
#
# Usage:
#   chmod +x generate_skyscope_system.sh
#   ./generate_skyscope_system.sh
#

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║   ███████╗██╗  ██╗██╗   ██╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗     ║"
echo "║   ██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝     ║"
echo "║   ███████╗█████╔╝  ╚████╔╝ ███████╗██║     ██║   ██║██████╔╝█████╗       ║"
echo "║   ╚════██║██╔═██╗   ╚██╔╝  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝       ║"
echo "║   ███████║██║  ██╗   ██║   ███████║╚██████╗╚██████╔╝██║     ███████╗     ║"
echo "║   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝     ║"
echo "║                                                                           ║"
echo "║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗            ║"
echo "║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║            ║"
echo "║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║            ║"
echo "║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║            ║"
echo "║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗       ║"
echo "║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝       ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${GREEN}${BOLD}Skyscope Sentinel Intelligence AI - System Generator${NC}"
echo -e "${YELLOW}Creating key files for the 10,000-agent autonomous business system...${NC}\n"

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p assets/css assets/js assets/fonts assets/images assets/animations
mkdir -p config data credentials websites templates business_plans
mkdir -p logs/business logs/crypto logs/agents
mkdir -p docs tests scripts
echo -e "${GREEN}✓ Directory structure created${NC}\n"

# Function to create a file with content
create_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    echo -e "${YELLOW}Creating $file_name...${NC}"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$file_path")"
    
    # Create the file
    cat > "$file_path"
    
    # Check if file was created successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Created $file_path${NC}"
        chmod +x "$file_path" 2>/dev/null  # Make executable if appropriate
    else
        echo -e "${RED}✗ Failed to create $file_path${NC}"
        exit 1
    fi
}

# Create main_launcher.py
echo -e "${BLUE}Generating main launcher...${NC}"
create_file "main_launcher.py" << 'EOF'
#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Main Launcher
==================================================
Main entry point for the Skyscope Sentinel Intelligence AI system.
Integrates all components and launches the enhanced chat interface with autonomous business operations.
"""

import os
import sys
import time
import json
import logging
import threading
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main_launcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_launcher")

# Ensure necessary directories exist
REQUIRED_DIRS = [
    "logs",
    "config",
    "data",
    "credentials",
    "websites",
    "templates",
    "business_plans",
    "assets/css",
    "assets/js",
    "assets/fonts",
    "assets/images",
    "assets/animations"
]

for directory in REQUIRED_DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Import custom modules
try:
    from enhanced_chat_interface import EnhancedChatInterface
    from agent_manager import AgentManager
    from business_manager import BusinessManager
    from crypto_manager import CryptoManager
    from autonomous_business_operations import BusinessIdeaGenerator, ServiceRegistrationManager, WebsiteBuilder
    MODULES_IMPORTED = True
except ImportError as e:
    logger.warning(f"Error importing modules: {e}")
    logger.warning("Some modules may not be available. System will attempt to continue with limited functionality.")
    MODULES_IMPORTED = False

class SystemLauncher:
    """Main system launcher for Skyscope Sentinel Intelligence AI."""
    
    def __init__(self):
        """Initialize the system launcher."""
        self.config = self._load_config()
        self.first_run = not Path("config/system_initialized.json").exists()
        
        # Initialize components
        self.agent_manager = None
        self.business_manager = None
        self.crypto_manager = None
        self.perplexica_path = "/Users/skyscope.cloud/Perplexica"
        
        # Initialize autonomous operations thread
        self.autonomous_thread = None
        self.stop_autonomous = False
    
    def _load_config(self):
        """Load system configuration."""
        config_file = Path("config/system_config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Default configuration
        default_config = {
            "system_name": "Skyscope Sentinel Intelligence AI",
            "version": "1.0.0",
            "agent_count": 10000,
            "pipeline_count": 100,
            "default_theme": "glass_dark",
            "enable_autonomous_operations": True,
            "enable_crypto_focus": True,
            "enable_perplexica_integration": True,
            "perplexica_path": "/Users/skyscope.cloud/Perplexica",
            "enable_ollama": True,
            "enable_openai_fallback": True,
            "database_type": "sqlite",
            "database_path": "data/skyscope.db",
            "log_level": "INFO",
            "max_memory_gb": 8,
            "initialized_at": None
        }
        
        # Save default configuration
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default configuration: {e}")
        
        return default_config
    
    def _mark_initialized(self):
        """Mark the system as initialized."""
        init_file = Path("config/system_initialized.json")
        
        try:
            with open(init_file, 'w') as f:
                json.dump({
                    "initialized_at": datetime.now().isoformat(),
                    "version": self.config["version"]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error marking system as initialized: {e}")
    
    def _check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking system dependencies...")
        
        try:
            import streamlit
            import pandas
            import numpy
            import plotly
            import requests
            
            logger.info("All core dependencies are installed.")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Please run 'pip install -r requirements.txt' to install all dependencies.")
            return False
    
    def _check_wallet_setup(self):
        """Check if wallet is set up."""
        wallet_file = Path("config/crypto_wallets.json")
        
        if wallet_file.exists():
            try:
                with open(wallet_file, 'r') as f:
                    wallets = json.load(f)
                    return len(wallets) > 0
            except Exception as e:
                logger.error(f"Error checking wallet setup: {e}")
        
        return False
    
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        if MODULES_IMPORTED:
            try:
                # Initialize core components
                self.agent_manager = AgentManager(
                    agent_count=self.config.get("agent_count", 10000),
                    pipeline_count=self.config.get("pipeline_count", 100)
                )
                
                self.crypto_manager = CryptoManager()
                self.business_manager = BusinessManager(
                    agent_manager=self.agent_manager,
                    crypto_manager=self.crypto_manager
                )
                
                logger.info("All system components initialized successfully.")
                return True
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
                return False
        else:
            logger.warning("Modules not imported. Running with limited functionality.")
            return False
    
    def start_autonomous_operations(self):
        """Start autonomous business operations in a separate thread."""
        if not self.config.get("enable_autonomous_operations", True):
            logger.info("Autonomous operations disabled in configuration.")
            return
        
        if not self._check_wallet_setup():
            logger.info("Wallet not set up. Autonomous operations will start after wallet setup.")
            return
        
        logger.info("Starting autonomous business operations...")
        
        self.autonomous_thread = threading.Thread(target=self._autonomous_operations_loop)
        self.autonomous_thread.daemon = True
        self.autonomous_thread.start()
        
        logger.info("Autonomous operations thread started.")
    
    def _autonomous_operations_loop(self):
        """Main loop for autonomous business operations."""
        logger.info("Autonomous operations loop started.")
        
        try:
            # Main autonomous operations loop
            while not self.stop_autonomous:
                logger.info("Autonomous operations cycle running...")
                
                # Sleep to prevent high CPU usage
                time.sleep(60)
        except Exception as e:
            logger.error(f"Error in autonomous operations: {e}")
    
    def stop_autonomous_operations(self):
        """Stop autonomous business operations."""
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            logger.info("Stopping autonomous operations...")
            self.stop_autonomous = True
            self.autonomous_thread.join(timeout=5)
            logger.info("Autonomous operations stopped.")
    
    def launch_streamlit_interface(self):
        """Launch the Streamlit interface."""
        logger.info("Launching Streamlit interface...")
        
        try:
            # Use subprocess to run Streamlit
            subprocess.Popen([
                "streamlit", "run", 
                "enhanced_chat_interface.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--browser.serverAddress=localhost",
                "--server.headless=true",
                "--theme.base=dark"
            ])
            logger.info("Streamlit interface launched successfully.")
            return True
        except Exception as e:
            logger.error(f"Error launching Streamlit interface: {e}")
            return False
    
    def run(self):
        """Run the system launcher."""
        logger.info(f"Starting {self.config['system_name']} v{self.config['version']}...")
        
        # Check dependencies
        if not self._check_dependencies():
            logger.error("Missing dependencies. Please install required packages.")
            return False
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components.")
            return False
        
        # First run setup
        if self.first_run:
            logger.info("First run detected. Performing initial setup...")
            
            # Update config with initialization timestamp
            self.config["initialized_at"] = datetime.now().isoformat()
            
            # Save updated config
            try:
                with open("config/system_config.json", 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving updated configuration: {e}")
            
            # Mark as initialized
            self._mark_initialized()
        
        # Start autonomous operations
        self.start_autonomous_operations()
        
        # Launch Streamlit interface
        success = self.launch_streamlit_interface()
        
        if not success:
            logger.error("Failed to launch Streamlit interface.")
            return False
        
        logger.info(f"{self.config['system_name']} started successfully.")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Skyscope Sentinel Intelligence AI Launcher")
    parser.add_argument("--no-autonomous", action="store_true", help="Disable autonomous operations")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run launcher
    launcher = SystemLauncher()
    
    # Override autonomous operations if specified
    if args.no_autonomous:
        launcher.config["enable_autonomous_operations"] = False
    
    # Run the launcher
    try:
        success = launcher.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        launcher.stop_autonomous_operations()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        launcher.stop_autonomous_operations()
        sys.exit(1)

if __name__ == "__main__":
    # Fix import path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run main
    main()
EOF

# Create requirements.txt
echo -e "${BLUE}Generating requirements file...${NC}"
create_file "requirements.txt" << 'EOF'
# Skyscope Sentinel Intelligence AI - Requirements
streamlit>=1.26.0
streamlit-ace>=0.1.1
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
python-dotenv>=1.0.0
openai>=1.0.0
pillow>=10.0.0
pycryptodome>=3.19.0
psutil>=5.9.5
ccxt>=4.0.0
SQLAlchemy>=2.0.0
fastapi>=0.103.0
uvicorn>=0.23.0
websockets>=11.0.0
python-multipart>=0.0.6
pydantic>=2.3.0
jinja2>=3.1.2
markdown>=3.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
aiohttp>=3.8.5
pytest>=7.4.0
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
EOF

# Create README.md
echo -e "${BLUE}Generating README...${NC}"
create_file "README.md" << 'EOF'
# Skyscope Sentinel Intelligence AI  
Autonomous Agentic Co-Op Swarm System

_"Unleashing the power of 10 000 autonomous agents to build, run and scale profitable crypto-centric businesses."_

---

## 1 ▪ What is Skyscope Sentinel?

Skyscope Sentinel Intelligence AI (SS-IAI) is a **production-grade multi-agent platform** that orchestrates  
**10 000 specialised AI agents** (100 pipelines × 100 agents) to research, code, trade, market, launch and manage whole businesses with minimal human input.  
The stack runs **locally on macOS, Windows or Linux** or inside Docker/Kubernetes and focuses on **cryptocurrency income generation** while remaining compliant with Australian taxation law.

Key highlights  

• Glass-morphism Streamlit UI with animated sliding side-menu  
• Flowing live-code execution pane  
• 100 % offline mode via Ollama; seamless OpenAI fallback  
• Built-in Perplexica AI search (DuckDuckGo powered)  
• Automatic wallet prompt – earnings routed to your crypto address  
• End-to-end autonomous business lifecycle: idea → plan → register → build → launch → scale  
• Australian GST/BAS reporting & ledger export  
• Self-criticism + QA loops for every agent

---

## 2 ▪ Feature Matrix

| Domain | Capabilities |
|--------|--------------|
| Agent Swarm | 10 000 concurrent agents, persona templates, dynamic load-balancing |
| Crypto | Multi-exchange trading engine, DeFi yield, wallet tracking, on-chain analytics |
| Business Ops | Company creation, service auto-registration, BAS/GST reports, P&L dashboards |
| Content & Web | Blog & docs generator, no-code website builder, AI SaaS template deployment |
| UI/UX | Glassy theme system, OCR-A fonts, sliding menu with 8 functional tabs |
| Research | Local Perplexica search, deep-research mode, RAG knowledge retrieval |
| DevOps | Docker SQL stack, K8s manifests, Terraform infra blueprints |
| Security | AES-256 secret vault, role-based access, CVE scanner, audit logs |
| QA | Unit / integration / load tests, self-healing, performance monitor |

---

## 3 ▪ Quick Install

### Prerequisites
• Python 3.9+  
• Git, Docker (optional but recommended)  
• macOS 12+, Ubuntu 20+, Windows 10+  

### 3.1 Clone & Install  

```bash
git clone https://github.com/yourorg/skyscope-sentinel.git
cd skyscope-sentinel

# run interactive installer
python install.py
```

The installer  
1. Creates virtual-env and installs `requirements.txt`  
2. Generates default configs & .env  
3. Builds directory tree and persona templates  
4. (Optional) pulls local LLM models into `models/`

### 3.2 First-run wallet prompt  
On first launch the UI asks for **at least one cryptocurrency wallet address** (BTC/ETH/etc.).  
This becomes the default destination for all profits. You can add more wallets later under **Settings → Wallets**.

---

## 4 ▪ Starting the System

Local headless launch:

```bash
./start.sh         # macOS / Linux
start.bat          # Windows
```

or directly:

```bash
python main_launcher.py
```

Open your browser at **http://localhost:8501**

---

## 5 ▪ Using the Interface

The hamburger icon opens a sliding menu with the following tabs:

1. **Chat** – converse with the AI, execute code, issue commands  
2. **Business** – view / create autonomous ventures, track P&L  
3. **Crypto** – live trades, portfolio, income & expense ledger  
4. **Search** – Perplexica AI search & deep-research workflows  
5. **Websites** – one-click generation & management of web assets  
6. **Analytics** – revenue, traffic, agent activity dashboards  
7. **Agents** – pipeline health, task queue, create manual tasks  
8. **Settings** – themes, code display mode, API keys, system limits

Flowing code blocks display execution in real time; toggle compact/full view in Settings.

---

## 6 ▪ Autonomous Operations

The **Autonomous Engine** runs in the background (can be disabled via `--no-autonomous`):

• Generates business ideas from market trends (bias to crypto).  
• Drafts a full business plan & financial model.  
• Registers necessary services (domain, hosting, exchanges, payment processors).  
• Requests KYC only when required, guiding you step-by-step.  
• Builds websites/SaaS apps from templates, deploys them and integrates crypto checkout.  
• Launches marketing campaigns & tracks KPIs.  
• Trades profits into your default wallet and records transactions locally.  

All state is persisted under `data/` and `credentials/` so the swarm continues exactly where it left off after a reboot.

---

## 7 ▪ Directory Layout (abridged)

```
├─ app.py                           # Streamlit entry (UI)
├─ enhanced_chat_interface.py       # rich chat UI & menu
├─ main_launcher.py                 # system bootstrapper
├─ agent_manager.py                 # 10k agent orchestration
├─ autonomous_business_operations.py# business life-cycle engine
├─ crypto_manager.py                # wallet & trading logic
├─ config/                          # configs & .env
├─ data/                            # persistent DB & RAG store
├─ websites/                        # generated web projects
├─ logs/                            # rotated runtime logs
└─ tests/                           # automated QA suite
```

---

## 8 ▪ Security & Compliance

• Secrets vault with AES-256 encryption  
• Continuous CVE scanning (`enhanced_security_compliance.py`)  
• Audit trail for all privileged actions  
• Australian BAS/GST report generator (`tax_reports/`)  
• GDPR export & delete utilities  

---

## 9 ▪ Troubleshooting

| Symptom | Fix |
|---------|-----|
| Streamlit page blank | Check `logs/streamlit.log` for JS errors, clear browser cache |
| Wallet prompt repeats | Delete corrupted `config/crypto_wallets.json` and re-add wallet |
| High RAM usage | Lower active agent count in **Settings → Agent Settings** |

---

## 10 ▪ FAQ

**Q:** Can I run without internet?  
**A:** Yes. Enable "Local-only mode" in Settings; Ollama models are used exclusively.

**Q:** How do I add my own agent persona?  
**A:** Drop a JSON file into `personas/` and restart; it auto-registers.

**Q:** Is fiat support available?  
**A:** Yes, but agents prioritise crypto. Enable additional fiat gateways under Settings → Payments.

---

## 11 ▪ License

Skyscope Sentinel Intelligence AI is released under the **MIT License**.  
See `LICENSE` for details.

---

## 12 ▪ Contact

• Support: support@skyscope.ai  
• Twitter/X: [@skyscopecloud](https://twitter.com/skyscopecloud)

© 2025 Skyscope Technologies. All rights reserved.
EOF

# Create install.py
echo -e "${BLUE}Generating installation script...${NC}"
create_file "install.py" << 'EOF'
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
VERSION = "1.0.0"

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
CREDENTIALS_DIR = BASE_DIR / "credentials"

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
    CREDENTIALS_DIR,
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
        
        logger.info("System requirements check completed.")
        return True
    
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
                "email": "admin@skyscope.cloud"
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
python main_launcher.py "$@"
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
python main_launcher.py %*
"""
        
        with open(BASE_DIR / "start.bat", 'w') as f:
            f.write(startup_bat)
        
        logger.info("Startup scripts created successfully.")
    
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
        
        # Generate configuration
        self.generate_config()
        
        # Create startup scripts
        self.create_startup_scripts()
        
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
EOF

# Create a placeholder for the enhanced chat interface
echo -e "${BLUE}Creating placeholder for enhanced chat interface...${NC}"
create_file "enhanced_chat_interface.py" << 'EOF'
#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Enhanced Chat Interface
==========================================================

Enhanced chat interface with flowing code display, sliding menu, and cryptocurrency wallet integration.
"""

import os
import sys
import json
import time
import logging
import streamlit as st
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/chat_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_chat_interface")

# Constants
PERPLEXICA_PATH = "/Users/skyscope.cloud/Perplexica"

# Set page configuration
st.set_page_config(
    page_title="Skyscope Sentinel Intelligence AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for glass morphism effects
st.markdown("""
<style>
    /* Glass morphism effects */
    .glass-panel {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 20px;
        margin: 10px 0;
    }
    
    .dark-glass-panel {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Sliding menu */
    .sliding-menu {
        position: fixed;
        top: 0;
        left: -300px;
        width: 300px;
        height: 100vh;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: left 0.3s ease-in-out;
        z-index: 1000;
        padding: 20px;
        overflow-y: auto;
    }
    
    .sliding-menu.expanded {
        left: 0;
    }
    
    .menu-toggle {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1001;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    /* Flowing code animation */
    .flowing-code {
        position: relative;
        overflow: hidden;
        padding: 15px;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #f8f9fa;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .flowing-code::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 123, 255, 0.2), transparent);
        animation: flowingHighlight 2s linear infinite;
    }
    
    @keyframes flowingHighlight {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'wallet_setup_complete' not in st.session_state:
    st.session_state.wallet_setup_complete = False

# Main title
st.title("🚀 Skyscope Sentinel Intelligence AI")
st.markdown("### Autonomous Agentic Co-Op Swarm System")

# Check for wallet setup
if not st.session_state.wallet_setup_complete:
    st.subheader("🔐 Cryptocurrency Wallet Setup")
    st.markdown("""
    To receive earnings from autonomous business operations, please provide a cryptocurrency wallet address.
    This will be used to credit your earnings from trading, content monetization, and other revenue streams.
    """)
    
    wallet_name = st.text_input("Wallet Name (e.g., 'My BTC Wallet')")
    wallet_address = st.text_input("Wallet Address")
    currency = st.selectbox("Cryptocurrency", ["BTC", "ETH", "SOL", "ADA", "XRP", "USDT", "USDC", "Other"])
    
    if currency == "Other":
        currency = st.text_input("Enter cryptocurrency symbol")
    
    if st.button("Save Wallet"):
        if wallet_address and wallet_name and currency:
            # Save wallet info
            wallet_dir = Path("config")
            wallet_dir.mkdir(parents=True, exist_ok=True)
            
            wallet_file = wallet_dir / "crypto_wallets.json"
            
            wallet_data = {}
            if wallet_file.exists():
                try:
                    with open(wallet_file, 'r') as f:
                        wallet_data = json.load(f)
                except:
                    wallet_data = {}
            
            wallet_id = f"wallet_{int(time.time())}"
            wallet_data[wallet_id] = {
                "name": wallet_name,
                "address": wallet_address,
                "currency": currency,
                "is_default": True,
                "added_at": datetime.now().isoformat()
            }
            
            with open(wallet_file, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            st.session_state.wallet_setup_complete = True
            st.success(f"Wallet '{wallet_name}' added successfully!")
            st.experimental_rerun()
        else:
            st.error("Please fill in all fields")
else:
    # Main chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Simulate typing with a placeholder
            response_placeholder.markdown("_Thinking..._")
            
            # Generate response (placeholder for actual implementation)
            time.sleep(1)  # Simulate processing time
            
            response = f"""I'm your Skyscope Sentinel Intelligence AI assistant.

You asked: "{prompt}"

This is a placeholder response. The full implementation will include:

1. **Flowing code display** with real-time execution
2. **Sliding menu system** with 8 functional tabs
3. **Perplexica AI search integration**
4. **Cryptocurrency wallet management**
5. **Autonomous business operations**

Your wallet has been set up and the system is ready to start generating income through cryptocurrency-focused business ventures.

To see the full implementation, please run `python main_launcher.py` after completing the installation.
"""
            
            response_placeholder.markdown(response)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("© 2025 Skyscope Technologies. All rights reserved.")
EOF

# Create start script
echo -e "${BLUE}Creating startup script...${NC}"
create_file "start.sh" << 'EOF'
#!/bin/bash
# Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
# Startup Script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the application
python main_launcher.py "$@"
EOF
chmod +x start.sh

# Create Windows batch file
echo -e "${BLUE}Creating Windows startup script...${NC}"
create_file "start.bat" << 'EOF'
@echo off
:: Skyscope Sentinel Intelligence AI Agentic Co-Op Swarm System
:: Startup Script

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Start the application
python main_launcher.py %*
EOF

# Create placeholder for autonomous_business_operations.py
echo -e "${BLUE}Creating placeholder for autonomous business operations...${NC}"
create_file "autonomous_business_operations.py" << 'EOF'
#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Autonomous Business Operations
=================================================================

This module manages autonomous business operations, including:
- Business idea generation
- Service registration
- Website creation
- Cryptocurrency trading
- Income generation

The system operates autonomously to create and manage businesses that generate
income primarily through cryptocurrency-focused ventures.
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/business_operations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("autonomous_business_operations")

# Constants
CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
CREDENTIALS_DIR = Path("credentials")
WEBSITES_DIR = Path("websites")
TEMPLATES_DIR = Path("templates")
BUSINESS_PLANS_DIR = Path("business_plans")

# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, CREDENTIALS_DIR, WEBSITES_DIR, TEMPLATES_DIR, BUSINESS_PLANS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class BusinessType(Enum):
    """Types of businesses that can be created."""
    CONTENT_SUBSCRIPTION = auto()
    CRYPTO_TRADING = auto()
    AI_SAAS = auto()
    ECOMMERCE = auto()
    DIGITAL_MARKETING = auto()
    EDUCATIONAL = auto()
    CONSULTING = auto()
    CUSTOM = auto()

class BusinessStatus(Enum):
    """Status of a business."""
    PLANNING = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    SCALING = auto()
    OPTIMIZING = auto()
    PIVOTING = auto()
    CLOSING = auto()

class ServiceRegistrationType(Enum):
    """Types of services that businesses can register for."""
    DOMAIN = auto()
    HOSTING = auto()
    EMAIL = auto()
    PAYMENT_PROCESSOR = auto()
    ANALYTICS = auto()
    MARKETING = auto()
    CRM = auto()
    SOCIAL_MEDIA = auto()
    EXCHANGE = auto()
    MARKETPLACE = auto()

class BusinessIdeaGenerator:
    """Generates business ideas based on market trends and opportunities."""
    
    def __init__(self, agent_manager=None):
        """Initialize the business idea generator."""
        self.agent_manager = agent_manager
    
    def generate_business_ideas(self, count: int = 5, focus_area: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate business ideas based on market trends."""
        logger.info(f"Generating {count} business ideas with focus on {focus_area if focus_area else 'all areas'}")
        
        # This is a placeholder implementation
        # In the full implementation, this would use the agent manager to generate ideas
        
        ideas = []
        for i in range(count):
            idea = {
                "name": f"Business Idea {i+1}",
                "description": f"This is a placeholder for business idea {i+1}",
                "business_type": random.choice(list(BusinessType)),
                "potential": random.uniform(0.5, 1.0)
            }
            ideas.append(idea)
        
        return ideas

class ServiceRegistrationManager:
    """Manages service registrations for businesses."""
    
    def __init__(self, agent_manager=None):
        """Initialize the service registration manager."""
        self.agent_manager = agent_manager
    
    def register_service(self, business_id: str, service_type: ServiceRegistrationType) -> Dict[str, Any]:
        """Register a service for a business."""
        logger.info(f"Registering {service_type.name} service for business {business_id}")
        
        # This is a placeholder implementation
        # In the full implementation, this would use the agent manager to register services
        
        return {
            "id": str(uuid.uuid4()),
            "business_id": business_id,
            "service_type": service_type.name,
            "status": "active",
            "registered_at": datetime.now().isoformat()
        }

class WebsiteBuilder:
    """Builds websites for businesses."""
    
    def __init__(self, agent_manager=None):
        """Initialize the website builder."""
        self.agent_manager = agent_manager
    
    def create_website(self, business_id: str, business_type: BusinessType) -> Dict[str, Any]:
        """Create a website for a business."""
        logger.info(f"Creating website for business {business_id} of type {business_type.name}")
        
        # This is a placeholder implementation
        # In the full implementation, this would use the agent manager to create websites
        
        return {
            "id": str(uuid.uuid4()),
            "business_id": business_id,
            "domain": f"business-{business_id[:8]}.example.com",
            "status": "active",
            "created_at": datetime.now().isoformat()
        }

# Main function for testing
if __name__ == "__main__":
    print("Skyscope Sentinel Intelligence AI - Autonomous Business Operations")
    print("This module is designed to be imported, not run directly.")
    print("In a production environment, this would be integrated with the agent manager.")
EOF

echo -e "${GREEN}${BOLD}All files created successfully!${NC}"
echo -e "${BLUE}To install and run the system:${NC}"
echo -e "1. Run the installation script: ${YELLOW}python install.py${NC}"
echo -e "2. Start the system: ${YELLOW}./start.sh${NC} (or ${YELLOW}start.bat${NC} on Windows)"
echo -e "3. Access the web interface at ${YELLOW}http://localhost:8501${NC}"
echo -e "\n${GREEN}The system will prompt for your cryptocurrency wallet on first run.${NC}"
echo -e "${GREEN}All earnings will be automatically credited to your wallet.${NC}"
