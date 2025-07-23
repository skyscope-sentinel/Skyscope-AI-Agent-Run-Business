#!/bin/bash
#
# generate_enhanced_system.sh
#
# This script generates all the enhanced Skyscope Sentinel Intelligence AI system files
# in the current directory. It creates the necessary directory structure and all files
# with their complete content.
#
# Usage:
#   ./generate_enhanced_system.sh
#
# Run from the root directory of your repository.
#

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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
echo -e "${GREEN}Enhanced System Generator${NC}"
echo -e "${YELLOW}Creating all files for the Skyscope Sentinel Intelligence AI system...${NC}\n"

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p assets/css assets/js assets/fonts assets/images assets/animations
mkdir -p config data credentials websites templates business_plans
mkdir -p logs/business logs/crypto logs/agents
mkdir -p docs
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
    else
        echo -e "${RED}✗ Failed to create $file_path${NC}"
    fi
}

# Create enhanced_chat_interface.py
echo -e "${BLUE}Generating enhanced chat interface...${NC}"
create_file "enhanced_chat_interface.py" << 'EOF'
import os
import sys
import json
import time
import uuid
import base64
import hashlib
import asyncio
import threading
import logging
import re
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

import streamlit as st
import streamlit.components.v1 as components
from streamlit_ace import st_ace
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageFilter, ImageEnhance

# Import custom components
try:
    from ui_themes import ThemeManager, UITheme
    from agent_manager import AgentManager
    from business_manager import BusinessManager
    from crypto_manager import CryptoManager
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from performance_monitor import PerformanceMonitor
except ImportError:
    print("Warning: Some modules could not be imported. Running in standalone mode.")

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
ASSETS_DIR = Path("assets")
CSS_DIR = ASSETS_DIR / "css"
JS_DIR = ASSETS_DIR / "js"
FONTS_DIR = ASSETS_DIR / "fonts"
IMAGES_DIR = ASSETS_DIR / "images"
ANIMATIONS_DIR = ASSETS_DIR / "animations"

# Ensure directories exist
for directory in [ASSETS_DIR, CSS_DIR, JS_DIR, FONTS_DIR, IMAGES_DIR, ANIMATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class MenuTab(Enum):
    """Menu tab options."""
    CHAT = auto()
    BUSINESS = auto()
    CRYPTO = auto()
    SEARCH = auto()
    WEBSITES = auto()
    SETTINGS = auto()
    ANALYTICS = auto()
    AGENTS = auto()

class CodeDisplayMode(Enum):
    """Code display modes."""
    HIDDEN = "hidden"
    COMPACT = "compact"
    FLOWING = "flowing"
    FULL = "full"

class ChatMessageType(Enum):
    """Types of chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CODE = "code"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"
    THINKING = "thinking"
    EXECUTION = "execution"

@st.cache_resource
class PerplexicaIntegration:
    """Integration with Perplexica AI search engine."""
    
    def __init__(self):
        """Initialize the Perplexica integration."""
        self.perplexica_path = Path(PERPLEXICA_PATH)
        self.available = self.perplexica_path.exists()
        self.process = None
        
        if self.available:
            logger.info(f"Perplexica found at {self.perplexica_path}")
            # Add Perplexica to Python path
            sys.path.append(str(self.perplexica_path))
            try:
                # Try to import Perplexica
                import perplexica
                self.perplexica = perplexica
                logger.info("Perplexica imported successfully")
            except ImportError:
                logger.warning("Could not import Perplexica module")
                self.perplexica = None
        else:
            logger.warning(f"Perplexica not found at {self.perplexica_path}")
            self.perplexica = None
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a search using Perplexica."""
        if not self.available or not self.perplexica:
            logger.warning("Perplexica not available for search")
            return [{"title": "Perplexica not available", "url": "", "snippet": "Search engine integration not available."}]
        
        try:
            # Use Perplexica's search function
            results = self.perplexica.search(query, num_results=num_results)
            return results
        except Exception as e:
            logger.error(f"Error searching with Perplexica: {e}")
            return [{"title": "Search Error", "url": "", "snippet": f"Error: {str(e)}"}]
    
    def deep_research(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Perform deep research on a topic using Perplexica."""
        if not self.available or not self.perplexica:
            logger.warning("Perplexica not available for deep research")
            return {"error": "Perplexica not available", "results": []}
        
        try:
            # Use Perplexica's deep research function if available
            if hasattr(self.perplexica, "deep_research"):
                research = self.perplexica.deep_research(topic, depth=depth)
                return research
            else:
                # Fall back to regular search with expanded queries
                expanded_queries = [
                    f"{topic} overview",
                    f"{topic} detailed explanation",
                    f"{topic} latest developments",
                    f"{topic} analysis",
                    f"{topic} examples"
                ]
                
                results = {}
                for query in expanded_queries:
                    results[query] = self.search(query, num_results=3)
                
                return {"topic": topic, "depth": depth, "results": results}
        except Exception as e:
            logger.error(f"Error performing deep research with Perplexica: {e}")
            return {"error": str(e), "results": []}

class CryptoWalletManager:
    """Manager for cryptocurrency wallets."""
    
    def __init__(self):
        """Initialize the crypto wallet manager."""
        self.wallets = self._load_wallets()
        self.default_wallet = self._get_default_wallet()
    
    def _load_wallets(self) -> Dict[str, Dict[str, Any]]:
        """Load saved wallets."""
        try:
            wallet_file = Path("config/crypto_wallets.json")
            if wallet_file.exists():
                with open(wallet_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading wallets: {e}")
            return {}
    
    def _save_wallets(self) -> None:
        """Save wallets to file."""
        try:
            wallet_file = Path("config/crypto_wallets.json")
            wallet_file.parent.mkdir(parents=True, exist_ok=True)
            with open(wallet_file, 'w') as f:
                json.dump(self.wallets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving wallets: {e}")
    
    def _get_default_wallet(self) -> Optional[str]:
        """Get the default wallet address."""
        try:
            for wallet_id, wallet in self.wallets.items():
                if wallet.get("is_default", False):
                    return wallet_id
            return None
        except Exception as e:
            logger.error(f"Error getting default wallet: {e}")
            return None
    
    def add_wallet(self, name: str, address: str, currency: str, is_default: bool = False) -> str:
        """Add a new wallet."""
        wallet_id = str(uuid.uuid4())
        self.wallets[wallet_id] = {
            "name": name,
            "address": address,
            "currency": currency,
            "is_default": is_default,
            "added_at": datetime.now().isoformat()
        }
        
        if is_default:
            # Unset previous default
            for wid, wallet in self.wallets.items():
                if wid != wallet_id:
                    wallet["is_default"] = False
            
            self.default_wallet = wallet_id
        
        self._save_wallets()
        return wallet_id
    
    def remove_wallet(self, wallet_id: str) -> bool:
        """Remove a wallet."""
        if wallet_id in self.wallets:
            was_default = self.wallets[wallet_id].get("is_default", False)
            del self.wallets[wallet_id]
            
            if was_default:
                self.default_wallet = None
                # Set a new default if there are other wallets
                if self.wallets:
                    new_default = next(iter(self.wallets.keys()))
                    self.wallets[new_default]["is_default"] = True
                    self.default_wallet = new_default
            
            self._save_wallets()
            return True
        return False
    
    def get_wallet(self, wallet_id: str) -> Optional[Dict[str, Any]]:
        """Get a wallet by ID."""
        return self.wallets.get(wallet_id)
    
    def get_default_wallet_address(self) -> Optional[str]:
        """Get the default wallet address."""
        if self.default_wallet:
            return self.wallets[self.default_wallet].get("address")
        return None
    
    def has_wallets(self) -> bool:
        """Check if any wallets are configured."""
        return len(self.wallets) > 0
    
    def prompt_for_wallet(self) -> Optional[str]:
        """Prompt the user to enter a wallet address."""
        if "wallet_prompt_completed" not in st.session_state:
            st.session_state.wallet_prompt_completed = False
            st.session_state.wallet_address = None
        
        if not st.session_state.wallet_prompt_completed:
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
            
            is_default = st.checkbox("Set as default wallet", value=True)
            
            if st.button("Save Wallet"):
                if wallet_address and wallet_name and currency:
                    self.add_wallet(wallet_name, wallet_address, currency, is_default)
                    st.session_state.wallet_prompt_completed = True
                    st.session_state.wallet_address = wallet_address
                    st.success(f"Wallet '{wallet_name}' added successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Please fill in all fields")
            
            return None
        
        return st.session_state.wallet_address

class CodeExecutor:
    """Execute and display code in real-time."""
    
    def __init__(self):
        """Initialize the code executor."""
        self.execution_history = []
        self.current_execution = None
        self.execution_thread = None
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code and return the result."""
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        try:
            if language.lower() == "python":
                # Create a temporary file to execute
                temp_file = Path("temp_execution.py")
                with open(temp_file, 'w') as f:
                    f.write(code)
                
                # Capture output
                process = subprocess.Popen(
                    [sys.executable, str(temp_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                # Clean up
                if temp_file.exists():
                    temp_file.unlink()
                
                if process.returncode == 0:
                    result["success"] = True
                    result["output"] = stdout
                else:
                    result["error"] = stderr
            else:
                result["error"] = f"Unsupported language: {language}"
        except Exception as e:
            result["error"] = str(e)
        
        result["execution_time"] = time.time() - start_time
        
        # Add to history
        self.execution_history.append(result)
        
        return result
    
    def execute_code_async(self, code: str, language: str = "python", callback: Callable = None) -> None:
        """Execute code asynchronously."""
        def _execute():
            result = self.execute_code(code, language)
            self.current_execution = None
            if callback:
                callback(result)
        
        self.current_execution = {
            "code": code,
            "language": language,
            "start_time": time.time(),
            "status": "running"
        }
        
        self.execution_thread = threading.Thread(target=_execute)
        self.execution_thread.daemon = True
        self.execution_thread.start()
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get the status of the current execution."""
        if self.current_execution:
            self.current_execution["elapsed_time"] = time.time() - self.current_execution["start_time"]
            return self.current_execution
        return None
    
    def format_code_for_display(self, code: str, language: str = "python", mode: CodeDisplayMode = CodeDisplayMode.FLOWING) -> str:
        """Format code for display with optional flowing effect."""
        if mode == CodeDisplayMode.HIDDEN:
            return ""
        
        if mode == CodeDisplayMode.COMPACT:
            # Return a compact version (first few lines)
            lines = code.split("\n")
            if len(lines) > 5:
                return "\n".join(lines[:5]) + f"\n... ({len(lines) - 5} more lines)"
            return code
        
        if mode == CodeDisplayMode.FLOWING:
            # Add HTML/CSS for flowing effect
            return f"""
            <div class="flowing-code {language.lower()}">
                <pre><code class="{language.lower()}">{code}</code></pre>
            </div>
            """
        
        # Full mode - return as is
        return code

class EnhancedChatInterface:
    """Enhanced chat interface with flowing code display and sliding menu."""
    
    def __init__(self):
        """Initialize the enhanced chat interface."""
        self.theme_manager = ThemeManager() if 'ThemeManager' in globals() else None
        self.agent_manager = AgentManager() if 'AgentManager' in globals() else None
        self.business_manager = BusinessManager() if 'BusinessManager' in globals() else None
        self.crypto_manager = CryptoManager() if 'CryptoManager' in globals() else None
        self.rag_system = LiveThinkingRAGSystem() if 'LiveThinkingRAGSystem' in globals() else None
        self.performance_monitor = PerformanceMonitor() if 'PerformanceMonitor' in globals() else None
        
        self.perplexica = PerplexicaIntegration()
        self.wallet_manager = CryptoWalletManager()
        self.code_executor = CodeExecutor()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load custom CSS and JavaScript
        self._load_custom_assets()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = MenuTab.CHAT
        
        if 'menu_expanded' not in st.session_state:
            st.session_state.menu_expanded = False
        
        if 'code_display_mode' not in st.session_state:
            st.session_state.code_display_mode = CodeDisplayMode.FLOWING
        
        if 'wallet_setup_complete' not in st.session_state:
            st.session_state.wallet_setup_complete = False
        
        if 'theme' not in st.session_state:
            st.session_state.theme = "glass_dark"
        
        if 'agent_tasks' not in st.session_state:
            st.session_state.agent_tasks = []
        
        if 'business_ventures' not in st.session_state:
            st.session_state.business_ventures = []
        
        if 'crypto_transactions' not in st.session_state:
            st.session_state.crypto_transactions = []
        
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'websites' not in st.session_state:
            st.session_state.websites = []
    
    def _load_custom_assets(self) -> None:
        """Load custom CSS and JavaScript assets."""
        # Create CSS file if it doesn't exist
        css_file = CSS_DIR / "enhanced_chat.css"
        if not css_file.exists():
            with open(css_file, 'w') as f:
                f.write("""
                /* Enhanced Chat Interface CSS */
                
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
                
                .menu-toggle:hover {
                    background: rgba(0, 0, 0, 0.8);
                }
                
                /* Message styles */
                .message-container {
                    margin-bottom: 15px;
                    animation: fadeIn 0.3s ease-in-out;
                }
                
                .user-message {
                    background: rgba(0, 123, 255, 0.1);
                    border-left: 4px solid #007bff;
                }
                
                .assistant-message {
                    background: rgba(40, 167, 69, 0.1);
                    border-left: 4px solid #28a745;
                }
                
                .system-message {
                    background: rgba(108, 117, 125, 0.1);
                    border-left: 4px solid #6c757d;
                    font-style: italic;
                }
                
                .error-message {
                    background: rgba(220, 53, 69, 0.1);
                    border-left: 4px solid #dc3545;
                }
                
                .thinking-message {
                    background: rgba(255, 193, 7, 0.1);
                    border-left: 4px solid #ffc107;
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
                
                @keyframes fadeIn {
                    from {
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                /* OCR A font for futuristic look */
                @font-face {
                    font-family: 'OCR A Extended';
                    src: url('assets/fonts/OCR-A.ttf') format('truetype');
                }
                
                .ocr-font {
                    font-family: 'OCR A Extended', monospace;
                }
                
                /* Tab styles */
                .tab-container {
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }
                
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 5px 5px 0 0;
                    margin-right: 5px;
                    background: rgba(0, 0, 0, 0.2);
                    transition: all 0.3s ease;
                }
                
                .tab:hover {
                    background: rgba(0, 0, 0, 0.4);
                }
                
                .tab.active {
                    background: rgba(0, 123, 255, 0.2);
                    border-bottom: 2px solid #007bff;
                }
                
                /* Crypto wallet display */
                .wallet-display {
                    background: rgba(0, 0, 0, 0.3);
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                
                .wallet-address {
                    font-family: 'OCR A Extended', monospace;
                    font-size: 0.8em;
                    color: #28a745;
                    word-break: break-all;
                }
                
                /* Animations */
                .pulse {
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0% {
                        opacity: 1;
                    }
                    50% {
                        opacity: 0.5;
                    }
                    100% {
                        opacity: 1;
                    }
                }
                """)
        
        # Create JavaScript file if it doesn't exist
        js_file = JS_DIR / "enhanced_chat.js"
        if not js_file.exists():
            with open(js_file, 'w') as f:
                f.write("""
                // Enhanced Chat Interface JavaScript
                
                // Toggle sliding menu
                function toggleMenu() {
                    const menu = document.querySelector('.sliding-menu');
                    menu.classList.toggle('expanded');
                    
                    // Update toggle button
                    const toggle = document.querySelector('.menu-toggle');
                    if (menu.classList.contains('expanded')) {
                        toggle.innerHTML = '&times;';
                    } else {
                        toggle.innerHTML = '&#9776;';
                    }
                }
                
                // Initialize syntax highlighting
                function initCodeHighlighting() {
                    if (typeof hljs !== 'undefined') {
                        document.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightBlock(block);
                        });
                    }
                }
                
                // Scroll to bottom of chat
                function scrollToBottom() {
                    const chatContainer = document.querySelector('.chat-container');
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
                
                // Initialize flowing code animation
                function initFlowingCode() {
                    document.querySelectorAll('.flowing-code').forEach((codeBlock) => {
                        // Add line numbers
                        const code = codeBlock.querySelector('code');
                        if (code) {
                            const lines = code.innerHTML.split('\\n');
                            let numberedLines = '';
                            lines.forEach((line, index) => {
                                numberedLines += `<span class="line-number">${index + 1}</span>${line}\\n`;
                            });
                            code.innerHTML = numberedLines;
                        }
                    });
                }
                
                // Document ready function
                document.addEventListener('DOMContentLoaded', function() {
                    initCodeHighlighting();
                    initFlowingCode();
                    scrollToBottom();
                    
                    // Set up menu toggle
                    const toggle = document.querySelector('.menu-toggle');
                    if (toggle) {
                        toggle.addEventListener('click', toggleMenu);
                    }
                });
                
                // Function to execute when Streamlit has loaded
                window.addEventListener('load', function() {
                    // Give Streamlit time to initialize
                    setTimeout(function() {
                        initCodeHighlighting();
                        initFlowingCode();
                        scrollToBottom();
                    }, 1000);
                });
                """)
        
        # Inject CSS
        with open(css_file, 'r') as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        
        # Inject JavaScript
        with open(js_file, 'r') as f:
            js_content = f.read()
            components.html(f"<script>{js_content}</script>", height=0)
    
    def _render_sliding_menu(self) -> None:
        """Render the sliding menu."""
        # Menu toggle button
        menu_toggle_html = """
        <div class="menu-toggle" onclick="toggleMenu()">&#9776;</div>
        <div class="sliding-menu{expanded}">
            <h2 class="ocr-font">Skyscope Sentinel</h2>
            <hr>
            <div class="menu-content">
                {menu_content}
            </div>
        </div>
        """
        
        # Menu content
        menu_items = []
        for tab in MenuTab:
            active_class = "active" if st.session_state.current_tab == tab else ""
            tab_name = tab.name.capitalize()
            menu_items.append(f'<div class="tab {active_class}" onclick="Streamlit.setComponentValue(\'{tab.name}\')">{tab_name}</div>')
        
        menu_content = "\n".join(menu_items)
        expanded_class = " expanded" if st.session_state.menu_expanded else ""
        
        # Render menu
        menu_html = menu_toggle_html.format(expanded=expanded_class, menu_content=menu_content)
        
        # Use a custom component to handle menu clicks
        clicked_tab = components.html(menu_html, height=50)
        
        if clicked_tab and clicked_tab in [tab.name for tab in MenuTab]:
            st.session_state.current_tab = MenuTab[clicked_tab]
            st.experimental_rerun()
    
    def _render_wallet_section(self) -> None:
        """Render the cryptocurrency wallet section."""
        st.subheader("💼 Cryptocurrency Wallets")
        
        # Check if wallet setup is complete
        if not self.wallet_manager.has_wallets():
            wallet_address = self.wallet_manager.prompt_for_wallet()
            if wallet_address:
                st.session_state.wallet_setup_complete = True
        else:
            # Display wallets
            wallets = self.wallet_manager.wallets
            
            for wallet_id, wallet in wallets.items():
                with st.expander(f"{wallet['name']} ({wallet['currency']})", expanded=wallet.get("is_default", False)):
                    st.markdown(f"**Address:** `{wallet['address']}`")
                    st.markdown(f"**Added:** {wallet['added_at']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if not wallet.get("is_default", False):
                            if st.button(f"Set as Default##wallet_{wallet_id}"):
                                # Update all wallets to not be default
                                for wid in wallets:
                                    wallets[wid]["is_default"] = False
                                # Set this one as default
                                wallets[wallet_id]["is_default"] = True
                                self.wallet_manager._save_wallets()
                                st.experimental_rerun()
                    
                    with col2:
                        if st.button(f"Remove##wallet_{wallet_id}"):
                            self.wallet_manager.remove_wallet(wallet_id)
                            st.experimental_rerun()
            
            # Add new wallet
            with st.expander("➕ Add New Wallet"):
                wallet_name = st.text_input("Wallet Name", key="new_wallet_name")
                wallet_address = st.text_input("Wallet Address", key="new_wallet_address")
                currency = st.selectbox("Cryptocurrency", ["BTC", "ETH", "SOL", "ADA", "XRP", "USDT", "USDC", "Other"], key="new_wallet_currency")
                
                if currency == "Other":
                    currency = st.text_input("Enter cryptocurrency symbol", key="new_wallet_currency_other")
                
                is_default = st.checkbox("Set as default wallet", value=not self.wallet_manager.has_wallets(), key="new_wallet_default")
                
                if st.button("Add Wallet"):
                    if wallet_address and wallet_name and currency:
                        self.wallet_manager.add_wallet(wallet_name, wallet_address, currency, is_default)
                        st.success(f"Wallet '{wallet_name}' added successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Please fill in all fields")
    
    def _render_perplexica_search(self) -> None:
        """Render the Perplexica search interface."""
        st.subheader("🔍 Perplexica AI Search")
        
        if not self.perplexica.available:
            st.warning(f"Perplexica not found at {PERPLEXICA_PATH}. Search functionality is limited.")
        
        # Search form
        search_query = st.text_input("Search Query")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            search_type = st.radio("Search Type", ["Quick Search", "Deep Research"])
        
        with col2:
            if search_type == "Quick Search":
                num_results = st.slider("Number of Results", 3, 20, 5)
            else:
                research_depth = st.slider("Research Depth", 1, 5, 3)
        
        if st.button("Search"):
            if search_query:
                with st.spinner("Searching..."):
                    if search_type == "Quick Search":
                        results = self.perplexica.search(search_query, num_results=num_results)
                        
                        # Save to search history
                        st.session_state.search_history.append({
                            "query": search_query,
                            "type": "quick",
                            "timestamp": datetime.now().isoformat(),
                            "results_count": len(results)
                        })
                        
                        # Display results
                        if results:
                            for i, result in enumerate(results):
                                with st.expander(f"{i+1}. {result['title']}", expanded=i==0):
                                    st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                                    st.markdown(result['snippet'])
                        else:
                            st.info("No results found")
                    else:
                        research = self.perplexica.deep_research(search_query, depth=research_depth)
                        
                        # Save to search history
                        st.session_state.search_history.append({
                            "query": search_query,
                            "type": "deep",
                            "timestamp": datetime.now().isoformat(),
                            "depth": research_depth
                        })
                        
                        # Display research results
                        if "error" in research:
                            st.error(f"Research error: {research['error']}")
                        else:
                            st.subheader(f"Deep Research: {search_query}")
                            
                            if "results" in research:
                                if isinstance(research["results"], dict):
                                    # Display results by query
                                    for query, query_results in research["results"].items():
                                        with st.expander(f"Results for: {query}", expanded=True):
                                            for i, result in enumerate(query_results):
                                                st.markdown(f"**{i+1}. {result['title']}**")
                                                st.markdown(f"[{result['url']}]({result['url']})")
                                                st.markdown(result['snippet'])
                                                st.markdown("---")
                                else:
                                    # Display flat results
                                    for i, result in enumerate(research["results"]):
                                        st.markdown(f"**{i+1}. {result['title']}**")
                                        st.markdown(f"[{result['url']}]({result['url']})")
                                        st.markdown(result['snippet'])
                                        st.markdown("---")
            else:
                st.warning("Please enter a search query")
        
        # Search history
        if st.session_state.search_history:
            with st.expander("Search History"):
                for i, search in enumerate(reversed(st.session_state.search_history[-10:])):
                    search_time = datetime.fromisoformat(search["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    if search["type"] == "quick":
                        st.markdown(f"**{i+1}.** {search['query']} - {search_time} ({search['results_count']} results)")
                    else:
                        st.markdown(f"**{i+1}.** {search['query']} - {search_time} (Deep Research, Depth: {search['depth']})")
    
    def _render_business_ventures(self) -> None:
        """Render the business ventures section."""
        st.subheader("💼 Autonomous Business Ventures")
        
        # Display existing ventures
        if not st.session_state.business_ventures:
            st.info("No active business ventures yet. The agents will automatically create and manage ventures to generate income.")
            
            # Add a sample venture for demonstration
            if st.button("Create Demo Venture"):
                st.session_state.business_ventures.append({
                    "id": str(uuid.uuid4()),
                    "name": "AI Content Subscription Service",
                    "description": "Automated content creation and distribution service with monthly subscription model.",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "revenue": 0.0,
                    "expenses": 0.0,
                    "profit": 0.0,
                    "tasks": [
                        {"name": "Website Setup", "status": "completed", "assigned_agent": "Agent-Executor-Technology-a1b2c3d4"},
                        {"name": "Content Pipeline Creation", "status": "in_progress", "assigned_agent": "Agent-Creator-Creative-e5f6g7h8"},
                        {"name": "Payment Integration", "status": "pending", "assigned_agent": "Agent-Integrator-Finance-i9j0k1l2"}
                    ]
                })
                st.experimental_rerun()
        else:
            for venture in st.session_state.business_ventures:
                with st.expander(f"{venture['name']} - {venture['status'].capitalize()}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {venture['description']}")
                        st.markdown(f"**Created:** {datetime.fromisoformat(venture['created_at']).strftime('%Y-%m-%d %H:%M')}")
                    
                    with col2:
                        st.metric("Revenue", f"${venture['revenue']:.2f}")
                        st.metric("Profit", f"${venture['profit']:.2f}")
                    
                    # Tasks
                    st.markdown("### Tasks")
                    for task in venture["tasks"]:
                        status_color = {
                            "completed": "green",
                            "in_progress": "orange",
                            "pending": "gray",
                            "failed": "red"
                        }.get(task["status"], "gray")
                        
                        st.markdown(f"- <span style='color: {status_color}'>{task['status'].upper()}</span>: {task['name']} (Assigned to: {task['assigned_agent']})", unsafe_allow_html=True)
        
        # Create new venture form
        with st.expander("➕ Create New Business Venture"):
            venture_name = st.text_input("Venture Name", key="new_venture_name")
            venture_description = st.text_area("Description", key="new_venture_description")
            venture_type = st.selectbox("Venture Type", [
                "Content Subscription Service", 
                "AI SaaS Product", 
                "Crypto Trading Bot", 
                "Automated E-commerce Store",
                "Digital Marketing Agency",
                "Educational Platform",
                "Custom"
            ], key="new_venture_type")
            
            if venture_type == "Custom":
                venture_type = st.text_input("Enter custom venture type", key="new_venture_type_custom")
            
            if st.button("Create Venture"):
                if venture_name and venture_description and venture_type:
                    # In a real implementation, this would use the agent manager to create a new venture
                    new_venture = {
                        "id": str(uuid.uuid4()),
                        "name": venture_name,
                        "description": venture_description,
                        "type": venture_type,
                        "status": "initializing",
                        "created_at": datetime.now().isoformat(),
                        "revenue": 0.0,
                        "expenses": 0.0,
                        "profit": 0.0,
                        "tasks": [
                            {"name": "Initial Planning", "status": "in_progress", "assigned_agent": "Agent-Planner-Business-a1b2c3d4"},
                            {"name": "Market Research", "status": "pending", "assigned_agent": "Agent-Researcher-Business-e5f6g7h8"},
                            {"name": "Implementation Strategy", "status": "pending", "assigned_agent": "Agent-Strategist-Business-i9j0k1l2"}
                        ]
                    }
                    
                    st.session_state.business_ventures.append(new_venture)
                    st.success(f"Business venture '{venture_name}' created successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Please fill in all fields")
    
    def _render_crypto_transactions(self) -> None:
        """Render the cryptocurrency transactions section."""
        st.subheader("💰 Cryptocurrency Transactions")
        
        # Display transactions
        if not st.session_state.crypto_transactions:
            st.info("No cryptocurrency transactions yet. The agents will automatically manage trading and payments.")
            
            # Add a sample transaction for demonstration
            if st.button("Create Demo Transaction"):
                st.session_state.crypto_transactions.append({
                    "id": str(uuid.uuid4()),
                    "type": "trade",
                    "status": "completed",
                    "from_currency": "USDT",
                    "to_currency": "BTC",
                    "amount_from": 1000.0,
                    "amount_to": 0.03,
                    "fee": 2.5,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "Binance",
                    "transaction_hash": "0x" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:64],
                    "wallet_id": list(self.wallet_manager.wallets.keys())[0] if self.wallet_manager.wallets else None
                })
                st.experimental_rerun()
        else:
            # Create a DataFrame for better display
            transactions_data = []
            for tx in st.session_state.crypto_transactions:
                tx_time = datetime.fromisoformat(tx["timestamp"]).strftime("%Y-%m-%d %H:%M")
                
                if tx["type"] == "trade":
                    description = f"Trade {tx['amount_from']} {tx['from_currency']} → {tx['amount_to']} {tx['to_currency']}"
                elif tx["type"] == "payment":
                    description = f"Payment {tx['amount_from']} {tx['from_currency']}"
                elif tx["type"] == "income":
                    description = f"Income {tx['amount_to']} {tx['to_currency']}"
                else:
                    description = f"{tx['type'].capitalize()} transaction"
                
                transactions_data.append({
                    "Time": tx_time,
                    "Type": tx["type"].capitalize(),
                    "Description": description,
                    "Status": tx["status"].capitalize(),
                    "Exchange/Platform": tx.get("exchange", "N/A"),
                    "ID": tx["id"]
                })
            
            tx_df = pd.DataFrame(transactions_data)
            st.dataframe(tx_df)
            
            # Transaction details
            if transactions_data:
                selected_tx_id = st.selectbox("Select transaction for details", 
                                            options=[tx["ID"] for tx in transactions_data],
                                            format_func=lambda x: next((tx["Description"] for tx in transactions_data if tx["ID"] == x), x))
                
                # Find the selected transaction
                selected_tx = next((tx for tx in st.session_state.crypto_transactions if tx["id"] == selected_tx_id), None)
                
                if selected_tx:
                    with st.expander("Transaction Details", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"**Type:** {selected_tx['type'].capitalize()}")
                            st.markdown(f"**Status:** {selected_tx['status'].capitalize()}")
                            st.markdown(f"**Time:** {datetime.fromisoformat(selected_tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            if "exchange" in selected_tx:
                                st.markdown(f"**Exchange:** {selected_tx['exchange']}")
                        
                        with col2:
                            if "from_currency" in selected_tx:
                                st.markdown(f"**From:** {selected_tx['amount_from']} {selected_tx['from_currency']}")
                            
                            if "to_currency" in selected_tx:
                                st.markdown(f"**To:** {selected_tx['amount_to']} {selected_tx['to_currency']}")
                            
                            if "fee" in selected_tx:
                                st.markdown(f"**Fee:** {selected_tx['fee']}")
                        
                        if "transaction_hash" in selected_tx:
                            st.markdown(f"**Transaction Hash:** `{selected_tx['transaction_hash']}`")
                        
                        if "notes" in selected_tx and selected_tx["notes"]:
                            st.markdown(f"**Notes:** {selected_tx['notes']}")
        
        # Manual transaction form
        with st.expander("➕ Add Manual Transaction"):
            tx_type = st.selectbox("Transaction Type", ["trade", "payment", "income", "withdrawal", "deposit"], key="new_tx_type")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                from_currency = st.text_input("From Currency", value="USDT", key="new_tx_from_currency")
                amount_from = st.number_input("Amount From", min_value=0.0, value=100.0, key="new_tx_amount_from")
            
            with col2:
                to_currency = st.text_input("To Currency", value="BTC", key="new_tx_to_currency")
                amount_to = st.number_input("Amount To", min_value=0.0, value=0.0, key="new_tx_amount_to")
            
            fee = st.number_input("Fee", min_value=0.0, value=1.0, key="new_tx_fee")
            exchange = st.text_input("Exchange/Platform", value="Binance", key="new_tx_exchange")
            tx_hash = st.text_input("Transaction Hash (optional)", key="new_tx_hash")
            notes = st.text_area("Notes", key="new_tx_notes")
            
            if st.button("Add Transaction"):
                # Create transaction
                new_tx = {
                    "id": str(uuid.uuid4()),
                    "type": tx_type,
                    "status": "completed",
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "amount_from": amount_from,
                    "amount_to": amount_to,
                    "fee": fee,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": exchange,
                    "notes": notes,
                    "wallet_id": self.wallet_manager.default_wallet
                }
                
                if tx_hash:
                    new_tx["transaction_hash"] = tx_hash
                else:
                    # Generate a dummy hash
                    new_tx["transaction_hash"] = "0x" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:64]
                
                st.session_state.crypto_transactions.append(new_tx)
                st.success("Transaction added successfully!")
                st.experimental_rerun()
    
    def _render_websites_section(self) -> None:
        """Render the websites section."""
        st.subheader("🌐 Automated Websites")
        
        # Display existing websites
        if not st.session_state.websites:
            st.info("No websites created yet. The agents will automatically create and manage websites for your business ventures.")
            
            # Add a sample website for demonstration
            if st.button("Create Demo Website"):
                st.session_state.websites.append({
                    "id": str(uuid.uuid4()),
                    "name": "AI Content Hub",
                    "domain": "aicontenthub.example.com",
                    "status": "live",
                    "created_at": datetime.now().isoformat(),
                    "type": "subscription_service",
                    "monthly_visitors": 1250,
                    "monthly_revenue": 750.0,
                    "features": [
                        "Content generation",
                        "SEO optimization",
                        "Social media scheduling",
                        "Analytics dashboard"
                    ],
                    "tech_stack": [
                        "React", "Node.js", "MongoDB", "AWS"
                    ]
                })
                st.experimental_rerun()
        else:
            for website in st.session_state.websites:
                with st.expander(f"{website['name']} - {website['domain']}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Status:** {website['status'].capitalize()}")
                        st.markdown(f"**Type:** {website['type'].replace('_', ' ').capitalize()}")
                        st.markdown(f"**Created:** {datetime.fromisoformat(website['created_at']).strftime('%Y-%m-%d')}")
                    
                    with col2:
                        st.metric("Monthly Visitors", f"{website['monthly_visitors']:,}")
                        st.metric("Monthly Revenue", f"${website['monthly_revenue']:.2f}")
                    
                    # Features and tech stack
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Features:**")
                        for feature in website["features"]:
                            st.markdown(f"- {feature}")
                    
                    with col2:
                        st.markdown("**Tech Stack:**")
                        for tech in website["tech_stack"]:
                            st.markdown(f"- {tech}")
                    
                    # Actions
                    st.markdown("### Actions")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.button(f"View Website##view_{website['id']}")
                    
                    with col2:
                        st.button(f"Edit Content##edit_{website['id']}")
                    
                    with col3:
                        st.button(f"Analytics##analytics_{website['id']}")
        
        # Create new website form
        with st.expander("➕ Create New Website"):
            website_name = st.text_input("Website Name", key="new_website_name")
            website_domain = st.text_input("Domain Name", key="new_website_domain", 
                                         help="Enter without http:// or https://")
            website_type = st.selectbox("Website Type", [
                "subscription_service", 
                "e_commerce", 
                "blog", 
                "portfolio",
                "saas_application",
                "landing_page",
                "custom"
            ], key="new_website_type")
            
            if website_type == "custom":
                website_type = st.text_input("Enter custom website type", key="new_website_type_custom")
            
            # Features
            st.markdown("### Features")
            feature1 = st.text_input("Feature 1", key="new_website_feature1")
            feature2 = st.text_input("Feature 2", key="new_website_feature2")
            feature3 = st.text_input("Feature 3", key="new_website_feature3")
            additional_features = st.text_area("Additional Features (one per line)", key="new_website_additional_features")
            
            # Tech stack
            st.markdown("### Tech Stack")
            frontend = st.selectbox("Frontend", ["React", "Vue.js", "Angular", "HTML/CSS/JS", "Other"], key="new_website_frontend")
            backend = st.selectbox("Backend", ["Node.js", "Python/Django", "Python/Flask", "PHP", "Ruby on Rails", "Other"], key="new_website_backend")
            database = st.selectbox("Database", ["MongoDB", "PostgreSQL", "MySQL", "SQLite", "Other"], key="new_website_database")
            hosting = st.selectbox("Hosting", ["AWS", "Google Cloud", "Azure", "Heroku", "Netlify", "Vercel", "Other"], key="new_website_hosting")
            
            if st.button("Create Website"):
                if website_name and website_domain:
                    # Collect features
                    features = [f for f in [feature1, feature2, feature3] if f]
                    if additional_features:
                        features.extend([f.strip() for f in additional_features.split("\n") if f.strip()])
                    
                    # Collect tech stack
                    tech_stack = [t for t in [frontend, backend, database, hosting] if t != "Other"]
                    
                    # Create website
                    new_website = {
                        "id": str(uuid.uuid4()),
                        "name": website_name,
                        "domain": website_domain,
                        "status": "initializing",
                        "created_at": datetime.now().isoformat(),
                        "type": website_type,
                        "monthly_visitors": 0,
                        "monthly_revenue": 0.0,
                        "features": features,
                        "tech_stack": tech_stack
                    }
                    
                    st.session_state.websites.append(new_website)
                    st.success(f"Website '{website_name}' creation initiated!")
                    st.experimental_rerun()
                else:
                    st.error("Please provide a website name and domain")
    
    def _render_settings_section(self) -> None:
        """Render the settings section."""
        st.subheader("⚙️ System Settings")
        
        # Theme settings
        st.markdown("### UI Theme")
        theme_options = ["glass_dark", "glass_light", "neon", "minimal", "terminal", "ocean", "sunset"]
        selected_theme = st.selectbox("Select Theme", theme_options, index=theme_options.index(st.session_state.theme))
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.success(f"Theme changed to {selected_theme}")
            st.experimental_rerun()
        
        # Code display settings
        st.markdown("### Code Display")
        code_display_options = [mode.value for mode in CodeDisplayMode]
        selected_code_display = st.selectbox("Code Display Mode", 
                                           code_display_options, 
                                           index=code_display_options.index(st.session_state.code_display_mode.value))
        
        if selected_code_display != st.session_state.code_display_mode.value:
            st.session_state.code_display_mode = CodeDisplayMode(selected_code_display)
            st.success(f"Code display mode changed to {selected_code_display}")
            st.experimental_rerun()
        
        # Agent settings
        st.markdown("### Agent Settings")
        agent_count = st.slider("Active Agents", min_value=100, max_value=10000, value=1000, step=100)
        
        # System settings
        st.markdown("### System Settings")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            auto_save = st.checkbox("Auto-save conversations", value=True)
            debug_mode = st.checkbox("Debug mode", value=False)
        
        with col2:
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], index=1)
            max_memory = st.slider("Max Memory Usage (GB)", min_value=1, max_value=32, value=8)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.markdown("### API Keys")
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            
            st.markdown("### Database Settings")
            db_connection = st.text_input("Database Connection String", value="postgresql://localhost:5432/skyscope")
            
            st.markdown("### Backup Settings")
            backup_frequency = st.selectbox("Backup Frequency", ["Hourly", "Daily", "Weekly"], index=1)
            backup_location = st.text_input("Backup Location", value="./backups")
            
            if st.button("Save Advanced Settings"):
                st.success("Advanced settings saved!")
        
        # Reset settings
        if st.button("Reset to Defaults"):
            st.warning("This will reset all settings to their default values. Are you sure?")
            confirm = st.checkbox("Yes, reset all settings")
            
            if confirm:
                # Reset settings
                st.session_state.theme = "glass_dark"
                st.session_state.code_display_mode = CodeDisplayMode.FLOWING
                st.success("Settings reset to defaults")
                st.experimental_rerun()
    
    def _render_analytics_section(self) -> None:
        """Render the analytics dashboard."""
        st.subheader("📊 Analytics Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("Total Revenue", "$1,250.75", delta="12.5%")
        
        with col2:
            st.metric("Active Ventures", "3", delta="1")
        
        with col3:
            st.metric("Website Visitors", "3,427", delta="8.2%")
        
        with col4:
            st.metric("Conversion Rate", "3.8%", delta="0.5%")
        
        # Revenue chart
        st.markdown("### Revenue Over Time")
        
        # Create sample data
        dates = pd.date_range(start='2025-01-01', end='2025-07-14', freq='D')
        revenue = np.cumsum(np.random.normal(50, 15, size=len(dates)))
        revenue = np.maximum(0, revenue)  # Ensure no negative revenue
        
        df = pd.DataFrame({
            'date': dates,
            'revenue': revenue
        })
        
        # Plot
        fig = px.line(df, x='date', y='revenue', title='Cumulative Revenue')
        st.plotly_chart(fig, use_container_width=True)
        
        # Business ventures performance
        st.markdown("### Business Ventures Performance")
        
        # Sample data
        ventures = ['AI Content Hub', 'Trading Bot', 'E-commerce Store']
        revenue = [750, 320, 180]
        expenses = [200, 50, 120]
        
        # Create DataFrame
        df_ventures = pd.DataFrame({
            'Venture': ventures,
            'Revenue': revenue,
            'Expenses': expenses,
            'Profit': [r - e for r, e in zip(revenue, expenses)]
        })
        
        # Plot
        fig = px.bar(df_ventures, x='Venture', y=['Revenue', 'Expenses', 'Profit'], 
                   title='Venture Performance', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic sources
        st.markdown("### Website Traffic Sources")
        
        # Sample data
        sources = ['Organic Search', 'Direct', 'Social Media', 'Referral', 'Email']
        traffic = [45, 25, 15, 10, 5]
        
        # Plot
        fig = px.pie(values=traffic, names=sources, title='Traffic Sources')
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent activity
        st.markdown("### Agent Activity")
        
        # Sample data
        agent_types = ['Executor', 'Researcher', 'Creator', 'Analyzer', 'Planner']
        tasks_completed = [120, 85, 65, 45, 30]
        
        # Plot
        fig = px.bar(x=agent_types, y=tasks_completed, title='Tasks Completed by Agent Type')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agents_section(self) -> None:
        """Render the agents management section."""
        st.subheader("🤖 Agent Management")
        
        # Agent stats
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Total Agents", "10,000")
        
        with col2:
            st.metric("Active Agents", "1,250", delta="50")
        
        with col3:
            st.metric("Tasks in Queue", "324", delta="-12")
        
        # Agent pipelines
        st.markdown("### Agent Pipelines")
        
        # Sample data
        pipelines = ['Business Operations', 'Content Creation', 'Trading', 'Research', 'Development']
        agents_per_pipeline = [2500, 2000, 2000, 1500, 2000]
        
        # Plot
        fig = px.bar(x=pipelines, y=agents_per_pipeline, title='Agents per Pipeline')
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent tasks
        st.markdown("### Recent Agent Tasks")
        
        # Sample data
        if not st.session_state.agent_tasks:
            # Generate sample tasks
            for i in range(10):
                task_types = ["research", "execute", "analyze", "create", "optimize"]
                status_types = ["completed", "in_progress", "queued", "failed"]
                weights = [0.6, 0.3, 0.08, 0.02]  # More completed than failed
                
                st.session_state.agent_tasks.append({
                    "id": str(uuid.uuid4()),
                    "name": f"Task {i+1}",
                    "type": random.choice(task_types),
                    "status": random.choices(status_types, weights=weights)[0],
                    "agent_id": f"agent-{random.randint(1000, 9999)}",
                    "created_at": (datetime.now() - datetime.timedelta(hours=random.randint(0, 48))).isoformat(),
                    "completed_at": (datetime.now() - datetime.timedelta(hours=random.randint(0, 24))).isoformat() if random.random() > 0.3 else None,
                    "priority": random.randint(1, 5)
                })
        
        # Create DataFrame
        tasks_data = []
        for task in st.session_state.agent_tasks:
            created_time = datetime.fromisoformat(task["created_at"]).strftime("%Y-%m-%d %H:%M")
            completed_time = datetime.fromisoformat(task["completed_at"]).strftime("%Y-%m-%d %H:%M") if task.get("completed_at") else "N/A"
            
            tasks_data.append({
                "Task": task["name"],
                "Type": task["type"].capitalize(),
                "Status": task["status"].capitalize(),
                "Agent": task["agent_id"],
                "Created": created_time,
                "Completed": completed_time,
                "Priority": task["priority"]
            })
        
        tasks_df = pd.DataFrame(tasks_data)
        st.dataframe(tasks_df)
        
        # Create new task
        with st.expander("➕ Create New Agent Task"):
            task_name = st.text_input("Task Name", key="new_task_name")
            task_type = st.selectbox("Task Type", ["research", "execute", "analyze", "create", "optimize"], key="new_task_type")
            task_description = st.text_area("Task Description", key="new_task_description")
            task_priority = st.slider("Priority", min_value=1, max_value=5, value=3, key="new_task_priority")
            
            if st.button("Create Task"):
                if task_name and task_description:
                    new_task = {
                        "id": str(uuid.uuid4()),
                        "name": task_name,
                        "type": task_type,
                        "description": task_description,
                        "status": "queued",
                        "agent_id": f"agent-{random.randint(1000, 9999)}",
                        "created_at": datetime.now().isoformat(),
                        "completed_at": None,
                        "priority": task_priority
                    }
                    
                    st.session_state.agent_tasks.append(new_task)
                    st.success(f"Task '{task_name}' created successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Please provide a task name and description")
    
    def _render_chat_interface(self) -> None:
        """Render the main chat interface."""
        st.subheader("💬 Skyscope Sentinel AI Assistant")
        
        # Display wallet status if available
        if self.wallet_manager.has_wallets():
            default_wallet = self.wallet_manager.get_default_wallet_address()
            if default_wallet:
                wallet_display = f"""
                <div class="wallet-display">
                    <div>
                        <span>💼 Default Wallet:</span>
                        <span class="wallet-address">{default_wallet[:10]}...{default_wallet[-8:]}</span>
                    </div>
                </div>
                """
                st.markdown(wallet_display, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display code if present
                if "code" in message:
                    if st.session_state.code_display_mode != CodeDisplayMode.HIDDEN:
                        st.code(message["code"], language=message.get("language", "python"))
        
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
                
                # Generate response (in a real implementation, this would use the agent manager)
                time.sleep(1)  # Simulate processing time
                
                # Check if this is a code execution request
                if prompt.lower().startswith(("run", "execute", "python", "code")):
                    # Extract code to execute
                    code_to_execute = None
                    
                    # Check if code is in backticks
                    code_blocks = re.findall(r"```(.*?)```", prompt, re.DOTALL)
                    if code_blocks:
                        code_to_execute = code_blocks[0]
                        if code_to_execute.startswith("python\n"):
                            code_to_execute = code_to_execute[7:]
                    else:
                        # Try to extract code without backticks
                        lines = prompt.split("\n")
                        if len(lines) > 1:
                            # Remove the first line (command) and join the rest
                            code_to_execute = "\n".join(lines[1:])
                    
                    if code_to_execute:
                        response_placeholder.markdown("Executing code...")
                        
                        # Execute code
                        result = self.code_executor.execute_code(code_to_execute)
                        
                        if result["success"]:
                            response = f"Code executed successfully in {result['execution_time']:.2f} seconds."
                            if result["output"]:
                                response += "\n\nOutput:\n```\n" + result["output"] + "\n```"
                        else:
                            response = f"Error executing code: {result['error']}"
                        
                        # Update response
                        response_placeholder.markdown(response)
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "code": code_to_execute,
                            "language": "python"
                        })
                    else:
                        response = "I couldn't find any code to execute. Please provide code between triple backticks."
                        response_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Check if this is a search request
                elif any(keyword in prompt.lower() for keyword in ["search", "find", "lookup", "research"]):
                    response_placeholder.markdown("Searching...")
                    
                    # Extract search query
                    search_query = prompt.lower()
                    for prefix in ["search", "find", "lookup", "research", "for"]:
                        search_query = search_query.replace(prefix + " ", "")
                    
                    # Perform search
                    results = self.perplexica.search(search_query, num_results=3)
                    
                    # Format response
                    response = f"Here are some results for '{search_query}':\n\n"
                    
                    for i, result in enumerate(results):
                        response += f"{i+1}. **{result['title']}**\n"
                        response += f"   {result['url']}\n"
                        response += f"   {result['snippet']}\n\n"
                    
                    # Update response
                    response_placeholder.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Check if this is a wallet request
                elif any(keyword in prompt.lower() for keyword in ["wallet", "crypto", "address", "payment"]):
                    if not self.wallet_manager.has_wallets():
                        
