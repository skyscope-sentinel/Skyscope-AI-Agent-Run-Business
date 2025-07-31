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
import streamlit.cli as stcli

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
    from database_manager import DatabaseManager
    from autonomous_business_operations import BusinessIdeaGenerator, ServiceRegistrationManager, WebsiteBuilder
    from performance_monitor import PerformanceMonitor
    from live_thinking_rag_system import LiveThinkingRAGSystem
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
        self.database_manager = None
        self.performance_monitor = None
        self.rag_system = None
        
        # Initialize business operations components
        self.business_idea_generator = None
        self.service_registration_manager = None
        self.website_builder = None
        
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
            import PIL
            from streamlit_ace import st_ace
            
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
                self.database_manager = DatabaseManager(
                    db_type=self.config.get("database_type", "sqlite"),
                    db_path=self.config.get("database_path", "data/skyscope.db")
                )
                
                self.agent_manager = AgentManager(
                    agent_count=self.config.get("agent_count", 10000),
                    pipeline_count=self.config.get("pipeline_count", 100),
                    database_manager=self.database_manager
                )
                
                self.crypto_manager = CryptoManager(
                    database_manager=self.database_manager
                )
                
                self.business_manager = BusinessManager(
                    agent_manager=self.agent_manager,
                    crypto_manager=self.crypto_manager,
                    database_manager=self.database_manager
                )
                
                self.performance_monitor = PerformanceMonitor(
                    agent_manager=self.agent_manager,
                    database_manager=self.database_manager
                )
                
                self.rag_system = LiveThinkingRAGSystem(
                    agent_manager=self.agent_manager,
                    database_manager=self.database_manager
                )
                
                # Initialize business operations components
                self.business_idea_generator = BusinessIdeaGenerator(
                    agent_manager=self.agent_manager
                )
                
                self.service_registration_manager = ServiceRegistrationManager(
                    agent_manager=self.agent_manager
                )
                
                self.website_builder = WebsiteBuilder(
                    agent_manager=self.agent_manager
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
            # Generate business ideas
            business_ideas = self.business_idea_generator.generate_business_ideas(
                count=3,
                focus_area="cryptocurrency" if self.config.get("enable_crypto_focus", True) else None
            )
            
            # Create business plans
            business_plans = []
            for idea in business_ideas:
                business_plan = self.business_idea_generator.create_business_plan(idea)
                business_plans.append(business_plan)
                
                # Save business plan
                plan_file = Path(f"business_plans/{business_plan.id}.json")
                try:
                    with open(plan_file, 'w') as f:
                        json.dump(business_plan.to_dict(), f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving business plan: {e}")
            
            # Create businesses and start operations
            for plan in business_plans:
                # Create business
                business = self.business_manager.create_business(
                    name=plan.name,
                    description=plan.description,
                    business_type=plan.business_type,
                    business_plan=plan
                )
                
                # Register services
                for service_type in plan.required_services:
                    self.service_registration_manager.register_service(
                        business=business,
                        service_type=service_type
                    )
                
                # Create website
                self.website_builder.create_website(business)
                
                # Start business operations
                self.business_manager.start_business_operations(business)
            
            # Main autonomous operations loop
            while not self.stop_autonomous:
                # Monitor businesses
                self.business_manager.monitor_businesses()
                
                # Generate reports
                self.business_manager.generate_reports()
                
                # Check for new opportunities
                if random.random() < 0.1:  # 10% chance to generate new ideas
                    new_ideas = self.business_idea_generator.generate_business_ideas(count=1)
                    for idea in new_ideas:
                        business_plan = self.business_idea_generator.create_business_plan(idea)
                        
                        # Save business plan
                        plan_file = Path(f"business_plans/{business_plan.id}.json")
                        try:
                            with open(plan_file, 'w') as f:
                                json.dump(business_plan.to_dict(), f, indent=2)
                        except Exception as e:
                            logger.error(f"Error saving business plan: {e}")
                        
                        # Create business
                        business = self.business_manager.create_business(
                            name=business_plan.name,
                            description=business_plan.description,
                            business_type=business_plan.business_type,
                            business_plan=business_plan
                        )
                        
                        # Register services
                        for service_type in business_plan.required_services:
                            self.service_registration_manager.register_service(
                                business=business,
                                service_type=service_type
                            )
                        
                        # Create website
                        self.website_builder.create_website(business)
                        
                        # Start business operations
                        self.business_manager.start_business_operations(business)
                
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
        
        # Prepare Streamlit arguments
        streamlit_args = [
            "streamlit", "run", 
            "enhanced_chat_interface.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.serverAddress=localhost",
            "--server.headless=true",
            "--theme.base=dark"
        ]
        
        try:
            # Use Streamlit CLI to run the app
            sys.argv = streamlit_args
            stcli.main()
        except Exception as e:
            logger.error(f"Error launching Streamlit interface: {e}")
            
            # Fallback to subprocess
            try:
                logger.info("Attempting to launch Streamlit using subprocess...")
                subprocess.Popen(streamlit_args)
            except Exception as e:
                logger.error(f"Error launching Streamlit using subprocess: {e}")
                return False
        
        return True
    
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
    
    # Import missing modules
    import random
    
    # Run main
    main()
