# OPTIMIZED BY SYSTEM INTEGRATION
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Sentinel Intelligence - Production Launcher
==================================================

Production-grade launcher for the autonomous AI agent swarm system.
Handles system initialization, monitoring, and management.

Business: Skyscope Sentinel Intelligence
Version: 2.0.0 Production
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/system_launcher.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SystemLauncher')

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("backups", exist_ok=True)

class SystemLauncher:
    """Production system launcher and manager"""
    
    def __init__(self):
        """Initialize the system launcher"""
        self.system = None
        self.dashboard_process = None
        self.running = False
        self.shutdown_requested = False
        
        # System configuration
        self.config = {
            "business_name": "Skyscope Sentinel Intelligence",
            "max_agents": 200000,
            "initial_agents": 1000,
            "auto_scale": True,
            "enable_gui": True,
            "enable_mev_bots": True,
            "enable_service_registration": True,
            "backup_interval": 3600,  # 1 hour
            "health_check_interval": 300,  # 5 minutes
        }
        
        # Environment validation
        self.required_env_vars = [
            "INFURA_API_KEY",
            "SKYSCOPE_WALLET_SEED_PHRASE",
            "DEFAULT_BTC_ADDRESS",
            "DEFAULT_ETH_ADDRESS"
        ]
        
        logger.info("System Launcher initialized")
    
    def validate_environment(self) -> bool:
        """Validate required environment variables"""
        missing_vars = []
        
        for var in self.required_env_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Please ensure all required variables are set in ~/.zshrc")
            return False
        
        logger.info("Environment validation passed")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        try:
            logger.info("Installing required dependencies...")
            
            # Core dependencies
            core_packages = [
                "web3>=6.0.0",
                "eth-account>=0.8.0",
                "ccxt>=4.0.0",
                "mnemonic>=0.20",
                "streamlit>=1.28.0",
                "plotly>=5.15.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "requests>=2.31.0",
                "schedule>=1.2.0"
            ]
            
            for package in core_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
            
            logger.info("Dependencies installation completed")
            return True
        
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def start_system(self, args: argparse.Namespace) -> bool:
        """Start the autonomous system"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 STARTING SKYSCOPE SENTINEL INTELLIGENCE")
            logger.info("=" * 60)
            logger.info(f"Business: {self.config['business_name']}")
            logger.info(f"Max Agents: {args.max_agents:,}")
            logger.info(f"Initial Agents: {args.agents:,}")
            logger.info(f"GUI Enabled: {not args.headless}")
            logger.info("=" * 60)
            
            # Import and initialize core system
            from core_autonomous_system import CoreAutonomousSystem, ServiceRegistrationSystem
            
            # Update configuration
            self.config["max_agents"] = args.max_agents
            self.config["initial_agents"] = args.agents
            self.config["enable_gui"] = not args.headless
            
            # Initialize system
            self.system = CoreAutonomousSystem()
            self.system.config["initial_agents"] = args.agents
            self.system.config["max_agents"] = args.max_agents
            
            # Register for services if requested
            if args.register_services or self.config["enable_service_registration"]:
                logger.info("Registering for required services...")
                registration_system = ServiceRegistrationSystem()
                registration_system.register_all_services()
            
            # Start the system
            self.system.start()
            self.running = True
            
            # Start GUI if enabled
            if self.config["enable_gui"] and not args.headless:
                self._start_gui()
            
            # Start monitoring threads
            self._start_monitoring()
            
            logger.info("✅ System started successfully!")
            logger.info(f"🌐 Dashboard: http://localhost:8501")
            logger.info(f"📊 Monitoring: Active")
            logger.info(f"💰 Target: $10,000 daily income")
            
            return True
        
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False
    
    def _start_gui(self):
        """Start the GUI dashboard"""
        try:
            logger.info("Starting GUI dashboard...")
            
            # Start Streamlit dashboard
            dashboard_cmd = [
                sys.executable, "-m", "streamlit", "run",
                "enhanced_gui_dashboard.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            self.dashboard_process = subprocess.Popen(
                dashboard_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            logger.info("GUI dashboard started on http://localhost:8501")
        
        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
    
    def _start_monitoring(self):
        """Start system monitoring threads"""
        try:
            # Health check thread
            health_thread = threading.Thread(target=self._health_check_loop)
            health_thread.daemon = True
            health_thread.start()
            
            # Backup thread
            backup_thread = threading.Thread(target=self._backup_loop)
            backup_thread.daemon = True
            backup_thread.start()
            
            # Performance monitoring thread
            perf_thread = threading.Thread(target=self._performance_monitoring_loop)
            perf_thread.daemon = True
            perf_thread.start()
            
            logger.info("Monitoring threads started")
        
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        while self.running and not self.shutdown_requested:
            try:
                if self.system:
                    status = self.system.get_status()
                    metrics = status["metrics"]
                    
                    # Check system health
                    health_issues = []
                    
                    # Check if agents are running
                    if metrics["total_agents"] == 0:
                        health_issues.append("No agents running")
                    
                    # Check if income is being generated
                    if metrics["total_income_usd"] == 0 and metrics["uptime_hours"] > 1:
                        health_issues.append("No income generated after 1 hour")
                    
                    # Check success rate
                    if metrics["success_rate"] < 0.5:
                        health_issues.append(f"Low success rate: {metrics['success_rate']:.1%}")
                    
                    # Log health status
                    if health_issues:
                        logger.warning(f"Health issues detected: {', '.join(health_issues)}")
                    else:
                        logger.info("System health: OK")
                
                time.sleep(self.config["health_check_interval"])
            
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                time.sleep(60)
    
    def _backup_loop(self):
        """Backup system data loop"""
        while self.running and not self.shutdown_requested:
            try:
                self._create_backup()
                time.sleep(self.config["backup_interval"])
            
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running and not self.shutdown_requested:
            try:
                if self.system:
                    status = self.system.get_status()
                    metrics = status["metrics"]
                    
                    # Log performance metrics
                    logger.info(f"PERFORMANCE - Agents: {metrics['active_agents']}/{metrics['total_agents']}, "
                              f"Income: ${metrics['total_income_usd']:.2f}, "
                              f"Daily Rate: ${metrics['daily_income_usd']:.2f}, "
                              f"Success: {metrics['success_rate']:.1%}")
                    
                    # Save metrics to file
                    self._save_metrics(metrics)
                
                time.sleep(300)  # Every 5 minutes
            
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(60)
    
    def _create_backup(self):
        """Create system backup"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("backups") / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup system status
            if self.system:
                status = self.system.get_status()
                with open(backup_dir / "system_status.json", 'w') as f:
                    json.dump(status, f, indent=2, default=str)
            
            # Backup configuration
            with open(backup_dir / "config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Backup logs (last 1000 lines)
            try:
                with open("logs/system_launcher.log", 'r') as f:
                    lines = f.readlines()
                    with open(backup_dir / "recent_logs.txt", 'w') as backup_f:
                        backup_f.writelines(lines[-1000:])
            except Exception:
                pass
            
            logger.info(f"Backup created: {backup_dir}")
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        try:
            metrics_file = Path("data") / "metrics.jsonl"
            
            metric_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **metrics
            }
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metric_entry, default=str) + '\n')
        
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def stop_system(self):
        """Stop the autonomous system"""
        try:
            logger.info("Stopping Skyscope Sentinel Intelligence...")
            
            self.shutdown_requested = True
            self.running = False
            
            # Stop the core system
            if self.system:
                self.system.stop()
            
            # Stop GUI dashboard
            if self.dashboard_process:
                self.dashboard_process.terminate()
                try:
                    self.dashboard_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.dashboard_process.kill()
            
            # Create final backup
            self._create_backup()
            
            logger.info("System stopped successfully")
        
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def run_interactive_mode(self):
        """Run in interactive mode with command interface"""
        logger.info("Starting interactive mode...")
        logger.info("Available commands: status, agents, income, stop, help")
        
        while self.running and not self.shutdown_requested:
            try:
                command = input("\nSkyscope> ").strip().lower()
                
                if command == "status":
                    self._show_status()
                elif command == "agents":
                    self._show_agents()
                elif command == "income":
                    self._show_income()
                elif command == "stop":
                    break
                elif command == "help":
                    self._show_help()
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
    
    def _show_status(self):
        """Show system status"""
        if self.system:
            status = self.system.get_status()
            metrics = status["metrics"]
            
            print(f"\n=== SYSTEM STATUS ===")
            print(f"Business: {status['business_name']}")
            print(f"Running: {status['running']}")
            print(f"Agents: {metrics['active_agents']}/{metrics['total_agents']}")
            print(f"Total Income: ${metrics['total_income_usd']:.2f}")
            print(f"Daily Rate: ${metrics['daily_income_usd']:.2f}")
            print(f"Success Rate: {metrics['success_rate']:.1%}")
            print(f"Uptime: {metrics['uptime_hours']:.1f} hours")
        else:
            print("System not running")
    
    def _show_agents(self):
        """Show agent information"""
        if self.system:
            status = self.system.get_status()
            top_agents = status["top_agents"]
            
            print(f"\n=== TOP PERFORMING AGENTS ===")
            for i, agent in enumerate(top_agents, 1):
                print(f"{i}. {agent['agent_id']} - {agent['strategy'].replace('_', ' ').title()}")
                print(f"   Income: ${agent['total_income']:.2f}, Success: {agent['success_rate']:.1%}")
        else:
            print("System not running")
    
    def _show_income(self):
        """Show income information"""
        if self.system:
            status = self.system.get_status()
            metrics = status["metrics"]
            wallet_balances = status["wallet_balances"]
            
            print(f"\n=== INCOME SUMMARY ===")
            print(f"Total Generated: ${metrics['total_income_usd']:.2f}")
            print(f"Daily Rate: ${metrics['daily_income_usd']:.2f}")
            print(f"Wallet Balance: ${wallet_balances.get('total_usd', 0):.2f}")
            
            if metrics.get('last_transfer_amount', 0) > 0:
                print(f"Last Transfer: ${metrics['last_transfer_amount']:.2f}")
                print(f"Transfer Time: {metrics.get('last_transfer_time', 'N/A')}")
        else:
            print("System not running")
    
    def _show_help(self):
        """Show help information"""
        print(f"\n=== AVAILABLE COMMANDS ===")
        print("status  - Show system status")
        print("agents  - Show top performing agents")
        print("income  - Show income summary")
        print("stop    - Stop the system")
        print("help    - Show this help message")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'launcher'):
        signal_handler.launcher.stop_system()
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Skyscope Sentinel Intelligence - Autonomous AI Agent Swarm System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_autonomous_system.py --agents 1000
  python launch_autonomous_system.py --max-agents 50000 --register-services
  python launch_autonomous_system.py --headless --agents 5000
        """
    )
    
    parser.add_argument("--agents", type=int, default=1000,
                       help="Initial number of agents (default: 1000)")
    parser.add_argument("--max-agents", type=int, default=200000,
                       help="Maximum number of agents (default: 200000)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI dashboard")
    parser.add_argument("--register-services", action="store_true",
                       help="Register for required services on startup")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install required dependencies")
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = SystemLauncher()
    signal_handler.launcher = launcher
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Install dependencies if requested
        if args.install_deps:
            if not launcher.install_dependencies():
                sys.exit(1)
        
        # Validate environment
        if not launcher.validate_environment():
            logger.error("Environment validation failed. Please check your ~/.zshrc configuration.")
            sys.exit(1)
        
        # Start system
        if not launcher.start_system(args):
            logger.error("Failed to start system")
            sys.exit(1)
        
        # Run interactive mode or keep running
        if args.interactive:
            launcher.run_interactive_mode()
        else:
            # Keep running until interrupted
            try:
                while launcher.running:
                    time.sleep(60)
                    
                    # Print status every 10 minutes
                    if time.time() % 600 < 60:
                        if launcher.system:
                            status = launcher.system.get_status()
                            metrics = status["metrics"]
                            print(f"\n🚀 SKYSCOPE STATUS: {metrics['active_agents']}/{metrics['total_agents']} agents, "
                                  f"${metrics['total_income_usd']:.2f} earned, "
                                  f"${metrics['daily_income_usd']:.2f}/day rate")
            
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    
    finally:
        launcher.stop_system()

if __name__ == "__main__":
    main()
