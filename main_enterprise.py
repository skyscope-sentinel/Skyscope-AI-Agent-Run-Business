#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope AI Agent Swarm - Main Application
==========================================

This is the main entry point for the Skyscope AI Agent Swarm system.
It initializes all components and starts the autonomous income generation process.
"""

import os
import sys
import time
import json
import logging
import signal
import threading
from datetime import datetime
import colorama
from colorama import Fore, Style, Back
import shutil
from tqdm import tqdm

# Import our components
from ai_client import ai_client
from crypto_wallet_manager import wallet_manager
from agent_swarm_manager import agent_swarm_manager

# Initialize colorama for colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("skyscope_ai_swarm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SkyscopeMain")

# ASCII Art for banner
BANNER = f"""
{Fore.CYAN}
 ███████╗██╗  ██╗██╗   ██╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗
 ██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ███████╗█████╔╝  ╚████╔╝ ███████╗██║     ██║   ██║██████╔╝█████╗  
 ╚════██║██╔═██╗   ╚██╔╝  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝  
 ███████║██║  ██╗   ██║   ███████║╚██████╗╚██████╔╝██║     ███████╗
 ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝
                                                                   
     █████╗ ██╗    ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗
    ██╔══██╗██║    ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║
    ███████║██║    ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║
    ██╔══██║██║    ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║
    ██║  ██║██████╗███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
    ╚═╝  ╚═╝╚═════╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
{Style.RESET_ALL}
"""

def print_header(text):
    """Print a formatted header."""
    terminal_width = shutil.get_terminal_size().columns
    print(f"\n{Fore.BLUE}{'═' * terminal_width}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}  {text}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'═' * terminal_width}{Style.RESET_ALL}\n")

def print_success(text):
    """Print a success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print an info message."""
    print(f"{Fore.CYAN}ℹ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print an error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutting down agent swarm...")
    agent_swarm_manager.stop()
    print("Agent swarm stopped.")
    
    # Save final earnings report
    earnings = agent_swarm_manager.get_earnings_summary()
    with open("final_earnings_report.json", "w") as f:
        json.dump(earnings, f, indent=2)
    
    print(f"\nFinal earnings report saved to: final_earnings_report.json")
    print(f"Total earnings: ${earnings['total_earnings']:.2f}")
    print("\nThank you for using Skyscope AI Agent Swarm!")
    sys.exit(0)

def display_system_status():
    """Display detailed system status in a separate thread."""
    try:
        # Get terminal size
        terminal_width = shutil.get_terminal_size().columns
        
        # Get status information
        earnings = agent_swarm_manager.get_earnings_summary()
        agent_status = agent_swarm_manager.get_agent_status_summary()
        department_status = agent_swarm_manager.get_department_status()
        queue_size = agent_swarm_manager.get_queue_size()
        active_tasks = agent_swarm_manager.get_active_tasks_count()
        
        # Clear screen
        print("\033[H\033[J", end="")
        
        # Print header
        print(f"{Fore.CYAN}{'=' * terminal_width}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}SKYSCOPE AI AGENT SWARM - SYSTEM STATUS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * terminal_width}{Style.RESET_ALL}")
        
        # Print time
        print(f"\n{Fore.YELLOW}Current Time:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print agent status
        print(f"\n{Fore.YELLOW}Agent Status:{Style.RESET_ALL}")
        print(f"Total Agents: {len(agent_swarm_manager.agents)}")
        print(f"Working: {agent_status.get('working', 0)}")
        print(f"Idle: {agent_status.get('idle', 0)}")
        
        # Print task status
        print(f"\n{Fore.YELLOW}Task Status:{Style.RESET_ALL}")
        print(f"Active Tasks: {active_tasks}")
        print(f"Queued Tasks: {queue_size}")
        print(f"Completed Tasks: {len(agent_swarm_manager.results)}")
        
        # Print earnings
        print(f"\n{Fore.YELLOW}Earnings:{Style.RESET_ALL}")
        print(f"Total Earnings: ${earnings['total_earnings']:.2f}")
        
        # Print wallet balances
        print(f"\n{Fore.YELLOW}Wallet Balances:{Style.RESET_ALL}")
        for name, wallet in earnings['wallet_balances'].items():
            print(f"{name}: {wallet['balance']} {wallet['cryptocurrency']}")
        
        # Print department performance
        print(f"\n{Fore.YELLOW}Department Performance:{Style.RESET_ALL}")
        for dept, info in department_status.items():
            if info['earnings'] > 0:
                earnings_bar = "█" * min(int(info['earnings'] / 10), 20)
                print(f"{dept:<20}: {Fore.GREEN}${info['earnings']:.2f}{Style.RESET_ALL} {Fore.GREEN}{earnings_bar}{Style.RESET_ALL} ({info['working']}/{info['total_agents']} agents working)")
        
        # Print footer
        print(f"\n{Fore.CYAN}{'=' * terminal_width}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Press Ctrl+C to exit{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"Error displaying system status: {e}")

def status_monitor_thread():
    """Thread that periodically updates the system status display."""
    while True:
        try:
            display_system_status()
            time.sleep(5)  # Update every 5 seconds
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in status monitor thread: {e}")
            time.sleep(10)  # Longer delay on error

def create_progress_bar(percentage, width=40):
    """Create a colorful progress bar."""
    filled_width = int(width * percentage / 100)
    empty_width = width - filled_width
    
    # Choose color based on percentage
    if percentage < 30:
        color = Fore.RED
    elif percentage < 70:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
        
    bar = f"{color}{'█' * filled_width}{Style.RESET_ALL}{'░' * empty_width}"
    return f"[{bar}] {percentage}%"

def main():
    """Main entry point for the application."""
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Display banner
    print(BANNER)
    
    print_header("AUTONOMOUS INCOME GENERATION SYSTEM")
    print_info("Initializing components...\n")
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("deployments", exist_ok=True)
    os.makedirs("wallets", exist_ok=True)
    
    # Initialize wallet manager and create wallets
    print_info("Setting up cryptocurrency wallets...")
    wallets = wallet_manager.setup_default_wallets()
    print_success(f"Created {len(wallets)} cryptocurrency wallets for receiving real-world income")
    
    # Initialize agent swarm
    print_info("\nInitializing AI agent swarm...")
    
    # Create initial agents
    print_info("Creating initial agent population...")
    num_agents = agent_swarm_manager.create_initial_swarm(num_agents=100)
    print_success(f"Created {num_agents} AI agents across various specializations")
    
    # Generate initial income tasks
    print_info("\nGenerating initial income-generating tasks...")
    income_tasks = agent_swarm_manager.generate_income_tasks(count=20)
    print_success(f"Created {len(income_tasks)} income-focused tasks")
    
    # Generate business creation tasks
    print_info("Generating business creation tasks...")
    business_tasks = agent_swarm_manager.generate_business_creation_tasks(count=5)
    print_success(f"Created {len(business_tasks)} business creation tasks")
    
    # Start the agent swarm
    print_info("\nStarting autonomous agent swarm operations...")
    agent_swarm_manager.start(num_workers=10)
    print_success("Agent swarm is now running and generating income")
    
    # Main monitoring loop
    print_header("SYSTEM IS NOW RUNNING AUTONOMOUSLY")
    print_info("The AI agent swarm is now operating autonomously to generate real-world income.")
    print_info("All earnings will be directed to the cryptocurrency wallets created earlier.")
    print_info("\nStarting real-time monitoring (press Ctrl+C to exit)...\n")
    
    # Start status monitor in a separate thread
    monitor_thread = threading.Thread(target=status_monitor_thread, daemon=True)
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
