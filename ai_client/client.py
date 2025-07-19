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
from datetime import datetime

# Import our components
from ai_client.client import ai_client
from crypto_wallet_manager.wallet import wallet_manager
from agent_swarm_manager.swarm import AgentSwarmManager

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

def main():
    """Main entry point for the application."""
    print("=" * 80)
    print("SKYSCOPE AI AGENT SWARM - AUTONOMOUS INCOME GENERATION SYSTEM")
    print("=" * 80)
    print("\nInitializing components...\n")
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("deployments", exist_ok=True)
    os.makedirs("wallets", exist_ok=True)
    
    # Initialize wallet manager and create wallets
    print("Setting up cryptocurrency wallets...")
    wallets = wallet_manager.setup_default_wallets()
    print(f"✅ Created {len(wallets)} cryptocurrency wallets for receiving real-world income")
    
    # Initialize agent swarm
    print("\nInitializing AI agent swarm...")
    swarm_manager = AgentSwarmManager(max_agents=10000)
    
    # Create initial agents
    print("Creating initial agent population...")
    num_agents = swarm_manager.create_initial_swarm(num_agents=100)
    print(f"✅ Created {num_agents} AI agents across various specializations")
    
    # Generate initial income tasks
    print("\nGenerating initial income-generating tasks...")
    income_tasks = swarm_manager.generate_income_tasks(count=20)
    print(f"✅ Created {len(income_tasks)} income-focused tasks")
    
    # Generate business creation tasks
    print("Generating business creation tasks...")
    business_tasks = swarm_manager.generate_business_creation_tasks(count=5)
    print(f"✅ Created {len(business_tasks)} business creation tasks")
    
    # Start the agent swarm
    print("\nStarting autonomous agent swarm operations...")
    swarm_manager.start(num_workers=10)
    print("✅ Agent swarm is now running and generating income")
    
    # Main monitoring loop
    print("\n" + "=" * 80)
    print("SYSTEM IS NOW RUNNING AUTONOMOUSLY")
    print("=" * 80)
    print("\nThe AI agent swarm is now operating autonomously to generate real-world income.")
    print("All earnings will be directed to the cryptocurrency wallets created earlier.")
    print("\nMonitoring system status (press Ctrl+C to exit)...\n")
    
    try:
        while True:
            # Get current earnings
            earnings = swarm_manager.get_earnings_summary()
            
            # Display status
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status Update:")
            print(f"Total Earnings: ${earnings['total_earnings']:.2f}")
            print("Wallet Balances:")
            for name, wallet in earnings['wallet_balances'].items():
                print(f"  - {name}: {wallet['balance']} {wallet['cryptocurrency']}")
            
            # Sleep for a while
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\nShutting down agent swarm...")
        swarm_manager.stop()
        print("Agent swarm stopped.")
        
        # Save final earnings report
        earnings = swarm_manager.get_earnings_summary()
        with open("final_earnings_report.json", "w") as f:
            json.dump(earnings, f, indent=2)
        
        print(f"\nFinal earnings report saved to: final_earnings_report.json")
        print(f"Total earnings: ${earnings['total_earnings']:.2f}")
        print("\nThank you for using Skyscope AI Agent Swarm!")

if __name__ == "__main__":
    main()

