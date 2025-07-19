#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup Real Cryptocurrency Trading
=================================

This script sets up real cryptocurrency wallets and trading capabilities
for the Skyscope AI Agentic Swarm Business/Enterprise system.

⚠️ WARNING: This will create real cryptocurrency wallets and enable real trading.
Only run this if you understand the risks and want to use real money.
"""

import os
import sys
import json
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_warning(message):
    print(f"⚠️  {message}")

def print_success(message):
    print(f"✅ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n]: ").lower().strip()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'")

def setup_real_crypto_system():
    """Set up the complete real cryptocurrency system"""
    
    print("🚀 Skyscope AI Agentic Swarm Business/Enterprise")
    print("Real Cryptocurrency Setup")
    print("=" * 50)
    
    print("\n🔐 IMPORTANT SECURITY NOTICE:")
    print("This setup will create REAL cryptocurrency wallets with REAL seed phrases.")
    print("These wallets can hold and transact REAL money.")
    print("You are responsible for securing your seed phrases and private keys.")
    
    if not ask_yes_no("\nDo you understand and want to continue?"):
        print("Setup cancelled.")
        return
    
    print_header("Installing Required Dependencies")
    
    # Install required packages
    required_packages = [
        "mnemonic",
        "eth-account",
        "bitcoinlib",
        "ccxt"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_success(f"{package} is already installed")
        except ImportError:
            print_info(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    print_header("Creating Real Cryptocurrency Wallets")
    
    try:
        from real_crypto_wallet_manager import setup_real_crypto_wallets
        wallet_manager = setup_real_crypto_wallets()
        print_success("Real cryptocurrency wallets created successfully!")
    except Exception as e:
        print(f"❌ Error creating wallets: {e}")
        return
    
    print_header("Setting Up Real Trading Engine")
    
    try:
        from real_trading_engine import setup_real_trading
        trading_engine = setup_real_trading()
        print_success("Real trading engine configured successfully!")
    except Exception as e:
        print(f"❌ Error setting up trading: {e}")
        return
    
    print_header("Configuration Instructions")
    
    print("\n📋 NEXT STEPS TO ENABLE REAL TRADING:")
    print("\n1. 📁 Check your wallet files:")
    print("   - Look in the 'wallets/' directory")
    print("   - Each wallet has a '_WALLET_INFO.txt' file")
    print("   - These contain your seed phrases and addresses")
    
    print("\n2. 🔐 Secure your seed phrases:")
    print("   - Write down your 24-word seed phrases")
    print("   - Store them in a secure location")
    print("   - Never share them with anyone")
    print("   - These are the keys to your money!")
    
    print("\n3. 💰 Fund your wallets (optional):")
    print("   - Send cryptocurrency to your wallet addresses")
    print("   - Start with small amounts for testing")
    print("   - The system can trade with any amount")
    
    print("\n4. 🔑 Configure exchange API keys:")
    print("   - Edit 'config/trading_config.json'")
    print("   - Add your exchange API keys")
    print("   - Start with sandbox/testnet mode")
    
    print("\n5. ⚡ Enable real trading:")
    print("   - In the GUI, go to Settings")
    print("   - Enable 'Real Trading Mode'")
    print("   - Monitor carefully!")
    
    print_header("Wallet Information")
    
    # Show wallet addresses
    wallets = wallet_manager.list_wallets()
    for wallet_name in wallets:
        wallet_info = wallet_manager.get_wallet(wallet_name)
        if wallet_info:
            print(f"\n💼 {wallet_info['cryptocurrency']} Wallet ({wallet_name}):")
            print(f"   Address: {wallet_info['address']}")
            print(f"   File: wallets/{wallet_name}_WALLET_INFO.txt")
    
    print_header("Security Reminders")
    
    print("\n🔒 CRITICAL SECURITY REMINDERS:")
    print("   • Your seed phrases are stored in the wallet files")
    print("   • Back up these files to a secure location")
    print("   • Never share your seed phrases or private keys")
    print("   • Start with small amounts for testing")
    print("   • Monitor all trades and transactions")
    print("   • You are responsible for your funds")
    
    print_header("Starting the System")
    
    print("\n🚀 TO START THE REAL CRYPTO SYSTEM:")
    print("   1. Run: ./START_SYSTEM.sh")
    print("   2. In the GUI, configure your API keys")
    print("   3. Enable real trading mode")
    print("   4. Monitor the debug console for real transactions")
    
    print("\n✨ Your AI agents will now:")
    print("   • Execute real cryptocurrency trades")
    print("   • Generate real income")
    print("   • Deposit earnings to your wallets")
    print("   • Show real-time profit/loss")
    
    if ask_yes_no("\nWould you like to start the system now?"):
        print("\n🚀 Starting Skyscope AI Agentic Swarm with Real Crypto...")
        os.system("./START_SYSTEM.sh")
    else:
        print("\n✅ Setup complete! Run './START_SYSTEM.sh' when ready.")

def create_example_config():
    """Create example configuration files"""
    
    # Create trading config
    trading_config = {
        "trading_enabled": False,
        "exchanges": {
            "binance": {
                "api_key": "YOUR_BINANCE_API_KEY",
                "api_secret": "YOUR_BINANCE_SECRET_KEY",
                "sandbox": True,
                "enabled": False
            },
            "coinbase": {
                "api_key": "YOUR_COINBASE_API_KEY",
                "api_secret": "YOUR_COINBASE_SECRET_KEY",
                "passphrase": "YOUR_COINBASE_PASSPHRASE",
                "sandbox": True,
                "enabled": False
            }
        },
        "trading_pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.05,
            "max_daily_trades": 50
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/trading_config.json", "w") as f:
        json.dump(trading_config, f, indent=2)
    
    print_success("Created config/trading_config.json")

if __name__ == "__main__":
    try:
        create_example_config()
        setup_real_crypto_system()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again.")