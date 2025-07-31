#!/bin/bash

# Skyscope Sentinel Intelligence - Production Startup Script
# Business: Skyscope Sentinel Intelligence
# Version: 2.0.0 Production

echo "🚀 Starting Skyscope Sentinel Intelligence..."
echo "Business: Skyscope Sentinel Intelligence"
echo "Version: 2.0.0 Production"
echo "Max Agents: 200,000"
echo "=================================="

# Source environment variables from ~/.zshrc
echo "📋 Loading environment variables..."
source ~/.zshrc

# Check environment variables
if [ -z "$INFURA_API_KEY" ]; then
    echo "❌ Error: INFURA_API_KEY not set"
    echo "Please add to ~/.zshrc: export INFURA_API_KEY=\"your_key_here\""
    exit 1
fi

if [ -z "$SKYSCOPE_WALLET_SEED_PHRASE" ]; then
    echo "❌ Error: SKYSCOPE_WALLET_SEED_PHRASE not set"
    echo "Please add to ~/.zshrc: export SKYSCOPE_WALLET_SEED_PHRASE=\"your_seed_phrase_here\""
    exit 1
fi

if [ -z "$DEFAULT_ETH_ADDRESS" ]; then
    echo "❌ Error: DEFAULT_ETH_ADDRESS not set"
    echo "Please add to ~/.zshrc: export DEFAULT_ETH_ADDRESS=\"your_eth_address_here\""
    exit 1
fi

echo "✅ Environment variables validated"
echo "🔑 INFURA API Key: ${INFURA_API_KEY:0:10}..."
echo "💰 ETH Address: $DEFAULT_ETH_ADDRESS"
echo "💰 BTC Address: $DEFAULT_BTC_ADDRESS"

# Install dependencies if needed
echo "📦 Installing dependencies..."
python3 launch_autonomous_system.py --install-deps

# Start the system
echo "🚀 Launching autonomous system..."
echo "🎯 Target: $10,000 daily income"
echo "💸 Transfer threshold: $1,000"
echo "🤖 Starting with 1,000 agents..."
echo "=================================="

python3 launch_autonomous_system.py --agents 1000 --register-services

echo "🎉 System startup completed!"
echo "📊 Dashboard: http://localhost:8501"
echo "💰 Monitor your autonomous income generation!"
