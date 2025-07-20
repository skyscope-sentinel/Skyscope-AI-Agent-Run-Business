#!/bin/bash
# =============================================================================
# Skyscope Enterprise Suite - Blockchain Environment Setup
# =============================================================================
#
# This script exports all required environment variables for blockchain integration.
#
# Usage:
#   source ./setup_blockchain_env.sh
#
# =============================================================================

# --- Infura API Credentials ---
export INFURA_API_KEY="a4e7502288814c10a9d2246a1dc7d977"
export INFURA_API_SECRET="dummy_secret"

# --- Bitcoin and Ethereum Network Settings ---
export BTC_NETWORK="main"
export ETH_NETWORK="mainnet"

# --- BlockCypher API Token (Optional) ---
export BLOCKCYPHER_TOKEN=""

# --- Wallet Settings ---
export SKYSCOPE_WALLET_MODE="real"
export SKYSCOPE_WALLET_KEY="dummy_key"

# --- Default Wallet Addresses ---
export DEFAULT_BTC_ADDRESS="1B2W1CcNUj15Mc9wYhdH8iR842JECLyo5d"
export DEFAULT_ETH_ADDRESS="0xecadc0c9d0f5c473d51b10d4c9994983cd9d3b8c"
