#!/bin/bash
# =============================================================================
# Skyscope Enterprise Suite - Run with Blockchain Integration
# =============================================================================
#
# This script launches the Skyscope Enterprise Suite with blockchain integration.
# It sources the blockchain environment variables and verifies they're properly
# set before starting the application.
#
# Usage:
#   ./run_with_blockchain.sh
#
# =============================================================================

# Terminal colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}     SKYSCOPE ENTERPRISE SUITE - BLOCKCHAIN INTEGRATION        ${NC}"
echo -e "${BLUE}===============================================================${NC}"

# Check if setup file exists
if [ ! -f "./setup_blockchain_env.sh" ]; then
    echo -e "${RED}ERROR: setup_blockchain_env.sh file not found!${NC}"
    echo -e "${YELLOW}Please ensure you're running this script from the SkyscopeEnterprise directory.${NC}"
    exit 1
fi

# Source the environment variables
echo -e "${YELLOW}Loading blockchain environment variables...${NC}"
source ./setup_blockchain_env.sh

# Verify required environment variables
echo -e "${YELLOW}Verifying blockchain environment setup...${NC}"

REQUIRED_VARS=(
    "INFURA_API_KEY"
    "INFURA_API_SECRET"
    "BTC_NETWORK"
    "ETH_NETWORK"
    "DEFAULT_BTC_ADDRESS"
    "DEFAULT_ETH_ADDRESS"
    "SKYSCOPE_WALLET_KEY"
)

MISSING_VARS=0
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo -e "${RED}ERROR: Required environment variable $VAR is not set!${NC}"
        MISSING_VARS=$((MISSING_VARS + 1))
    fi
done

# Check if wallet mode is set to real
if [ "$SKYSCOPE_WALLET_MODE" != "real" ]; then
    echo -e "${YELLOW}WARNING: SKYSCOPE_WALLET_MODE is not set to 'real'. Using simulation mode.${NC}"
    echo -e "${YELLOW}Set SKYSCOPE_WALLET_MODE=real in setup_blockchain_env.sh for real blockchain transactions.${NC}"
    sleep 2
fi

# Exit if required variables are missing
if [ $MISSING_VARS -gt 0 ]; then
    echo -e "${RED}ERROR: $MISSING_VARS required environment variables are missing.${NC}"
    echo -e "${YELLOW}Please check setup_blockchain_env.sh and ensure all required variables are set.${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Please install Python 3.${NC}"
    exit 1
fi

# Check if main application exists
if [ ! -f "./main_enterprise.py" ]; then
    echo -e "${RED}ERROR: main_enterprise.py file not found!${NC}"
    echo -e "${YELLOW}Please ensure you're running this script from the SkyscopeEnterprise directory.${NC}"
    exit 1
fi

# Check for required Python packages
echo -e "${YELLOW}Checking for required Python packages...${NC}"
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

# Try to import web3 package (required for blockchain integration)
$PYTHON_CMD -c "import web3" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: web3 package not found. Ethereum blockchain integration will be limited.${NC}"
    echo -e "${YELLOW}Consider installing it with: pip install web3${NC}"
    sleep 2
fi

# All checks passed, start the application
echo -e "${GREEN}Blockchain environment verified successfully!${NC}"
echo -e "${GREEN}Starting Skyscope Enterprise Suite with blockchain integration...${NC}"
echo -e "${BLUE}===============================================================${NC}"

# Run the main application
if [ "$PYTHON_CMD" = "python3" ]; then
    python3 main_enterprise.py
else
    python main_enterprise.py
fi

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Application exited with error code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

exit 0
