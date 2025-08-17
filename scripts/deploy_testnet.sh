#!/bin/bash
# Testnet Deployment Script for Luminar Subnet
# Copyright © 2025 Luminar Network

set -e

echo "🚀 Luminar Subnet Testnet Deployment"
echo "======================================"

# Configuration
NETUID=999  # Default testnet netuid (adjust as needed)
WALLET_NAME="testnet_wallet"
MINER_HOTKEY="miner_hotkey"
VALIDATOR_HOTKEY="validator_hotkey"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Setup Environment
setup_environment() {
    echo -e "\n${YELLOW}📋 Step 1: Setting up environment${NC}"
    
    # Copy testnet environment
    cp .env.testnet .env
    print_status "Testnet environment configured"
    
    # Create logs directory
    mkdir -p logs
    print_status "Logs directory created"
    
    # Setup database
    echo "Setting up testnet database..."
    LUMINAR_ENV=testnet python scripts/setup_database.py setup
    print_status "Database setup completed"
}

# Step 2: Check Bittensor Installation
check_bittensor() {
    echo -e "\n${YELLOW}🔧 Step 2: Checking Bittensor installation${NC}"
    
    if ! command -v btcli &> /dev/null; then
        print_error "btcli not found. Installing Bittensor..."
        pip install bittensor
    fi
    
    print_status "Bittensor CLI available"
    btcli --version
}

# Step 3: Setup Wallet
setup_wallet() {
    echo -e "\n${YELLOW}💳 Step 3: Setting up testnet wallet${NC}"
    
    # Check if wallet exists
    if btcli wallet list | grep -q "$WALLET_NAME"; then
        print_status "Wallet '$WALLET_NAME' already exists"
    else
        print_warning "Creating new wallet: $WALLET_NAME"
        btcli wallet new_coldkey --wallet.name $WALLET_NAME
        btcli wallet new_hotkey --wallet.name $WALLET_NAME --wallet.hotkey $MINER_HOTKEY
        btcli wallet new_hotkey --wallet.name $WALLET_NAME --wallet.hotkey $VALIDATOR_HOTKEY
        print_status "Wallet created successfully"
    fi
    
    # Show wallet info
    echo -e "\n📊 Wallet Information:"
    btcli wallet overview --wallet.name $WALLET_NAME --subtensor.network test
}

# Step 4: Get Testnet TAO
get_testnet_tao() {
    echo -e "\n${YELLOW}🪙 Step 4: Getting testnet TAO${NC}"
    
    print_warning "You need testnet TAO to register on the subnet"
    echo "1. Visit the faucet: https://faucet.bittensor.com/"
    echo "2. Enter your coldkey address"
    echo "3. Request testnet TAO"
    
    # Get coldkey address
    COLDKEY_ADDRESS=$(btcli wallet overview --wallet.name $WALLET_NAME --subtensor.network test | grep "COLDKEY" -A 1 | tail -1 | awk '{print $1}')
    echo -e "\n🔑 Your coldkey address: ${GREEN}$COLDKEY_ADDRESS${NC}"
    
    read -p "Press Enter after getting testnet TAO from the faucet..."
}

# Step 5: Register on Subnet
register_subnet() {
    echo -e "\n${YELLOW}📝 Step 5: Registering on subnet${NC}"
    
    # Register miner
    echo "Registering miner..."
    btcli subnet register --wallet.name $WALLET_NAME --wallet.hotkey $MINER_HOTKEY --subtensor.network test --netuid $NETUID
    
    # Register validator (if you have enough TAO)
    echo "Registering validator..."
    btcli subnet register --wallet.name $WALLET_NAME --wallet.hotkey $VALIDATOR_HOTKEY --subtensor.network test --netuid $NETUID
    
    print_status "Registration completed"
}

# Step 6: Start Miner
start_miner() {
    echo -e "\n${YELLOW}⛏️  Step 6: Starting Luminar Miner${NC}"
    
    echo "Starting miner in background..."
    nohup python neurons/miner.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $WALLET_NAME \
        --wallet.hotkey $MINER_HOTKEY \
        --logging.debug \
        --axon.port 8091 > logs/miner_testnet.log 2>&1 &
    
    MINER_PID=$!
    echo $MINER_PID > logs/miner.pid
    print_status "Miner started (PID: $MINER_PID)"
}

# Step 7: Start Validator
start_validator() {
    echo -e "\n${YELLOW}✅ Step 7: Starting Luminar Validator${NC}"
    
    echo "Starting validator in background..."
    nohup python neurons/validator.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $WALLET_NAME \
        --wallet.hotkey $VALIDATOR_HOTKEY \
        --logging.debug \
        --neuron.sample_size 10 > logs/validator_testnet.log 2>&1 &
    
    VALIDATOR_PID=$!
    echo $VALIDATOR_PID > logs/validator.pid
    print_status "Validator started (PID: $VALIDATOR_PID)"
}

# Step 8: Monitor
monitor_subnet() {
    echo -e "\n${YELLOW}📊 Step 8: Monitoring${NC}"
    
    print_status "Luminar subnet is now running on testnet!"
    
    echo -e "\n📋 Quick Commands:"
    echo "• Monitor miner: tail -f logs/miner_testnet.log"
    echo "• Monitor validator: tail -f logs/validator_testnet.log"
    echo "• Check subnet: btcli subnet list --subtensor.network test"
    echo "• Check wallet: btcli wallet overview --wallet.name $WALLET_NAME --subtensor.network test"
    
    echo -e "\n🔗 Useful Links:"
    echo "• Testnet Explorer: https://explorer.bittensor.com/"
    echo "• TAO Faucet: https://faucet.bittensor.com/"
    echo "• Documentation: https://docs.luminar.network/"
    
    echo -e "\n${GREEN}🎉 Testnet deployment completed successfully!${NC}"
}

# Stop function
stop_subnet() {
    echo "Stopping Luminar subnet..."
    
    if [ -f logs/miner.pid ]; then
        kill $(cat logs/miner.pid) 2>/dev/null || true
        rm logs/miner.pid
        print_status "Miner stopped"
    fi
    
    if [ -f logs/validator.pid ]; then
        kill $(cat logs/validator.pid) 2>/dev/null || true
        rm logs/validator.pid
        print_status "Validator stopped"
    fi
}

# Main execution
case "${1:-deploy}" in
    "deploy")
        setup_environment
        check_bittensor
        setup_wallet
        get_testnet_tao
        register_subnet
        start_miner
        start_validator
        monitor_subnet
        ;;
    "stop")
        stop_subnet
        ;;
    "monitor")
        monitor_subnet
        ;;
    *)
        echo "Usage: $0 {deploy|stop|monitor}"
        echo "  deploy  - Full testnet deployment"
        echo "  stop    - Stop running subnet"
        echo "  monitor - Show monitoring info"
        exit 1
        ;;
esac
