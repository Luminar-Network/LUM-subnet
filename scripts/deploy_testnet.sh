#!/bin/bash
# Testnet Deployment Script for Luminar Subnet
# Deploys on Luminar Network (netuid 414) 
# Copyright © 2025 Luminar Network

set -e

echo "🚀 Luminar Subnet Testnet Deployment (netuid 414)"
echo "=================================================="

# Configuration
NETUID=414  # Luminar Network testnet netuid
WALLET_NAME="miner"  # Default wallet name as per README
VALIDATOR_WALLET="validator"  # Validator wallet name
HOTKEY_NAME="default"  # Default hotkey name

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
    if [ -f .env.testnet ]; then
        cp .env.testnet .env
        print_status "Testnet environment configured (.env.testnet → .env)"
    else
        print_warning ".env.testnet not found, using default configuration"
    fi
    
    # Create logs directory
    mkdir -p logs
    print_status "Logs directory created"
    
    # Setup database
    echo "Setting up testnet database..."
    if [ -f scripts/setup_database.py ]; then
        LUMINAR_ENV=testnet python scripts/setup_database.py setup
        print_status "Database setup completed"
    else
        print_warning "Database setup script not found, skipping database setup"
    fi
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
    echo -e "\n${YELLOW}💳 Step 3: Setting up testnet wallets${NC}"
    
    # Setup miner wallet
    if btcli wallet list | grep -q "$WALLET_NAME"; then
        print_status "Miner wallet '$WALLET_NAME' already exists"
    else
        print_warning "Creating new miner wallet: $WALLET_NAME"
        btcli wallet new_coldkey --wallet.name $WALLET_NAME
        btcli wallet new_hotkey --wallet.name $WALLET_NAME --wallet.hotkey $HOTKEY_NAME
        print_status "Miner wallet created successfully"
    fi
    
    # Setup validator wallet  
    if btcli wallet list | grep -q "$VALIDATOR_WALLET"; then
        print_status "Validator wallet '$VALIDATOR_WALLET' already exists"
    else
        print_warning "Creating new validator wallet: $VALIDATOR_WALLET"
        btcli wallet new_coldkey --wallet.name $VALIDATOR_WALLET
        btcli wallet new_hotkey --wallet.name $VALIDATOR_WALLET --wallet.hotkey $HOTKEY_NAME
        print_status "Validator wallet created successfully"
    fi
    
    # Show wallet info
    echo -e "\n📊 Wallet Information:"
    btcli wallet balance --subtensor.network test --wallet.name $WALLET_NAME
    btcli wallet balance --subtensor.network test --wallet.name $VALIDATOR_WALLET
}

# Step 4: Get Testnet TAO
get_testnet_tao() {
    echo -e "\n${YELLOW}🪙 Step 4: Getting testnet TAO${NC}"
    
    print_warning "You need testnet TAO to register on subnet 414 (Luminar Network)"
    echo "Registration cost: ~0.0717 τ per registration"
    echo ""
    echo "Options to get testnet TAO:"
    echo "1. Faucet (if available): btcli wallet faucet --subtensor.network test --wallet.name $WALLET_NAME"
    echo "2. Transfer between wallets: btcli wallet transfer --subtensor.network test --wallet.name source --dest destination_address --amount 0.1"
    
    # Get coldkey addresses
    echo -e "\n🔑 Your wallet addresses:"
    btcli wallet overview --wallet.name $WALLET_NAME --subtensor.network test | grep -A 1 "COLDKEY"
    btcli wallet overview --wallet.name $VALIDATOR_WALLET --subtensor.network test | grep -A 1 "COLDKEY"
    
    read -p "Press Enter after getting testnet TAO..."
}

# Step 5: Register on Subnet
register_subnet() {
    echo -e "\n${YELLOW}📝 Step 5: Registering on Luminar Network (subnet 414)${NC}"
    
    # Register miner
    echo "Registering miner on subnet 414..."
    btcli subnet register \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $WALLET_NAME \
        --wallet.hotkey $HOTKEY_NAME
    
    # Register validator
    echo "Registering validator on subnet 414..."
    btcli subnet register \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $VALIDATOR_WALLET \
        --wallet.hotkey $HOTKEY_NAME
    
    print_status "Registration completed"
    
    # Verify registration
    echo -e "\n📊 Verifying registration..."
    btcli subnet metagraph --subtensor.network test --netuid $NETUID
}

# Step 6: Start Miner
start_miner() {
    echo -e "\n${YELLOW}⛏️  Step 6: Starting Luminar Miner${NC}"
    
    echo "Starting miner in background..."
    nohup python neurons/miner.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $WALLET_NAME \
        --wallet.hotkey $HOTKEY_NAME \
        --logging.debug \
        --axon.port 8091 > logs/miner_testnet.log 2>&1 &
    
    MINER_PID=$!
    echo $MINER_PID > logs/miner.pid
    print_status "Miner started (PID: $MINER_PID)"
    
    # Show miner startup info
    echo "Miner log: tail -f logs/miner_testnet.log"
}

# Step 7: Start Validator
start_validator() {
    echo -e "\n${YELLOW}✅ Step 7: Starting Luminar Validator${NC}"
    
    echo "Starting validator in background..."
    nohup python neurons/validator.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name $VALIDATOR_WALLET \
        --wallet.hotkey $HOTKEY_NAME \
        --logging.debug \
        --neuron.sample_size 10 > logs/validator_testnet.log 2>&1 &
    
    VALIDATOR_PID=$!
    echo $VALIDATOR_PID > logs/validator.pid
    print_status "Validator started (PID: $VALIDATOR_PID)"
    
    # Show validator startup info
    echo "Validator log: tail -f logs/validator_testnet.log"
}

# Step 8: Monitor
monitor_subnet() {
    echo -e "\n${YELLOW}📊 Step 8: Monitoring${NC}"
    
    print_status "Luminar subnet is now running on testnet (subnet 414)!"
    
    echo -e "\n📋 Quick Commands:"
    echo "• Monitor miner: tail -f logs/miner_testnet.log"
    echo "• Monitor validator: tail -f logs/validator_testnet.log"
    echo "• Check subnet 414: btcli subnet metagraph --subtensor.network test --netuid 414"
    echo "• Check miner wallet: btcli wallet balance --subtensor.network test --wallet.name $WALLET_NAME"
    echo "• Check validator wallet: btcli wallet balance --subtensor.network test --wallet.name $VALIDATOR_WALLET"
    
    echo -e "\n🔗 Useful Links:"
    echo "• Testnet Explorer: https://explorer.bittensor.com/"
    echo "• Luminar Network Documentation: https://docs.luminar.network/"
    echo "• Subnet 414 (Luminar Network) Status: https://taostats.io/subnets/netuid-414/"
    
    echo -e "\n${GREEN}🎉 Testnet deployment completed successfully!${NC}"
    echo -e "You are now running on ${GREEN}Luminar Network (subnet 414)${NC}"
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
