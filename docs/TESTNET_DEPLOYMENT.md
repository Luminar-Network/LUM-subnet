# Luminar Subnet Testnet Deployment Guide

## 🚀 Quick Testnet Setup

This guide will help you deploy and test the Luminar subnet on Bittensor testnet.

### Prerequisites

- Python 3.9+
- Git
- 16GB+ RAM recommended
- Internet connection

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify Bittensor installation
btcli --version
```

### Step 2: Automated Deployment

```bash
# Run the automated testnet deployment
./scripts/deploy_testnet.sh deploy
```

This script will:
1. ✅ Setup testnet environment
2. ✅ Check Bittensor installation  
3. ✅ Create testnet wallet
4. ✅ Guide you to get testnet TAO
5. ✅ Register on subnet
6. ✅ Start miner and validator
7. ✅ Show monitoring commands

### Step 3: Manual Setup (Alternative)

If you prefer manual setup:

#### 3.1 Environment Setup
```bash
# Copy testnet config
cp .env.testnet .env

# Setup database
make db-setup
```

#### 3.2 Create Wallet
```bash
# Create testnet wallet
btcli wallet new_coldkey --wallet.name testnet_wallet
btcli wallet new_hotkey --wallet.name testnet_wallet --wallet.hotkey miner_hotkey
btcli wallet new_hotkey --wallet.name testnet_wallet --wallet.hotkey validator_hotkey
```

#### 3.3 Get Testnet TAO
1. Visit: https://faucet.bittensor.com/
2. Enter your coldkey address
3. Request testnet TAO

#### 3.4 Register on Subnet
```bash
# Register miner
btcli subnet register \
    --wallet.name testnet_wallet \
    --wallet.hotkey miner_hotkey \
    --subtensor.network test \
    --netuid 999

# Register validator  
btcli subnet register \
    --wallet.name testnet_wallet \
    --wallet.hotkey validator_hotkey \
    --subtensor.network test \
    --netuid 999
```

#### 3.5 Start Nodes

**Start Miner:**
```bash
python neurons/miner.py \
    --netuid 999 \
    --subtensor.network test \
    --wallet.name testnet_wallet \
    --wallet.hotkey miner_hotkey \
    --logging.debug
```

**Start Validator:**
```bash
python neurons/validator.py \
    --netuid 999 \
    --subtensor.network test \
    --wallet.name testnet_wallet \
    --wallet.hotkey validator_hotkey \
    --logging.debug
```

### Step 4: Monitor Your Subnet

#### Check Subnet Status
```bash
btcli subnet list --subtensor.network test
```

#### Check Your Registration
```bash
btcli wallet overview --wallet.name testnet_wallet --subtensor.network test
```

#### Monitor Logs
```bash
# Miner logs
tail -f logs/miner_testnet.log

# Validator logs  
tail -f logs/validator_testnet.log
```

#### Database Status
```bash
make db-status
```

### Step 5: Test the Complete Flow

#### Simulate User Submissions
The validator automatically generates synthetic user submissions to test the complete flow:

1. **User uploads** (simulated) - Media + text + metadata
2. **Validator distribution** - Sends to miners
3. **Miner processing** - AI analysis of text + visuals
4. **Event generation** - Creates events like "truck and motorbike accident near balaju at 4pm"
5. **Metadata validation** - Compares with original submissions
6. **Scoring** - Updates miner weights based on accuracy

#### Expected Log Output
```
🎯 Processing 4 user submissions
📥 Processing user submission USR_20250817_001
🔍 Analyzing text: "Car accident on main street"
👁️ Processing image: accident_001.jpg
📝 Generated event: "car accident on main street at 3pm"
✅ Timestamp validation: 0.95
✅ Geotag validation: 0.88
📊 Final miner score: 0.91
```

### Troubleshooting

#### Common Issues

1. **Registration Failed**
   - Ensure you have enough testnet TAO
   - Check if netuid 999 is available
   - Verify wallet has sufficient balance

2. **Connection Issues**
   - Check internet connection
   - Verify testnet endpoint: `wss://test.finney.opentensor.ai:443`
   - Try different port for axon

3. **Database Issues**
   ```bash
   # Reset database
   make db-reset
   
   # Check database status
   make db-status
   ```

4. **Dependencies Issues**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

### Useful Commands

```bash
# Stop subnet
./scripts/deploy_testnet.sh stop

# Show monitoring info
./scripts/deploy_testnet.sh monitor

# Check processes
ps aux | grep luminar

# Check open ports
lsof -i :8091
```

### Next Steps

After successful testnet deployment:

1. **Monitor Performance** - Watch miner scores and validator accuracy
2. **Test Features** - Verify media processing and metadata validation
3. **Scale Testing** - Add more miners/validators
4. **Optimize** - Tune parameters based on performance
5. **Mainnet Prep** - Prepare for mainnet deployment

### Support

- **Documentation**: `docs/`
- **Issues**: Check logs in `logs/` directory
- **Discord**: Join Luminar community
- **GitHub**: Report issues on repository

---

🎉 **Your Luminar subnet is now running on testnet!**
