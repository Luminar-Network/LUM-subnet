# Luminar Subnet Testnet Deployment Guide

## 🚀 Quick Testnet Setup (Subnet 414)

This guide will help you deploy and test the Luminar subnet on Bittensor testnet using **Luminar Network (netuid 414)**.

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
3. ✅ Create miner and validator wallets
4. ✅ Guide you to get testnet TAO
5. ✅ Register on subnet 414 (Luminar Network)
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
# Create miner wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Create validator wallet  
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

#### 3.3 Get Testnet TAO
Options to get testnet TAO:
- Get from testnet faucet (if available): `btcli wallet faucet --subtensor.network test --wallet.name miner`
- Transfer between your wallets: `btcli wallet transfer --subtensor.network test --wallet.name source_wallet --dest destination_address --amount 0.1`

Registration cost: ~0.0717 τ per registration

#### 3.4 Register on Subnet 414 (Luminar Network)
```bash
# Register miner (cost: ~0.0717 τ)
btcli subnet register \
    --netuid 414 \
    --subtensor.network test \
    --wallet.name miner \
    --wallet.hotkey default

# Register validator (cost: ~0.0717 τ)
btcli subnet register \
    --netuid 414 \
    --subtensor.network test \
    --wallet.name validator \
    --wallet.hotkey default
```

#### 3.5 Start Nodes

**Start Miner:**
```bash
python neurons/miner.py \
    --netuid 414 \
    --subtensor.network test \
    --wallet.name miner \
    --wallet.hotkey default \
    --logging.debug
```

**Start Validator:**
```bash
python neurons/validator.py \
    --netuid 414 \
    --subtensor.network test \
    --wallet.name validator \
    --wallet.hotkey default \
    --logging.debug
```

### Step 4: Monitor Your Subnet

#### Check Subnet Status
```bash
# Check subnet 414 (Luminar Network)
btcli subnet metagraph --subtensor.network test --netuid 414
```

#### Check Your Registration
```bash
# Check miner wallet
btcli wallet balance --subtensor.network test --wallet.name miner

# Check validator wallet
btcli wallet balance --subtensor.network test --wallet.name validator
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
   - Ensure you have enough testnet TAO (minimum ~0.0717 τ per registration)
   - Check if netuid 414 is available: `btcli subnet metagraph --subtensor.network test --netuid 414`
   - Verify wallet has sufficient balance
   - Try using faucet: `btcli wallet faucet --subtensor.network test --wallet.name miner`

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
ps aux | grep python

# Check open ports
lsof -i :8091

# Check wallet balances
btcli wallet balance --subtensor.network test --wallet.name miner
btcli wallet balance --subtensor.network test --wallet.name validator
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

🎉 **Your Luminar subnet is now running on testnet subnet 414 (Luminar Network)!**
