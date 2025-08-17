# Scripts Directory

Essential scripts for Luminar subnet deployment and management.

## 🚀 Deployment Scripts

### `deploy_testnet.sh`
**Purpose:** Complete testnet deployment automation
- Sets up testnet environment
- Creates wallets and gets testnet TAO  
- Registers on subnet
- Starts miner and validator nodes
- Shows monitoring commands

**Usage:**
```bash
./scripts/deploy_testnet.sh deploy  # Full deployment
./scripts/deploy_testnet.sh stop    # Stop subnet
./scripts/deploy_testnet.sh monitor # Show status
```

### `install_staging.sh`
**Purpose:** System dependencies installation
- Installs required system packages (macOS/Ubuntu)
- Sets up development environment
- Installs Homebrew, LLVM, protobuf, etc.

**Usage:**
```bash
./scripts/install_staging.sh
```

## 🗄️ Database Scripts

### `setup_database.py`
**Purpose:** Production database setup and management
- Uses asyncpg with environment-based configuration
- Supports multiple environments (development, staging, production)
- Advanced database operations and migrations

**Usage:**
```bash
python scripts/setup_database.py setup
```

### `setup_test_database.py`
**Purpose:** Test database setup with sample data
- Creates tables for testnet/development
- Populates with sample media processing data
- Simple psycopg2-based implementation

**Usage:**
```bash
python scripts/setup_test_database.py
```

### `test_database.py`
**Purpose:** Database connectivity and data testing
- Tests database connection
- Fetches and displays sample data
- Validates database setup

**Usage:**
```bash
python scripts/test_database.py
```

## 📁 File Purpose Summary

| Script | Environment | Purpose |
|--------|-------------|---------|
| `deploy_testnet.sh` | Testnet | Full deployment automation |
| `install_staging.sh` | Development | System setup |
| `setup_database.py` | Production | Advanced DB management |
| `setup_test_database.py` | Development/Test | Simple DB setup with data |
| `test_database.py` | All | Database validation |

## 🗑️ Removed Files

The following files were removed to reduce confusion:
- `check_compatibility.sh` - CI/CD specific
- `check_requirements_changes.sh` - CI/CD specific  
- `local_miner.py` - Testing code (moved to test suite)
