# Luminar Subnet Database Architecture

## 🏗️ **Database Architecture Overview**

The Luminar subnet uses a **hybrid database architecture** with:

1. **Centralized Database** - Production database hosted by Luminar Network
2. **Local Replicas** - For subnet development and testing
3. **Synchronization Layer** - Keeps local replicas updated

```mermaid
graph TB
    subgraph "Production Environment"
        A[Luminar Central DB] --> B[Primary PostgreSQL + PostGIS]
        B --> C[Read Replicas]
        C --> D[Subnet Validators/Miners]
    end
    
    subgraph "Development Environment"
        E[Local Development] --> F[Local Replica DB]
        B -.->|Sync| F
    end
    
    subgraph "Testing Environment"
        G[Test Suite] --> H[Test Database]
        I[Synthetic Data] --> H
    end
```

## 📊 **Database Schema**

### Core Tables:

1. **`raw_incident_reports`** - Unprocessed crime reports from mobile apps
2. **`crime_events`** - AI-processed, clustered crime intelligence
3. **`miner_performance`** - Tracks miner scoring and performance
4. **`validator_performance`** - Tracks validator consensus accuracy
5. **`consensus_records`** - Detailed scoring history for transparency
6. **`crime_clusters`** - Geographic and temporal crime patterns
7. **`risk_zones`** - Dynamic risk assessment areas

### Key Features:
- **PostGIS** for geospatial operations
- **JSONB** for flexible metadata storage
- **UUID** primary keys for distributed systems
- **Triggers** for automatic timestamp updates
- **Views** for common analytics queries

## 🚀 **Quick Start**

### 1. **Local Development Setup**

```bash
# Clone and setup
git clone https://github.com/Luminar-Network/LUM-subnet.git
cd LUM-subnet

# Setup development environment
make dev-setup

# Start database
make db-start

# Check status
make db-status
```

### 2. **Production Connection**

```bash
# Copy production environment
cp .env.production .env

# Edit with your credentials
vim .env

# Connect to centralized database
LUMINAR_ENV=production python neurons/luminar_miner.py
```

## 🔧 **Configuration**

### Environment Variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `LUMINAR_ENV` | Environment type | `local`, `production` |
| `LUMINAR_DB_HOST` | Database host | `central-db.luminar.network` |
| `LUMINAR_DB_NAME` | Database name | `luminar_production` |
| `LUMINAR_DB_USER` | Database user | `luminar_subnet` |
| `LUMINAR_DB_PASSWORD` | Database password | `secure_password` |

### Database Configurations:

#### **Production (Centralized)**
```python
# Connects to Luminar's central database
DATABASE_URL = "postgresql://luminar_subnet:password@central-db.luminar.network:5432/luminar_production"
SSL_MODE = "require"
POOL_SIZE = 50
```

#### **Local (Replica)**
```python
# Local development with replica
DATABASE_URL = "postgresql://luminar_dev:dev_password@localhost:5432/luminar_local"
SSL_MODE = "disable"
POOL_SIZE = 10
```

## 🛠️ **Database Management**

### Migration System:

```bash
# Create new migration
make migrate-create NAME=add_new_feature

# Run migrations
make db-migrate

# Check migration status
make migrate-status
```

### Example Migration:
```sql
-- 20250817120000_add_risk_scoring.sql
BEGIN;

ALTER TABLE crime_events 
ADD COLUMN risk_score DECIMAL(3,2) CHECK (risk_score >= 0 AND risk_score <= 1);

CREATE INDEX idx_events_risk_score ON crime_events(risk_score);

COMMIT;
```

### Backup & Restore:

```bash
# Create backup
make db-backup

# Reset database (WARNING: Destroys data)
make db-reset

# Restore from backup
pg_restore -d luminar_local backup_file.sql
```

## 📈 **Performance Optimization**

### Indexes for Crime Data:

```sql
-- Geographic queries (crime hotspots)
CREATE INDEX idx_events_geotag ON crime_events USING GIST(geotag);

-- Time-based queries (recent incidents)
CREATE INDEX idx_events_timestamp ON crime_events(timestamp);

-- Metadata searches (crime types, severity)
CREATE INDEX idx_events_metadata ON crime_events USING GIN(metadata);
```

### Connection Pooling:
```python
# Production settings
POOL_SIZE = 50          # Concurrent connections
MAX_OVERFLOW = 20       # Additional connections during peak
POOL_TIMEOUT = 60       # Connection timeout
POOL_RECYCLE = 7200     # Recycle connections every 2 hours
```

## 🔍 **Data Access Patterns**

### Reading Crime Data:
```python
from database.access import db

# Get recent crime events
events = await db.get_recent_events(hours=24, limit=100)

# Get crime hotspots
hotspots = await db.get_crime_hotspots(days=7)

# Get miner performance
stats = await db.get_miner_stats(miner_uid=123)
```

### Writing Crime Data:
```python
# Store processed crime event
success = await db.store_crime_event(crime_event, miner_uid=123)

# Update miner performance
await db.update_miner_performance(
    miner_uid=123,
    hotkey="hotkey_string",
    scores={
        "integrity": 0.95,
        "clustering": 0.88,
        "novelty": 0.75,
        "quality": 0.92
    }
)
```

## 🧪 **Testing**

### Test Database Setup:
```bash
# Setup test environment
make db-test-setup

# Run tests
make test

# Generate test data
python scripts/setup_database.py sample-data --env=testing
```

### Synthetic Data Generation:
```python
# Creates realistic crime reports for testing
await populate_sample_data()

# Generates:
# - 100+ sample incident reports
# - Geographic clusters around major cities
# - Realistic timestamps and metadata
# - Sample miner/validator performance data
```

## 🔒 **Security**

### Production Security:
- **SSL/TLS** required for all connections
- **Certificate-based** authentication
- **IP whitelisting** for subnet nodes
- **Role-based** access control
- **Audit logging** for all operations

### Local Security:
- **No SSL** for development ease
- **Local-only** access
- **Development** credentials

## 📊 **Monitoring**

### Health Checks:
```bash
# Check database connectivity
make db-status

# View real-time statistics
make db-stats

# Monitor performance
docker-compose --profile monitoring up -d
# Access Grafana at http://localhost:3000
```

### Key Metrics:
- **Processing Throughput**: Reports/hour
- **Consensus Accuracy**: Validator agreement %
- **Geographic Coverage**: Crime events by region
- **Miner Performance**: Success rates and scoring

### Database Views:
```sql
-- Active miners performance
SELECT * FROM active_miners WHERE success_rate > 80;

-- Recent crime hotspots
SELECT * FROM crime_hotspots WHERE incident_count >= 5;

-- Network health
SELECT COUNT(*) as active_nodes FROM miner_performance WHERE is_active = true;
```

## 🔄 **Data Synchronization**

### Centralized → Local Sync:
```bash
# Sync recent data for testing
python scripts/setup_database.py sync --hours=24

# Full replica setup
python scripts/setup_database.py setup --env=local
```

### Real-time Streaming (Future):
```python
# Will implement real-time data streaming
# for production subnet deployment
await sync_manager.start_real_time_sync()
```

## 🤝 **Integration with Subnet**

### Miner Integration:
```python
# In luminar_miner.py
from database.access import db

async def process_reports(self, reports):
    # Store raw reports
    for report in reports:
        await db.store_raw_report(report)
    
    # Process and store events
    events = await self.cluster_reports(reports)
    for event in events:
        await db.store_crime_event(event, self.uid)
```

### Validator Integration:
```python
# In luminar_validator.py
async def validate_events(self, events):
    # Retrieve consensus data
    for event in events:
        scores = await self.score_event(event)
        await db.store_consensus_record(
            event.event_id, self.uid, scores, round_number
        )
```

## 🚨 **Troubleshooting**

### Common Issues:

1. **Connection Refused**
   ```bash
   # Check if database is running
   make db-status
   
   # Restart database
   make db-stop && make db-start
   ```

2. **Migration Failures**
   ```bash
   # Check migration status
   make migrate-status
   
   # Manual migration
   python migrations/manager.py migrate
   ```

3. **Permission Errors**
   ```bash
   # Fix file permissions
   chmod +x init-scripts/*.sh
   
   # Check database permissions
   docker-compose logs luminar-db
   ```

4. **Out of Memory**
   ```bash
   # Reduce connection pool size
   export DB_POOL_SIZE=10
   
   # Clear old data
   make clean
   ```

## 📚 **Additional Resources**

- **PostGIS Documentation**: https://postgis.net/docs/
- **AsyncPG Guide**: https://magicstack.github.io/asyncpg/
- **Docker Compose**: https://docs.docker.com/compose/
- **PostgreSQL Performance**: https://wiki.postgresql.org/wiki/Performance_Optimization

---

*For support, join our Discord: https://discord.gg/luminar*
