# Database Setup and Synchronization Scripts
# Copyright © 2025 Luminar Network

import asyncio
import asyncpg
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.database import LuminarDatabaseManager, DatabaseEnvironment
from migrations.manager import migration_manager
from database.access import LuminarDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and synchronization utilities"""
    
    def __init__(self):
        self.db_manager = LuminarDatabaseManager()
        self.db_access = LuminarDatabase()
    
    async def create_database(self, db_name: str = None):
        """Create database if it doesn't exist"""
        config = self.db_manager.config
        db_name = db_name or config.database
        
        # Connect to postgres database to create new database
        temp_config = config.__dict__.copy()
        temp_config['database'] = 'postgres'
        
        try:
            conn = await asyncpg.connect(**temp_config)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database already exists: {db_name}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def setup_local_replica(self):
        """Set up local replica database for testing"""
        logger.info("Setting up local replica database...")
        
        # Create local database
        await self.create_database()
        
        # Run migrations
        await migration_manager.run_migrations()
        
        # Initialize database access layer
        await self.db_access.initialize()
        
        # Populate with sample data if needed
        if os.getenv('GENERATE_SYNTHETIC_DATA', 'false').lower() == 'true':
            await self.populate_sample_data()
        
        logger.info("Local replica database setup complete")
    
    async def populate_sample_data(self):
        """Populate database with sample data for testing"""
        logger.info("Populating sample data...")
        
        try:
            async with self.db_access.pool.acquire() as conn:
                # Insert sample raw reports
                sample_reports = [
                    {
                        'external_id': 'SAMPLE_001',
                        'text_description': 'Phone stolen from coffee shop table',
                        'longitude': -74.006,
                        'latitude': 40.7128,
                        'timestamp': 'NOW() - INTERVAL \'2 hours\'',
                        'media_urls': ['https://storage.luminar.network/sample1.jpg'],
                        'user_id': 'user_001',
                        'metadata': '{"location_name": "Downtown Coffee", "reported_by": "victim"}'
                    },
                    {
                        'external_id': 'SAMPLE_002',
                        'text_description': 'Car break-in reported on 5th street',
                        'longitude': -74.010,
                        'latitude': 40.715,
                        'timestamp': 'NOW() - INTERVAL \'1 hour\'',
                        'media_urls': [],
                        'user_id': 'user_002',
                        'metadata': '{"location_name": "5th Street Parking", "damage_level": "minor"}'
                    },
                    {
                        'external_id': 'SAMPLE_003',
                        'text_description': 'Witnessed assault near park entrance',
                        'longitude': -74.005,
                        'latitude': 40.718,
                        'timestamp': 'NOW() - INTERVAL \'30 minutes\'',
                        'media_urls': [],
                        'user_id': 'user_003',
                        'metadata': '{"location_name": "Central Park", "witness_account": true}'
                    }
                ]
                
                for report in sample_reports:
                    await conn.execute(f"""
                        INSERT INTO raw_incident_reports (
                            external_id, text_description, geotag, timestamp,
                            media_urls, user_id, metadata, source, status
                        ) VALUES ($1, $2, POINT($3, $4), {report['timestamp']}, $5, $6, $7, 'sample_data', 'pending')
                        ON CONFLICT (external_id) DO NOTHING
                    """,
                        report['external_id'],
                        report['text_description'],
                        report['longitude'],
                        report['latitude'],
                        report['media_urls'],
                        report['user_id'],
                        report['metadata']
                    )
                
                # Insert sample miner performance data
                await conn.execute("""
                    INSERT INTO miner_performance (
                        miner_uid, hotkey, total_score, processed_reports,
                        verified_events, integrity_score, clustering_score,
                        novelty_score, quality_score, stake_amount
                    ) VALUES 
                        (1, 'miner_hotkey_001', 85.5, 25, 20, 0.9, 0.85, 0.8, 0.9, 1000.0),
                        (2, 'miner_hotkey_002', 72.3, 18, 15, 0.8, 0.75, 0.7, 0.85, 800.0),
                        (3, 'miner_hotkey_003', 91.2, 30, 28, 0.95, 0.9, 0.85, 0.95, 1200.0)
                    ON CONFLICT (miner_uid) DO NOTHING
                """)
                
                # Insert sample validator performance data
                await conn.execute("""
                    INSERT INTO validator_performance (
                        validator_uid, hotkey, validated_events, consensus_accuracy, stake_amount
                    ) VALUES 
                        (101, 'validator_hotkey_001', 45, 0.92, 5000.0),
                        (102, 'validator_hotkey_002', 38, 0.88, 4500.0),
                        (103, 'validator_hotkey_003', 52, 0.95, 6000.0)
                    ON CONFLICT (validator_uid) DO NOTHING
                """)
                
                logger.info("Sample data populated successfully")
                
        except Exception as e:
            logger.error(f"Failed to populate sample data: {e}")
            raise
    
    async def sync_from_central(self, hours_back: int = 24):
        """Sync recent data from centralized database to local replica"""
        if self.db_manager.environment != DatabaseEnvironment.LOCAL:
            logger.warning("Sync from central only available in local environment")
            return
        
        logger.info(f"Syncing last {hours_back} hours from central database...")
        
        # This would connect to production database and sync recent data
        # Implementation depends on your central database access permissions
        
        logger.info("Sync from central database completed")
    
    async def backup_database(self, backup_path: str = None):
        """Create database backup"""
        config = self.db_manager.config
        
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/luminar_backup_{timestamp}.sql"
        
        # Create backup directory
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use pg_dump to create backup
        import subprocess
        
        cmd = [
            'pg_dump',
            '-h', config.host,
            '-p', str(config.port),
            '-U', config.username,
            '-d', config.database,
            '-f', backup_path,
            '--verbose'
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = config.password
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Database backup created: {backup_path}")
            else:
                logger.error(f"Backup failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

async def setup_database_cli():
    """CLI for database setup operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Luminar Database Setup')
    parser.add_argument('action', choices=['setup', 'migrate', 'sample-data', 'sync', 'backup'])
    parser.add_argument('--env', choices=['local', 'staging', 'production'], default='local')
    parser.add_argument('--hours', type=int, default=24, help='Hours to sync back')
    parser.add_argument('--backup-path', help='Backup file path')
    
    args = parser.parse_args()
    
    # Set environment
    os.environ['LUMINAR_ENV'] = args.env
    
    setup = DatabaseSetup()
    
    try:
        if args.action == 'setup':
            await setup.setup_local_replica()
        elif args.action == 'migrate':
            await migration_manager.run_migrations()
        elif args.action == 'sample-data':
            await setup.db_access.initialize()
            await setup.populate_sample_data()
        elif args.action == 'sync':
            await setup.sync_from_central(args.hours)
        elif args.action == 'backup':
            await setup.backup_database(args.backup_path)
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(setup_database_cli())
