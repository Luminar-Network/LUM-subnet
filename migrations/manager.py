# Database Migration Management for Luminar Subnet
# Copyright © 2025 Luminar Network

import asyncio
import asyncpg
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from config.database import get_connection_string, get_connection_params, is_production

logger = logging.getLogger(__name__)

class LuminarMigrationManager:
    """Manages database migrations for Luminar subnet"""
    
    def __init__(self):
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
        
    async def create_migration_table(self, conn: asyncpg.Connection):
        """Create migrations tracking table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(20) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64),
                execution_time_ms INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
            ON schema_migrations(version);
        """)
    
    async def get_applied_migrations(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of applied migration versions"""
        rows = await conn.fetch(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        return [row['version'] for row in rows]
    
    async def apply_migration(self, conn: asyncpg.Connection, migration_file: Path):
        """Apply a single migration file"""
        version = migration_file.stem.split('_')[0]
        name = '_'.join(migration_file.stem.split('_')[1:])
        
        logger.info(f"Applying migration {version}: {name}")
        
        # Read migration SQL
        sql_content = migration_file.read_text()
        
        # Calculate checksum
        import hashlib
        checksum = hashlib.sha256(sql_content.encode()).hexdigest()
        
        # Execute migration
        start_time = datetime.now()
        
        try:
            await conn.execute(sql_content)
            
            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await conn.execute("""
                INSERT INTO schema_migrations (version, name, checksum, execution_time_ms)
                VALUES ($1, $2, $3, $4)
            """, version, name, checksum, execution_time)
            
            logger.info(f"Migration {version} applied successfully in {execution_time}ms")
            
        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            raise
    
    async def run_migrations(self, target_version: str = None):
        """Run pending migrations"""
        # Connect to database
        conn_params = get_connection_params()
        conn = await asyncpg.connect(**conn_params)
        
        try:
            # Create migration table if it doesn't exist
            await self.create_migration_table(conn)
            
            # Get applied migrations
            applied = await self.get_applied_migrations(conn)
            
            # Get available migration files
            migration_files = sorted(self.migrations_dir.glob("*.sql"))
            
            # Filter pending migrations
            pending = []
            for migration_file in migration_files:
                version = migration_file.stem.split('_')[0]
                if version not in applied:
                    if target_version is None or version <= target_version:
                        pending.append(migration_file)
            
            if not pending:
                logger.info("No pending migrations")
                return
            
            # Apply pending migrations
            async with conn.transaction():
                for migration_file in pending:
                    await self.apply_migration(conn, migration_file)
            
            logger.info(f"Applied {len(pending)} migrations successfully")
            
        finally:
            await conn.close()
    
    async def rollback_migration(self, target_version: str):
        """Rollback to a specific migration version"""
        if is_production():
            raise RuntimeError("Rollbacks not allowed in production")
        
        # Implementation for rollback
        logger.warning(f"Rolling back to version {target_version}")
        # TODO: Implement rollback logic
    
    def create_migration_file(self, name: str) -> Path:
        """Create a new migration file"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{name}.sql"
        filepath = self.migrations_dir / filename
        
        # Create migration template
        template = f"""-- Migration: {name}
-- Created: {datetime.now().isoformat()}
-- Description: {name.replace('_', ' ').title()}

-- Forward migration
BEGIN;

-- Add your migration SQL here


COMMIT;

-- Rollback SQL (for reference)
-- BEGIN;
-- 
-- -- Add rollback SQL here
-- 
-- COMMIT;
"""
        
        filepath.write_text(template)
        logger.info(f"Created migration file: {filepath}")
        return filepath

# Singleton instance
migration_manager = LuminarMigrationManager()

# CLI-like functions
async def migrate(target_version: str = None):
    """Run database migrations"""
    await migration_manager.run_migrations(target_version)

async def rollback(target_version: str):
    """Rollback database to specific version"""
    await migration_manager.rollback_migration(target_version)

def create_migration(name: str) -> Path:
    """Create new migration file"""
    return migration_manager.create_migration_file(name)

# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations/manager.py [migrate|rollback|create] [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "migrate":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(migrate(target))
    
    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: python migrations/manager.py rollback <version>")
            sys.exit(1)
        asyncio.run(rollback(sys.argv[2]))
    
    elif command == "create":
        if len(sys.argv) < 3:
            print("Usage: python migrations/manager.py create <migration_name>")
            sys.exit(1)
        create_migration(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
