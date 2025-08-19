# Database Configuration for Luminar Subnet
# Copyright © 2025 Luminar Network

import os
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class DatabaseEnvironment(Enum):
    """Database environment types"""
    PRODUCTION = "production"
    STAGING = "staging"
    TESTING = "testing"
    LOCAL = "local"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

class LuminarDatabaseManager:
    """Manages database connections for different environments"""
    
    def __init__(self, environment: DatabaseEnvironment = None):
        self.environment = environment or self._detect_environment()
        self.config = self._load_config()
    
    def _detect_environment(self) -> DatabaseEnvironment:
        """Auto-detect environment based on ENV variables"""
        env = os.getenv('LUMINAR_ENV', 'local').lower()
        try:
            return DatabaseEnvironment(env)
        except ValueError:
            return DatabaseEnvironment.LOCAL
    
    def _load_config(self) -> DatabaseConfig:
        """Load database configuration based on environment"""
        configs = {
            DatabaseEnvironment.PRODUCTION: self._production_config(),
            DatabaseEnvironment.STAGING: self._staging_config(),
            DatabaseEnvironment.TESTING: self._testing_config(),
            DatabaseEnvironment.LOCAL: self._local_config()
        }
        return configs[self.environment]
    
    def _production_config(self) -> DatabaseConfig:
        """Production database configuration - Centralized Database"""
        return DatabaseConfig(
            host=os.getenv('LUMINAR_DB_HOST', 'central-db.luminar.network'),
            port=int(os.getenv('LUMINAR_DB_PORT', '5432')),
            database=os.getenv('LUMINAR_DB_NAME', 'luminar_production'),
            username=os.getenv('LUMINAR_DB_USER', 'luminar_subnet'),
            password=os.getenv('LUMINAR_DB_PASSWORD'),
            ssl_mode='require',
            pool_size=50,  # Higher for production
            max_overflow=20,
            pool_timeout=60,
            pool_recycle=7200
        )
    
    def _staging_config(self) -> DatabaseConfig:
        """Staging database configuration - Central DB staging replica"""
        return DatabaseConfig(
            host=os.getenv('LUMINAR_STAGING_DB_HOST', 'staging-db.luminar.network'),
            port=int(os.getenv('LUMINAR_STAGING_DB_PORT', '5432')),
            database=os.getenv('LUMINAR_STAGING_DB_NAME', 'luminar_staging'),
            username=os.getenv('LUMINAR_STAGING_DB_USER', 'luminar_staging'),
            password=os.getenv('LUMINAR_STAGING_DB_PASSWORD'),
            ssl_mode='prefer',
            pool_size=20,
            max_overflow=10,
            pool_timeout=30
        )
    
    def _testing_config(self) -> DatabaseConfig:
        """Testing database configuration - Local replica with test data"""
        return DatabaseConfig(
            host=os.getenv('LUMINAR_TEST_DB_HOST', 'localhost'),
            port=int(os.getenv('LUMINAR_TEST_DB_PORT', '5433')),  # Different port
            database=os.getenv('LUMINAR_TEST_DB_NAME', 'luminar_test'),
            username=os.getenv('LUMINAR_TEST_DB_USER', 'luminar_test'),
            password=os.getenv('LUMINAR_TEST_DB_PASSWORD', 'test_password'),
            ssl_mode='disable',  # No SSL for local testing
            pool_size=5,  # Small pool for testing
            max_overflow=5,
            pool_timeout=10
        )
    
    def _local_config(self) -> DatabaseConfig:
        """Local development configuration - Subnet replica database"""
        return DatabaseConfig(
            host=os.getenv('LUMINAR_LOCAL_DB_HOST', 'localhost'),
            port=int(os.getenv('LUMINAR_LOCAL_DB_PORT', '5432')),
            database=os.getenv('LUMINAR_LOCAL_DB_NAME', 'luminar_local'),
            username=os.getenv('LUMINAR_LOCAL_DB_USER', 'luminar_dev'),
            password=os.getenv('LUMINAR_LOCAL_DB_PASSWORD', 'dev_password'),
            ssl_mode='disable',
            pool_size=10,
            max_overflow=5,
            pool_timeout=20
        )
    
    def get_connection_string(self, read_only: bool = False) -> str:
        """Generate PostgreSQL connection string"""
        config = self.config
        
        # Use read replica for read-only operations in production
        if read_only and self.environment == DatabaseEnvironment.PRODUCTION:
            host = os.getenv('LUMINAR_DB_READ_HOST', config.host)
        else:
            host = config.host
        
        return (
            f"postgresql://{config.username}:{config.password}@"
            f"{host}:{config.port}/{config.database}"
            f"?sslmode={config.ssl_mode}"
        )
    
    def get_async_connection_string(self, read_only: bool = False) -> str:
        """Generate asyncpg connection string"""
        return self.get_connection_string(read_only).replace('postgresql://', 'postgresql+asyncpg://')
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for direct database connections"""
        config = self.config
        return {
            'host': config.host,
            'port': config.port,
            'database': config.database,
            'user': config.username,
            'password': config.password,
            'ssl': 'prefer' if config.ssl_mode == 'require' else 'disable',
        }

# Singleton instance
db_manager = LuminarDatabaseManager()

# Convenience functions
def get_db_config() -> DatabaseConfig:
    """Get current database configuration"""
    return db_manager.config

def get_connection_string(read_only: bool = False) -> str:
    """Get database connection string"""
    return db_manager.get_connection_string(read_only)

def get_connection_params() -> Dict[str, Any]:
    """Get connection parameters for direct database connections"""
    return db_manager.get_connection_params()

def get_async_connection_string(read_only: bool = False) -> str:
    """Get async database connection string"""
    return db_manager.get_async_connection_string(read_only)

def is_production() -> bool:
    """Check if running in production environment"""
    return db_manager.environment == DatabaseEnvironment.PRODUCTION

def is_local() -> bool:
    """Check if running in local development environment"""
    return db_manager.environment == DatabaseEnvironment.LOCAL
