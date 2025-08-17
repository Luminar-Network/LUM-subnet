#!/bin/bash
# Database initialization script for Docker
# This script runs when the PostgreSQL container starts for the first time

set -e

# Create additional databases if needed
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Enable PostGIS extension
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Create additional roles if needed
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'luminar_readonly') THEN
            CREATE ROLE luminar_readonly;
        END IF;
    END
    \$\$;
    
    -- Grant permissions
    GRANT CONNECT ON DATABASE $POSTGRES_DB TO luminar_readonly;
    GRANT USAGE ON SCHEMA public TO luminar_readonly;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO luminar_readonly;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO luminar_readonly;
    
    -- Log successful initialization
    SELECT 'Luminar database initialized successfully' as message;
EOSQL

echo "Luminar PostgreSQL initialization completed"
