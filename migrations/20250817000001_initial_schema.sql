-- Migration: Initial Luminar Subnet Schema
-- Created: 2025-08-17T00:00:00
-- Description: Create core tables for Luminar crime intelligence subnet

-- Forward migration
BEGIN;

-- Enable PostGIS extension for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Raw incident reports table (from mobile app / external sources)
CREATE TABLE raw_incident_reports (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE, -- ID from source system
    text_description TEXT NOT NULL,
    geotag POINT NOT NULL, -- Geographic coordinates
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    media_urls TEXT[], -- Array of media file URLs
    media_hashes TEXT[], -- Array of media file hashes for verification
    user_id VARCHAR(255), -- Anonymous user identifier
    metadata JSONB DEFAULT '{}', -- Flexible metadata storage
    source VARCHAR(100) DEFAULT 'mobile_app', -- Data source identifier
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, processed, failed
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Processed crime events table (output from miners)
CREATE TABLE crime_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_id UUID, -- Group related incidents
    event_type VARCHAR(100) NOT NULL, -- theft, assault, vandalism, etc.
    summary TEXT NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    geotag POINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    severity_level INTEGER CHECK (severity_level >= 1 AND severity_level <= 5),
    related_reports UUID[] NOT NULL, -- Array of related raw report IDs
    metadata JSONB DEFAULT '{}',
    miner_uid INTEGER NOT NULL, -- Bittensor UID of processing miner
    validator_uid INTEGER, -- Bittensor UID of validating validator
    verification_status VARCHAR(50) DEFAULT 'pending', -- pending, verified, rejected
    verification_score DECIMAL(3,2), -- Score from validators
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Miner performance tracking
CREATE TABLE miner_performance (
    id SERIAL PRIMARY KEY,
    miner_uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    coldkey VARCHAR(255),
    processed_reports INTEGER DEFAULT 0,
    verified_events INTEGER DEFAULT 0,
    total_score DECIMAL(10,4) DEFAULT 0,
    integrity_score DECIMAL(5,4) DEFAULT 0,
    clustering_score DECIMAL(5,4) DEFAULT 0,
    novelty_score DECIMAL(5,4) DEFAULT 0,
    quality_score DECIMAL(5,4) DEFAULT 0,
    last_activity TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    stake_amount DECIMAL(18,9), -- TAO stake amount
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Validator performance tracking
CREATE TABLE validator_performance (
    id SERIAL PRIMARY KEY,
    validator_uid INTEGER NOT NULL,
    hotkey VARCHAR(255) NOT NULL,
    coldkey VARCHAR(255),
    validated_events INTEGER DEFAULT 0,
    consensus_accuracy DECIMAL(5,4) DEFAULT 0,
    last_activity TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    stake_amount DECIMAL(18,9), -- TAO stake amount
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Network consensus data
CREATE TABLE consensus_records (
    id SERIAL PRIMARY KEY,
    event_id UUID REFERENCES crime_events(event_id),
    validator_uid INTEGER NOT NULL,
    integrity_score DECIMAL(3,2),
    clustering_score DECIMAL(3,2),
    novelty_score DECIMAL(3,2),
    quality_score DECIMAL(3,2),
    final_score DECIMAL(3,2),
    consensus_round INTEGER,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Crime event clusters for pattern analysis
CREATE TABLE crime_clusters (
    cluster_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_type VARCHAR(100), -- geographic, temporal, behavioral
    center_point POINT, -- Geographic center of cluster
    radius_meters INTEGER, -- Cluster radius in meters
    time_window_start TIMESTAMPTZ,
    time_window_end TIMESTAMPTZ,
    event_count INTEGER DEFAULT 0,
    confidence_score DECIMAL(3,2),
    pattern_description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Risk assessment zones
CREATE TABLE risk_zones (
    zone_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    zone_name VARCHAR(255),
    boundary POLYGON NOT NULL, -- Geographic boundary
    risk_level INTEGER CHECK (risk_level >= 1 AND risk_level <= 5),
    crime_density DECIMAL(8,4), -- Crimes per square km
    recent_incidents INTEGER DEFAULT 0,
    trend_direction VARCHAR(20), -- increasing, decreasing, stable
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Audit log for all subnet operations
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- report, event, miner, validator
    entity_id VARCHAR(255) NOT NULL,
    uid INTEGER, -- Bittensor UID when applicable
    details JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance

-- Raw reports indexes
CREATE INDEX idx_raw_reports_status ON raw_incident_reports(status);
CREATE INDEX idx_raw_reports_timestamp ON raw_incident_reports(timestamp);
CREATE INDEX idx_raw_reports_geotag ON raw_incident_reports USING GIST(geotag);
CREATE INDEX idx_raw_reports_metadata ON raw_incident_reports USING GIN(metadata);

-- Crime events indexes
CREATE INDEX idx_events_type ON crime_events(event_type);
CREATE INDEX idx_events_timestamp ON crime_events(timestamp);
CREATE INDEX idx_events_geotag ON crime_events USING GIST(geotag);
CREATE INDEX idx_events_miner ON crime_events(miner_uid);
CREATE INDEX idx_events_validator ON crime_events(validator_uid);
CREATE INDEX idx_events_cluster ON crime_events(cluster_id);
CREATE INDEX idx_events_verification ON crime_events(verification_status);

-- Performance indexes
CREATE INDEX idx_miner_perf_uid ON miner_performance(miner_uid);
CREATE INDEX idx_miner_perf_hotkey ON miner_performance(hotkey);
CREATE INDEX idx_validator_perf_uid ON validator_performance(validator_uid);
CREATE INDEX idx_validator_perf_hotkey ON validator_performance(hotkey);

-- Consensus indexes
CREATE INDEX idx_consensus_event ON consensus_records(event_id);
CREATE INDEX idx_consensus_validator ON consensus_records(validator_uid);
CREATE INDEX idx_consensus_round ON consensus_records(consensus_round);

-- Cluster indexes
CREATE INDEX idx_clusters_type ON crime_clusters(cluster_type);
CREATE INDEX idx_clusters_center ON crime_clusters USING GIST(center_point);
CREATE INDEX idx_clusters_time ON crime_clusters(time_window_start, time_window_end);

-- Risk zone indexes
CREATE INDEX idx_risk_zones_boundary ON risk_zones USING GIST(boundary);
CREATE INDEX idx_risk_zones_level ON risk_zones(risk_level);

-- Audit log indexes
CREATE INDEX idx_audit_operation ON audit_log(operation);
CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_uid ON audit_log(uid);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_raw_reports_updated_at 
    BEFORE UPDATE ON raw_incident_reports 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crime_events_updated_at 
    BEFORE UPDATE ON crime_events 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_miner_performance_updated_at 
    BEFORE UPDATE ON miner_performance 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_validator_performance_updated_at 
    BEFORE UPDATE ON validator_performance 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crime_clusters_updated_at 
    BEFORE UPDATE ON crime_clusters 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- Active miners view
CREATE VIEW active_miners AS
SELECT 
    miner_uid,
    hotkey,
    total_score,
    processed_reports,
    verified_events,
    last_activity,
    stake_amount,
    (verified_events::decimal / NULLIF(processed_reports, 0)) * 100 as success_rate
FROM miner_performance 
WHERE is_active = true;

-- Recent crime events view
CREATE VIEW recent_crime_events AS
SELECT 
    event_id,
    event_type,
    summary,
    confidence_score,
    ST_X(geotag) as longitude,
    ST_Y(geotag) as latitude,
    timestamp,
    severity_level,
    verification_status
FROM crime_events 
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Crime hotspots view (last 7 days)
CREATE VIEW crime_hotspots AS
SELECT 
    event_type,
    ST_X(geotag) as longitude,
    ST_Y(geotag) as latitude,
    COUNT(*) as incident_count,
    AVG(severity_level) as avg_severity
FROM crime_events 
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND verification_status = 'verified'
GROUP BY event_type, geotag
HAVING COUNT(*) >= 3
ORDER BY incident_count DESC;

COMMIT;

-- Rollback SQL (for reference)
-- BEGIN;
-- 
-- DROP VIEW IF EXISTS crime_hotspots;
-- DROP VIEW IF EXISTS recent_crime_events;
-- DROP VIEW IF EXISTS active_miners;
-- 
-- DROP TRIGGER IF EXISTS update_crime_clusters_updated_at ON crime_clusters;
-- DROP TRIGGER IF EXISTS update_validator_performance_updated_at ON validator_performance;
-- DROP TRIGGER IF EXISTS update_miner_performance_updated_at ON miner_performance;
-- DROP TRIGGER IF EXISTS update_crime_events_updated_at ON crime_events;
-- DROP TRIGGER IF EXISTS update_raw_reports_updated_at ON raw_incident_reports;
-- 
-- DROP FUNCTION IF EXISTS update_updated_at_column();
-- 
-- DROP TABLE IF EXISTS audit_log;
-- DROP TABLE IF EXISTS risk_zones;
-- DROP TABLE IF EXISTS crime_clusters;
-- DROP TABLE IF EXISTS consensus_records;
-- DROP TABLE IF EXISTS validator_performance;
-- DROP TABLE IF EXISTS miner_performance;
-- DROP TABLE IF EXISTS crime_events;
-- DROP TABLE IF EXISTS raw_incident_reports;
-- 
-- DROP EXTENSION IF EXISTS "uuid-ossp";
-- DROP EXTENSION IF EXISTS postgis;
-- 
-- COMMIT;
