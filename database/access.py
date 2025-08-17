# Database Access Layer for Luminar Subnet
# Copyright © 2025 Luminar Network

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from uuid import UUID

from config.database import get_connection_params, get_connection_string, is_production
from template.protocol import RawIncidentReport, CrimeEvent

logger = logging.getLogger(__name__)

@dataclass
class DatabaseStats:
    """Database statistics"""
    total_reports: int
    processed_events: int
    active_miners: int
    active_validators: int
    verification_rate: float
    avg_processing_time: float

class LuminarDatabase:
    """Database access layer for Luminar subnet"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        if self.pool is None:
            try:
                conn_params = get_connection_params()
                self.pool = await asyncpg.create_pool(
                    **conn_params,
                    min_size=5,
                    max_size=20,
                    command_timeout=60
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        if not self.pool:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    # ==================== RAW INCIDENT REPORTS ====================
    
    async def store_raw_report(self, report: RawIncidentReport) -> bool:
        """Store a raw incident report"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO raw_incident_reports (
                        external_id, text_description, geotag, timestamp,
                        media_urls, media_hashes, user_id, metadata, source
                    ) VALUES ($1, $2, POINT($3, $4), $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (external_id) DO NOTHING
                """, 
                    report.report_id,
                    report.text_description,
                    report.geotag.get("lng", 0),
                    report.geotag.get("lat", 0),
                    report.timestamp,
                    report.media_urls,
                    report.media_hashes,
                    report.user_id,
                    json.dumps(report.metadata),
                    "subnet_synthetic"  # Mark as synthetic for testing
                )
                logger.debug(f"Stored raw report: {report.report_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store raw report: {e}")
            return False
    
    async def get_pending_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending incident reports for processing"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT report_id, external_id, text_description, 
                           ST_X(geotag) as longitude, ST_Y(geotag) as latitude,
                           timestamp, media_urls, media_hashes, user_id, metadata
                    FROM raw_incident_reports 
                    WHERE status = 'pending'
                    ORDER BY timestamp ASC
                    LIMIT $1
                """, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get pending reports: {e}")
            return []
    
    async def update_report_status(self, report_id: str, status: str) -> bool:
        """Update report processing status"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE raw_incident_reports 
                    SET status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE external_id = $2
                """, status, report_id)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update report status: {e}")
            return False
    
    # ==================== CRIME EVENTS ====================
    
    async def store_crime_event(self, event: CrimeEvent, miner_uid: int) -> bool:
        """Store a processed crime event"""
        try:
            async with self.pool.acquire() as conn:
                # Convert related_reports list to array of UUIDs
                related_uuids = [str(rid) for rid in event.related_reports]
                
                await conn.execute("""
                    INSERT INTO crime_events (
                        cluster_id, event_type, summary, confidence_score,
                        geotag, timestamp, severity_level, related_reports,
                        metadata, miner_uid
                    ) VALUES ($1, $2, $3, $4, POINT($5, $6), $7, $8, $9, $10, $11)
                """,
                    str(event.cluster_id) if event.cluster_id else None,
                    event.event_type,
                    event.summary,
                    float(event.confidence_score),
                    event.geotag.get("lng", 0),
                    event.geotag.get("lat", 0),
                    event.timestamp,
                    event.severity_level,
                    related_uuids,
                    json.dumps(event.metadata),
                    miner_uid
                )
                logger.debug(f"Stored crime event: {event.event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store crime event: {e}")
            return False
    
    async def get_recent_events(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent crime events"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT event_id, event_type, summary, confidence_score,
                           ST_X(geotag) as longitude, ST_Y(geotag) as latitude,
                           timestamp, severity_level, verification_status,
                           miner_uid, verification_score
                    FROM crime_events 
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT $1
                """ % hours, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    async def update_event_verification(self, event_id: UUID, validator_uid: int, 
                                       verification_status: str, score: float) -> bool:
        """Update event verification status"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE crime_events 
                    SET validator_uid = $1, verification_status = $2, 
                        verification_score = $3, updated_at = CURRENT_TIMESTAMP
                    WHERE event_id = $4
                """, validator_uid, verification_status, score, event_id)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update event verification: {e}")
            return False
    
    # ==================== MINER PERFORMANCE ====================
    
    async def update_miner_performance(self, miner_uid: int, hotkey: str,
                                     scores: Dict[str, float]) -> bool:
        """Update miner performance metrics"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO miner_performance (
                        miner_uid, hotkey, total_score, integrity_score,
                        clustering_score, novelty_score, quality_score,
                        processed_reports, verified_events
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, 1, 0)
                    ON CONFLICT (miner_uid) DO UPDATE SET
                        total_score = miner_performance.total_score + EXCLUDED.total_score,
                        integrity_score = EXCLUDED.integrity_score,
                        clustering_score = EXCLUDED.clustering_score,
                        novelty_score = EXCLUDED.novelty_score,
                        quality_score = EXCLUDED.quality_score,
                        processed_reports = miner_performance.processed_reports + 1,
                        last_activity = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    miner_uid, hotkey,
                    scores.get('total', 0.0),
                    scores.get('integrity', 0.0),
                    scores.get('clustering', 0.0),
                    scores.get('novelty', 0.0),
                    scores.get('quality', 0.0)
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to update miner performance: {e}")
            return False
    
    async def get_miner_stats(self, miner_uid: int) -> Optional[Dict[str, Any]]:
        """Get miner performance statistics"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT miner_uid, hotkey, total_score, processed_reports,
                           verified_events, last_activity,
                           (verified_events::decimal / NULLIF(processed_reports, 0)) * 100 as success_rate
                    FROM miner_performance 
                    WHERE miner_uid = $1
                """, miner_uid)
                
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get miner stats: {e}")
            return None
    
    # ==================== CONSENSUS TRACKING ====================
    
    async def store_consensus_record(self, event_id: UUID, validator_uid: int,
                                   scores: Dict[str, float], consensus_round: int) -> bool:
        """Store consensus scoring record"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO consensus_records (
                        event_id, validator_uid, integrity_score, clustering_score,
                        novelty_score, quality_score, final_score, consensus_round
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    event_id, validator_uid,
                    scores.get('integrity', 0.0),
                    scores.get('clustering', 0.0),
                    scores.get('novelty', 0.0),
                    scores.get('quality', 0.0),
                    scores.get('final', 0.0),
                    consensus_round
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to store consensus record: {e}")
            return False
    
    # ==================== ANALYTICS ====================
    
    async def get_database_stats(self) -> DatabaseStats:
        """Get overall database statistics"""
        try:
            async with self.pool.acquire() as conn:
                stats_row = await conn.fetchrow("""
                    SELECT 
                        (SELECT COUNT(*) FROM raw_incident_reports) as total_reports,
                        (SELECT COUNT(*) FROM crime_events) as processed_events,
                        (SELECT COUNT(*) FROM miner_performance WHERE is_active = true) as active_miners,
                        (SELECT COUNT(*) FROM validator_performance WHERE is_active = true) as active_validators,
                        (SELECT AVG(verification_score) FROM crime_events WHERE verification_score IS NOT NULL) as avg_verification_score
                """)
                
                # Calculate verification rate
                verified_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM crime_events WHERE verification_status = 'verified'
                """)
                total_events = stats_row['processed_events'] or 1
                verification_rate = (verified_count / total_events) * 100 if total_events > 0 else 0
                
                return DatabaseStats(
                    total_reports=stats_row['total_reports'] or 0,
                    processed_events=stats_row['processed_events'] or 0,
                    active_miners=stats_row['active_miners'] or 0,
                    active_validators=stats_row['active_validators'] or 0,
                    verification_rate=verification_rate,
                    avg_processing_time=0.0  # TODO: Calculate from audit logs
                )
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return DatabaseStats(0, 0, 0, 0, 0.0, 0.0)
    
    async def get_crime_hotspots(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get crime hotspots for the last N days"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        event_type,
                        ST_X(geotag) as longitude,
                        ST_Y(geotag) as latitude,
                        COUNT(*) as incident_count,
                        AVG(severity_level) as avg_severity,
                        AVG(confidence_score) as avg_confidence
                    FROM crime_events 
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        AND verification_status = 'verified'
                    GROUP BY event_type, geotag
                    HAVING COUNT(*) >= 2
                    ORDER BY incident_count DESC
                    LIMIT 50
                """ % days)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get crime hotspots: {e}")
            return []

# Singleton instance
db = LuminarDatabase()

# Convenience functions
async def initialize_database():
    """Initialize database connection"""
    await db.initialize()

async def close_database():
    """Close database connection"""
    await db.close()

async def health_check() -> bool:
    """Check database health"""
    return await db.health_check()

# Context manager for database operations
class DatabaseTransaction:
    """Context manager for database transactions"""
    
    def __init__(self):
        self.conn = None
        self.transaction = None
    
    async def __aenter__(self):
        if not db.pool:
            await db.initialize()
        
        self.conn = await db.pool.acquire()
        self.transaction = self.conn.transaction()
        await self.transaction.start()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.transaction.rollback()
        else:
            await self.transaction.commit()
        
        await db.pool.release(self.conn)
