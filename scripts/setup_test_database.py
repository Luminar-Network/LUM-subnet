#!/usr/bin/env python3
"""
Setup script for Luminar testnet database with sample data
Creates tables and populates them with test media processing data
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime, timedelta
import json

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'luminar_testnet',
    'user': os.getenv('USER', 'khemrajregmi'),  # Use current user
}

def create_tables(conn):
    """Create all necessary tables for Luminar media processing"""
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            wallet_address VARCHAR(64) UNIQUE NOT NULL,
            username VARCHAR(50) UNIQUE,
            email VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Media submissions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media_submissions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            file_size INTEGER,
            original_url TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status VARCHAR(50) DEFAULT 'pending',
            metadata JSONB,
            hash_id VARCHAR(64) UNIQUE
        );
    """)
    
    # Processing events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_events (
            id SERIAL PRIMARY KEY,
            submission_id INTEGER REFERENCES media_submissions(id),
            event_type VARCHAR(50) NOT NULL,
            miner_hotkey VARCHAR(64),
            validator_hotkey VARCHAR(64),
            processing_result JSONB,
            confidence_score FLOAT,
            processing_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Validation results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id SERIAL PRIMARY KEY,
            event_id INTEGER REFERENCES processing_events(id),
            validator_hotkey VARCHAR(64) NOT NULL,
            is_valid BOOLEAN NOT NULL,
            quality_score FLOAT,
            feedback TEXT,
            validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Rewards table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            id SERIAL PRIMARY KEY,
            miner_hotkey VARCHAR(64) NOT NULL,
            validator_hotkey VARCHAR(64),
            reward_amount DECIMAL(18, 8),
            reward_type VARCHAR(50),
            submission_id INTEGER REFERENCES media_submissions(id),
            distributed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    print("✅ Database tables created successfully!")

def insert_sample_data(conn):
    """Insert sample data for testing"""
    cursor = conn.cursor()
    
    # Sample users
    sample_users = [
        ('5GBW9uruRyKbRjEgAvSvJakkP3NAg8GXdBVFd9CSHuTg56dD', 'test_miner', 'miner@luminar.network'),
        ('5HDoBRU31YP8sTrZvxVD6qcCZWzUu7Ye3bbkQFKHJUUBYgUa', 'test_validator', 'validator@luminar.network'),
        ('5ECvrzNV4Tk84mPRN3ZVESvcEBApq8RZYLsQwjME3YrxP2Xy', 'content_creator', 'creator@example.com'),
        ('5Gb4LeyynhXfdShZuEiQ9yt3oM3aNtY2fBcJCpyu56SJnJQj', 'media_uploader', 'uploader@example.com'),
    ]
    
    for wallet, username, email in sample_users:
        cursor.execute("""
            INSERT INTO users (wallet_address, username, email) 
            VALUES (%s, %s, %s) ON CONFLICT (wallet_address) DO NOTHING;
        """, (wallet, username, email))
    
    # Sample media submissions
    sample_media = [
        {
            'filename': 'sunset_beach.jpg',
            'file_type': 'image/jpeg',
            'file_size': 2048576,
            'original_url': 'https://example.com/images/sunset_beach.jpg',
            'metadata': {
                'width': 1920,
                'height': 1080,
                'format': 'JPEG',
                'camera': 'iPhone 14 Pro',
                'location': 'Malibu Beach, CA',
                'tags': ['sunset', 'beach', 'ocean', 'nature']
            },
            'hash_id': 'hash_sunset_beach_001'
        },
        {
            'filename': 'city_night.mp4',
            'file_type': 'video/mp4',
            'file_size': 52428800,
            'original_url': 'https://example.com/videos/city_night.mp4',
            'metadata': {
                'duration': 30,
                'resolution': '4K',
                'fps': 60,
                'codec': 'H.264',
                'location': 'New York City',
                'tags': ['city', 'night', 'urban', 'lights']
            },
            'hash_id': 'hash_city_night_002'
        },
        {
            'filename': 'podcast_episode.mp3',
            'file_type': 'audio/mp3',
            'file_size': 25165824,
            'original_url': 'https://example.com/audio/podcast_episode.mp3',
            'metadata': {
                'duration': 1800,
                'bitrate': '320kbps',
                'artist': 'Luminar Podcast',
                'title': 'AI and Blockchain: The Future',
                'tags': ['podcast', 'ai', 'blockchain', 'technology']
            },
            'hash_id': 'hash_podcast_003'
        },
        {
            'filename': 'product_demo.gif',
            'file_type': 'image/gif',
            'file_size': 1048576,
            'original_url': 'https://example.com/gifs/product_demo.gif',
            'metadata': {
                'width': 800,
                'height': 600,
                'frames': 120,
                'duration': 4,
                'tags': ['demo', 'product', 'animation', 'ui']
            },
            'hash_id': 'hash_product_demo_004'
        }
    ]
    
    for i, media in enumerate(sample_media, 1):
        cursor.execute("""
            INSERT INTO media_submissions 
            (user_id, filename, file_type, file_size, original_url, metadata, hash_id, processing_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hash_id) DO NOTHING;
        """, (
            (i % 4) + 1,  # Cycle through users
            media['filename'],
            media['file_type'],
            media['file_size'],
            media['original_url'],
            json.dumps(media['metadata']),
            media['hash_id'],
            'processed' if i <= 2 else 'pending'
        ))
    
    # Sample processing events
    sample_events = [
        {
            'submission_id': 1,
            'event_type': 'image_analysis',
            'miner_hotkey': '5HNi3h9G1WzKbHvcoxVbwT5qMbpkLZS7CqbGLeh4WioiKZSN',
            'validator_hotkey': '5EP6R7c6FSL4DKS1115XyqeHpmR2VeYGxbFaShGNqXJphg6F',
            'processing_result': {
                'objects_detected': ['sunset', 'ocean', 'beach', 'palm_trees'],
                'scene_description': 'A beautiful sunset over a tropical beach with palm trees',
                'sentiment': 'positive',
                'artistic_score': 8.5,
                'technical_quality': 9.2
            },
            'confidence_score': 0.94,
            'processing_time_ms': 1250
        },
        {
            'submission_id': 2,
            'event_type': 'video_analysis',
            'miner_hotkey': '5HNi3h9G1WzKbHvcoxVbwT5qMbpkLZS7CqbGLeh4WioiKZSN',
            'validator_hotkey': '5EP6R7c6FSL4DKS1115XyqeHpmR2VeYGxbFaShGNqXJphg6F',
            'processing_result': {
                'scene_changes': 8,
                'dominant_colors': ['blue', 'yellow', 'white'],
                'motion_analysis': 'smooth_panning',
                'audio_present': False,
                'content_rating': 'family_friendly'
            },
            'confidence_score': 0.87,
            'processing_time_ms': 5420
        }
    ]
    
    for event in sample_events:
        cursor.execute("""
            INSERT INTO processing_events 
            (submission_id, event_type, miner_hotkey, validator_hotkey, 
             processing_result, confidence_score, processing_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (
            event['submission_id'],
            event['event_type'],
            event['miner_hotkey'],
            event['validator_hotkey'],
            json.dumps(event['processing_result']),
            event['confidence_score'],
            event['processing_time_ms']
        ))
    
    conn.commit()
    print("✅ Sample data inserted successfully!")

def create_indexes(conn):
    """Create indexes for better performance"""
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_media_status ON media_submissions(processing_status);",
        "CREATE INDEX IF NOT EXISTS idx_media_user ON media_submissions(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_events_submission ON processing_events(submission_id);",
        "CREATE INDEX IF NOT EXISTS idx_events_miner ON processing_events(miner_hotkey);",
        "CREATE INDEX IF NOT EXISTS idx_validation_event ON validation_results(event_id);",
        "CREATE INDEX IF NOT EXISTS idx_rewards_miner ON rewards(miner_hotkey);",
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    conn.commit()
    print("✅ Database indexes created successfully!")

def main():
    """Main setup function"""
    print("🚀 Setting up Luminar testnet database...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Setup database
        create_tables(conn)
        insert_sample_data(conn)
        create_indexes(conn)
        
        # Verify setup
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM media_submissions;")
        media_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processing_events;")
        event_count = cursor.fetchone()[0]
        
        print(f"\n🎉 Database setup complete!")
        print(f"📊 Users: {user_count}")
        print(f"📷 Media submissions: {media_count}")  
        print(f"⚡ Processing events: {event_count}")
        print(f"\n🔗 Database: postgresql://localhost:5432/luminar_testnet")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
