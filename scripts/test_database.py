#!/usr/bin/env python3
"""
Test script to fetch data from local Luminar database
This demonstrates the media processing flow with local data
"""

import os
import sys
import psycopg2
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'luminar_testnet',
    'user': os.getenv('USER', 'khemrajregmi'),
}

class LuminarDatabaseClient:
    """Client for interacting with Luminar test database"""
    
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
    
    def get_pending_submissions(self):
        """Get all pending media submissions for processing"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ms.id, ms.filename, ms.file_type, ms.file_size, 
                   ms.original_url, ms.metadata, ms.hash_id,
                   u.wallet_address, u.username
            FROM media_submissions ms
            JOIN users u ON ms.user_id = u.id
            WHERE ms.processing_status = 'pending'
            ORDER BY ms.upload_timestamp;
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'submission_id': row[0],
                'filename': row[1],
                'file_type': row[2],
                'file_size': row[3],
                'original_url': row[4],
                'metadata': row[5],
                'hash_id': row[6],
                'user_wallet': row[7],
                'username': row[8]
            })
        
        return results
    
    def get_processing_history(self, limit=10):
        """Get recent processing events"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT pe.id, pe.event_type, pe.miner_hotkey, pe.validator_hotkey,
                   pe.processing_result, pe.confidence_score, pe.processing_time_ms,
                   pe.created_at, ms.filename, ms.file_type
            FROM processing_events pe
            JOIN media_submissions ms ON pe.submission_id = ms.id
            ORDER BY pe.created_at DESC
            LIMIT %s;
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'event_id': row[0],
                'event_type': row[1],
                'miner_hotkey': row[2][:12] + '...',  # Truncate for display
                'validator_hotkey': row[3][:12] + '...',
                'processing_result': row[4],
                'confidence_score': row[5],
                'processing_time_ms': row[6],
                'created_at': row[7],
                'filename': row[8],
                'file_type': row[9]
            })
        
        return results
    
    def simulate_media_processing(self, submission_id, miner_hotkey, processing_result):
        """Simulate processing a media submission"""
        cursor = self.conn.cursor()
        
        # Get submission details
        cursor.execute("""
            SELECT filename, file_type FROM media_submissions WHERE id = %s;
        """, (submission_id,))
        
        submission = cursor.fetchone()
        if not submission:
            return None
        
        filename, file_type = submission
        
        # Determine event type based on file type
        if file_type.startswith('image'):
            event_type = 'image_analysis'
        elif file_type.startswith('video'):
            event_type = 'video_analysis'
        elif file_type.startswith('audio'):
            event_type = 'audio_analysis'
        else:
            event_type = 'general_analysis'
        
        # Insert processing event
        cursor.execute("""
            INSERT INTO processing_events 
            (submission_id, event_type, miner_hotkey, processing_result, 
             confidence_score, processing_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            submission_id,
            event_type,
            miner_hotkey,
            json.dumps(processing_result),
            processing_result.get('confidence', 0.8),
            processing_result.get('processing_time_ms', 1000)
        ))
        
        event_id = cursor.fetchone()[0]
        
        # Update submission status
        cursor.execute("""
            UPDATE media_submissions 
            SET processing_status = 'processed' 
            WHERE id = %s;
        """, (submission_id,))
        
        self.conn.commit()
        
        return {
            'event_id': event_id,
            'submission_id': submission_id,
            'filename': filename,
            'event_type': event_type
        }
    
    def get_user_stats(self):
        """Get user statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT u.username, u.wallet_address,
                   COUNT(ms.id) as total_submissions,
                   COUNT(CASE WHEN ms.processing_status = 'processed' THEN 1 END) as processed,
                   COUNT(CASE WHEN ms.processing_status = 'pending' THEN 1 END) as pending
            FROM users u
            LEFT JOIN media_submissions ms ON u.id = ms.user_id
            GROUP BY u.id, u.username, u.wallet_address
            ORDER BY total_submissions DESC;
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'username': row[0],
                'wallet_address': row[1][:12] + '...',
                'total_submissions': row[2],
                'processed': row[3],
                'pending': row[4]
            })
        
        return results
    
    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    """Test database connectivity and data fetching"""
    print("🔍 Testing Luminar Database Connectivity...")
    
    try:
        db = LuminarDatabaseClient()
        
        # Test 1: Get pending submissions
        print("\n📋 Pending Media Submissions:")
        pending = db.get_pending_submissions()
        for item in pending:
            print(f"  • {item['filename']} ({item['file_type']}) - {item['username']}")
            print(f"    Hash: {item['hash_id']}")
            print(f"    Size: {item['file_size'] / 1024 / 1024:.1f} MB")
            if item['metadata'] and 'tags' in item['metadata']:
                print(f"    Tags: {', '.join(item['metadata']['tags'])}")
            print()
        
        # Test 2: Get processing history
        print("📈 Recent Processing Events:")
        history = db.get_processing_history(5)
        for event in history:
            print(f"  • {event['filename']} - {event['event_type']}")
            print(f"    Miner: {event['miner_hotkey']}")
            print(f"    Confidence: {event['confidence_score']:.2f}")
            print(f"    Time: {event['processing_time_ms']}ms")
            print(f"    Processed: {event['created_at']}")
            print()
        
        # Test 3: Simulate processing a pending submission
        if pending:
            print("⚡ Simulating Media Processing...")
            test_submission = pending[0]
            
            # Mock processing result
            mock_result = {
                'confidence': 0.92,
                'processing_time_ms': 1500,
                'analysis': {
                    'detected_objects': ['text', 'graphics', 'ui_elements'],
                    'quality_score': 8.7,
                    'content_type': 'demo_animation'
                },
                'miner_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
            
            result = db.simulate_media_processing(
                test_submission['submission_id'],
                '5HNi3h9G1WzKbRjEgAvSvJakkP3NAg8GXdBVFd9CSHuTg56dD',  # Test miner
                mock_result
            )
            
            if result:
                print(f"  ✅ Processed: {result['filename']}")
                print(f"  📋 Event ID: {result['event_id']}")
                print(f"  🎯 Type: {result['event_type']}")
        
        # Test 4: Get user statistics
        print("\n👥 User Statistics:")
        stats = db.get_user_stats()
        for user in stats:
            print(f"  • {user['username']} ({user['wallet_address']})")
            print(f"    Total: {user['total_submissions']}, Processed: {user['processed']}, Pending: {user['pending']}")
        
        db.close()
        print("\n🎉 Database connectivity test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
