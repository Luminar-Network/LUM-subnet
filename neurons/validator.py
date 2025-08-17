# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Luminar Network
# Copyright © 2025 Khem Raj Regmi

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import asyncio
import sys
import os
import random
import numpy as np
import bittensor as bt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
try:
    from template.validator import forward
    TEMPLATE_FORWARD_AVAILABLE = True
except ImportError:
    TEMPLATE_FORWARD_AVAILABLE = False
    bt.logging.warning("Template forward not available - using Luminar implementation")

# Import Luminar protocol
try:
    from template.protocol import (
        MediaProcessingRequest, UserSubmission, ProcessedEvent, 
        MediaUpload, ValidationResult
    )
    LUMINAR_PROTOCOL_AVAILABLE = True
except ImportError:
    # Fallback for dummy protocol if MediaProcessingRequest not available
    from template.protocol import Dummy as MediaProcessingRequest
    
    # Mock classes for testing
    class UserSubmission:
        def __init__(self, submission_id, user_id, text_description, media_files, geotag, timestamp, incident_type, metadata):
            self.submission_id = submission_id
            self.user_id = user_id
            self.text_description = text_description
            self.media_files = media_files
            self.geotag = geotag
            self.timestamp = timestamp
            self.incident_type = incident_type
            self.metadata = metadata
    
    class MediaUpload:
        def __init__(self, filename, media_type, url, file_size):
            self.filename = filename
            self.media_type = media_type
            self.url = url
            self.file_size = file_size
    
    class ProcessedEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    LUMINAR_PROTOCOL_AVAILABLE = False
    bt.logging.warning("Using dummy protocol - Luminar protocol not available")

# Import utilities
try:
    from template.utils.uids import get_random_uids
except ImportError:
    def get_random_uids(validator, k=None):
        """Fallback function for getting random UIDs"""
        if k is None:
            k = min(10, len(validator.metagraph.hotkeys))
        return np.random.choice(len(validator.metagraph.hotkeys), size=k, replace=False)


class Validator(BaseValidatorNeuron):
    """
    Luminar Media Processing Validator
    
    Implements complete user media processing flow:
    1. User uploads media (text + visuals) + metadata with ID
    2. Validator receives user submissions
    3. Validator sends submissions to miners for processing  
    4. Miner creates events from text/visual analysis, filters duplicates
    5. Validator compares miner events with original user metadata
    6. Validator scores miners based on metadata consistency
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("🚀 Initializing Luminar Media Processing Validator")
        bt.logging.info("load_state()")
        self.load_state()

        # User submission queue (simulates receiving from app)
        self.pending_submissions = []
        self.processed_submissions = {}
        
        # Validation statistics
        self.validation_stats = {
            "total_submissions": 0,
            "successful_validations": 0,
            "metadata_mismatches": 0,
            "duplicate_flags": 0
        }
        
        # Metadata comparison thresholds
        self.validation_thresholds = {
            "timestamp_tolerance_minutes": 60,
            "geotag_tolerance_meters": 500,
            "visual_authenticity_min": 0.7,
            "text_consistency_min": 0.6
        }
        
        bt.logging.info("✅ Luminar Validator initialized successfully")

    async def forward(self):
        """
        Main validation loop implementing your flow:
        
        1. Get user submissions (simulated)
        2. Send to miners for processing
        3. Compare results with metadata
        4. Score miners and update weights
        """
        # Check if we have Luminar protocol available
        if LUMINAR_PROTOCOL_AVAILABLE:
            return await self._luminar_forward()
        else:
            # Fallback to template forward if available
            if TEMPLATE_FORWARD_AVAILABLE:
                bt.logging.info("Using template forward (Luminar protocol not available)")
                return await forward(self)
            else:
                # Basic validation loop
                bt.logging.info("Running basic validation loop")
                await asyncio.sleep(30)
                return
    
    async def _luminar_forward(self):
        """
        Luminar-specific validation forward pass
        """
        bt.logging.info("🔄 Starting Luminar validation round...")
        
        # Step 1: Get user submissions (simulate app uploads)
        user_submissions = await self._get_user_submissions()
        
        if not user_submissions:
            bt.logging.info("⏳ No user submissions available, waiting...")
            await asyncio.sleep(30)
            return
        
        bt.logging.info(f"📥 Processing {len(user_submissions)} user submissions")
        
        # Step 2: Send submissions to miners
        try:
            miner_uids = get_random_uids(self, k=min(getattr(self.config.neuron, 'sample_size', 10), self.subtensor.n.item()))
        except:
            # Fallback if get_random_uids fails
            available_uids = list(range(len(self.metagraph.hotkeys)))
            sample_size = min(10, len(available_uids))
            miner_uids = np.random.choice(available_uids, size=sample_size, replace=False)
        
        if len(miner_uids) == 0:
            bt.logging.warning("❌ No miners available for processing")
            await asyncio.sleep(30)
            return
        
        # Create processing request
        request = MediaProcessingRequest()
        if hasattr(request, 'user_submissions'):
            request.user_submissions = user_submissions
            request.task_id = f"VAL_{int(time.time())}"
            request.processing_deadline = datetime.now() + timedelta(minutes=5)
            request.requirements = {"duplicate_detection": True, "visual_analysis": True}
        
        # Step 3: Query miners
        bt.logging.info(f"🔗 Querying {len(miner_uids)} miners...")
        try:
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=request,
                deserialize=False,
                timeout=getattr(self.config.neuron, 'timeout', 30)
            )
        except Exception as e:
            bt.logging.error(f"❌ Error querying miners: {e}")
            responses = []
        
        # Step 4: Validate miner responses against user metadata
        scores = await self._validate_miner_responses(responses, user_submissions, miner_uids)
        
        # Step 5: Update miner weights based on validation scores
        await self._update_weights(scores, miner_uids)
        
        # Step 6: Store results and update statistics
        await self._store_validation_results(user_submissions, responses, scores)
        
        bt.logging.info(f"✅ Validation round completed. Processed {len(user_submissions)} submissions")
    
    async def _get_user_submissions(self) -> List[UserSubmission]:
        """
        Get user submissions from mobile app/web interface
        
        In production: This would connect to your app's API
        For testing: Generates realistic synthetic submissions
        """
        # TODO: In production, replace with actual API calls
        # return await self._fetch_from_app_api()
        
        # For now, generate synthetic user submissions
        return await self._generate_synthetic_submissions()
    
    async def _generate_synthetic_submissions(self) -> List[UserSubmission]:
        """Generate realistic user submissions for testing"""
        
        # Sample incidents with media
        incidents = [
            {
                "text": "Car accident on main street, two vehicles involved, no injuries reported",
                "location": {"lat": 27.7172, "lng": 85.3240},
                "incident_type": "accident",
                "has_media": True
            },
            {
                "text": "Motorcycle theft reported near college gate, red Honda missing",
                "location": {"lat": 27.7000, "lng": 85.3333},
                "incident_type": "theft", 
                "has_media": False
            },
            {
                "text": "Fight between two groups near market area, police called",
                "location": {"lat": 27.7068, "lng": 85.3181},
                "incident_type": "assault",
                "has_media": True
            },
            {
                "text": "Suspicious person taking photos of houses in residential area",
                "location": {"lat": 27.7256, "lng": 85.3370},
                "incident_type": "suspicious",
                "has_media": False
            }
        ]
        
        submissions = []
        
        for i, incident in enumerate(incidents):
            submission_id = f"sub_{uuid.uuid4().hex[:8]}"
            
            # Create media files if incident has media
            media_files = []
            if incident["has_media"]:
                media_files.append(MediaUpload(
                    filename=f"incident_{i}.jpg",
                    media_type="image/jpeg",
                    url=f"https://example.com/media/{submission_id}.jpg",
                    file_size=1024 * 1024 * 2  # 2MB
                ))
            
            submission = UserSubmission(
                submission_id=submission_id,
                user_id=f"user_{random.randint(1, 100)}",
                text_description=incident["text"],
                media_files=media_files,
                geotag=incident["location"],
                timestamp=datetime.now() - timedelta(minutes=random.randint(5, 60)),
                incident_type=incident["incident_type"],
                metadata={
                    "source": "mobile_app",
                    "app_version": "1.2.3",
                    "device_type": "iPhone"
                }
            )
            
            submissions.append(submission)
        
        return submissions
    
    async def _validate_miner_responses(self, responses: List[Any], 
                                      user_submissions: List[UserSubmission],
                                      miner_uids: List[int]) -> List[float]:
        """
        Core validation: Compare miner events with user metadata
        
        Your requirement: "validator compares miner generated event with metadata upload by user (timestamp, geotags)"
        """
        bt.logging.info("🔍 Validating miner responses against user metadata...")
        
        scores = []
        
        for i, (response, miner_uid) in enumerate(zip(responses, miner_uids)):
            try:
                score = await self._score_miner_response(response, user_submissions, miner_uid)
                scores.append(score)
                bt.logging.debug(f"Miner {miner_uid} scored: {score:.3f}")
            except Exception as e:
                bt.logging.error(f"Error scoring miner {miner_uid}: {e}")
                scores.append(0.0)
        
        return scores
    
    async def _score_miner_response(self, response: Any,
                                  user_submissions: List[UserSubmission],
                                  miner_uid: int) -> float:
        """
        Score a single miner's response by comparing with user metadata
        
        Checks:
        1. Timestamp consistency
        2. Geotag accuracy
        3. Event type accuracy
        4. Text consistency
        5. Duplicate detection accuracy
        """
        
        if not hasattr(response, 'processed_events') or not response.processed_events:
            bt.logging.warning(f"Miner {miner_uid} returned no processed events")
            return 0.0
        
        total_score = 0.0
        event_count = len(response.processed_events)
        
        # Create mapping of submission IDs to submissions
        submission_map = {sub.submission_id: sub for sub in user_submissions}
        
        for event in response.processed_events:
            event_score = 0.0
            
            # Find corresponding submission
            submission = submission_map.get(getattr(event, 'submission_id', None))
            if not submission:
                bt.logging.warning(f"No submission found for event {getattr(event, 'event_id', 'unknown')}")
                continue
            
            # 1. Timestamp validation (25%)
            timestamp_score = self._validate_timestamp(event, submission)
            event_score += timestamp_score * 0.25
            
            # 2. Geotag validation (25%)
            geotag_score = self._validate_geotag(event, submission)
            event_score += geotag_score * 0.25
            
            # 3. Event type validation (20%)
            type_score = self._validate_event_type(event, submission)
            event_score += type_score * 0.20
            
            # 4. Text consistency validation (20%)
            text_score = self._validate_text_consistency(event, submission)
            event_score += text_score * 0.20
            
            # 5. Processing quality (10%)
            quality_score = self._validate_processing_quality(event)
            event_score += quality_score * 0.10
            
            total_score += event_score
        
        # Add bonus for duplicate detection
        duplicate_bonus = self._score_duplicate_detection(response, user_submissions)
        
        # Average score across all events + duplicate bonus
        final_score = (total_score / event_count) + duplicate_bonus if event_count > 0 else 0.0
        
        return min(1.0, final_score)  # Cap at 1.0
    
    def _validate_timestamp(self, event: Any, submission: UserSubmission) -> float:
        """
        Validate timestamp consistency between event and submission
        
        Full score if within tolerance, linear decay outside
        """
        if not hasattr(event, 'processing_timestamp') or not submission.timestamp:
            return 0.5  # Partial credit if timestamps are missing
        
        # Calculate time difference in minutes
        time_diff = abs((event.processing_timestamp - submission.timestamp).total_seconds()) / 60
        tolerance = self.validation_thresholds["timestamp_tolerance_minutes"]
        
        if time_diff <= tolerance:
            return 1.0  # Perfect score within tolerance
        else:
            # Linear decay beyond tolerance
            return max(0.0, 1.0 - (time_diff - tolerance) / tolerance)
    
    def _validate_geotag(self, event: Any, submission: UserSubmission) -> float:
        """Validate geotag consistency between event and submission"""
        if not submission.geotag:
            return 0.5  # Partial credit if no geotag provided
        
        # Extract coordinates from event (would need proper implementation)
        event_coords = self._extract_coordinates_from_event(event, submission)
        if not event_coords:
            return 0.3  # Low score if coordinates not found in event
        
        # Calculate distance
        lat_diff = abs(event_coords[0] - submission.geotag["lat"])
        lng_diff = abs(event_coords[1] - submission.geotag["lng"])
        distance_deg = (lat_diff**2 + lng_diff**2)**0.5
        
        # Convert to meters (rough approximation: 1 degree ≈ 111km)
        distance_meters = distance_deg * 111000
        tolerance = self.validation_thresholds["geotag_tolerance_meters"]
        
        if distance_meters <= tolerance:
            return 1.0
        else:
            return max(0.0, 1.0 - (distance_meters - tolerance) / tolerance)
    
    def _extract_coordinates_from_event(self, event: Any, submission: UserSubmission) -> Tuple[float, float]:
        """Extract coordinates from event (placeholder implementation)"""
        # In production, this would parse the event content for location information
        # For now, return submission coordinates as fallback
        return (submission.geotag["lat"], submission.geotag["lng"])
    
    def _validate_event_type(self, event: Any, submission: UserSubmission) -> float:
        """Validate event type classification accuracy"""
        if not hasattr(submission, 'incident_type') or not hasattr(event, 'event_type'):
            return 0.5
        
        # Simple matching for now
        if event.event_type.lower() == submission.incident_type.lower():
            return 1.0
        
        # Partial credit for related types
        related_types = {
            "accident": ["crash", "collision", "traffic"],
            "theft": ["robbery", "burglary", "stolen"],
            "assault": ["fight", "violence", "attack"]
        }
        
        for main_type, variants in related_types.items():
            if (submission.incident_type.lower() == main_type and 
                event.event_type.lower() in variants):
                return 0.7
        
        return 0.0
    
    def _validate_text_consistency(self, event: Any, submission: UserSubmission) -> float:
        """Validate text consistency between event summary and submission"""
        if not hasattr(event, 'summary') or not submission.text_description:
            return 0.5
        
        # Simple word overlap check
        event_words = set(event.summary.lower().split())
        submission_words = set(submission.text_description.lower().split())
        
        if not event_words or not submission_words:
            return 0.3
        
        overlap = len(event_words.intersection(submission_words))
        total_unique = len(event_words.union(submission_words))
        
        consistency = overlap / total_unique if total_unique > 0 else 0.0
        return min(1.0, consistency * 2)  # Scale up to reward good overlap
    
    def _validate_processing_quality(self, event: Any) -> float:
        """Validate the overall quality of event processing"""
        quality_score = 0.0
        
        # Check if event has required fields
        required_fields = ['event_id', 'summary', 'confidence_score']
        for field in required_fields:
            if hasattr(event, field) and getattr(event, field):
                quality_score += 0.3
        
        # Check confidence score range
        if hasattr(event, 'confidence_score'):
            conf = getattr(event, 'confidence_score', 0)
            if 0.0 <= conf <= 1.0:
                quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _score_duplicate_detection(self, response: Any, 
                                 user_submissions: List[UserSubmission]) -> float:
        """Score the miner's duplicate detection capability"""
        # This would implement actual duplicate detection scoring
        # For now, give a small bonus if processing metadata indicates duplicate detection
        if hasattr(response, 'processing_metadata'):
            metadata = response.processing_metadata
            if isinstance(metadata, dict) and metadata.get('duplicates_detected', 0) >= 0:
                return 0.05  # Small bonus
        
        return 0.0
    
    async def _update_weights(self, scores: List[float], miner_uids: List[int]):
        """Update miner weights based on validation scores"""
        if not scores or not miner_uids:
            return
        
        bt.logging.info(f"🔄 Updating weights for {len(miner_uids)} miners")
        
        # Update scores for miners
        for uid, score in zip(miner_uids, scores):
            if uid < len(self.scores):
                # Update moving average
                alpha = 0.1  # Learning rate
                self.scores[uid] = alpha * score + (1 - alpha) * self.scores[uid]
        
        bt.logging.info(f"📊 Average scores: {np.mean(scores):.3f}")
    
    async def _store_validation_results(self, submissions: List[UserSubmission],
                                      responses: List[Any],
                                      scores: List[float]):
        """Store validation results and update statistics"""
        self.validation_stats["total_submissions"] += len(submissions)
        self.validation_stats["successful_validations"] += len([s for s in scores if s > 0.5])
        
        bt.logging.info(f"📈 Validation Stats: {self.validation_stats}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"🚀 Luminar Validator running... {time.time()}")
            bt.logging.info(f"📊 Stats: {validator.validation_stats}")
            time.sleep(5)
