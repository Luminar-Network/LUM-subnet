# Enhanced Luminar Validator with Media Processing and Metadata Comparison
# Copyright © 2025 Luminar Network
# Implements complete flow: User uploads → Validator → Miner → Metadata comparison

import time
import asyncio
import threading
import bittensor as bt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import hashlib
import numpy as np

from template.protocol import (
    MediaProcessingRequest, UserSubmission, ProcessedEvent, 
    MediaUpload, ValidationResult
)
from template.base.validator import BaseValidatorNeuron
from template.validator.luminar_reward import get_luminar_rewards
from template.utils.uids import get_random_uids

class LuminarMediaValidator(BaseValidatorNeuron):
    """
    Enhanced Luminar Validator implementing complete user media processing flow
    
    Your Complete Flow:
    1. User uploads media (text + visuals) + metadata with ID
    2. Validator receives user submissions
    3. Validator sends submissions to miners for processing  
    4. Miner creates events from text/visual analysis, filters duplicates
    5. Validator compares miner events with original user metadata
    6. Validator scores miners based on metadata consistency
    """
    
    def __init__(self, config=None):
        super(LuminarMediaValidator, self).__init__(config=config)
        
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
            "timestamp_tolerance_minutes": 60,  # 1 hour tolerance
            "geotag_tolerance_meters": 500,     # 500m tolerance
            "visual_authenticity_min": 0.7,    # Minimum authenticity score
            "text_consistency_min": 0.6        # Minimum text consistency
        }
        
        bt.logging.info("🎯 Luminar Media Validator initialized")
    
    async def forward(self):
        """
        Main validation loop implementing your flow:
        
        1. Get user submissions (simulated)
        2. Send to miners for processing
        3. Compare results with metadata
        4. Score miners and update weights
        """
        bt.logging.info("🔄 Starting validation round...")
        
        # Step 1: Get user submissions (simulate app uploads)
        user_submissions = await self._get_user_submissions()
        
        if not user_submissions:
            bt.logging.info("No pending user submissions")
            await asyncio.sleep(12)
            return
        
        bt.logging.info(f"📥 Processing {len(user_submissions)} user submissions")
        
        # Step 2: Send submissions to miners
        miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.subtensor.n.item()))
        
        if not miner_uids:
            bt.logging.warning("No miners available")
            return
        
        # Create processing request
        request = MediaProcessingRequest(
            user_submissions=user_submissions,
            task_id=f"VAL_{int(time.time())}",
            processing_deadline=datetime.now() + timedelta(minutes=5),
            requirements={"duplicate_detection": True, "visual_analysis": True}
        )
        
        # Step 3: Query miners
        bt.logging.info(f"🔗 Querying {len(miner_uids)} miners...")
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=request,
            deserialize=False,
            timeout=self.config.neuron.timeout
        )
        
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
                "location": {"lat": 27.7172, "lng": 85.3240},  # Kathmandu
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
            submission_id = f"USR_{int(time.time())}_{i:03d}"
            
            # Create media files if incident has media
            media_files = []
            if incident["has_media"]:
                media_id = f"MED_{uuid.uuid4().hex[:12]}"
                media = MediaUpload(
                    media_id=media_id,
                    media_type="image",
                    media_url=f"https://storage.luminar.network/uploads/{media_id}.jpg",
                    media_hash=hashlib.sha256(f"media_{media_id}".encode()).hexdigest(),
                    content_description=incident["text"],
                    file_size=2048576,  # 2MB
                    mime_type="image/jpeg",
                    upload_timestamp=datetime.now()
                )
                media_files.append(media)
            
            # Create user submission
            submission = UserSubmission(
                submission_id=submission_id,
                text_description=incident["text"],
                geotag=incident["location"],
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(10, 180)),
                submission_timestamp=datetime.now(),
                user_id=f"user_{np.random.randint(1000, 9999)}",
                media_files=media_files,
                metadata={
                    "incident_type": incident["incident_type"],
                    "confidence": np.random.uniform(0.7, 1.0),
                    "app_version": "1.2.3",
                    "device_info": "iOS 17.0"
                }
            )
            
            submissions.append(submission)
        
        return submissions
    
    async def _validate_miner_responses(self, responses: List[MediaProcessingRequest], 
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
                if not response or not hasattr(response, 'processed_events'):
                    scores.append(0.0)
                    continue
                
                # Validate each processed event against original submissions
                validation_score = await self._score_miner_response(
                    response, user_submissions, miner_uid
                )
                
                scores.append(validation_score)
                
                bt.logging.debug(f"Miner {miner_uid}: {validation_score:.3f}")
                
            except Exception as e:
                bt.logging.error(f"Validation failed for miner {miner_uid}: {e}")
                scores.append(0.0)
        
        return scores
    
    async def _score_miner_response(self, response: MediaProcessingRequest,
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
        
        if not response.processed_events:
            return 0.0
        
        total_score = 0.0
        event_count = len(response.processed_events)
        
        # Create mapping of submission IDs to submissions
        submission_map = {sub.submission_id: sub for sub in user_submissions}
        
        for event in response.processed_events:
            event_score = 0.0
            
            # Find corresponding user submission(s)
            source_submissions = []
            for source_id in event.source_submissions:
                if source_id in submission_map:
                    source_submissions.append(submission_map[source_id])
            
            if not source_submissions:
                continue  # No source submissions found
            
            # Use primary source submission for validation
            primary_submission = source_submissions[0]
            
            # 1. Timestamp Consistency (25%)
            timestamp_score = self._validate_timestamp(event, primary_submission)
            
            # 2. Geotag Accuracy (25%)
            geotag_score = self._validate_geotag(event, primary_submission)
            
            # 3. Event Type Accuracy (20%)
            event_type_score = self._validate_event_type(event, primary_submission)
            
            # 4. Text Consistency (20%)
            text_score = self._validate_text_consistency(event, primary_submission)
            
            # 5. Processing Quality (10%)
            quality_score = self._validate_processing_quality(event)
            
            # Calculate weighted score
            event_score = (
                timestamp_score * 0.25 +
                geotag_score * 0.25 +
                event_type_score * 0.20 +
                text_score * 0.20 +
                quality_score * 0.10
            )
            
            total_score += event_score
        
        # Add bonus for duplicate detection
        duplicate_bonus = self._score_duplicate_detection(response, user_submissions)
        
        # Average score across all events + duplicate bonus
        final_score = (total_score / event_count) + duplicate_bonus if event_count > 0 else 0.0
        
        return min(1.0, final_score)  # Cap at 1.0
    
    def _validate_timestamp(self, event: ProcessedEvent, submission: UserSubmission) -> float:
        """
        Validate timestamp consistency between event and submission
        
        Full score if within tolerance, linear decay outside
        """
        if not event.processing_timestamp or not submission.timestamp:
            return 0.5  # Neutral score if timestamps missing
        
        # Calculate time difference in minutes
        time_diff = abs((event.processing_timestamp - submission.timestamp).total_seconds()) / 60
        tolerance = self.validation_thresholds["timestamp_tolerance_minutes"]
        
        if time_diff <= tolerance:
            return 1.0
        elif time_diff <= tolerance * 3:  # Grace period
            return 1.0 - ((time_diff - tolerance) / (tolerance * 2))
        else:
            return 0.0
    
    def _validate_geotag(self, event: ProcessedEvent, submission: UserSubmission) -> float:
        """
        Validate geotag accuracy between event and submission
        """
        try:
            from geopy.distance import geodesic
            
            # Extract coordinates from event entities or use submission coords
            event_coords = self._extract_coordinates_from_event(event, submission)
            submission_coords = (submission.geotag["lat"], submission.geotag["lng"])
            
            # Calculate distance
            distance_meters = geodesic(event_coords, submission_coords).meters
            tolerance = self.validation_thresholds["geotag_tolerance_meters"]
            
            if distance_meters <= tolerance:
                return 1.0
            elif distance_meters <= tolerance * 3:  # Grace period
                return 1.0 - ((distance_meters - tolerance) / (tolerance * 2))
            else:
                return 0.0
                
        except Exception as e:
            bt.logging.error(f"Geotag validation failed: {e}")
            return 0.5
    
    def _extract_coordinates_from_event(self, event: ProcessedEvent, submission: UserSubmission) -> Tuple[float, float]:
        """Extract coordinates from event or fall back to submission coords"""
        # For now, use submission coordinates
        # TODO: Implement coordinate extraction from event text/entities
        return (submission.geotag["lat"], submission.geotag["lng"])
    
    def _validate_event_type(self, event: ProcessedEvent, submission: UserSubmission) -> float:
        """
        Validate event type classification accuracy
        """
        predicted_type = event.event_type.lower()
        
        # Get expected type from submission metadata
        expected_type = submission.metadata.get("incident_type", "").lower()
        
        if not expected_type:
            return 0.7  # Neutral score if no expected type
        
        # Direct match
        if predicted_type == expected_type:
            return 1.0
        
        # Check for similar types
        type_similarities = {
            "accident": ["crash", "collision", "incident"],
            "theft": ["robbery", "stealing", "burglary"],
            "assault": ["fight", "violence", "attack"],
            "suspicious": ["unusual", "concerning", "strange"]
        }
        
        for main_type, similar_types in type_similarities.items():
            if expected_type == main_type and predicted_type in similar_types:
                return 0.8
            if predicted_type == main_type and expected_type in similar_types:
                return 0.8
        
        return 0.3  # Low score for mismatch
    
    def _validate_text_consistency(self, event: ProcessedEvent, submission: UserSubmission) -> float:
        """
        Validate consistency between generated event summary and original text
        """
        generated = event.generated_summary.lower()
        original = submission.text_description.lower()
        
        # Simple word overlap calculation
        generated_words = set(generated.split())
        original_words = set(original.split())
        
        # Calculate Jaccard similarity
        intersection = len(generated_words.intersection(original_words))
        union = len(generated_words.union(original_words))
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Ensure minimum threshold
        min_threshold = self.validation_thresholds["text_consistency_min"]
        if similarity >= min_threshold:
            return similarity
        else:
            return similarity * 0.5  # Penalty for below threshold
    
    def _validate_processing_quality(self, event: ProcessedEvent) -> float:
        """
        Validate overall processing quality of the event
        """
        quality_score = 0.0
        
        # Check confidence score
        if event.confidence_score >= 0.8:
            quality_score += 0.4
        elif event.confidence_score >= 0.6:
            quality_score += 0.2
        
        # Check summary quality (length and structure)
        summary_words = len(event.generated_summary.split())
        if 5 <= summary_words <= 20:  # Reasonable length
            quality_score += 0.3
        
        # Check if entities were extracted
        if event.extracted_entities and len(event.extracted_entities) > 0:
            quality_score += 0.3
        
        return min(1.0, quality_score)
    
    def _score_duplicate_detection(self, response: MediaProcessingRequest, 
                                 user_submissions: List[UserSubmission]) -> float:
        """
        Score the accuracy of duplicate detection
        
        Returns bonus points for correct duplicate detection
        """
        if not response.duplicate_flags:
            return 0.0  # No duplicates detected
        
        # For synthetic data, we don't have real duplicates
        # In production, this would check against known duplicates
        
        # Simple heuristic: bonus for detecting potential duplicates
        duplicate_count = len(response.duplicate_flags)
        submission_count = len(user_submissions)
        
        # Reasonable duplicate rate (5-15%)
        duplicate_rate = duplicate_count / submission_count if submission_count > 0 else 0
        
        if 0.05 <= duplicate_rate <= 0.15:
            return 0.1  # Small bonus for reasonable duplicate detection
        else:
            return 0.0
    
    async def _update_weights(self, scores: List[float], miner_uids: List[int]):
        """Update miner weights based on validation scores"""
        
        if len(scores) != len(miner_uids):
            bt.logging.error(f"Score/UID mismatch: {len(scores)} scores, {len(miner_uids)} UIDs")
            return
        
        # Convert scores to rewards using Luminar reward function
        rewards = torch.tensor(scores, dtype=torch.float32)
        
        bt.logging.info(f"📊 Miner scores: avg={rewards.mean():.3f}, std={rewards.std():.3f}")
        
        # Set weights on chain
        (
            processed_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=miner_uids,
            weights=rewards,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        
        # Update weights
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_uids,
            weights=processed_weights,
            wait_for_finalization=False,
        )
        
        if result:
            bt.logging.info(f"✅ Weights updated successfully")
        else:
            bt.logging.error(f"❌ Weight update failed: {msg}")
    
    async def _store_validation_results(self, submissions: List[UserSubmission],
                                      responses: List[MediaProcessingRequest],
                                      scores: List[float]):
        """Store validation results for analysis and debugging"""
        
        # Update statistics
        self.validation_stats["total_submissions"] += len(submissions)
        self.validation_stats["successful_validations"] += sum(1 for s in scores if s > 0.5)
        
        # Count metadata mismatches
        for response in responses:
            if response and hasattr(response, 'duplicate_flags'):
                self.validation_stats["duplicate_flags"] += len(response.duplicate_flags or [])
        
        # Log summary
        success_rate = (self.validation_stats["successful_validations"] / 
                       max(1, self.validation_stats["total_submissions"])) * 100
        
        bt.logging.info(f"📈 Validation stats: {success_rate:.1f}% success rate, "
                       f"{self.validation_stats['duplicate_flags']} duplicates detected")
        
        # TODO: Store detailed results in database for analysis
        # await self._store_to_database(submissions, responses, scores)

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with LuminarMediaValidator() as validator:
        while True:
            bt.logging.info("🎯 Luminar Media Validator is running...")
            time.sleep(12)
