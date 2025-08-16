# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Luminar Subnet Contributors
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
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
import bittensor as bt

# Luminar Subnet imports
import template
from template.base.validator import BaseValidatorNeuron
from template.protocol import DataProcessingRequest, CrimeEvent, RawIncidentReport, VerificationRequest
from template.utils.uids import get_random_uids


class LuminarValidator(BaseValidatorNeuron):
    """
    Luminar Subnet Validator: Verifies and scores crime data processing by miners.
    
    Core responsibilities:
    1. Request clustered crime events from miners
    2. Verify data integrity, clustering accuracy, and authenticity  
    3. Score miners based on quality metrics
    4. Set weights to reward high-performing miners
    5. Manage the validated dataset of crime events
    """

    def __init__(self, config=None):
        super(LuminarValidator, self).__init__(config=config)
        
        # Initialize verification systems
        self.setup_verification_systems()
        self.setup_database_connection()
        
        # Performance tracking
        self.validation_history = []
        self.miner_performance = {}
        self.verified_events_count = 0
        
        # Load any previous state
        self.load_validation_state()
        
        bt.logging.info("Luminar Validator initialized successfully")

    def setup_verification_systems(self):
        """Initialize systems for verifying crime event authenticity and accuracy."""
        try:
            # TODO: Initialize actual verification models and systems
            self.authenticity_checker = None  # Media authenticity verification
            self.geo_verifier = None  # Geographic consistency checker
            self.temporal_verifier = None  # Temporal consistency checker
            self.clustering_evaluator = None  # Semantic clustering quality evaluator
            bt.logging.info("Verification systems initialized (placeholder mode)")
        except Exception as e:
            bt.logging.error(f"Failed to initialize verification systems: {e}")

    def setup_database_connection(self):
        """Setup connection to PostgreSQL database for state management."""
        try:
            # TODO: Setup actual PostgreSQL connection for verified events storage
            self.db_connection = None  # Placeholder
            bt.logging.info("Database connection established (placeholder mode)")
        except Exception as e:
            bt.logging.error(f"Failed to setup database connection: {e}")

    def load_validation_state(self):
        """Load previous validation state and miner performance data."""
        try:
            # TODO: Load from persistent storage
            bt.logging.info("Validation state loaded")
        except Exception as e:
            bt.logging.warning(f"Could not load validation state: {e}")

    async def forward(self):
        """
        Main validator forward pass for the Luminar subnet.
        
        Process:
        1. Generate or fetch raw incident reports
        2. Send processing requests to miners
        3. Verify and score miner responses
        4. Update weights based on performance
        """
        try:
            bt.logging.info("Starting validator forward pass")
            
            # Step 1: Prepare batch of raw incident reports for processing
            raw_reports = await self.prepare_incident_reports()
            
            if not raw_reports:
                bt.logging.warning("No incident reports available for processing")
                await asyncio.sleep(10)
                return
            
            # Step 2: Select miners for processing
            miner_uids = get_random_uids(self, k=min(len(raw_reports) // 5 + 1, self.config.neuron.sample_size))
            
            if not miner_uids.size:
                bt.logging.warning("No available miners found")
                await asyncio.sleep(10)
                return
            
            bt.logging.info(f"Requesting processing from {len(miner_uids)} miners for {len(raw_reports)} reports")
            
            # Step 3: Send processing requests to miners
            task_id = str(uuid.uuid4())
            deadline = datetime.now() + timedelta(minutes=5)
            
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=DataProcessingRequest(
                    raw_reports=raw_reports,
                    task_id=task_id,
                    deadline=deadline,
                    processing_requirements={
                        "min_confidence": 0.3,
                        "max_events_per_report": 2
                    }
                ),
                deserialize=True,
                timeout=300  # 5 minute timeout
            )
            
            bt.logging.info(f"Received {len(responses)} miner responses")
            
            # Step 4: Verify and score miner outputs
            scores = await self.verify_and_score_responses(raw_reports, responses, miner_uids)
            
            # Step 5: Update miner weights
            self.update_scores(scores, miner_uids)
            
            # Step 6: Store verified events
            await self.store_verified_events(responses, scores)
            
            bt.logging.info(f"Validator forward pass completed. Processed {len(raw_reports)} reports.")
            
        except Exception as e:
            bt.logging.error(f"Error in validator forward pass: {e}")
        
        await asyncio.sleep(30)  # Wait before next iteration

    async def prepare_incident_reports(self) -> List[RawIncidentReport]:
        """
        Prepare a batch of raw incident reports for processing.
        
        In production, this would:
        1. Fetch from user submission API
        2. Pull from pending queue
        3. Generate synthetic data for testing
        """
        # For demo purposes, generate synthetic incident reports
        reports = []
        
        # Sample locations (major cities)
        locations = [
            {"lat": 28.6139, "lng": 77.2090, "city": "Delhi"},
            {"lat": 19.0760, "lng": 72.8777, "city": "Mumbai"}, 
            {"lat": 13.0827, "lng": 80.2707, "city": "Chennai"},
            {"lat": 22.5726, "lng": 88.3639, "city": "Kolkata"},
            {"lat": 12.9716, "lng": 77.5946, "city": "Bangalore"}
        ]
        
        # Sample incident types and descriptions
        incident_templates = [
            {
                "type": "theft",
                "descriptions": [
                    "Phone was stolen from my pocket while walking",
                    "Bicycle theft from parking area",
                    "Purse snatching incident on main road"
                ]
            },
            {
                "type": "vandalism", 
                "descriptions": [
                    "Graffiti spray painted on building wall",
                    "Public property damaged in park",
                    "Car windows broken in parking lot"
                ]
            },
            {
                "type": "noise",
                "descriptions": [
                    "Loud music disturbance at night",
                    "Construction noise during quiet hours",
                    "Party causing noise complaints"
                ]
            }
        ]
        
        # Generate 10-20 reports
        num_reports = random.randint(10, 20)
        
        for i in range(num_reports):
            location = random.choice(locations)
            incident = random.choice(incident_templates)
            
            # Add some geographic clustering by creating nearby incidents
            if i > 0 and random.random() < 0.3:  # 30% chance of clustering
                prev_report = reports[-1]
                location = {
                    "lat": prev_report.geotag["lat"] + random.uniform(-0.01, 0.01),
                    "lng": prev_report.geotag["lng"] + random.uniform(-0.01, 0.01), 
                    "city": prev_report.geotag.get("city", location["city"])
                }
            
            report = RawIncidentReport(
                report_id=f"report_{uuid.uuid4().hex[:8]}",
                text_description=random.choice(incident["descriptions"]),
                geotag={"lat": location["lat"], "lng": location["lng"]},
                timestamp=datetime.now() - timedelta(minutes=random.randint(10, 1440)),  # Last 24 hours
                media_urls=[f"https://storage.luminar.ai/media/{uuid.uuid4().hex}.jpg"] if random.random() < 0.3 else [],
                media_hashes=[hashlib.sha256(f"media_{i}".encode()).hexdigest()] if random.random() < 0.3 else [],
                user_id=f"user_{random.randint(1000, 9999)}",
                metadata={"incident_type": incident["type"], "location_name": location.get("city")}
            )
            
            reports.append(report)
        
        bt.logging.info(f"Prepared {len(reports)} synthetic incident reports")
        return reports

    async def verify_and_score_responses(
        self, 
        raw_reports: List[RawIncidentReport], 
        responses: List[DataProcessingRequest],
        miner_uids: np.ndarray
    ) -> np.ndarray:
        """
        Verify miner responses and calculate quality scores.
        
        Scoring criteria:
        1. Data Integrity (30%): Timestamp/geotag consistency, media verification
        2. Clustering Accuracy (40%): Semantic relevance, appropriate grouping
        3. Novelty & Redundancy (20%): New events vs duplicates  
        4. Processing Quality (10%): Summary quality, metadata completeness
        """
        scores = np.zeros(len(miner_uids))
        
        for i, (response, uid) in enumerate(zip(responses, miner_uids)):
            try:
                if not response.processed_events:
                    scores[i] = 0.0
                    bt.logging.warning(f"Miner {uid} returned no processed events")
                    continue
                
                # Calculate component scores
                integrity_score = await self.verify_data_integrity(raw_reports, response.processed_events)
                clustering_score = await self.evaluate_clustering_accuracy(raw_reports, response.processed_events)
                novelty_score = await self.check_novelty_and_redundancy(response.processed_events)
                quality_score = await self.assess_processing_quality(response.processed_events, response)
                
                # Weight and combine scores
                final_score = (
                    integrity_score * 0.30 +
                    clustering_score * 0.40 + 
                    novelty_score * 0.20 +
                    quality_score * 0.10
                )
                
                scores[i] = final_score
                
                # Update miner performance tracking
                self.update_miner_performance(uid, {
                    "integrity": integrity_score,
                    "clustering": clustering_score, 
                    "novelty": novelty_score,
                    "quality": quality_score,
                    "final": final_score,
                    "timestamp": datetime.now()
                })
                
                bt.logging.info(f"Miner {uid} scored {final_score:.3f} "
                              f"(I:{integrity_score:.2f} C:{clustering_score:.2f} "
                              f"N:{novelty_score:.2f} Q:{quality_score:.2f})")
                
            except Exception as e:
                bt.logging.error(f"Error scoring miner {uid}: {e}")
                scores[i] = 0.0
        
        return scores

    async def verify_data_integrity(
        self, 
        raw_reports: List[RawIncidentReport], 
        crime_events: List[CrimeEvent]
    ) -> float:
        """Verify data integrity: timestamps, geotags, and media authenticity."""
        if not crime_events:
            return 0.0
        
        integrity_checks = []
        
        for event in crime_events:
            score = 0.0
            checks = 0
            
            # Check if source reports exist and are valid
            valid_sources = 0
            for report_id in event.source_reports:
                if any(r.report_id == report_id for r in raw_reports):
                    valid_sources += 1
            
            if event.source_reports:
                score += (valid_sources / len(event.source_reports)) * 0.4
                checks += 1
            
            # Check geographic consistency
            source_reports = [r for r in raw_reports if r.report_id in event.source_reports]
            if source_reports and event.location.get("coordinates"):
                geo_consistency = self.check_geographic_consistency(source_reports, event.location["coordinates"])
                score += geo_consistency * 0.3
                checks += 1
            
            # Check temporal consistency
            if source_reports and event.timestamp_range:
                temporal_consistency = self.check_temporal_consistency(source_reports, event.timestamp_range)
                score += temporal_consistency * 0.3
                checks += 1
            
            if checks > 0:
                integrity_checks.append(score / checks)
        
        return sum(integrity_checks) / len(integrity_checks) if integrity_checks else 0.0

    def check_geographic_consistency(self, reports: List[RawIncidentReport], event_coords: Dict) -> float:
        """Check if event location is consistent with source report locations."""
        if not reports or not event_coords:
            return 0.0
        
        event_lat, event_lng = event_coords["lat"], event_coords["lng"]
        distances = []
        
        for report in reports:
            lat_diff = abs(report.geotag["lat"] - event_lat)
            lng_diff = abs(report.geotag["lng"] - event_lng)
            distance = (lat_diff**2 + lng_diff**2)**0.5
            distances.append(distance)
        
        # Good if all reports are within reasonable distance (0.05 degrees ~ 5km)
        avg_distance = sum(distances) / len(distances)
        return max(0.0, 1.0 - (avg_distance / 0.05))

    def check_temporal_consistency(self, reports: List[RawIncidentReport], time_range: Dict) -> float:
        """Check if event time range is consistent with source reports."""
        if not reports or not time_range:
            return 0.0
        
        report_times = [r.timestamp for r in reports]
        min_time, max_time = min(report_times), max(report_times)
        
        # Check if event time range encompasses all reports
        range_start = time_range.get("start", min_time)
        range_end = time_range.get("end", max_time)
        
        if range_start <= min_time and range_end >= max_time:
            # Check if range is reasonable (not too wide)
            range_duration = (range_end - range_start).total_seconds()
            reasonable_duration = 86400  # 24 hours
            
            if range_duration <= reasonable_duration:
                return 1.0
            else:
                return max(0.0, 1.0 - ((range_duration - reasonable_duration) / reasonable_duration))
        
        return 0.5  # Partial credit if partially overlapping

    async def evaluate_clustering_accuracy(
        self,
        raw_reports: List[RawIncidentReport],
        crime_events: List[CrimeEvent]
    ) -> float:
        """Evaluate the accuracy of semantic clustering performed by miner."""
        if not crime_events:
            return 0.0
        
        clustering_scores = []
        
        for event in crime_events:
            score = 0.0
            checks = 0
            
            # Get source reports for this event
            source_reports = [r for r in raw_reports if r.report_id in event.source_reports]
            
            if not source_reports:
                clustering_scores.append(0.0)
                continue
            
            # Check semantic similarity of clustered reports
            if len(source_reports) > 1:
                similarity_score = self.calculate_semantic_similarity(source_reports)
                score += similarity_score * 0.4
                checks += 1
            else:
                score += 0.4  # Single reports are inherently consistent
                checks += 1
            
            # Check if event type classification is appropriate
            type_accuracy = self.verify_event_type_classification(source_reports, event.event_type)
            score += type_accuracy * 0.3
            checks += 1
            
            # Check clustering completeness (are similar reports properly grouped?)
            completeness = self.check_clustering_completeness(raw_reports, event)
            score += completeness * 0.3
            checks += 1
            
            if checks > 0:
                clustering_scores.append(score / checks)
        
        return sum(clustering_scores) / len(clustering_scores) if clustering_scores else 0.0

    def calculate_semantic_similarity(self, reports: List[RawIncidentReport]) -> float:
        """Calculate semantic similarity between clustered reports."""
        if len(reports) <= 1:
            return 1.0
        
        # Placeholder implementation - would use actual NLP similarity in production
        descriptions = [r.text_description.lower() for r in reports]
        
        # Simple keyword overlap method
        all_words = set()
        word_sets = []
        
        for desc in descriptions:
            words = set(desc.split())
            word_sets.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0.0
        
        # Calculate average pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

    def verify_event_type_classification(self, reports: List[RawIncidentReport], event_type: str) -> float:
        """Verify if the event type classification is appropriate for the reports."""
        if not reports:
            return 0.0
        
        # Simple keyword-based verification
        crime_keywords = {
            "theft": ["theft", "steal", "stolen", "robbery", "rob"],
            "vandalism": ["vandalism", "graffiti", "damage", "break", "broken"],
            "assault": ["assault", "fight", "attack", "violence", "hit"],
            "fraud": ["fraud", "scam", "cheat", "fake"],
            "burglary": ["burglary", "break-in", "breaking", "entry"],
            "traffic": ["accident", "crash", "traffic", "collision"],
            "drug": ["drug", "substance", "overdose", "dealing"],
            "noise": ["noise", "loud", "disturbance", "party"],
            "other": []
        }
        
        expected_keywords = crime_keywords.get(event_type.lower(), [])
        if not expected_keywords:  # "other" category
            return 0.8  # Reasonable default for miscellaneous
        
        descriptions = [r.text_description.lower() for r in reports]
        combined_text = " ".join(descriptions)
        
        matches = sum(1 for keyword in expected_keywords if keyword in combined_text)
        return min(1.0, matches / max(1, len(expected_keywords) * 0.5))

    def check_clustering_completeness(self, all_reports: List[RawIncidentReport], event: CrimeEvent) -> float:
        """Check if similar reports are properly included in the cluster."""
        # This is a simplified check - in production would be more sophisticated
        source_report_ids = set(event.source_reports)
        
        # Find reports that might be similar but weren't clustered
        event_location = event.location.get("coordinates", {})
        if not event_location:
            return 0.8  # Default if no location data
        
        nearby_reports = []
        for report in all_reports:
            if report.report_id in source_report_ids:
                continue
                
            # Check if report is nearby and similar time
            lat_diff = abs(report.geotag["lat"] - event_location["lat"])
            lng_diff = abs(report.geotag["lng"] - event_location["lng"]) 
            distance = (lat_diff**2 + lng_diff**2)**0.5
            
            if distance < 0.01:  # Within ~1km
                time_diff = abs((report.timestamp - event.timestamp_range["start"]).total_seconds())
                if time_diff < 86400:  # Within 24 hours
                    nearby_reports.append(report)
        
        # If there are many similar reports not clustered, reduce score
        if len(nearby_reports) > len(source_report_ids):
            return 0.5  # Possible under-clustering
        
        return 1.0

    async def check_novelty_and_redundancy(self, crime_events: List[CrimeEvent]) -> float:
        """Check for novelty and detect redundant/duplicate events."""
        if not crime_events:
            return 0.0
        
        novelty_scores = []
        
        for event in crime_events:
            score = 1.0  # Start with full novelty
            
            # Check against recently verified events (would query database in production)
            duplicate_penalty = self.check_for_duplicates(event)
            score -= duplicate_penalty
            
            # Bonus for events with multiple confirming sources
            if len(event.source_reports) > 1:
                score += 0.1
            
            # Penalty for events with very low confidence
            if event.confidence_score < 0.3:
                score -= 0.2
            
            novelty_scores.append(max(0.0, min(1.0, score)))
        
        return sum(novelty_scores) / len(novelty_scores)

    def check_for_duplicates(self, event: CrimeEvent) -> float:
        """Check if this event is a duplicate of a recently processed one."""
        # Placeholder - would check against database of recent events
        # For now, just check within current batch
        return 0.0  # No duplicates detected

    async def assess_processing_quality(self, events: List[CrimeEvent], response: DataProcessingRequest) -> float:
        """Assess overall processing quality including summaries and metadata."""
        if not events:
            return 0.0
        
        quality_scores = []
        
        for event in events:
            score = 0.0
            checks = 0
            
            # Summary quality
            if event.summary_tag and len(event.summary_tag) > 5:
                score += 0.3
                checks += 1
            
            # Detailed summary quality
            if event.clustered_summary and len(event.clustered_summary) > 20:
                score += 0.3
                checks += 1
            
            # Metadata completeness
            required_fields = ['event_id', 'event_type', 'location', 'timestamp_range', 'confidence_score']
            complete_fields = sum(1 for field in required_fields if getattr(event, field, None) is not None)
            score += (complete_fields / len(required_fields)) * 0.2
            checks += 1
            
            # Confidence score reasonableness
            if 0.0 <= event.confidence_score <= 1.0:
                score += 0.2
                checks += 1
            
            if checks > 0:
                quality_scores.append(score / checks)
        
        # Factor in processing time (faster is better, within reason)
        time_bonus = 0.0
        if response.processing_time:
            if response.processing_time < 30:  # Very fast
                time_bonus = 0.1
            elif response.processing_time < 60:  # Reasonable
                time_bonus = 0.05
        
        base_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        return min(1.0, base_quality + time_bonus)

    def update_miner_performance(self, uid: int, performance_data: Dict):
        """Update performance tracking for a specific miner."""
        if uid not in self.miner_performance:
            self.miner_performance[uid] = []
        
        self.miner_performance[uid].append(performance_data)
        
        # Keep only last 100 entries per miner
        if len(self.miner_performance[uid]) > 100:
            self.miner_performance[uid] = self.miner_performance[uid][-100:]

    async def store_verified_events(self, responses: List[DataProcessingRequest], scores: np.ndarray):
        """Store high-quality verified events to the database."""
        stored_count = 0
        
        for response, score in zip(responses, scores):
            if score > 0.7 and response.processed_events:  # Only store high-quality events
                for event in response.processed_events:
                    try:
                        # TODO: Store to actual PostgreSQL database
                        # For now, just log and count
                        event.verified = True
                        event.verification_metadata = {
                            "validator_score": float(score),
                            "verified_at": datetime.now().isoformat(),
                            "validator_id": self.wallet.hotkey.ss58_address
                        }
                        stored_count += 1
                        
                    except Exception as e:
                        bt.logging.error(f"Error storing event {event.event_id}: {e}")
        
        self.verified_events_count += stored_count
        bt.logging.info(f"Stored {stored_count} verified crime events. Total: {self.verified_events_count}")


# This is the main function, which runs the Luminar validator.
if __name__ == "__main__":
    with LuminarValidator() as validator:
        while True:
            bt.logging.info(f"Luminar Validator running... {time.time()}")
            time.sleep(5)
