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
import typing
import hashlib
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import bittensor as bt

# Luminar Subnet imports
import template
from template.base.miner import BaseMinerNeuron
from template.protocol import DataProcessingRequest, CrimeEvent, RawIncidentReport


class LuminarMiner(BaseMinerNeuron):
    """
    Luminar Subnet Miner: Processes raw incident reports into structured crime events.
    
    Core responsibilities:
    1. Receive raw incident reports from validators
    2. Apply AI models for semantic clustering and multimodal processing  
    3. Generate structured CrimeEvent objects with summaries and metadata
    4. Ensure data quality and authenticity verification
    """

    def __init__(self, config=None):
        super(LuminarMiner, self).__init__(config=config)
        
        # Initialize AI models and processing components
        self.setup_ai_models()
        self.setup_database_connection()
        
        # Performance metrics
        self.processed_reports_count = 0
        self.clustering_accuracy_scores = []
        self.processing_times = []
        
        bt.logging.info("Luminar Miner initialized successfully")

    def setup_ai_models(self):
        """Initialize AI models for text processing and clustering."""
        try:
            # TODO: Initialize actual AI models (CLIP, semantic clustering, etc.)
            # For now, using placeholder implementations
            self.text_processor = None  # Will be actual NLP model
            self.image_processor = None  # Will be actual CLIP model  
            self.clustering_model = None  # Will be actual clustering algorithm
            bt.logging.info("AI models initialized (placeholder mode)")
        except Exception as e:
            bt.logging.error(f"Failed to initialize AI models: {e}")

    def setup_database_connection(self):
        """Setup connection to PostgreSQL database for state management."""
        try:
            # TODO: Setup actual PostgreSQL connection
            self.db_connection = None  # Placeholder
            bt.logging.info("Database connection established (placeholder mode)")
        except Exception as e:
            bt.logging.error(f"Failed to setup database connection: {e}")

    async def forward(
        self, synapse: DataProcessingRequest
    ) -> DataProcessingRequest:
        """
        Process raw incident reports into structured crime events.
        
        This is the core miner function that receives raw reports from validators
        and applies AI clustering to create structured crime events.
        
        Args:
            synapse: DataProcessingRequest containing raw incident reports
            
        Returns:
            DataProcessingRequest with processed crime events
        """
        start_time = time.time()
        
        try:
            bt.logging.info(f"Processing {len(synapse.raw_reports)} raw incident reports")
            
            # Validate input reports
            validated_reports = self.validate_reports(synapse.raw_reports)
            
            # Apply semantic clustering
            clustered_groups = await self.cluster_reports(validated_reports)
            
            # Generate structured crime events
            crime_events = await self.generate_crime_events(clustered_groups)
            
            # Quality assurance and verification
            verified_events = self.verify_events(crime_events)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self.processed_reports_count += len(synapse.raw_reports)
            self.processing_times.append(processing_time)
            
            # Set response data
            synapse.processed_events = verified_events
            synapse.processing_time = processing_time
            synapse.miner_metadata = {
                "miner_id": self.wallet.hotkey.ss58_address,
                "processed_count": len(verified_events),
                "clustering_confidence": self.calculate_clustering_confidence(verified_events),
                "timestamp": datetime.now().isoformat()
            }
            
            bt.logging.info(f"Successfully processed {len(verified_events)} crime events in {processing_time:.2f}s")
            
        except Exception as e:
            bt.logging.error(f"Error processing reports: {e}")
            synapse.processed_events = []
            synapse.processing_time = time.time() - start_time
            synapse.miner_metadata = {"error": str(e)}
        
        return synapse

    def validate_reports(self, reports: List[RawIncidentReport]) -> List[RawIncidentReport]:
        """Validate and clean raw incident reports."""
        validated = []
        
        for report in reports:
            # Basic validation checks
            if not report.text_description or len(report.text_description.strip()) < 10:
                bt.logging.warning(f"Skipping report {report.report_id}: insufficient description")
                continue
                
            if not report.geotag or 'lat' not in report.geotag or 'lng' not in report.geotag:
                bt.logging.warning(f"Skipping report {report.report_id}: invalid geotag")
                continue
                
            # Verify timestamp is reasonable (not too old or in future)
            if report.timestamp > datetime.now() + timedelta(hours=1):
                bt.logging.warning(f"Skipping report {report.report_id}: future timestamp")
                continue
                
            validated.append(report)
            
        bt.logging.info(f"Validated {len(validated)}/{len(reports)} reports")
        return validated

    async def cluster_reports(self, reports: List[RawIncidentReport]) -> List[List[RawIncidentReport]]:
        """Apply semantic clustering to group similar incident reports."""
        if not reports:
            return []
            
        # Placeholder clustering logic - In production, this would use:
        # 1. Text embeddings (e.g., BERT, RoBERTa)
        # 2. Geospatial clustering (DBSCAN with geographic distance)
        # 3. Temporal clustering (reports within time windows)
        # 4. Multimodal embeddings for images/videos (CLIP)
        
        clusters = []
        
        # Simple geographic clustering for demo (group by proximity)
        remaining_reports = reports.copy()
        
        while remaining_reports:
            base_report = remaining_reports.pop(0)
            cluster = [base_report]
            
            # Find nearby reports (within ~1km and 24 hours)
            to_remove = []
            for i, report in enumerate(remaining_reports):
                if self.are_reports_similar(base_report, report):
                    cluster.append(report)
                    to_remove.append(i)
            
            # Remove clustered reports
            for i in reversed(to_remove):
                remaining_reports.pop(i)
                
            clusters.append(cluster)
            
        bt.logging.info(f"Created {len(clusters)} clusters from {len(reports)} reports")
        return clusters

    def are_reports_similar(self, report1: RawIncidentReport, report2: RawIncidentReport) -> bool:
        """Determine if two reports should be clustered together."""
        # Geographic proximity (simple Euclidean distance for demo)
        lat_diff = abs(report1.geotag['lat'] - report2.geotag['lat'])
        lng_diff = abs(report1.geotag['lng'] - report2.geotag['lng'])
        geo_distance = (lat_diff**2 + lng_diff**2)**0.5
        
        # Time proximity
        time_diff = abs((report1.timestamp - report2.timestamp).total_seconds())
        
        # Simple similarity thresholds (would be more sophisticated in production)
        return geo_distance < 0.01 and time_diff < 86400  # ~1km and 24 hours

    async def generate_crime_events(self, clusters: List[List[RawIncidentReport]]) -> List[CrimeEvent]:
        """Generate structured crime events from clustered reports."""
        crime_events = []
        
        for cluster in clusters:
            if not cluster:
                continue
                
            try:
                # Generate event ID
                event_id = str(uuid.uuid4())
                
                # Determine event type and severity
                event_type = self.classify_event_type(cluster)
                severity = self.calculate_severity(cluster)
                
                # Generate summary
                summary = self.generate_summary(cluster, event_type)
                
                # Calculate location (centroid of reports)
                location = self.calculate_location(cluster)
                
                # Determine time range
                timestamps = [report.timestamp for report in cluster]
                time_range = {
                    "start": min(timestamps),
                    "end": max(timestamps)
                }
                
                # Create crime event
                crime_event = CrimeEvent(
                    event_id=event_id,
                    summary_tag=summary,
                    event_type=event_type,
                    location=location,
                    timestamp_range=time_range,
                    confidence_score=self.calculate_confidence_score(cluster),
                    source_reports=[report.report_id for report in cluster],
                    clustered_summary=self.generate_detailed_summary(cluster),
                    severity_level=severity,
                    verified=False,  # Will be verified by validators
                    verification_metadata={}
                )
                
                crime_events.append(crime_event)
                
            except Exception as e:
                bt.logging.error(f"Error generating crime event from cluster: {e}")
                continue
                
        return crime_events

    def classify_event_type(self, cluster: List[RawIncidentReport]) -> str:
        """Classify the type of crime event based on report content."""
        # Placeholder implementation - would use NLP classification in production
        descriptions = [report.text_description.lower() for report in cluster]
        combined_text = " ".join(descriptions)
        
        # Simple keyword-based classification
        crime_keywords = {
            "theft": ["theft", "steal", "stolen", "robbery", "rob"],
            "vandalism": ["vandalism", "graffiti", "damage", "break", "broken"],
            "assault": ["assault", "fight", "attack", "violence", "hit"],
            "fraud": ["fraud", "scam", "cheat", "fake"],
            "burglary": ["burglary", "break-in", "breaking", "entry"],
            "traffic": ["accident", "crash", "traffic", "collision"],
            "drug": ["drug", "substance", "overdose", "dealing"],
            "noise": ["noise", "loud", "disturbance", "party"]
        }
        
        for crime_type, keywords in crime_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return crime_type
                
        return "other"

    def calculate_severity(self, cluster: List[RawIncidentReport]) -> int:
        """Calculate severity level (1-5) based on report content and cluster size."""
        # Placeholder logic - would use more sophisticated analysis in production
        severity_keywords = {
            5: ["murder", "death", "killed", "weapon", "gun", "knife"],
            4: ["assault", "violence", "injured", "hospital", "blood"],
            3: ["theft", "robbery", "burglary", "stolen", "loss"],
            2: ["vandalism", "damage", "graffiti", "noise"],
            1: ["suspicious", "minor", "disturbance"]
        }
        
        descriptions = [report.text_description.lower() for report in cluster]
        combined_text = " ".join(descriptions)
        
        for severity, keywords in severity_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return severity
                
        # Default severity based on cluster size
        return min(len(cluster), 3)

    def generate_summary(self, cluster: List[RawIncidentReport], event_type: str) -> str:
        """Generate a concise summary tag for the crime event."""
        if not cluster:
            return "Unknown incident"
            
        # Get primary location
        location = self.calculate_location(cluster)
        city = location.get("city", "Unknown location")
        
        # Get time
        primary_time = cluster[0].timestamp
        date_str = primary_time.strftime("%d %B")
        
        # Generate summary
        count = len(cluster)
        if count == 1:
            return f"{event_type.title()} in {city} on {date_str}"
        else:
            return f"Multiple {event_type} incidents in {city} on {date_str}"

    def calculate_location(self, cluster: List[RawIncidentReport]) -> Dict[str, Any]:
        """Calculate the representative location for a cluster of reports."""
        if not cluster:
            return {}
            
        # Calculate centroid
        lats = [report.geotag['lat'] for report in cluster]
        lngs = [report.geotag['lng'] for report in cluster]
        
        centroid_lat = sum(lats) / len(lats)
        centroid_lng = sum(lngs) / len(lngs)
        
        # TODO: Reverse geocoding to get city/address
        # For now, using placeholder
        return {
            "city": "Unknown City",  # Would be reverse geocoded
            "coordinates": {"lat": centroid_lat, "lng": centroid_lng},
            "address": "Unknown Address"  # Would be reverse geocoded
        }

    def calculate_confidence_score(self, cluster: List[RawIncidentReport]) -> float:
        """Calculate confidence score for the clustering and event generation."""
        if not cluster:
            return 0.0
            
        # Factors affecting confidence:
        # 1. Number of reports (more reports = higher confidence)
        # 2. Geographic consistency
        # 3. Temporal consistency
        # 4. Media evidence availability
        
        base_confidence = min(len(cluster) * 0.2, 0.8)  # Max 0.8 from count
        
        # Boost for media evidence
        media_count = sum(1 for report in cluster if report.media_urls)
        media_boost = min(media_count * 0.1, 0.2)
        
        return min(base_confidence + media_boost, 1.0)

    def generate_detailed_summary(self, cluster: List[RawIncidentReport]) -> str:
        """Generate a detailed summary of the clustered reports."""
        if not cluster:
            return "No reports available"
            
        # Combine key information from all reports
        descriptions = [report.text_description for report in cluster]
        
        # Simple summary for now - would use LLM summarization in production
        summary = f"Event based on {len(cluster)} report(s). "
        
        if len(descriptions) == 1:
            summary += descriptions[0]
        else:
            summary += f"Multiple reports describe similar incidents. Key details: {descriptions[0][:200]}..."
            
        return summary

    def verify_events(self, events: List[CrimeEvent]) -> List[CrimeEvent]:
        """Perform basic verification and quality checks on generated events."""
        verified_events = []
        
        for event in events:
            # Basic validation checks
            if not event.summary_tag or not event.event_type:
                bt.logging.warning(f"Skipping invalid event {event.event_id}")
                continue
                
            if event.confidence_score < 0.1:
                bt.logging.warning(f"Skipping low-confidence event {event.event_id}")
                continue
                
            verified_events.append(event)
            
        return verified_events

    def calculate_clustering_confidence(self, events: List[CrimeEvent]) -> float:
        """Calculate overall clustering confidence for this processing batch."""
        if not events:
            return 0.0
            
        scores = [event.confidence_score for event in events]
        return sum(scores) / len(scores)

    async def blacklist(
        self, synapse: DataProcessingRequest
    ) -> typing.Tuple[bool, str]:
        """
        Enhanced blacklisting for Luminar subnet with additional crime data specific checks.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # Check if hotkey is registered
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unregistered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        # Force validator permit for data processing requests
        if not self.metagraph.validator_permit[uid]:
            bt.logging.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        # Check stake requirements (higher for crime data processing)
        min_stake = getattr(self.config.blacklist, 'min_stake_threshold', 1000)
        if self.metagraph.S[uid] < min_stake:
            bt.logging.warning(f"Blacklisting low-stake validator {synapse.dendrite.hotkey}")
            return True, f"Insufficient stake: {self.metagraph.S[uid]} < {min_stake}"

        # Additional checks for data processing requests
        if hasattr(synapse, 'raw_reports') and synapse.raw_reports:
            # Check for reasonable batch size
            if len(synapse.raw_reports) > 1000:  # Configurable limit
                bt.logging.warning(f"Blacklisting oversized batch: {len(synapse.raw_reports)} reports")
                return True, "Batch size too large"

        bt.logging.trace(f"Accepting request from validator {synapse.dendrite.hotkey}")
        return False, "Validator approved"

    async def priority(self, synapse: DataProcessingRequest) -> float:
        """
        Priority function for Luminar subnet processing requests.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        base_priority = float(self.metagraph.S[uid])
        
        # Boost priority for urgent/recent reports
        if hasattr(synapse, 'raw_reports') and synapse.raw_reports:
            now = datetime.now()
            recent_reports = sum(1 for report in synapse.raw_reports 
                               if (now - report.timestamp).total_seconds() < 3600)  # 1 hour
            urgency_boost = recent_reports * 0.1
            base_priority += urgency_boost
            
        # Boost priority based on deadline proximity
        if hasattr(synapse, 'deadline') and synapse.deadline:
            time_remaining = (synapse.deadline - datetime.now()).total_seconds()
            if time_remaining < 300:  # 5 minutes
                base_priority *= 2.0
            elif time_remaining < 900:  # 15 minutes
                base_priority *= 1.5

        bt.logging.trace(f"Priority for {synapse.dendrite.hotkey}: {base_priority}")
        return base_priority


# This is the main function, which runs the Luminar miner.
if __name__ == "__main__":
    with LuminarMiner() as miner:
        while True:
            bt.logging.info(f"Luminar Miner running... {time.time()}")
            time.sleep(5)
