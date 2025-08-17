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

import typing
import bittensor as bt
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Luminar Subnet Protocol Definition
# This protocol defines the communication between miners and validators for crime data processing
# in the Luminar decentralized intelligence platform.

@dataclass
class MediaUpload:
    """
    Media file uploaded by user with metadata
    """
    media_id: str  # Unique ID for this upload
    media_type: str  # "image", "video", "audio"
    media_url: str  # URL to access the media
    media_hash: str  # Content hash for verification
    content_description: str  # Text description from user
    file_size: int  # Size in bytes
    mime_type: str  # e.g., "image/jpeg", "video/mp4"
    upload_timestamp: datetime

@dataclass
class UserSubmission:
    """
    Complete user submission with text, media, and metadata
    """
    submission_id: str  # Unique ID for this submission
    text_description: str  # User's text description
    geotag: Dict[str, float]  # {"lat": float, "lng": float}
    timestamp: datetime  # When incident occurred
    submission_timestamp: datetime  # When user submitted
    user_id: str  # Anonymous user identifier
    media_files: List[MediaUpload] = None  # Associated media
    metadata: Dict[str, Any] = None  # Additional metadata
    verification_status: str = "pending"  # pending, processing, verified, rejected

@dataclass
class RawIncidentReport:
    """
    A raw incident report submitted by users (legacy compatibility).
    """
    report_id: str
    text_description: str
    geotag: Dict[str, float]  # {"lat": float, "lng": float}
    timestamp: datetime
    media_urls: List[str] = None  # URLs to photos/videos
    media_hashes: List[str] = None  # Content hashes for verification
    user_id: str = None
    metadata: Dict[str, Any] = None

@dataclass
class ProcessedEvent:
    """
    Event created by miner from text and visual analysis
    """
    event_id: str
    generated_summary: str  # e.g., "truck and motorbike accident near balaju at 4pm, June 11"
    event_type: str  # accident, theft, assault, etc.
    confidence_score: float  # 0.0 to 1.0
    extracted_entities: Dict[str, Any]  # {"vehicles": ["truck", "motorbike"], "location": "balaju", "time": "4pm"}
    visual_analysis: Dict[str, Any]  # Results from image/video processing
    source_submissions: List[str]  # List of submission IDs used
    processing_timestamp: datetime
    miner_uid: int  # UID of processing miner

@dataclass 
class ValidationResult:
    """
    Result of validator comparing miner event with user metadata
    """
    validation_id: str
    event_id: str
    submission_id: str
    metadata_consistency: Dict[str, float]  # Scores for timestamp, geotag, etc.
    visual_authenticity: float  # Score for media authenticity
    duplicate_detection: bool  # Is this a duplicate?
    overall_score: float  # Final validation score
    validator_uid: int
    validation_timestamp: datetime

@dataclass
class CrimeEvent:
    """
    A structured crime event object created by miners from clustered reports.
    """
    event_id: str
    summary_tag: str  # e.g., "Vandalism in Delhi on 14 June"
    event_type: str  # e.g., "Vandalism", "Theft", "Assault"
    location: Dict[str, Any]  # {"city": str, "coordinates": {"lat": float, "lng": float}, "address": str}
    timestamp_range: Dict[str, datetime]  # {"start": datetime, "end": datetime}
    confidence_score: float  # 0.0 to 1.0
    source_reports: List[str]  # List of report_ids that were clustered
    clustered_summary: str  # AI-generated detailed summary
    severity_level: int  # 1-5 scale
    verified: bool = False
    verification_metadata: Dict[str, Any] = None

# Luminar Subnet Communication Protocols

class MediaProcessingRequest(bt.Synapse):
    """
    Validator -> Miner: Send user submissions with media for processing
    
    This implements your desired flow:
    1. Validator receives user submissions with media + metadata
    2. Validator sends to miner for processing
    3. Miner analyzes text + visuals, creates events, filters duplicates
    4. Validator compares results with original metadata
    """
    
    # Request input (filled by validator)
    user_submissions: List[UserSubmission]  # User uploads with media + metadata
    task_id: str
    processing_deadline: datetime
    requirements: Dict[str, Any] = None  # Special processing requirements
    
    # Response output (filled by miner)
    processed_events: Optional[List[ProcessedEvent]] = None
    duplicate_flags: Optional[List[str]] = None  # IDs of detected duplicates
    processing_metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def deserialize(self) -> List[ProcessedEvent]:
        """
        Deserialize the processed events from miner
        """
        return self.processed_events or []

class ValidationRequest(bt.Synapse):
    """
    Validator internal: Compare miner events with user metadata
    
    Validates:
    - Timestamp consistency
    - Geotag accuracy  
    - Visual authenticity
    - Duplicate detection
    """
    
    # Request input
    miner_events: List[ProcessedEvent]
    original_submissions: List[UserSubmission]
    validation_criteria: Dict[str, float]  # Thresholds for validation
    
    # Response output
    validation_results: Optional[List[ValidationResult]] = None
    overall_scores: Optional[Dict[str, float]] = None
    
    def deserialize(self) -> List[ValidationResult]:
        """
        Deserialize validation results
        """
        return self.validation_results or []

class DataProcessingRequest(bt.Synapse):
    """
    Legacy: Validator -> Miner processing request (for compatibility)
    """
    
    # Required request input, filled by validator
    raw_reports: List[RawIncidentReport]
    task_id: str
    deadline: datetime
    processing_requirements: Dict[str, Any] = None
    
    # Optional response output, filled by miner
    processed_events: Optional[List[CrimeEvent]] = None
    processing_time: Optional[float] = None
    miner_metadata: Optional[Dict[str, Any]] = None
    
    def deserialize(self) -> List[CrimeEvent]:
        """
        Deserialize the processed crime events from the miner.
        """
        return self.processed_events or []

# Legacy compatibility
class Dummy(bt.Synapse):
    """Legacy dummy protocol for testing"""
    dummy_input: int
    dummy_output: typing.Optional[int] = None
    
    def deserialize(self) -> int:
        return self.dummy_output
