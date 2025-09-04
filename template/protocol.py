# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Luminar Network

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
# Simple crime report analysis system where validators send crime reports
# and miners analyze them to extract structured events.

@dataclass
class RawIncidentReport:
    """
    Raw incident report structure used by the validator system
    """
    report_id: str
    text_description: str
    geotag: Dict[str, float]  # {"lat": float, "lng": float}
    timestamp: datetime
    media_urls: List[str]
    media_hashes: List[str]
    user_id: str
    metadata: Dict[str, Any]

class CrimeReportAnalysisRequest(bt.Synapse):
    """
    Version 1 protocol for crime report analysis.
    Validators send text crime reports to miners for analysis.
    """
    
    # Request fields (sent by validator)
    crime_report_text: str = ""  # The crime report text to analyze
    request_id: str = ""  # Unique identifier for this request
    timestamp: float = 0.0  # When the request was made
    
    # Response fields (filled by miner) 
    analyzed_events: List[Dict[str, Any]] = []  # Extracted crime events
    analysis_confidence: float = 0.0  # Miner's confidence in analysis (0.0-1.0)
    processing_time: float = 0.0  # Time taken to process
    miner_version: str = "1.0.0"  # Version of miner that processed this
    
    def deserialize(self) -> "CrimeReportAnalysisRequest":
        return self

@dataclass
class CrimeEvent:
    """
    crime event structure extracted from text reports
    """
    event_type: str  # Type of crime (theft, assault, burglary, etc.)
    severity: str  # low, medium, high
    location: str  # Extracted location information
    time_info: str  # Extracted time/date information  
    entities: List[str]  # People, vehicles, objects mentioned
    summary: str  # Brief summary of the event
    confidence: float  # Confidence in this extraction (0.0-1.0)

# Legacy protocol support (for backward compatibility)
@dataclass  
class MediaUpload:
    """Legacy media upload class - kept for compatibility"""
    media_id: str = ""
    media_type: str = ""
    media_url: str = ""
    media_hash: str = ""
    content_description: str = ""
    file_size: int = 0
    mime_type: str = ""
    upload_timestamp: datetime = datetime.now()

@dataclass
class UserSubmission:
    """Legacy user submission class - kept for compatibility"""
    submission_id: str = ""
    text_description: str = ""
    geotag: Dict[str, float] = None
    timestamp: datetime = datetime.now()
    submission_timestamp: datetime = datetime.now()
    user_id: str = ""
    media_files: List[MediaUpload] = None
    metadata: Dict[str, Any] = None
    verification_status: str = "pending"

# Dummy/Mock synapse for compatibility
class Dummy(bt.Synapse):
    """
    Version 1 dummy synapse for backward compatibility
    """
    dummy_input: int = 1
    dummy_output: Optional[int] = None

    def deserialize(self) -> int:
        return self.dummy_output

# Legacy classes for backward compatibility
@dataclass
class ProcessedEvent:
    """Legacy processed event"""
    event_id: str = ""
    generated_summary: str = ""
    event_type: str = ""
    confidence_score: float = 0.0
    extracted_entities: Dict[str, Any] = None
    processing_timestamp: datetime = datetime.now()

@dataclass
class ValidationResult:
    """Legacy validation result"""
    validation_id: str = ""
    event_id: str = ""
    overall_score: float = 0.0
    validation_timestamp: datetime = datetime.now()

# Main protocol class for Version 1
class MediaProcessingRequest(bt.Synapse):
    """
    Legacy protocol
    """
    # Keep for backward compatibility but simplify
    user_submissions: List[UserSubmission] = []
    task_id: str = ""
    
    # Response
    processed_events: Optional[List[ProcessedEvent]] = None
    processing_time: Optional[float] = None
    
    def deserialize(self) -> List[ProcessedEvent]:
        return self.processed_events or []

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
