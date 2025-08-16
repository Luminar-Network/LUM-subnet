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
class RawIncidentReport:
    """
    A raw incident report submitted by users.
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

class DataProcessingRequest(bt.Synapse):

import typing
import bittensor as bt

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class Dummy(bt.Synapse):
class DataProcessingRequest(bt.Synapse):
    """
    Validator -> Miner: Request to process raw incident reports into structured crime events.
    
    Validators send batches of raw incident reports to miners for processing.
    Miners respond with clustered and structured crime events.
    """
    
    # Required request input, filled by validator
    raw_reports: List[RawIncidentReport]
    task_id: str
    deadline: datetime
    processing_requirements: Dict[str, Any] = None  # Special requirements for processing
    
    # Optional response output, filled by miner
    processed_events: Optional[List[CrimeEvent]] = None
    processing_time: Optional[float] = None
    miner_metadata: Optional[Dict[str, Any]] = None
    
    def deserialize(self) -> List[CrimeEvent]:
        """
        Deserialize the processed crime events from the miner.
        
        Returns:
        - List[CrimeEvent]: The clustered and structured crime events.
        """
        return self.processed_events or []


class VerificationRequest(bt.Synapse):
    """
    Validator -> Validator: Request for cross-validation of crime events.
    
    Used for consensus building and verification of miner outputs.
    """
    
    # Required request input
    crime_events: List[CrimeEvent]
    verification_type: str  # "data_integrity", "clustering_accuracy", "novelty_check"
    validator_id: str
    
    # Optional response output
    verification_results: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    
    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the verification results.
        
        Returns:
        - Dict[str, Any]: The verification results and scores.
        """
        return self.verification_results or {}


class ReputationQuery(bt.Synapse):
    """
    Query for miner reputation and performance metrics.
    """
    
    # Required request input
    miner_hotkey: str
    time_range: Dict[str, datetime]  # {"start": datetime, "end": datetime}
    
    # Optional response output
    reputation_score: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    historical_data: Optional[Dict[str, Any]] = None
    
    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the reputation and performance data.
        
        Returns:
        - Dict[str, Any]: Combined reputation and performance metrics.
        """
        return {
            "reputation_score": self.reputation_score,
            "performance_metrics": self.performance_metrics,
            "historical_data": self.historical_data
        }


# Legacy protocol for backward compatibility during migration
class Dummy(bt.Synapse):
    """
    Legacy dummy protocol for testing and migration purposes.
    This will be deprecated once Luminar protocol is fully implemented.
    """
    
    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by receiving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output.
        
        Returns:
        - int: The deserialized response.
        """
        return self.dummy_output
