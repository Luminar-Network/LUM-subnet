# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Luminar Network
# Copyright © 2025 Khem Raj Regmi
# Copyright © 2025 diwas7777

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
import torch
import hashlib
import asyncio
import typing
import bittensor as bt
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import re
import sys
import os
import json
import uuid

# Add template path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    bt.logging.warning("OpenAI not available - using rule-based processing")

# Import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from template.protocol import CrimeReportAnalysisRequest, CrimeEvent

class Miner(BaseMinerNeuron):
    """
    Luminar Subnet Miner
    
    Simple crime report analysis system:
    - Receives text crime reports from validators
    - Uses OpenAI to analyze and extract structured events 
    - Returns crime events with confidence scores
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Version 1 configuration
        self.version = "1.0.0"
        
        # Initialize OpenAI client for crime report analysis
        self._initialize_openai()
        
        # Crime type classifications
        self.crime_types = [
            "theft", "burglary", "robbery", "assault", "vandalism", 
            "fraud", "drug_offense", "traffic_violation", "domestic_violence",
            "cybercrime", "harassment", "trespassing", "arson", "other"
        ]
        
        # Severity levels
        self.severity_levels = ["low", "medium", "high"]
        
        bt.logging.info(f"🚀 Luminar Miner Version {self.version} initialized")
        
    def blacklist(self, synapse: CrimeReportAnalysisRequest) -> typing.Tuple[bool, str]:
        """
        Blacklist function to determine if a request should be blocked.
        Returns a tuple of (should_blacklist, reason)
        """
        try:
            # Check if the requester is a validator
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"
            
            # Get the UID of the requester
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            
            # Check if force validator permit is enabled
            if self.config.blacklist.force_validator_permit:
                if not self.metagraph.validator_permit[uid]:
                    return True, f"Hotkey {synapse.dendrite.hotkey} does not have validator permit"
            
            # Check if non-registered entities are allowed
            if not self.config.blacklist.allow_non_registered:
                if uid >= len(self.metagraph.hotkeys):
                    return True, f"Hotkey {synapse.dendrite.hotkey} is not registered"
            
            # Blacklist requests from entities with too much stake (spam protection)
            stake = self.metagraph.S[uid].item()
            if stake > 100000:  # Configurable threshold
                return True, f"Hotkey {synapse.dendrite.hotkey} has too much stake: {stake}"
                
            return False, "Allowed"
            
        except Exception as e:
            bt.logging.error(f"Error in blacklist function: {e}")
            return True, f"Blacklist error: {e}"

    def priority(self, synapse: CrimeReportAnalysisRequest) -> float:
        """
        Priority function to determine the order of processing requests.
        Higher values = higher priority.
        """
        try:
            # Get the UID of the requester
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                return 0.0
                
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            
            # Base priority
            priority = 1.0
            
            # Higher priority for validators
            if self.metagraph.validator_permit[uid]:
                priority += 10.0
                
            # Priority based on stake (higher stake = higher priority)
            stake = self.metagraph.S[uid].item()
            priority += min(stake / 1000, 5.0)  # Cap at 5.0 additional priority
            
            # Priority based on trust/incentive
            trust = self.metagraph.T[uid].item() if uid < len(self.metagraph.T) else 0.0
            incentive = self.metagraph.I[uid].item() if uid < len(self.metagraph.I) else 0.0
            priority += (trust + incentive) * 2.0
            
            return priority
            
        except Exception as e:
            bt.logging.error(f"Error in priority function: {e}")
            return 0.0
        
    def _initialize_openai(self):
        """Initialize OpenAI client for crime analysis"""
        if not OPENAI_AVAILABLE:
            bt.logging.info("🤖 OpenAI: Not available - using rule-based processing")
            self.openai_client = None
            return
            
        # Get OpenAI API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            bt.logging.warning("🔑 OPENAI_API_KEY not found in environment")
            self.openai_client = None
            return
            
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            
            # OpenAI configuration
            self.openai_config = {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "500")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
            }
            
            bt.logging.info(f"✅ OpenAI initialized with model: {self.openai_config['model']}")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize OpenAI: {e}")
            self.openai_client = None

    async def forward(
        self, synapse: CrimeReportAnalysisRequest
    ) -> CrimeReportAnalysisRequest:
        """
        forward function - analyze crime reports and extract events
        
        Args:
            synapse: CrimeReportAnalysisRequest containing crime report text
            
        Returns:
            synapse: Same synapse with analyzed_events filled
        """
        bt.logging.info(f"📝 Processing crime report analysis request: {synapse.request_id}")
        
        start_time = time.time()
        
        try:
            # Analyze the crime report text
            analyzed_events = await self._analyze_crime_report(synapse.crime_report_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate overall confidence based on events
            overall_confidence = self._calculate_overall_confidence(analyzed_events)
            
            # Fill response
            synapse.analyzed_events = analyzed_events
            synapse.analysis_confidence = overall_confidence
            synapse.processing_time = processing_time
            synapse.miner_version = self.version
            
            bt.logging.info(f"✅ Analysis completed in {processing_time:.2f}s with {len(analyzed_events)} events")
            
        except Exception as e:
            bt.logging.error(f"❌ Error processing crime report: {e}")
            # Return empty results on error
            synapse.analyzed_events = []
            synapse.analysis_confidence = 0.0
            synapse.processing_time = time.time() - start_time
            
        return synapse

    async def _analyze_crime_report(self, report_text: str) -> List[Dict[str, Any]]:
        """
        Crime report analysis using OpenAI

        Args:
            report_text: The crime report text to analyze
            
        Returns:
            List of extracted crime events
        """
        if not report_text or not report_text.strip():
            return []
            
        # Try OpenAI analysis first
        if self.openai_client:
            try:
                return await self._openai_analyze_crime_report(report_text)
            except Exception as e:
                bt.logging.warning(f"⚠️ OpenAI analysis failed: {e}, falling back to rule-based")
                
        # Fallback to rule-based analysis
        return self._rule_based_analyze_crime_report(report_text)

    async def _openai_analyze_crime_report(self, report_text: str) -> List[Dict[str, Any]]:
        """
        Use OpenAI to analyze crime report and extract structured events
        """
        prompt = f"""
        Analyze this crime report and extract structured information. Return a JSON array of crime events.

        Crime Report:
        {report_text}

        For each event found, extract:
        - event_type: Type of crime (theft, assault, burglary, vandalism, fraud, drug_offense, traffic_violation, domestic_violence, cybercrime, harassment, trespassing, arson, other)
        - severity: Severity level (low, medium, high)
        - location: Location information mentioned
        - time_info: Time/date information mentioned  
        - entities: List of people, vehicles, objects mentioned
        - summary: Brief summary of the event
        - confidence: Your confidence in this extraction (0.0 to 1.0)

        Return valid JSON only. Example:
        [{{
            "event_type": "theft",
            "severity": "medium", 
            "location": "downtown parking lot",
            "time_info": "yesterday evening around 7pm",
            "entities": ["red car", "suspect in black jacket", "wallet"],
            "summary": "Theft of wallet from red car in downtown parking lot",
            "confidence": 0.85
        }}]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_config["model"],
                messages=[
                    {"role": "system", "content": "You are a crime analysis expert. Extract structured information from crime reports and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.openai_config["max_tokens"],
                temperature=self.openai_config["temperature"]
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's valid JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            events = json.loads(content)
            
            # Validate and clean the events
            validated_events = []
            for event in events:
                if self._validate_event(event):
                    validated_events.append(event)
                    
            bt.logging.info(f"🤖 OpenAI extracted {len(validated_events)} valid events")
            return validated_events
            
        except json.JSONDecodeError as e:
            bt.logging.error(f"❌ Failed to parse OpenAI JSON response: {e}")
            raise
        except Exception as e:
            bt.logging.error(f"❌ OpenAI analysis error: {e}")
            raise

    def _rule_based_analyze_crime_report(self, report_text: str) -> List[Dict[str, Any]]:
        """
        Rule-based fallback analysis for crime reports
        """
        events = []
        
        # Simple pattern matching for crime types
        crime_patterns = {
            "theft": r"(stol|theft|steal|rob|burglar|loot)",
            "assault": r"(assault|attack|hit|punch|fight|violence)",
            "vandalism": r"(vandal|damage|destroy|graffiti|break)",
            "fraud": r"(fraud|scam|cheat|deceive|fake)",
            "drug_offense": r"(drug|narcotic|cocaine|marijuana|heroin)",
            "traffic_violation": r"(speeding|traffic|accident|crash|collision)",
        }
        
        text_lower = report_text.lower()
        
        # Extract events based on patterns
        for crime_type, pattern in crime_patterns.items():
            if re.search(pattern, text_lower):
                # Extract basic information
                location = self._extract_location(report_text)
                time_info = self._extract_time(report_text)
                entities = self._extract_entities(report_text)
                
                event = {
                    "event_type": crime_type,
                    "severity": self._assess_severity(report_text),
                    "location": location,
                    "time_info": time_info,
                    "entities": entities,
                    "summary": f"{crime_type.title()} incident" + (f" at {location}" if location else ""),
                    "confidence": 0.6  # Lower confidence for rule-based
                }
                
                events.append(event)
                
        # If no patterns matched, create a generic event
        if not events:
            events.append({
                "event_type": "other",
                "severity": "low",
                "location": self._extract_location(report_text),
                "time_info": self._extract_time(report_text),
                "entities": self._extract_entities(report_text),
                "summary": "General incident report",
                "confidence": 0.3
            })
            
        bt.logging.info(f"🔍 Rule-based analysis extracted {len(events)} events")
        return events

    def _validate_event(self, event: Dict[str, Any]) -> bool:
        """Validate that an event has required fields and valid values"""
        required_fields = ["event_type", "severity", "location", "time_info", "entities", "summary", "confidence"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in event:
                return False
                
        # Validate event_type
        if event["event_type"] not in self.crime_types:
            event["event_type"] = "other"
            
        # Validate severity
        if event["severity"] not in self.severity_levels:
            event["severity"] = "low"
            
        # Validate confidence range
        try:
            confidence = float(event["confidence"])
            if confidence < 0.0 or confidence > 1.0:
                event["confidence"] = 0.5
            else:
                event["confidence"] = confidence
        except (ValueError, TypeError):
            event["confidence"] = 0.5
            
        # Ensure entities is a list
        if not isinstance(event["entities"], list):
            event["entities"] = []
            
        return True

    def _extract_location(self, text: str) -> str:
        """Extract location information from text"""
        # Simple location extraction patterns
        location_patterns = [
            r"(?:at|in|on|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+\s+(?:street|road|avenue|blvd|park|school|store|mall))",
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]
                
        return "location not specified"

    def _extract_time(self, text: str) -> str:
        """Extract time/date information from text"""
        # Time extraction patterns
        time_patterns = [
            r"(yesterday|today|tomorrow|last\s+\w+|this\s+\w+)",
            r"(\d{1,2}:\d{2}\s*(?:am|pm)?)",
            r"(morning|afternoon|evening|night)",
            r"(\d{1,2}/\d{1,2}/\d{2,4})",
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
            
        return " ".join(times) if times else "time not specified"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (people, vehicles, objects) from text"""
        entities = []
        
        # Vehicle patterns
        vehicle_patterns = r"(car|truck|motorcycle|bike|vehicle|van|suv|sedan)"
        vehicles = re.findall(vehicle_patterns, text, re.IGNORECASE)
        entities.extend([f"{v} (vehicle)" for v in vehicles])
        
        # Color + object patterns
        color_patterns = r"(red|blue|green|yellow|black|white|gray|silver)\s+(\w+)"
        colored_objects = re.findall(color_patterns, text, re.IGNORECASE)
        entities.extend([f"{color} {obj}" for color, obj in colored_objects])
        
        # Person indicators
        person_patterns = r"(suspect|person|man|woman|individual|guy|girl)"
        persons = re.findall(person_patterns, text, re.IGNORECASE)
        entities.extend([f"{p} (person)" for p in persons])
        
        return list(set(entities))  # Remove duplicates

    def _assess_severity(self, text: str) -> str:
        """Assess severity based on text content"""
        text_lower = text.lower()
        
        # High severity indicators
        high_severity = ["weapon", "gun", "knife", "injured", "hospital", "blood", "violent", "serious"]
        if any(word in text_lower for word in high_severity):
            return "high"
            
        # Medium severity indicators
        medium_severity = ["damage", "hurt", "threat", "broke", "stolen", "lost"]
        if any(word in text_lower for word in medium_severity):
            return "medium"
            
        return "low"

    def _calculate_overall_confidence(self, events: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not events:
            return 0.0
            
        confidences = [event.get("confidence", 0.0) for event in events]
        return sum(confidences) / len(confidences)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"🔄 Miner Version {miner.version} running...")
            time.sleep(30)
