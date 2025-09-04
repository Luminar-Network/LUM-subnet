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
import asyncio
import sys
import os
import random
import numpy as np
import torch
import bittensor as bt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import hashlib
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

# OpenAI integration for Version 1 crime report generation
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    bt.logging.warning("OpenAI not available - using rule-based crime report generation")

# Import Luminar protocol
from template.protocol import CrimeReportAnalysisRequest, CrimeEvent
from template.utils.uids import get_random_uids

class Validator(BaseValidatorNeuron):
    """
    Luminar Subnet Validator
    
    Simple crime report analysis system:
    - Generates realistic crime reports using OpenAI
    - Sends reports to miners for analysis
    - Scores miners based on analysis quality
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        
        # Version 1 configuration
        self.version = "1.0.0"
        
        # Initialize OpenAI client for crime report generation
        self._initialize_openai()
        
        # Crime scenarios
        self.crime_scenarios = [
            "theft", "burglary", "robbery", "assault", "vandalism", 
            "fraud", "drug_offense", "traffic_violation", "domestic_violence",
            "cybercrime", "harassment", "trespassing", "arson"
        ]
        
        # Common locations for scenarios
        self.locations = [
            "downtown area", "residential neighborhood", "shopping mall", 
            "parking lot", "school campus", "public park", "subway station",
            "office building", "grocery store", "gas station"
        ]
        
        # Time references
        self.time_references = [
            "yesterday evening", "this morning", "last night", "earlier today",
            "around 3pm", "late at night", "during rush hour", "weekend afternoon"
        ]
        
        bt.logging.info(f"🚀 Luminar Validator Version {self.version} initialized")
        
    def _initialize_openai(self):
        """Initialize OpenAI client for crime report generation"""
        if not OPENAI_AVAILABLE:
            bt.logging.info("🤖 OpenAI: Not available - using template-based reports")
            self.openai_client = None
            return
            
        # Get OpenAI API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            bt.logging.warning("🔑 OPENAI_API_KEY not found in environment")
            self.openai_client = None
            return
            
        # Initialize OpenAI client for Version 1
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            
            # Version 1 OpenAI configuration
            self.openai_config = {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "300")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # Higher temperature for creative reports
            }
            
            bt.logging.info(f"✅ OpenAI initialized with model: {self.openai_config['model']}")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize OpenAI: {e}")
            self.openai_client = None

    async def forward(self):
        """
        Forward function - generate crime reports and test miners
        """
        bt.logging.info(f"🔄 Starting Version {self.version} validation round")
        
        try:
            # Get a sample of miners to test
            miner_uids = get_random_uids(self, k=min(10, len(self.metagraph.hotkeys)))
            
            if len(miner_uids) == 0:
                bt.logging.warning("⚠️ No miners available for testing")
                return
                
            bt.logging.info(f"🎯 Testing {len(miner_uids)} miners")
            
            # Generate crime report for testing
            crime_report = await self._generate_crime_report()
            
            # Send report to miners and collect responses
            responses = await self._query_miners(miner_uids, crime_report)
            
            # Score the responses
            scores = self._score_responses(responses, crime_report)
            
            # Set weights based on scores
            self._set_weights(scores)
            
            bt.logging.info(f"✅ Validation round completed for {len(miner_uids)} miners")
            
        except Exception as e:
            bt.logging.error(f"❌ Error in validation round: {e}")

    async def _generate_crime_report(self) -> Dict[str, Any]:
        """
        Generate a realistic crime report for testing miners
        """
        # Select random scenario components
        crime_type = random.choice(self.crime_scenarios)
        location = random.choice(self.locations)
        time_ref = random.choice(self.time_references)
        
        # Generate report using OpenAI if available
        if self.openai_client:
            try:
                report_text = await self._openai_generate_crime_report(crime_type, location, time_ref)
            except Exception as e:
                bt.logging.warning(f"⚠️ OpenAI report generation failed: {e}, using template")
                report_text = self._template_generate_crime_report(crime_type, location, time_ref)
        else:
            report_text = self._template_generate_crime_report(crime_type, location, time_ref)
            
        # Create the expected answer for scoring
        expected_answer = {
            "crime_type": crime_type,
            "location": location,
            "time_ref": time_ref,
            "severity": self._determine_expected_severity(crime_type),
            "expected_entities": self._get_expected_entities(crime_type, location)
        }
        
        report = {
            "text": report_text,
            "expected": expected_answer,
            "request_id": str(uuid.uuid4()),
            "generated_at": time.time()
        }
        
        bt.logging.info(f"📝 Generated {crime_type} report: {report_text[:100]}...")
        return report

    async def _openai_generate_crime_report(self, crime_type: str, location: str, time_ref: str) -> str:
        """
        Use OpenAI to generate realistic crime reports
        """
        prompt = f"""
        Generate a realistic crime report for law enforcement training. 

        Details:
        - Crime type: {crime_type}
        - Location: {location}
        - Time: {time_ref}

        Write a 2-3 sentence crime report that sounds like it came from a police report or witness statement. 
        Include specific details about what happened, who was involved, and any evidence.
        Make it realistic but not graphic.

        Example format: "Witness reported seeing two individuals breaking into a red sedan in the downtown parking lot yesterday evening around 7pm. One suspect was wearing a black hoodie and the other had a blue baseball cap. The suspects fled on foot when approached by security."

        Write ONLY the crime report text, no additional formatting or explanation.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_config["model"],
                messages=[
                    {"role": "system", "content": "You are a police report writer. Generate realistic but training-appropriate crime reports."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.openai_config["max_tokens"],
                temperature=self.openai_config["temperature"]
            )
            
            report_text = response.choices[0].message.content.strip()
            bt.logging.info(f"🤖 OpenAI generated crime report")
            return report_text
            
        except Exception as e:
            bt.logging.error(f"❌ OpenAI report generation error: {e}")
            raise

    def _template_generate_crime_report(self, crime_type: str, location: str, time_ref: str) -> str:
        """
        Generate crime report using templates (fallback)
        """
        templates = {
            "theft": f"Victim reported that their wallet was stolen from their car in the {location} {time_ref}. Suspect described as wearing dark clothing and fled on foot.",
            "assault": f"Witness reported seeing an individual being attacked in the {location} {time_ref}. Victim sustained minor injuries and suspect fled the scene.",
            "vandalism": f"Property damage reported at {location} {time_ref}. Windows were broken and graffiti was spray-painted on the building walls.",
            "burglary": f"Homeowner discovered break-in at their residence in {location} {time_ref}. Items missing include electronics and jewelry.",
            "fraud": f"Victim reported receiving suspicious phone calls requesting personal information {time_ref}. Caller claimed to be from bank security.",
            "traffic_violation": f"Multiple witnesses reported reckless driving incident in {location} {time_ref}. Vehicle was speeding and ran red light.",
        }
        
        report = templates.get(crime_type, f"Incident reported in {location} {time_ref}. Details are being investigated by local authorities.")
        bt.logging.info(f"📋 Generated template-based {crime_type} report")
        return report

    async def _query_miners(self, miner_uids: List[int], crime_report: Dict[str, Any]) -> List[Tuple[int, CrimeReportAnalysisRequest]]:
        """
        Send crime report to miners and collect their responses
        """
        bt.logging.info(f"📤 Sending crime report to {len(miner_uids)} miners")
        
        # Create the request synapse
        request = CrimeReportAnalysisRequest(
            crime_report_text=crime_report["text"],
            request_id=crime_report["request_id"],
            timestamp=time.time()
        )
        
        responses = []
        
        # Query miners asynchronously
        try:
            # Send requests to miners using dendrite
            miner_responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=request,
                deserialize=True,
                timeout=30.0
            )
            
            # Collect valid responses
            for i, response in enumerate(miner_responses):
                uid = miner_uids[i]
                if response is not None:
                    responses.append((uid, response))
                    bt.logging.info(f"✅ Received response from miner {uid}")
                else:
                    bt.logging.warning(f"⚠️ No response from miner {uid}")
                    
        except Exception as e:
            bt.logging.error(f"❌ Error querying miners: {e}")
            
        return responses

    def _score_responses(self, responses: List[Tuple[int, CrimeReportAnalysisRequest]], crime_report: Dict[str, Any]) -> Dict[int, float]:
        """
        Score miner responses based on analysis quality
        """
        scores = {}
        expected = crime_report["expected"]
        
        bt.logging.info(f"📊 Scoring {len(responses)} miner responses")
        
        for uid, response in responses:
            try:
                score = self._score_single_response(response, expected)
                scores[uid] = score
                bt.logging.info(f"🎯 Miner {uid} scored: {score:.3f}")
            except Exception as e:
                bt.logging.error(f"❌ Error scoring miner {uid}: {e}")
                scores[uid] = 0.0
                
        return scores

    def _score_single_response(self, response: CrimeReportAnalysisRequest, expected: Dict[str, Any]) -> float:
        """
        Score a single miner response
        """
        if not response.analyzed_events:
            return 0.0
            
        total_score = 0.0
        max_score = 0.0
        
        # Score each analyzed event
        for event in response.analyzed_events:
            event_score = 0.0
            max_event_score = 100.0
            
            # Crime type accuracy (30 points)
            if event.get("event_type") == expected["crime_type"]:
                event_score += 30.0
            elif event.get("event_type") in ["other", expected["crime_type"]]:
                event_score += 15.0  # Partial credit
                
            # Location detection (20 points)
            event_location = event.get("location", "").lower()
            expected_location = expected["location"].lower()
            if expected_location in event_location or event_location in expected_location:
                event_score += 20.0
            elif event_location != "location not specified":
                event_score += 10.0  # Partial credit for any location
                
            # Time detection (20 points)  
            event_time = event.get("time_info", "").lower()
            expected_time = expected["time_ref"].lower()
            if any(word in event_time for word in expected_time.split()):
                event_score += 20.0
            elif event_time != "time not specified":
                event_score += 10.0  # Partial credit for any time
                
            # Entity extraction (15 points)
            event_entities = event.get("entities", [])
            if len(event_entities) > 0:
                entity_score = min(15.0, len(event_entities) * 3.0)
                event_score += entity_score
                
            # Confidence calibration (10 points)
            confidence = event.get("confidence", 0.0)
            if 0.3 <= confidence <= 0.9:  # Reasonable confidence range
                event_score += 10.0
            elif confidence > 0.0:
                event_score += 5.0  # Partial credit
                
            # Summary quality (5 points)
            summary = event.get("summary", "")
            if len(summary) > 10:  # Non-trivial summary
                event_score += 5.0
                
            total_score += event_score
            max_score += max_event_score
            
        # Normalize to 0-1 range
        if max_score > 0:
            final_score = total_score / max_score
        else:
            final_score = 0.0
            
        # Bonus for processing time (fast but accurate)
        processing_time = response.processing_time
        if processing_time > 0 and processing_time < 10.0:  # Under 10 seconds
            time_bonus = max(0, (10.0 - processing_time) / 100.0)  # Small bonus
            final_score += time_bonus
            
        # Penalty for too many events (likely hallucination)
        if len(response.analyzed_events) > 3:
            final_score *= 0.8
            
        return min(1.0, max(0.0, final_score))

    def _determine_expected_severity(self, crime_type: str) -> str:
        """Determine expected severity based on crime type"""
        high_severity = ["assault", "robbery", "arson"]
        medium_severity = ["burglary", "theft", "fraud", "drug_offense"]
        
        if crime_type in high_severity:
            return "high"
        elif crime_type in medium_severity:
            return "medium"
        else:
            return "low"

    def _get_expected_entities(self, crime_type: str, location: str) -> List[str]:
        """Get expected entities based on crime type and location"""
        entities = []
        
        # Location-based entities
        if "parking" in location:
            entities.extend(["car", "vehicle"])
        elif "store" in location or "mall" in location:
            entities.extend(["merchandise", "cash register"])
        elif "school" in location:
            entities.extend(["student", "teacher"])
            
        # Crime-type based entities
        if crime_type in ["theft", "burglary", "robbery"]:
            entities.extend(["suspect", "victim", "stolen items"])
        elif crime_type == "assault":
            entities.extend(["attacker", "victim", "witness"])
        elif crime_type == "vandalism":
            entities.extend(["damage", "graffiti", "property"])
            
        return entities

    def _set_weights(self, scores: Dict[int, float]):
        """Set weights based on miner scores"""
        bt.logging.info(f"⚖️ Setting weights for {len(scores)} miners")
        
        # Create weight vector
        weights = torch.zeros(len(self.metagraph.hotkeys))
        
        for uid, score in scores.items():
            if uid < len(weights):
                weights[uid] = score
                
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Set weights on chain
        try:
            success = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=torch.arange(len(weights)),
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            
            if success:
                bt.logging.info(f"✅ Successfully set weights on chain")
            else:
                bt.logging.error(f"❌ Failed to set weights on chain")
                
        except Exception as e:
            bt.logging.error(f"❌ Error setting weights: {e}")

    def run(self):
        """
        Run method for Luminar Validator
        """
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Luminar Validator Version {self.version} starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"🔄 Validation step({self.step}) block({self.block})")

                # Run single forward for crime report analysis
                self.loop.run_until_complete(self.forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1
                
                # Sleep between validation rounds to avoid overwhelming miners
                # Crime report analysis doesn't need to be as frequent as other subnets
                time.sleep(60)  # 1 minute between validation rounds

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(err))


# This is the main function, which runs the validator.
if __name__ == "__main__":
    validator = Validator()
    bt.logging.info(f"🔄 Validator Version {validator.version} starting...")
    try:
        validator.run()
    except KeyboardInterrupt:
        bt.logging.success("Validator killed by keyboard interrupt.")
        exit()
    except Exception as e:
        bt.logging.error(f"Error starting validator: {e}")
        exit(1)
