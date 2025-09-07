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
            "cybercrime", "harassment", "trespassing", "arson", "homicide",
            "sexual_assault", "kidnapping", "extortion", "money_laundering",
            "identity_theft", "embezzlement", "forgery", "counterfeiting",
            "weapon_offense", "public_disorder", "loitering", "stalking",
            "child_abuse", "elder_abuse", "hate_crime", "terrorism",
            "smuggling", "human_trafficking", "prostitution", "gambling",
            "bribery", "corruption", "tax_evasion", "conspiracy",
            "noise_complaint", "littering", "public_intoxication"
        ]
        
        # Enhanced locations for scenarios
        self.locations = [
            "1245 Main Street, Downtown District", "789 Oak Avenue, Residential Area",
            "456 Commerce Boulevard, Shopping District", "321 University Drive, Campus Area",
            "654 Park Avenue, Central Park District", "987 Industrial Road, Warehouse District",
            "147 Hospital Drive, Medical Center", "258 School Street, Education District",
            "369 Financial Plaza, Business District", "741 Sunset Boulevard, Entertainment District",
            "852 Harbor View, Waterfront Area", "963 Mountain Road, Suburban Hills",
            "159 Tech Center, Innovation District", "357 Government Plaza, Civic Center"
        ]
        
        # Dynamic time references will be generated in _get_dynamic_time_reference()
        # This ensures times stay current and don't become outdated
        
        bt.logging.info(f"🚀 Luminar Validator Version {self.version} initialized")
        
    def _get_dynamic_time_reference(self) -> str:
        """Generate dynamic time references that stay current"""
        now = datetime.now()
        
        # Calculate relative dates dynamically
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        three_days_ago = now - timedelta(days=3)
        last_week = now - timedelta(days=7)
        
        # Get day names
        yesterday_name = yesterday.strftime("%A")
        two_days_ago_name = two_days_ago.strftime("%A")
        
        # Dynamic time reference templates
        time_templates = [
            f"{yesterday.strftime('%B %d, %Y')} at approximately 7:00 PM",
            f"{yesterday.strftime('%B %d, %Y')} around 10:30 AM", 
            f"{two_days_ago.strftime('%B %d, %Y')} during late evening hours",
            f"{now.strftime('%B %d, %Y')} at 2:15 PM",
            f"{three_days_ago.strftime('%B %d, %Y')} in the early morning",
            f"{yesterday.strftime('%B %d, %Y')} at midnight",
            f"{yesterday.strftime('%B %d, %Y')} during rush hour traffic",
            f"{last_week.strftime('%B %d, %Y')} on weekend afternoon",
            f"{two_days_ago.strftime('%B %d, %Y')} between 11:00 PM and 1:00 AM",
            f"{now.strftime('%B %d, %Y')} at dawn",
            f"{now.strftime('%B %d, %Y')} during business hours",
            f"{yesterday.strftime('%B %d, %Y')} late at night",
            f"last {yesterday_name} evening",
            f"this past {two_days_ago_name} morning",
            f"over the weekend",
            f"last week around noon",
            f"a few hours ago",
            f"yesterday afternoon",
            f"last {yesterday_name} night",
            f"this past {two_days_ago_name} evening",
            f"early yesterday morning"
        ]
        
        return random.choice(time_templates)
        
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
        time_ref = self._get_dynamic_time_reference()  # Generate dynamic time reference
        
        # Generate report using OpenAI if available
        if self.openai_client:
            try:
                report_text = await self._openai_generate_crime_report(crime_type, location, time_ref)
            except Exception as e:
                bt.logging.warning(f"⚠️ Report generation failed: {e}, using template")
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

        Example format: "Witness reported seeing two individuals breaking into a red sedan in the parking lot at 1245 Main Street, Downtown District, on September 6, 2025 at approximately 7:00 PM. One suspect was wearing a black hoodie and the other had a blue baseball cap. The suspects fled on foot eastbound on Main Street when approached by security personnel."

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
            bt.logging.info(f"Generated Crime Report: {report_text}")
            return report_text
            
        except Exception as e:
            bt.logging.error(f"❌ Report generation error: {e}")
            raise

    def _template_generate_crime_report(self, crime_type: str, location: str, time_ref: str) -> str:
        """
        Generate crime report using templates (fallback)
        """
        templates = {
            "theft": f"Victim reported that their wallet and smartphone were stolen from their vehicle at {location} on {time_ref}. Suspect described as wearing dark clothing and fled on foot toward the nearby intersection.",
            "burglary": f"Homeowner discovered break-in at their residence at {location} on {time_ref}. Items missing include electronics, jewelry, and cash. Entry was gained through rear window.",
            "robbery": f"Armed robbery reported at {location} on {time_ref}. Suspect approached victim demanding money while displaying what appeared to be a handgun. Victim complied and suspect fled the scene.",
            "assault": f"Witness reported seeing an individual being attacked at {location} on {time_ref}. Victim sustained minor injuries and was transported to hospital. Suspect fled before police arrival.",
            "vandalism": f"Property damage reported at {location} on {time_ref}. Windows were broken and graffiti was spray-painted on building walls. Estimated damage exceeds $500.",
            "fraud": f"Victim reported receiving suspicious phone calls requesting personal banking information on {time_ref}. Caller claimed to be from bank security department and requested account verification.",
            "drug_offense": f"Suspicious activity reported at {location} on {time_ref}. Officers observed what appeared to be drug transaction taking place. Two individuals detained for questioning.",
            "traffic_violation": f"Multiple witnesses reported reckless driving incident at {location} on {time_ref}. Vehicle was speeding excessively and ran multiple red lights before disappearing.",
            "domestic_violence": f"Domestic disturbance reported at {location} on {time_ref}. Neighbors reported loud arguing and sounds of altercation. Officers responded and separated the parties involved.",
            "cybercrime": f"Business owner reported computer system breach at {location} discovered on {time_ref}. Unauthorized access to customer database and potential data theft under investigation.",
            "harassment": f"Victim reported ongoing harassment via phone calls and text messages beginning {time_ref}. Suspect has been making threatening communications despite cease and desist request.",
            "trespassing": f"Property owner reported unauthorized individuals on private property at {location} on {time_ref}. Suspects were asked to leave but returned later the same evening.",
            "arson": f"Fire department responded to suspicious fire at {location} on {time_ref}. Investigation revealed evidence of accelerant use. Fire marshal ruling incident as arson.",
            "homicide": f"Body discovered at {location} on {time_ref}. Victim showed signs of trauma consistent with foul play. Homicide detectives have taken over the investigation.",
            "weapon_offense": f"Individual reported carrying concealed weapon without permit at {location} on {time_ref}. Suspect was detained and weapon was confiscated by responding officers.",
            "public_disorder": f"Large disturbance reported at {location} on {time_ref}. Multiple individuals involved in altercation requiring police intervention to restore order.",
            "noise_complaint": f"Residents reported excessive noise coming from {location} on {time_ref}. Loud music and shouting continued past city noise ordinance hours."
        }
        
        report = templates.get(crime_type, f"Incident of {crime_type} reported at {location} on {time_ref}. Details are being investigated by local authorities and additional information will be released as it becomes available.")
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
        high_severity = [
            "homicide", "sexual_assault", "kidnapping", "terrorism", "arson",
            "armed_robbery", "robbery", "assault", "weapon_offense", "human_trafficking"
        ]
        medium_severity = [
            "burglary", "theft", "fraud", "drug_offense", "domestic_violence",
            "cybercrime", "harassment", "stalking", "extortion", "money_laundering",
            "embezzlement", "forgery", "counterfeiting", "hate_crime", "child_abuse",
            "elder_abuse", "identity_theft", "bribery", "corruption"
        ]
        low_severity = [
            "vandalism", "trespassing", "traffic_violation", "public_disorder",
            "loitering", "noise_complaint", "littering", "public_intoxication",
            "gambling", "prostitution", "tax_evasion", "conspiracy"
        ]
        
        if crime_type in high_severity:
            return "high"
        elif crime_type in medium_severity:
            return "medium"
        elif crime_type in low_severity:
            return "low"
        else:
            return "medium"  # Default for unknown crime types

    def _get_expected_entities(self, crime_type: str, location: str) -> List[str]:
        """Get expected entities based on crime type and location"""
        entities = []
        
        # Location-based entities (enhanced)
        if "parking" in location.lower() or "street" in location.lower():
            entities.extend(["vehicle", "car", "license plate"])
        elif "store" in location.lower() or "mall" in location.lower() or "shopping" in location.lower():
            entities.extend(["merchandise", "cash register", "security camera", "customer"])
        elif "school" in location.lower() or "campus" in location.lower():
            entities.extend(["student", "teacher", "security guard"])
        elif "hospital" in location.lower() or "medical" in location.lower():
            entities.extend(["patient", "medical staff", "security"])
        elif "bank" in location.lower() or "financial" in location.lower():
            entities.extend(["teller", "security guard", "cash", "surveillance"])
        elif "residential" in location.lower() or "house" in location.lower():
            entities.extend(["homeowner", "neighbor", "property"])
        elif "office" in location.lower() or "business" in location.lower():
            entities.extend(["employee", "computer", "documents"])
            
        # Crime-type based entities (enhanced)
        if crime_type in ["theft", "burglary", "robbery"]:
            entities.extend(["suspect", "victim", "stolen items", "witness"])
            if crime_type == "robbery":
                entities.extend(["weapon", "threat"])
        elif crime_type == "assault":
            entities.extend(["attacker", "victim", "witness", "injury"])
        elif crime_type == "vandalism":
            entities.extend(["damage", "graffiti", "property", "spray paint"])
        elif crime_type in ["drug_offense"]:
            entities.extend(["drugs", "paraphernalia", "dealer", "buyer"])
        elif crime_type == "traffic_violation":
            entities.extend(["vehicle", "driver", "license plate", "traffic light"])
        elif crime_type == "domestic_violence":
            entities.extend(["victim", "perpetrator", "family member", "residence"])
        elif crime_type == "cybercrime":
            entities.extend(["computer", "internet", "data", "hacker"])
        elif crime_type in ["harassment", "stalking"]:
            entities.extend(["victim", "perpetrator", "phone", "messages"])
        elif crime_type == "arson":
            entities.extend(["fire", "accelerant", "building", "arsonist"])
        elif crime_type == "homicide":
            entities.extend(["victim", "body", "weapon", "crime scene"])
        elif crime_type == "weapon_offense":
            entities.extend(["weapon", "firearm", "permit", "suspect"])
        elif crime_type in ["fraud", "identity_theft", "embezzlement"]:
            entities.extend(["documents", "credit card", "bank account", "victim"])
        elif crime_type in ["noise_complaint", "public_disorder"]:
            entities.extend(["noise", "disturbance", "residents", "loud music"])
            
        return list(set(entities))  # Remove duplicates

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
