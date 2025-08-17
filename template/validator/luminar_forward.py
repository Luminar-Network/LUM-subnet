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
import uuid
from datetime import datetime, timedelta
from typing import List
import bittensor as bt

from template.protocol import DataProcessingRequest, RawIncidentReport
from template.validator.luminar_reward import get_luminar_rewards
from template.utils.uids import get_random_uids


async def forward(self):
    """
    Luminar subnet validator forward pass.
    
    This function implements the core validation loop for the Luminar crime intelligence subnet:
    1. Prepare batch of raw incident reports
    2. Send processing requests to miners
    3. Verify and score miner responses using comprehensive reward system
    4. Update miner weights based on performance
    5. Store verified events to database
    
    Args:
        self: The validator neuron instance
    """
    try:
        bt.logging.info("Starting Luminar validator forward pass")
        
        # Step 1: Prepare raw incident reports for processing
        raw_reports = await prepare_incident_reports()
        
        if not raw_reports:
            bt.logging.warning("No incident reports available for processing")
            await asyncio.sleep(30)
            return
        
        bt.logging.info(f"Prepared {len(raw_reports)} incident reports for processing")
        
        # Step 2: Select miners for processing task
        miner_uids = get_random_uids(self, k=min(len(raw_reports) // 5 + 1, self.config.neuron.sample_size))
        
        if not miner_uids.size:
            bt.logging.warning("No available miners found")
            await asyncio.sleep(30)
            return
        
        bt.logging.info(f"Selected {len(miner_uids)} miners for processing task")
        
        # Step 3: Send processing requests to miners
        task_id = str(uuid.uuid4())
        deadline = datetime.now() + timedelta(minutes=5)
        
        bt.logging.info(f"Sending processing requests (Task ID: {task_id})")
        
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=DataProcessingRequest(
                raw_reports=raw_reports,
                task_id=task_id,
                deadline=deadline,
                processing_requirements={
                    "min_confidence": 0.3,
                    "max_events_per_report": 2,
                    "clustering_method": "semantic"
                }
            ),
            deserialize=True,
            timeout=300  # 5 minute timeout
        )
        
        bt.logging.info(f"Received {len(responses)} miner responses")
        
        # Step 4: Calculate comprehensive rewards using Luminar reward system
        scores = get_luminar_rewards(raw_reports, responses)
        
        # Log detailed scoring results
        for i, (uid, score) in enumerate(zip(miner_uids, scores)):
            events_count = len(responses[i].processed_events) if responses[i].processed_events else 0
            processing_time = responses[i].processing_time or 0
            bt.logging.info(
                f"Miner {uid}: Score={score:.3f}, Events={events_count}, "
                f"ProcessingTime={processing_time:.2f}s"
            )
        
        # Step 5: Update miner scores and weights
        self.update_scores(scores, miner_uids)
        
        # Step 6: Store high-quality verified events
        await store_verified_events(responses, scores, self)
        
        bt.logging.info(f"Luminar forward pass completed successfully")
        
    except Exception as e:
        bt.logging.error(f"Error in Luminar validator forward pass: {e}")
        
    # Wait before next iteration
    await asyncio.sleep(45)


async def prepare_incident_reports() -> List[RawIncidentReport]:
    """
    Prepare a batch of raw incident reports for processing.
    
    In production, this would:
    1. Fetch from user submission API
    2. Pull from pending processing queue
    3. Load from incident reporting web app
    
    For development/testing, generates synthetic crime incident reports.
    """
    reports = []
    
    # Sample geographic locations (major cities in India for initial deployment)
    locations = [
        {"lat": 28.6139, "lng": 77.2090, "city": "New Delhi", "state": "Delhi"},
        {"lat": 19.0760, "lng": 72.8777, "city": "Mumbai", "state": "Maharashtra"}, 
        {"lat": 13.0827, "lng": 80.2707, "city": "Chennai", "state": "Tamil Nadu"},
        {"lat": 22.5726, "lng": 88.3639, "city": "Kolkata", "state": "West Bengal"},
        {"lat": 12.9716, "lng": 77.5946, "city": "Bangalore", "state": "Karnataka"},
        {"lat": 17.3850, "lng": 78.4867, "city": "Hyderabad", "state": "Telangana"},
        {"lat": 23.0225, "lng": 72.5714, "city": "Ahmedabad", "state": "Gujarat"},
        {"lat": 18.5204, "lng": 73.8567, "city": "Pune", "state": "Maharashtra"}
    ]
    
    # Crime incident templates with realistic descriptions
    incident_templates = [
        {
            "type": "theft",
            "descriptions": [
                "Mobile phone snatched while walking on main road near metro station",
                "Bicycle stolen from apartment parking area overnight", 
                "Purse snatching incident outside shopping mall entrance",
                "Laptop bag theft from car parked on street",
                "Cash stolen from shop counter during busy hours"
            ]
        },
        {
            "type": "vandalism",
            "descriptions": [
                "Graffiti spray painted on government building wall",
                "Public park benches damaged and broken",
                "Car windows smashed in residential parking",
                "Bus stop glass shelter vandalized with stones",
                "Street light poles damaged in housing society"
            ]
        },
        {
            "type": "noise",
            "descriptions": [
                "Loud music and DJ playing past midnight causing disturbance",
                "Construction work noise during restricted hours",
                "Wedding celebration exceeding permitted sound levels",
                "Bar/restaurant noise complaints from neighbors",
                "Traffic horn noise excessive near hospital zone"
            ]
        },
        {
            "type": "assault",
            "descriptions": [
                "Physical altercation between two individuals near market",
                "Domestic violence incident reported by neighbors",
                "Road rage incident involving vehicle drivers",
                "Group conflict outside college campus",
                "Workplace harassment and physical threat reported"
            ]
        },
        {
            "type": "fraud",
            "descriptions": [
                "Online payment fraud through fake website",
                "Credit card cloning at ATM machine",
                "Investment scam through phone calls",
                "Fake job offer demanding money upfront",
                "Identity theft for loan applications"
            ]
        },
        {
            "type": "traffic",
            "descriptions": [
                "Two-wheeler accident at busy intersection",
                "Hit and run case involving pedestrian",
                "Multiple vehicle collision on highway",
                "Traffic signal violation causing accident", 
                "Reckless driving incident in residential area"
            ]
        }
    ]
    
    # Generate 15-25 realistic incident reports
    num_reports = random.randint(15, 25)
    
    for i in range(num_reports):
        location = random.choice(locations)
        incident = random.choice(incident_templates)
        
        # Create geographic clustering (30% chance of nearby incidents)
        if i > 0 and random.random() < 0.3:
            prev_report = reports[-1]
            # Generate nearby location (within 2km)
            location = {
                "lat": prev_report.geotag["lat"] + random.uniform(-0.02, 0.02),
                "lng": prev_report.geotag["lng"] + random.uniform(-0.02, 0.02),
                "city": prev_report.metadata.get("city", location["city"]),
                "state": prev_report.metadata.get("state", location["state"])
            }
        
        # Generate realistic timestamp (last 48 hours with peak times)
        hours_ago = random.choices(
            range(1, 49),  # 1-48 hours ago
            weights=[3 if 18 <= (24 - h % 24) <= 23 or 8 <= (24 - h % 24) <= 10 else 1 for h in range(1, 49)]
        )[0]
        
        timestamp = datetime.now() - timedelta(hours=hours_ago, minutes=random.randint(0, 59))
        
        # Generate media URLs for 40% of reports (more realistic)
        has_media = random.random() < 0.4
        media_urls = []
        media_hashes = []
        
        if has_media:
            media_count = random.randint(1, 3)
            for j in range(media_count):
                media_id = uuid.uuid4().hex[:12]
                media_urls.append(f"https://storage.luminar.network/evidence/{media_id}.jpg")
                media_hashes.append(f"sha256:{uuid.uuid4().hex}")
        
        report = RawIncidentReport(
            report_id=f"LMR_{timestamp.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8].upper()}",
            text_description=random.choice(incident["descriptions"]),
            geotag={
                "lat": location["lat"], 
                "lng": location["lng"]
            },
            timestamp=timestamp,
            media_urls=media_urls,
            media_hashes=media_hashes,
            user_id=f"user_{random.randint(10000, 99999)}",
            metadata={
                "incident_type": incident["type"],
                "city": location["city"],
                "state": location["state"],
                "severity": random.randint(1, 5),
                "verified_user": random.random() < 0.7,  # 70% verified users
                "report_source": random.choice(["mobile_app", "web_portal", "api_integration"])
            }
        )
        
        reports.append(report)
    
    # Sort by timestamp to simulate realistic processing queue
    reports.sort(key=lambda x: x.timestamp, reverse=True)
    
    bt.logging.info(f"Generated {len(reports)} synthetic incident reports across {len(set(r.metadata['city'] for r in reports))} cities")
    return reports


async def store_verified_events(responses: List[DataProcessingRequest], scores, validator_instance):
    """
    Store high-quality verified crime events to the database.
    
    Args:
        responses: Miner responses containing processed crime events
        scores: Quality scores for each response
        validator_instance: Reference to validator for database access
    """
    stored_count = 0
    high_quality_threshold = 0.7
    
    for response, score in zip(responses, scores):
        if score >= high_quality_threshold and response.processed_events:
            for event in response.processed_events:
                try:
                    # Mark event as verified
                    event.verified = True
                    event.verification_metadata = {
                        "validator_score": float(score),
                        "verified_at": datetime.now().isoformat(),
                        "validator_id": validator_instance.wallet.hotkey.ss58_address,
                        "verification_method": "luminar_consensus",
                        "quality_threshold": high_quality_threshold
                    }
                    
                    # TODO: Store to actual PostgreSQL database
                    # INSERT INTO verified_crime_events (event_id, event_data, verification_data, created_at)
                    # VALUES (event.event_id, json.dumps(event.__dict__), json.dumps(event.verification_metadata), NOW())
                    
                    stored_count += 1
                    
                    bt.logging.debug(f"Stored verified event: {event.event_id} - {event.summary_tag}")
                    
                except Exception as e:
                    bt.logging.error(f"Error storing event {event.event_id}: {e}")
    
    if hasattr(validator_instance, 'verified_events_count'):
        validator_instance.verified_events_count += stored_count
    
    bt.logging.info(f"Stored {stored_count} verified crime events with quality score >= {high_quality_threshold}")
    
    if stored_count > 0:
        # Log summary statistics
        total_events = sum(len(r.processed_events) for r in responses if r.processed_events)
        storage_rate = (stored_count / total_events * 100) if total_events > 0 else 0
        
        bt.logging.info(f"Event storage rate: {storage_rate:.1f}% ({stored_count}/{total_events})")


# Legacy forward function for backward compatibility
async def legacy_forward(self):
    """
    Legacy forward function for backward compatibility with template structure.
    This maintains the original dummy protocol while Luminar is being developed.
    """
    from template.protocol import Dummy
    from template.validator.reward import get_rewards
    
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=Dummy(dummy_input=self.step),
        deserialize=True,
    )

    bt.logging.info(f"Received responses: {responses}")
    rewards = get_rewards(self, query=self.step, responses=responses)
    bt.logging.info(f"Scored responses: {rewards}")
    
    self.update_scores(rewards, miner_uids)
    time.sleep(5)
