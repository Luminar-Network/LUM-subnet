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

import numpy as np
from typing import List, Dict, Any
import bittensor as bt
from template.protocol import DataProcessingRequest, CrimeEvent, RawIncidentReport


def calculate_data_integrity_reward(
    raw_reports: List[RawIncidentReport], 
    crime_events: List[CrimeEvent]
) -> float:
    """
    Calculate reward based on data integrity verification.
    
    Checks:
    - Consistency of timestamps and geotags
    - Media authenticity verification
    - Source report validation
    
    Weight: 30% of total score
    """
    if not crime_events:
        return 0.0
    
    integrity_scores = []
    
    for event in crime_events:
        score = 0.0
        
        # Verify source reports exist
        valid_sources = sum(1 for report_id in event.source_reports 
                          if any(r.report_id == report_id for r in raw_reports))
        if event.source_reports:
            score += (valid_sources / len(event.source_reports)) * 0.4
        
        # Check geographic consistency
        source_reports = [r for r in raw_reports if r.report_id in event.source_reports]
        if source_reports and event.location.get("coordinates"):
            geo_score = verify_geographic_consistency(source_reports, event.location["coordinates"])
            score += geo_score * 0.3
        
        # Check temporal consistency  
        if source_reports and event.timestamp_range:
            temporal_score = verify_temporal_consistency(source_reports, event.timestamp_range)
            score += temporal_score * 0.3
        
        integrity_scores.append(score)
    
    return np.mean(integrity_scores) if integrity_scores else 0.0


def calculate_clustering_accuracy_reward(
    raw_reports: List[RawIncidentReport],
    crime_events: List[CrimeEvent]
) -> float:
    """
    Calculate reward based on clustering accuracy.
    
    Checks:
    - Semantic similarity of clustered reports
    - Appropriate event type classification
    - Clustering completeness
    
    Weight: 40% of total score
    """
    if not crime_events:
        return 0.0
    
    clustering_scores = []
    
    for event in crime_events:
        score = 0.0
        source_reports = [r for r in raw_reports if r.report_id in event.source_reports]
        
        if not source_reports:
            clustering_scores.append(0.0)
            continue
        
        # Semantic similarity (40% of clustering score)
        if len(source_reports) > 1:
            similarity = calculate_semantic_similarity(source_reports)
            score += similarity * 0.4
        else:
            score += 0.4  # Single reports are perfectly consistent
        
        # Event type classification accuracy (30% of clustering score)
        type_accuracy = verify_event_classification(source_reports, event.event_type)
        score += type_accuracy * 0.3
        
        # Clustering completeness (30% of clustering score)
        completeness = assess_clustering_completeness(raw_reports, event)
        score += completeness * 0.3
        
        clustering_scores.append(score)
    
    return np.mean(clustering_scores) if clustering_scores else 0.0


def calculate_novelty_reward(crime_events: List[CrimeEvent], historical_events: List[CrimeEvent] = None) -> float:
    """
    Calculate reward based on novelty and redundancy detection.
    
    Checks:
    - Duplicate detection
    - Novel event identification
    - Quality over quantity
    
    Weight: 20% of total score
    """
    if not crime_events:
        return 0.0
    
    novelty_scores = []
    
    for event in crime_events:
        score = 1.0  # Start with perfect novelty
        
        # Penalty for low confidence events
        if event.confidence_score < 0.3:
            score -= 0.3
        
        # Bonus for multi-source events (more reliable)
        if len(event.source_reports) > 1:
            score += 0.1
        
        # Check for duplicates within current batch
        duplicates = sum(1 for other in crime_events 
                        if other != event and events_are_similar(event, other))
        if duplicates > 0:
            score -= duplicates * 0.2
        
        # TODO: Check against historical events database
        # if historical_events:
        #     historical_duplicates = sum(1 for hist_event in historical_events 
        #                                if events_are_similar(event, hist_event))
        #     score -= historical_duplicates * 0.1
        
        novelty_scores.append(max(0.0, min(1.0, score)))
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def calculate_processing_quality_reward(
    crime_events: List[CrimeEvent],
    processing_metadata: Dict[str, Any]
) -> float:
    """
    Calculate reward based on processing quality.
    
    Checks:
    - Summary quality and completeness
    - Metadata richness
    - Processing efficiency
    
    Weight: 10% of total score
    """
    if not crime_events:
        return 0.0
    
    quality_scores = []
    
    for event in crime_events:
        score = 0.0
        
        # Summary quality (40% of quality score)
        if event.summary_tag and len(event.summary_tag.strip()) > 5:
            score += 0.2
        if event.clustered_summary and len(event.clustered_summary.strip()) > 20:
            score += 0.2
        
        # Metadata completeness (40% of quality score)
        required_fields = ['event_id', 'event_type', 'location', 'timestamp_range', 'confidence_score']
        complete_fields = sum(1 for field in required_fields 
                            if getattr(event, field, None) is not None)
        score += (complete_fields / len(required_fields)) * 0.4
        
        # Confidence score reasonableness (20% of quality score)
        if 0.0 <= event.confidence_score <= 1.0:
            if 0.1 <= event.confidence_score <= 0.9:  # Reasonable confidence range
                score += 0.2
            else:
                score += 0.1  # Too low or too high confidence
        
        quality_scores.append(score)
    
    # Processing efficiency bonus
    efficiency_bonus = 0.0
    processing_time = processing_metadata.get("processing_time", float('inf'))
    if processing_time < 30:  # Very fast
        efficiency_bonus = 0.1
    elif processing_time < 60:  # Reasonable
        efficiency_bonus = 0.05
    
    base_quality = np.mean(quality_scores) if quality_scores else 0.0
    return min(1.0, base_quality + efficiency_bonus)


def get_luminar_rewards(
    raw_reports: List[RawIncidentReport],
    responses: List[DataProcessingRequest],
    historical_events: List[CrimeEvent] = None
) -> np.ndarray:
    """
    Calculate comprehensive rewards for Luminar subnet miners.
    
    Combines all reward components with appropriate weights:
    - Data Integrity: 30%
    - Clustering Accuracy: 40%  
    - Novelty & Redundancy: 20%
    - Processing Quality: 10%
    
    Args:
        raw_reports: Original incident reports sent to miners
        responses: Miner responses with processed crime events
        historical_events: Previously verified events for duplicate detection
        
    Returns:
        np.ndarray: Reward scores for each miner response
    """
    rewards = []
    
    for response in responses:
        if not response.processed_events:
            rewards.append(0.0)
            continue
        
        try:
            # Calculate component rewards
            integrity_reward = calculate_data_integrity_reward(raw_reports, response.processed_events)
            clustering_reward = calculate_clustering_accuracy_reward(raw_reports, response.processed_events)
            novelty_reward = calculate_novelty_reward(response.processed_events, historical_events)
            quality_reward = calculate_processing_quality_reward(
                response.processed_events, 
                response.miner_metadata or {}
            )
            
            # Combine with weights
            total_reward = (
                integrity_reward * 0.30 +
                clustering_reward * 0.40 +
                novelty_reward * 0.20 +
                quality_reward * 0.10
            )
            
            # Apply performance modifiers
            total_reward = apply_performance_modifiers(total_reward, response)
            
            rewards.append(total_reward)
            
            bt.logging.info(
                f"Miner reward breakdown - "
                f"Integrity: {integrity_reward:.3f}, "
                f"Clustering: {clustering_reward:.3f}, "
                f"Novelty: {novelty_reward:.3f}, "
                f"Quality: {quality_reward:.3f}, "
                f"Total: {total_reward:.3f}"
            )
            
        except Exception as e:
            bt.logging.error(f"Error calculating reward: {e}")
            rewards.append(0.0)
    
    return np.array(rewards)


# Helper functions

def verify_geographic_consistency(reports: List[RawIncidentReport], event_coords: Dict) -> float:
    """Verify geographic consistency between reports and event location."""
    if not reports or not event_coords:
        return 0.0
    
    event_lat, event_lng = event_coords.get("lat", 0), event_coords.get("lng", 0)
    distances = []
    
    for report in reports:
        lat_diff = abs(report.geotag["lat"] - event_lat)
        lng_diff = abs(report.geotag["lng"] - event_lng)
        distance = (lat_diff**2 + lng_diff**2)**0.5
        distances.append(distance)
    
    avg_distance = np.mean(distances)
    return max(0.0, 1.0 - (avg_distance / 0.05))  # 0.05 degrees ~ 5km threshold


def verify_temporal_consistency(reports: List[RawIncidentReport], time_range: Dict) -> float:
    """Verify temporal consistency between reports and event time range."""
    if not reports or not time_range:
        return 0.0
    
    report_times = [r.timestamp for r in reports]
    min_time, max_time = min(report_times), max(report_times)
    
    range_start = time_range.get("start", min_time)
    range_end = time_range.get("end", max_time)
    
    if range_start <= min_time and range_end >= max_time:
        range_duration = (range_end - range_start).total_seconds()
        reasonable_duration = 86400  # 24 hours
        
        if range_duration <= reasonable_duration:
            return 1.0
        else:
            return max(0.0, 1.0 - ((range_duration - reasonable_duration) / reasonable_duration))
    
    return 0.5


def calculate_semantic_similarity(reports: List[RawIncidentReport]) -> float:
    """Calculate semantic similarity between reports (simplified implementation)."""
    if len(reports) <= 1:
        return 1.0
    
    descriptions = [r.text_description.lower() for r in reports]
    word_sets = [set(desc.split()) for desc in descriptions]
    
    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
    
    return np.mean(similarities) if similarities else 0.0


def verify_event_classification(reports: List[RawIncidentReport], event_type: str) -> float:
    """Verify if event type classification matches report content."""
    if not reports:
        return 0.0
    
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
    
    expected_keywords = crime_keywords.get(event_type.lower(), [])
    if not expected_keywords:
        return 0.8  # Reasonable default for "other" category
    
    combined_text = " ".join(r.text_description.lower() for r in reports)
    matches = sum(1 for keyword in expected_keywords if keyword in combined_text)
    
    return min(1.0, matches / max(1, len(expected_keywords) * 0.5))


def assess_clustering_completeness(all_reports: List[RawIncidentReport], event: CrimeEvent) -> float:
    """Assess if similar reports are properly included in the cluster."""
    # Simplified implementation - would be more sophisticated in production
    return 1.0  # Placeholder


def events_are_similar(event1: CrimeEvent, event2: CrimeEvent) -> bool:
    """Check if two events are duplicates or very similar."""
    if not event1.location or not event2.location:
        return False
    
    coords1 = event1.location.get("coordinates", {})
    coords2 = event2.location.get("coordinates", {})
    
    if not coords1 or not coords2:
        return False
    
    # Geographic similarity
    lat_diff = abs(coords1.get("lat", 0) - coords2.get("lat", 0))
    lng_diff = abs(coords1.get("lng", 0) - coords2.get("lng", 0))
    geo_distance = (lat_diff**2 + lng_diff**2)**0.5
    
    # Temporal similarity
    time1 = event1.timestamp_range.get("start") if event1.timestamp_range else None
    time2 = event2.timestamp_range.get("start") if event2.timestamp_range else None
    
    if time1 and time2:
        time_diff = abs((time1 - time2).total_seconds())
        return geo_distance < 0.01 and time_diff < 3600  # ~1km and 1 hour
    
    return geo_distance < 0.01


def apply_performance_modifiers(base_reward: float, response: DataProcessingRequest) -> float:
    """Apply additional performance modifiers to the base reward."""
    modified_reward = base_reward
    
    # Processing time modifier
    if response.processing_time:
        if response.processing_time > 300:  # > 5 minutes
            modified_reward *= 0.8  # Penalty for slow processing
        elif response.processing_time < 10:  # < 10 seconds
            modified_reward *= 1.1  # Bonus for fast processing
    
    # Event count modifier (quality over quantity)
    if response.processed_events:
        event_count = len(response.processed_events)
        if event_count > 50:  # Too many events might indicate poor clustering
            modified_reward *= 0.9
        elif 5 <= event_count <= 20:  # Good clustering range
            modified_reward *= 1.05
    
    return min(1.0, max(0.0, modified_reward))
