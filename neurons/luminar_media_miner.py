# Enhanced Luminar Miner with Media Processing
# Copyright © 2025 Luminar Network
# Implements complete flow: User media → Validator → Miner → Event generation

import time
import torch
import hashlib
import asyncio
import bittensor as bt
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re

# AI/ML imports for visual processing
try:
    import clip
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    VISUAL_PROCESSING_AVAILABLE = True
except ImportError:
    VISUAL_PROCESSING_AVAILABLE = False
    bt.logging.warning("Visual processing libraries not installed. Using text-only mode.")

from template.protocol import (
    MediaProcessingRequest, UserSubmission, ProcessedEvent, 
    MediaUpload, ValidationResult
)
from template.base.miner import BaseMinerNeuron

class LuminarMediaMiner(BaseMinerNeuron):
    """
    Enhanced Luminar Miner that processes user submissions with media
    
    Flow Implementation:
    1. Receives user submissions from validator (text + media + metadata)
    2. Analyzes text content using NLP
    3. Processes images/videos using computer vision
    4. Combines analysis to generate structured events
    5. Filters duplicates and validates consistency
    6. Returns events for validator comparison with metadata
    """
    
    def __init__(self, config=None):
        super(LuminarMediaMiner, self).__init__(config=config)
        
        # Initialize AI models
        self._initialize_models()
        
        # Duplicate detection cache
        self.processed_events_cache = {}
        self.duplicate_threshold = 0.85
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "duplicates_detected": 0,
            "processing_errors": 0,
            "avg_processing_time": 0.0
        }
    
    def _initialize_models(self):
        """Initialize AI models for text and visual processing"""
        bt.logging.info("Initializing Luminar AI models...")
        
        # Text processing models
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            bt.logging.info("✅ Text embedding model loaded")
        except Exception as e:
            bt.logging.error(f"Failed to load text model: {e}")
            self.text_model = None
        
        # Visual processing models (if available)
        if VISUAL_PROCESSING_AVAILABLE:
            try:
                # CLIP for image understanding
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
                bt.logging.info("✅ CLIP visual model loaded")
                
                # BLIP for image captioning
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                bt.logging.info("✅ BLIP captioning model loaded")
                
            except Exception as e:
                bt.logging.error(f"Failed to load visual models: {e}")
                self.clip_model = None
                self.blip_model = None
        else:
            self.clip_model = None
            self.blip_model = None
    
    async def forward(self, synapse: MediaProcessingRequest) -> MediaProcessingRequest:
        """
        Main processing function for media submissions
        
        Your Flow:
        1. User uploads media + metadata → Validator
        2. Validator sends to this miner
        3. Miner processes text + visuals
        4. Creates structured events
        5. Filters duplicates
        6. Returns to validator for metadata comparison
        """
        bt.logging.info(f"🎯 Processing {len(synapse.user_submissions)} user submissions")
        
        start_time = time.time()
        processed_events = []
        duplicate_flags = []
        
        try:
            for submission in synapse.user_submissions:
                # Process each user submission
                event = await self._process_user_submission(submission)
                
                if event:
                    # Check for duplicates
                    is_duplicate = await self._check_duplicate(event, submission)
                    
                    if is_duplicate:
                        duplicate_flags.append(submission.submission_id)
                        bt.logging.debug(f"🔄 Duplicate detected: {submission.submission_id}")
                    else:
                        processed_events.append(event)
                        # Cache for future duplicate detection
                        self._cache_event(event)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(synapse.user_submissions), len(duplicate_flags), processing_time)
            
            # Fill response
            synapse.processed_events = processed_events
            synapse.duplicate_flags = duplicate_flags
            synapse.processing_time = processing_time
            synapse.processing_metadata = {
                "miner_uid": self.uid,
                "model_versions": self._get_model_versions(),
                "processing_stats": self.stats
            }
            
            bt.logging.info(f"✅ Processed {len(processed_events)} events, detected {len(duplicate_flags)} duplicates")
            
        except Exception as e:
            bt.logging.error(f"❌ Processing failed: {e}")
            self.stats["processing_errors"] += 1
            
        return synapse
    
    async def _process_user_submission(self, submission: UserSubmission) -> Optional[ProcessedEvent]:
        """
        Process a single user submission with text and media
        
        Returns structured event like: "truck and motorbike accident near balaju at 4pm, June 11"
        """
        try:
            # 1. Analyze text description
            text_analysis = await self._analyze_text(submission.text_description)
            
            # 2. Process media files (images/videos)
            visual_analysis = await self._analyze_media(submission.media_files)
            
            # 3. Combine text + visual analysis
            combined_analysis = self._combine_analysis(text_analysis, visual_analysis)
            
            # 4. Generate structured event
            event = self._generate_event(submission, combined_analysis)
            
            return event
            
        except Exception as e:
            bt.logging.error(f"Failed to process submission {submission.submission_id}: {e}")
            return None
    
    async def _analyze_text(self, text_description: str) -> Dict[str, Any]:
        """
        Analyze text description using NLP
        
        Extracts:
        - Event type (accident, theft, assault)
        - Entities (vehicles, locations, people)
        - Time references
        - Severity indicators
        """
        if not self.text_model:
            return {"event_type": "unknown", "entities": {}, "confidence": 0.5}
        
        # Basic entity extraction using patterns
        entities = {}
        
        # Vehicle detection
        vehicles = re.findall(r'\b(car|truck|bike|motorcycle|motorbike|bus|van|auto|taxi)\b', 
                             text_description.lower())
        if vehicles:
            entities["vehicles"] = list(set(vehicles))
        
        # Location detection (simple pattern)
        location_patterns = [
            r'\b(near|at|in|around)\s+([A-Za-z\s]{2,20})\b',
            r'\b([A-Za-z]{3,15})\s+(road|street|area|market|park|station)\b'
        ]
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text_description, re.IGNORECASE)
            locations.extend([match[1] if isinstance(match, tuple) else match for match in matches])
        
        if locations:
            entities["locations"] = list(set(locations))
        
        # Time extraction
        time_patterns = [
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\b(\d{1,2}:\d{2})\s*(am|pm)?\b',
            r'\b(morning|afternoon|evening|night)\b'
        ]
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text_description.lower())
            times.extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        if times:
            entities["time_references"] = list(set(times))
        
        # Event type classification
        event_type = self._classify_event_type(text_description)
        
        # Generate text embedding for similarity comparison
        try:
            embedding = self.text_model.encode(text_description)
            entities["text_embedding"] = embedding.tolist()
        except:
            entities["text_embedding"] = None
        
        return {
            "event_type": event_type,
            "entities": entities,
            "confidence": 0.8,  # TODO: Implement proper confidence scoring
            "original_text": text_description
        }
    
    def _classify_event_type(self, text: str) -> str:
        """Classify event type from text description"""
        text_lower = text.lower()
        
        # Define keywords for different event types
        keywords = {
            "accident": ["accident", "collision", "crash", "hit", "injured", "ambulance"],
            "theft": ["stolen", "theft", "robbed", "missing", "pickpocket", "burglary"],
            "assault": ["fight", "attacked", "assault", "beaten", "violence", "aggression"],
            "vandalism": ["damaged", "broken", "vandalism", "graffiti", "destroyed"],
            "suspicious": ["suspicious", "strange", "unusual", "concerning", "weird"],
            "fire": ["fire", "smoke", "burning", "flames", "firefighters"],
            "medical": ["medical", "emergency", "ambulance", "injured", "hurt", "pain"],
            "traffic": ["traffic", "jam", "blocked", "congestion", "stuck"]
        }
        
        # Score each event type
        scores = {}
        for event_type, keyword_list in keywords.items():
            score = sum(1 for keyword in keyword_list if keyword in text_lower)
            if score > 0:
                scores[event_type] = score
        
        # Return highest scoring event type
        if scores:
            return max(scores, key=scores.get)
        else:
            return "incident"  # Default
    
    async def _analyze_media(self, media_files: List[MediaUpload]) -> Dict[str, Any]:
        """
        Analyze media files (images/videos) using computer vision
        
        Returns analysis of visual content
        """
        if not media_files or not VISUAL_PROCESSING_AVAILABLE:
            return {"visual_content": None, "confidence": 0.0}
        
        visual_analysis = {
            "image_captions": [],
            "detected_objects": [],
            "scene_description": "",
            "confidence": 0.0
        }
        
        for media in media_files:
            if media.media_type == "image":
                analysis = await self._analyze_image(media)
                if analysis:
                    visual_analysis["image_captions"].append(analysis.get("caption", ""))
                    visual_analysis["detected_objects"].extend(analysis.get("objects", []))
        
        # Combine all captions into scene description
        if visual_analysis["image_captions"]:
            visual_analysis["scene_description"] = ". ".join(visual_analysis["image_captions"])
            visual_analysis["confidence"] = 0.8
        
        return visual_analysis
    
    async def _analyze_image(self, media: MediaUpload) -> Optional[Dict[str, Any]]:
        """Analyze single image using CLIP and BLIP"""
        if not self.blip_model:
            return None
        
        try:
            # Download image
            response = requests.get(media.media_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Generate caption using BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # TODO: Add object detection using CLIP
            # For now, extract simple objects from caption
            objects = self._extract_objects_from_caption(caption)
            
            return {
                "caption": caption,
                "objects": objects,
                "media_id": media.media_id
            }
            
        except Exception as e:
            bt.logging.error(f"Failed to analyze image {media.media_id}: {e}")
            return None
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects from image caption"""
        # Simple noun extraction (can be improved with proper NLP)
        common_objects = [
            "car", "truck", "motorcycle", "bike", "bus", "person", "people",
            "road", "street", "building", "tree", "sign", "light", "vehicle"
        ]
        
        found_objects = []
        caption_lower = caption.lower()
        
        for obj in common_objects:
            if obj in caption_lower:
                found_objects.append(obj)
        
        return found_objects
    
    def _combine_analysis(self, text_analysis: Dict[str, Any], visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine text and visual analysis to create comprehensive understanding
        """
        combined = {
            "event_type": text_analysis.get("event_type", "incident"),
            "confidence": (text_analysis.get("confidence", 0.5) + visual_analysis.get("confidence", 0.0)) / 2,
            "entities": text_analysis.get("entities", {}),
            "visual_content": visual_analysis.get("scene_description", ""),
            "all_objects": text_analysis.get("entities", {}).get("vehicles", []) + visual_analysis.get("detected_objects", [])
        }
        
        # Enhance event type if visual evidence supports it
        visual_desc = visual_analysis.get("scene_description", "").lower()
        if "accident" in visual_desc or "crash" in visual_desc:
            if combined["event_type"] in ["incident", "unknown"]:
                combined["event_type"] = "accident"
                combined["confidence"] = min(0.9, combined["confidence"] + 0.2)
        
        return combined
    
    def _generate_event(self, submission: UserSubmission, analysis: Dict[str, Any]) -> ProcessedEvent:
        """
        Generate structured event like: "truck and motorbike accident near balaju at 4pm, June 11"
        """
        # Extract components
        event_type = analysis.get("event_type", "incident")
        entities = analysis.get("entities", {})
        visual_content = analysis.get("visual_content", "")
        
        # Build natural language summary
        summary_parts = []
        
        # Add vehicles/objects
        vehicles = entities.get("vehicles", []) + analysis.get("all_objects", [])
        if vehicles:
            unique_vehicles = list(set(vehicles))
            if len(unique_vehicles) == 1:
                summary_parts.append(unique_vehicles[0])
            else:
                summary_parts.append(" and ".join(unique_vehicles))
        
        # Add event type
        summary_parts.append(event_type)
        
        # Add location
        locations = entities.get("locations", [])
        if locations:
            summary_parts.append(f"near {locations[0]}")
        elif submission.geotag:
            # TODO: Reverse geocode coordinates to location name
            summary_parts.append(f"at coordinates ({submission.geotag['lat']:.4f}, {submission.geotag['lng']:.4f})")
        
        # Add time
        time_refs = entities.get("time_references", [])
        if time_refs:
            summary_parts.append(f"at {time_refs[0]}")
        elif submission.timestamp:
            time_str = submission.timestamp.strftime("%I%p, %B %d")
            summary_parts.append(f"on {time_str}")
        
        # Combine into natural summary
        generated_summary = " ".join(summary_parts)
        
        # Create ProcessedEvent
        event = ProcessedEvent(
            event_id=f"EVT_{int(time.time())}_{submission.submission_id[:8]}",
            generated_summary=generated_summary,
            event_type=event_type,
            confidence_score=analysis.get("confidence", 0.5),
            extracted_entities=entities,
            visual_analysis=analysis.get("visual_content", ""),
            source_submissions=[submission.submission_id],
            processing_timestamp=datetime.now(),
            miner_uid=self.uid
        )
        
        return event
    
    async def _check_duplicate(self, event: ProcessedEvent, submission: UserSubmission) -> bool:
        """
        Check if this event is a duplicate of previously processed events
        
        Uses:
        - Text similarity
        - Geographic proximity
        - Time proximity
        """
        if not self.processed_events_cache:
            return False
        
        # Check against recent events (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for cached_event_id, cached_data in self.processed_events_cache.items():
            cached_event = cached_data["event"]
            cached_submission = cached_data["submission"]
            
            # Skip old events
            if cached_event.processing_timestamp < cutoff_time:
                continue
            
            # Calculate similarity scores
            text_similarity = self._calculate_text_similarity(
                event.generated_summary, cached_event.generated_summary
            )
            
            geo_similarity = self._calculate_geo_similarity(
                submission.geotag, cached_submission.geotag
            )
            
            time_similarity = self._calculate_time_similarity(
                submission.timestamp, cached_submission.timestamp
            )
            
            # Combined similarity score
            overall_similarity = (text_similarity * 0.5 + geo_similarity * 0.3 + time_similarity * 0.2)
            
            if overall_similarity > self.duplicate_threshold:
                bt.logging.debug(f"Duplicate detected: {overall_similarity:.3f} similarity")
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using embeddings or simple comparison"""
        if not self.text_model:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        try:
            emb1 = self.text_model.encode(text1)
            emb2 = self.text_model.encode(text2)
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_geo_similarity(self, geo1: Dict[str, float], geo2: Dict[str, float]) -> float:
        """Calculate geographic similarity (1.0 if within 100m, 0.0 if > 1km)"""
        if not geo1 or not geo2:
            return 0.0
        
        from geopy.distance import geodesic
        
        try:
            point1 = (geo1["lat"], geo1["lng"])
            point2 = (geo2["lat"], geo2["lng"])
            distance_km = geodesic(point1, point2).kilometers
            
            if distance_km <= 0.1:  # Within 100m
                return 1.0
            elif distance_km >= 1.0:  # Over 1km
                return 0.0
            else:
                return 1.0 - (distance_km / 1.0)  # Linear decay
                
        except:
            return 0.0
    
    def _calculate_time_similarity(self, time1: datetime, time2: datetime) -> float:
        """Calculate time similarity (1.0 if within 30 min, 0.0 if > 6 hours)"""
        if not time1 or not time2:
            return 0.0
        
        time_diff = abs((time1 - time2).total_seconds()) / 60  # minutes
        
        if time_diff <= 30:  # Within 30 minutes
            return 1.0
        elif time_diff >= 360:  # Over 6 hours
            return 0.0
        else:
            return 1.0 - (time_diff / 360)  # Linear decay
    
    def _cache_event(self, event: ProcessedEvent):
        """Cache event for duplicate detection"""
        # Keep only last 100 events to prevent memory issues
        if len(self.processed_events_cache) > 100:
            # Remove oldest event
            oldest_key = min(self.processed_events_cache.keys())
            del self.processed_events_cache[oldest_key]
        
        self.processed_events_cache[event.event_id] = {
            "event": event,
            "submission": None,  # Would need to pass this
            "timestamp": datetime.now()
        }
    
    def _update_stats(self, total_submissions: int, duplicates: int, processing_time: float):
        """Update processing statistics"""
        self.stats["total_processed"] += total_submissions
        self.stats["duplicates_detected"] += duplicates
        
        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (current_avg + processing_time) / 2
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of loaded models"""
        versions = {}
        
        if self.text_model:
            versions["text_model"] = "all-MiniLM-L6-v2"
        
        if self.clip_model:
            versions["clip_model"] = "ViT-B/32"
        
        if self.blip_model:
            versions["blip_model"] = "Salesforce/blip-image-captioning-base"
        
        return versions

# The main function parses the configuration and runs the miner.
if __name__ == "__main__":
    with LuminarMediaMiner() as miner:
        while True:
            bt.logging.info("🎯 Luminar Media Miner is running...")
            time.sleep(60)
