# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Luminar Network
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
import torch
import hashlib
import asyncio
import typing
import bittensor as bt
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import sys
import os

# Add template path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# AI/ML imports for visual processing
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    bt.logging.warning("CLIP not available - using mock processing")
    
    class MockCLIP:
        @staticmethod
        def load(model_name, device="cpu"):
            return None, None
        @staticmethod
        def tokenize(text):
            return torch.zeros(1, 77)
    clip = MockCLIP()

# OpenAI integration for enhanced text and image analysis
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    bt.logging.warning("OpenAI not available - using rule-based processing")

# Visual processing availability
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    VISUAL_PROCESSING_AVAILABLE = True
except ImportError:
    VISUAL_PROCESSING_AVAILABLE = False
    bt.logging.warning("BLIP not available - using mock visual processing")

# Bittensor Miner Template:
import template

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron

# Import Luminar protocol
try:
    from template.protocol import (
        MediaProcessingRequest, UserSubmission, ProcessedEvent, 
        MediaUpload, ValidationResult, Dummy
    )
    LUMINAR_PROTOCOL_AVAILABLE = True
except ImportError:
    # Fallback for dummy protocol if MediaProcessingRequest not available
    from template.protocol import Dummy as MediaProcessingRequest
    LUMINAR_PROTOCOL_AVAILABLE = False
    bt.logging.warning("Using dummy protocol - MediaProcessingRequest not available")
    
    # Mock classes for testing
    class UserSubmission:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ProcessedEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MediaUpload:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class Miner(BaseMinerNeuron):
    """
    Luminar Media Processing Miner
    
    Processes user submissions with media (images/videos) and text to generate structured events.
    
    Flow Implementation:
    1. Receives user submissions from validator (text + media + metadata)
    2. Analyzes text content using NLP
    3. Processes images/videos using computer vision
    4. Combines analysis to generate structured events
    5. Filters duplicates and validates consistency
    6. Returns events for validator comparison with metadata
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        bt.logging.info("🚀 Initializing Luminar Media Processing Miner")
        
        # Initialize OpenAI client
        self._initialize_openai()
        
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
        
        bt.logging.info("✅ Luminar Miner initialized successfully")
    
    def _initialize_openai(self):
        """Initialize OpenAI client for enhanced text and image analysis"""
        if not OPENAI_AVAILABLE:
            bt.logging.info("🤖 OpenAI: Not available - using rule-based processing")
            self.openai_client = None
            return
        
        try:
            # Get OpenAI API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                bt.logging.warning("🔑 OPENAI_API_KEY not found in environment")
                self.openai_client = None
                return
            
            # Initialize OpenAI client
            openai.api_key = api_key
            self.openai_client = openai
            
            # Configuration from environment
            self.openai_config = {
                "model": os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                "vision_model": os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview'),
                "max_tokens": int(os.getenv('OPENAI_MAX_TOKENS', 500)),
                "temperature": float(os.getenv('OPENAI_TEMPERATURE', 0.3))
            }
            
            bt.logging.info(f"🤖 OpenAI initialized: {self.openai_config['model']}")
            
        except Exception as e:
            bt.logging.error(f"Failed to initialize OpenAI: {e}")
            self.openai_client = None
    
    def _initialize_models(self):
        """Initialize AI models for text and visual processing"""
        bt.logging.info("🤖 Initializing Luminar AI models...")
        
        # Text processing models
        try:
            # In production, you'd initialize actual NLP models here
            # For now, using rule-based processing
            self.text_model = None
            bt.logging.info("📝 Text processing initialized (rule-based)")
        except Exception as e:
            bt.logging.error(f"Failed to initialize text models: {e}")
            self.text_model = None
        
        # Visual processing models (if available)
        if VISUAL_PROCESSING_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                bt.logging.info("👁️ Visual processing initialized (BLIP)")
            except Exception as e:
                bt.logging.error(f"Failed to initialize visual models: {e}")
                self.blip_model = None
                self.blip_processor = None
        else:
            self.blip_model = None
            self.blip_processor = None
            bt.logging.info("👁️ Visual processing: Mock mode (BLIP not available)")

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Main processing function for media submissions or dummy requests
        
        Handles both:
        1. MediaProcessingRequest - Your Luminar flow
        2. Dummy requests - Standard Bittensor template compatibility
        """
        # Check if this is a Luminar MediaProcessingRequest
        if hasattr(synapse, 'user_submissions'):
            return await self._process_media_request(synapse)
        
        # Fallback to dummy processing for template compatibility
        elif hasattr(synapse, 'dummy_input'):
            bt.logging.info("Processing dummy request (template compatibility)")
            synapse.dummy_output = synapse.dummy_input * 2
            return synapse
        
        else:
            bt.logging.warning(f"Unknown synapse type: {type(synapse)}")
            return synapse
    
    async def _process_media_request(self, synapse) -> Any:
        """
        Process Luminar media submissions
        
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
                # Process each submission
                event = await self._process_user_submission(submission)
                
                if event:
                    # Check for duplicates
                    is_duplicate = await self._check_duplicate(event, submission)
                    
                    processed_events.append(event)
                    duplicate_flags.append(is_duplicate)
                    
                    if is_duplicate:
                        bt.logging.info(f"🔄 Duplicate detected for: {submission.submission_id}")
            
            # Update synapse with results
            synapse.processed_events = processed_events
            synapse.processing_complete = True
            synapse.processing_metadata = {
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "events_generated": len(processed_events),
                "duplicates_detected": sum(duplicate_flags),
                "model_versions": self._get_model_versions()
            }
            
            # Update stats
            self._update_stats(len(synapse.user_submissions), sum(duplicate_flags), time.time() - start_time)
            
            bt.logging.info(f"✅ Processing complete: {len(processed_events)} events generated")
            
        except Exception as e:
            bt.logging.error(f"❌ Processing error: {e}")
            synapse.processing_complete = False
            synapse.processing_metadata = {
                "error": str(e),
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            
        return synapse
    
    async def _process_user_submission(self, submission) -> Optional[Any]:
        """
        Process a single user submission with text and media
        
        Returns structured event like: "truck and motorbike accident near balaju at 4pm, June 11"
        """
        try:
            bt.logging.debug(f"📝 Processing submission: {submission.submission_id}")
            
            # Analyze text description
            text_analysis = await self._analyze_text(submission.text_description)
            
            # Analyze media files if present
            visual_analysis = await self._analyze_media(getattr(submission, 'media_files', []))
            
            # Combine analysis
            combined_analysis = self._combine_analysis(text_analysis, visual_analysis)
            
            # Generate structured event
            event = self._generate_event(submission, combined_analysis)
            
            return event
            
        except Exception as e:
            bt.logging.error(f"Error processing submission {submission.submission_id}: {e}")
            return None
    
    async def _analyze_text(self, text_description: str) -> Dict[str, Any]:
        """
        Analyze text description using OpenAI or rule-based NLP
        
        Extracts:
        - Event type (accident, theft, assault)
        - Entities (vehicles, locations, people)
        - Time references
        - Severity indicators
        """
        # Try OpenAI first for enhanced analysis
        if self.openai_client:
            try:
                return await self._analyze_text_with_openai(text_description)
            except Exception as e:
                bt.logging.warning(f"OpenAI analysis failed, falling back to rule-based: {e}")
        
        # Fallback to rule-based analysis
        return await self._analyze_text_rule_based(text_description)
    
    async def _analyze_text_with_openai(self, text_description: str) -> Dict[str, Any]:
        """Enhanced text analysis using OpenAI GPT models"""
        
        system_prompt = """You are an expert incident analyzer. Analyze the provided incident report and extract structured information.

        Extract and return ONLY a JSON object with these fields:
        {
            "event_type": "accident|theft|assault|vandalism|suspicious|fire|medical|traffic|incident",
            "entities": {
                "vehicles": ["list of vehicles mentioned"],
                "locations": ["list of locations mentioned"],
                "times": ["list of times/temporal references"],
                "people": ["list of people or groups mentioned"]
            },
            "severity": "low|medium|high",
            "confidence": 0.0-1.0,
            "summary": "concise incident summary"
        }
        
        Focus on factual extraction, not speculation."""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=self.openai_config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Incident report: {text_description}"}
                ],
                max_tokens=self.openai_config["max_tokens"],
                temperature=self.openai_config["temperature"]
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            import json
            analysis = json.loads(content)
            
            # Add original text and embedding
            analysis["original_text"] = text_description
            analysis["text_embedding"] = hashlib.md5(text_description.encode()).hexdigest()
            
            bt.logging.debug(f"🤖 OpenAI analysis: {analysis['event_type']} (confidence: {analysis.get('confidence', 0.0)})")
            return analysis
            
        except Exception as e:
            bt.logging.error(f"OpenAI text analysis error: {e}")
            raise
    
    async def _analyze_text_rule_based(self, text_description: str) -> Dict[str, Any]:
        """Fallback rule-based text analysis using regex patterns"""
        # Basic entity extraction using patterns
        entities = {}
        confidence_factors = []
        
        # Vehicle detection
        vehicles = re.findall(r'\b(car|truck|bike|motorcycle|motorbike|bus|van|auto|taxi)\b', 
                             text_description.lower())
        if vehicles:
            entities["vehicles"] = list(set(vehicles))
            confidence_factors.append(0.2)  # Vehicles increase confidence
        
        # Location detection (simple pattern)
        location_patterns = [
            r'\b(near|at|in|around)\s+([A-Za-z\s]{2,20})\b',
            r'\b([A-Za-z]{3,15})\s+(road|street|area|market|park|station)\b'
        ]
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text_description.lower())
            locations.extend([match[1] if isinstance(match, tuple) else match for match in matches])
        
        if locations:
            entities["locations"] = list(set(locations))
            confidence_factors.append(0.15)  # Locations increase confidence
        
        # Time extraction
        time_patterns = [
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\b(\d{1,2}:\d{2})\s*(am|pm)?\b',
            r'\b(morning|afternoon|evening|night)\b'
        ]
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text_description.lower())
            times.extend([match[0] if isinstance(match, tuple) else match for match in matches])
        
        if times:
            entities["times"] = list(set(times))
            confidence_factors.append(0.1)  # Time references increase confidence
        
        # Event type classification
        event_type, type_confidence = self._classify_event_type_with_confidence(text_description)
        confidence_factors.append(type_confidence)
        
        # Text quality factors
        text_length = len(text_description.split())
        if text_length > 10:
            confidence_factors.append(0.1)  # Longer descriptions are more detailed
        elif text_length < 3:
            confidence_factors.append(-0.2)  # Very short descriptions are less reliable
        
        # Calculate final confidence
        base_confidence = 0.4  # Base confidence for rule-based analysis
        final_confidence = min(0.95, base_confidence + sum(confidence_factors))
        
        # Generate text embedding for similarity comparison
        try:
            # In production, you'd use actual embeddings
            text_embedding = hashlib.md5(text_description.encode()).hexdigest()
        except:
            text_embedding = None
        
        return {
            "event_type": event_type,
            "entities": entities,
            "confidence": final_confidence,
            "text_embedding": text_embedding,
            "original_text": text_description
        }
    
    def _classify_event_type(self, text: str) -> str:
        """Classify event type from text description"""
        event_type, _ = self._classify_event_type_with_confidence(text)
        return event_type
    
    def _classify_event_type_with_confidence(self, text: str) -> tuple[str, float]:
        """Classify event type from text description with confidence score"""
        text_lower = text.lower()
        
        # Define keywords for different event types with weights
        keywords = {
            "accident": {
                "strong": ["accident", "collision", "crash", "hit", "injured", "ambulance"],
                "weak": ["bump", "scrape", "damage"]
            },
            "theft": {
                "strong": ["stolen", "theft", "robbed", "burglary"],
                "weak": ["missing", "pickpocket", "lost"]
            },
            "assault": {
                "strong": ["fight", "attacked", "assault", "beaten", "violence"],
                "weak": ["aggression", "confrontation", "argument"]
            },
            "vandalism": {
                "strong": ["vandalism", "graffiti", "destroyed"],
                "weak": ["damaged", "broken", "defaced"]
            },
            "suspicious": {
                "strong": ["suspicious", "concerning"],
                "weak": ["strange", "unusual", "weird"]
            },
            "fire": {
                "strong": ["fire", "burning", "flames", "firefighters"],
                "weak": ["smoke", "smoldering"]
            },
            "medical": {
                "strong": ["medical emergency", "ambulance", "injured", "hurt"],
                "weak": ["pain", "unwell", "sick"]
            },
            "traffic": {
                "strong": ["traffic jam", "blocked", "congestion"],
                "weak": ["stuck", "slow", "heavy traffic"]
            }
        }
        
        # Score each event type with weighted confidence
        scores = {}
        confidences = {}
        
        for event_type, keyword_categories in keywords.items():
            strong_matches = sum(1 for keyword in keyword_categories["strong"] if keyword in text_lower)
            weak_matches = sum(1 for keyword in keyword_categories["weak"] if keyword in text_lower)
            
            total_score = strong_matches * 2 + weak_matches  # Strong keywords worth more
            if total_score > 0:
                scores[event_type] = total_score
                # Confidence based on match strength and quantity
                confidence = min(0.9, 0.3 + (strong_matches * 0.2) + (weak_matches * 0.1))
                confidences[event_type] = confidence
        
        # Return highest scoring event type with its confidence
        if scores:
            best_event = max(scores, key=scores.get)
            return best_event, confidences[best_event]
        else:
            return "incident", 0.2  # Low confidence for generic classification
    
    async def _analyze_media(self, media_files: List[Any]) -> Dict[str, Any]:
        """
        Analyze media files (images/videos) using computer vision
        
        Returns analysis of visual content
        """
        if not media_files or not VISUAL_PROCESSING_AVAILABLE:
            return {
                "image_captions": [],
                "detected_objects": [],
                "scene_description": "",
                "confidence": 0.0,
                "analysis_quality": "no_media"
            }
        
        visual_analysis = {
            "image_captions": [],
            "detected_objects": [],
            "scene_description": "",
            "confidence": 0.0,
            "analysis_quality": "processing"
        }
        
        for media in media_files:
            if hasattr(media, 'media_type') and media.media_type.startswith('image/'):
                image_analysis = await self._analyze_image(media)
                if image_analysis:
                    visual_analysis["image_captions"].append(image_analysis["caption"])
                    visual_analysis["detected_objects"].extend(image_analysis["objects"])
                    visual_analysis["confidence"] = max(visual_analysis["confidence"], image_analysis["confidence"])
        
        # Combine all captions into scene description
        if visual_analysis["image_captions"]:
            visual_analysis["scene_description"] = ". ".join(visual_analysis["image_captions"])
        
        return visual_analysis
    
    async def _analyze_image(self, media: Any) -> Optional[Dict[str, Any]]:
        """Analyze single image using OpenAI Vision or BLIP"""
        
        # Try OpenAI Vision first for enhanced analysis
        if self.openai_client:
            try:
                return await self._analyze_image_with_openai(media)
            except Exception as e:
                bt.logging.warning(f"OpenAI Vision analysis failed, falling back to BLIP: {e}")
        
        # Fallback to BLIP analysis
        return await self._analyze_image_with_blip(media)
    
    async def _analyze_image_with_openai(self, media: Any) -> Optional[Dict[str, Any]]:
        """Enhanced image analysis using OpenAI GPT-4 Vision"""
        
        try:
            # Download image
            response = requests.get(media.url, timeout=10)
            image_data = response.content
            
            # Convert to base64 for OpenAI
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            system_prompt = """You are an expert incident scene analyzer. Analyze this image and extract structured information.

            Return ONLY a JSON object with these fields:
            {
                "caption": "detailed description of what you see",
                "objects": ["list of objects/vehicles/people detected"],
                "incident_type": "accident|theft|assault|vandalism|suspicious|fire|medical|traffic|normal",
                "confidence": 0.0-1.0,
                "safety_concern": "none|low|medium|high",
                "location_type": "street|building|parking|residential|commercial|unknown"
            }
            
            Focus on safety-relevant details and factual observations."""
            
            vision_response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=self.openai_config["vision_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this incident scene image:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "low"  # Use low detail for faster processing
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Parse JSON response
            content = vision_response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            import json
            vision_analysis = json.loads(content)
            
            bt.logging.debug(f"👁️ OpenAI Vision: {vision_analysis.get('incident_type', 'unknown')} (confidence: {vision_analysis.get('confidence', 0.0)})")
            
            # Use OpenAI's confidence if available, otherwise calculate based on response quality
            openai_confidence = vision_analysis.get("confidence", 0.0)
            if openai_confidence > 0:
                final_confidence = openai_confidence
            else:
                # Calculate confidence based on response completeness
                caption_length = len(vision_analysis.get("caption", "").split())
                objects_count = len(vision_analysis.get("objects", []))
                final_confidence = min(0.95, 0.7 + (caption_length * 0.01) + (objects_count * 0.05))
            
            return {
                "caption": vision_analysis.get("caption", ""),
                "objects": vision_analysis.get("objects", []),
                "confidence": final_confidence,
                "incident_type": vision_analysis.get("incident_type", "unknown"),
                "safety_concern": vision_analysis.get("safety_concern", "unknown"),
                "analysis_quality": "openai_vision"
            }
            
        except Exception as e:
            bt.logging.error(f"OpenAI Vision analysis error: {e}")
            raise
    
    async def _analyze_image_with_blip(self, media: Any) -> Optional[Dict[str, Any]]:
        """Fallback image analysis using BLIP"""
        if not self.blip_model:
            # Mock analysis for testing - lower confidence for mock
            return {
                "caption": f"Image content from {getattr(media, 'filename', 'unknown')}",
                "objects": ["vehicle", "road"],
                "confidence": 0.3,  # Lower confidence for mock analysis
                "analysis_quality": "mock"
            }
        
        try:
            # Download and process image
            response = requests.get(media.url, timeout=10)
            image = Image.open(BytesIO(response.content))
            
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Extract objects from caption
            objects = self._extract_objects_from_caption(caption)
            
            # Calculate confidence based on caption quality and object detection
            caption_confidence = min(0.9, 0.6 + (len(objects) * 0.05) + (len(caption.split()) * 0.01))
            
            return {
                "caption": caption,
                "objects": objects,
                "confidence": caption_confidence,
                "analysis_quality": "blip_model"
            }
            
        except Exception as e:
            bt.logging.error(f"Error analyzing image {getattr(media, 'filename', 'unknown')}: {e}")
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
        text_confidence = text_analysis.get("confidence", 0.5)
        visual_confidence = visual_analysis.get("confidence", 0.0)
        
        # Weighted combination - text analysis is primary, visual is supporting
        if visual_confidence > 0:
            # Both analyses available - weighted average with text being more important
            combined_confidence = (text_confidence * 0.7) + (visual_confidence * 0.3)
        else:
            # Only text analysis available
            combined_confidence = text_confidence
        
        combined = {
            "event_type": text_analysis.get("event_type", "incident"),
            "confidence": combined_confidence,
            "entities": text_analysis.get("entities", {}),
            "visual_content": visual_analysis.get("scene_description", ""),
            "all_objects": text_analysis.get("entities", {}).get("vehicles", []) + visual_analysis.get("detected_objects", []),
            "analysis_sources": {
                "text_confidence": text_confidence,
                "visual_confidence": visual_confidence,
                "text_quality": "openai" if self.openai_client else "rule_based",
                "visual_quality": visual_analysis.get("analysis_quality", "none")
            }
        }
        
        # Enhance event type if visual evidence supports it
        visual_desc = visual_analysis.get("scene_description", "").lower()
        if "accident" in visual_desc or "crash" in visual_desc:
            combined["event_type"] = "accident"
            # Boost confidence when visual evidence supports text analysis
            combined["confidence"] = min(combined["confidence"] + 0.1, 1.0)
            combined["analysis_sources"]["visual_support"] = True
        
        return combined
    
    def _generate_event(self, submission: Any, analysis: Dict[str, Any]) -> Any:
        """Generate a structured ProcessedEvent from the analysis"""
        
        # Generate event summary like: "truck and motorbike accident near balaju at 4pm, June 11"
        event_type = analysis["event_type"]
        entities = analysis.get("entities", {})
        
        # Build summary components
        summary_parts = []
        
        # Add vehicles if any
        if "vehicles" in entities:
            vehicles_str = " and ".join(entities["vehicles"])
            summary_parts.append(vehicles_str)
        
        # Add event type
        summary_parts.append(event_type)
        
        # Add location if any
        if "locations" in entities:
            location_str = f"near {entities['locations'][0]}"
            summary_parts.append(location_str)
        
        # Add time if any
        if "times" in entities:
            time_str = f"at {entities['times'][0]}"
            summary_parts.append(time_str)
        
        # Add date
        event_date = getattr(submission, 'timestamp', datetime.now()).strftime("%B %d")
        summary_parts.append(event_date)
        
        summary = " ".join(summary_parts)
        
        # Create ProcessedEvent
        try:
            if LUMINAR_PROTOCOL_AVAILABLE:
                event = ProcessedEvent(
                    event_id=f"evt_{submission.submission_id}_{int(time.time())}",
                    submission_id=submission.submission_id,
                    event_type=event_type,
                    summary=summary,
                    confidence_score=analysis["confidence"],
                    processing_timestamp=datetime.now(),
                    extracted_entities=entities,
                    visual_analysis=analysis.get("visual_content", ""),
                    processing_metadata={
                        "miner_version": "1.0.0",
                        "processing_time": time.time(),
                        "analysis_confidence": analysis["confidence"]
                    }
                )
            else:
                # Fallback to dict if ProcessedEvent class is not available
                event = {
                    "event_id": f"evt_{submission.submission_id}_{int(time.time())}",
                    "submission_id": submission.submission_id,
                    "summary": summary,
                    "confidence": analysis["confidence"],
                    "timestamp": datetime.now()
                }
            
            # Cache for duplicate detection
            self._cache_event(event)
            
            return event
            
        except Exception as e:
            bt.logging.error(f"Error creating ProcessedEvent: {e}")
            # Return a basic event structure
            return {
                "event_id": f"evt_{getattr(submission, 'submission_id', 'unknown')}_{int(time.time())}",
                "submission_id": getattr(submission, 'submission_id', 'unknown'),
                "summary": summary,
                "confidence": analysis["confidence"],
                "timestamp": datetime.now()
            }
    
    async def _check_duplicate(self, event: Any, submission: Any) -> bool:
        """Check if this event is a duplicate of a previously processed event"""
        for cached_event in self.processed_events_cache.values():
            # Text similarity
            text_sim = self._calculate_text_similarity(
                getattr(submission, 'text_description', ''),
                cached_event.get("original_text", "")
            )
            
            # Geographic similarity
            geo_sim = self._calculate_geo_similarity(
                getattr(submission, 'geotag', {}),
                cached_event.get("geotag", {})
            )
            
            # Time similarity
            time_sim = self._calculate_time_similarity(
                getattr(submission, 'timestamp', datetime.now()),
                cached_event.get("timestamp", datetime.now())
            )
            
            # Combined similarity
            combined_sim = (text_sim + geo_sim + time_sim) / 3
            
            if combined_sim > self.duplicate_threshold:
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def _calculate_geo_similarity(self, geo1: Dict[str, float], geo2: Dict[str, float]) -> float:
        """Calculate geographic similarity"""
        if not geo1 or not geo2:
            return 0.0
        
        try:
            # Simple Euclidean distance
            lat_diff = abs(geo1.get("lat", 0) - geo2.get("lat", 0))
            lng_diff = abs(geo1.get("lng", 0) - geo2.get("lng", 0))
            distance = (lat_diff**2 + lng_diff**2)**0.5
            
            # Convert to similarity (closer = more similar)
            # 0.01 degrees ≈ 1km, consider similar if within 1km
            similarity = max(0.0, 1.0 - (distance / 0.01))
            return similarity
            
        except:
            return 0.0
    
    def _calculate_time_similarity(self, time1: datetime, time2: datetime) -> float:
        """Calculate temporal similarity"""
        try:
            time_diff = abs((time1 - time2).total_seconds())
            # Consider similar if within 1 hour
            similarity = max(0.0, 1.0 - (time_diff / 3600))
            return similarity
        except:
            return 0.0
    
    def _cache_event(self, event: Any):
        """Cache event for duplicate detection"""
        event_id = getattr(event, 'event_id', str(time.time()))
        self.processed_events_cache[event_id] = {
            "event": event,
            "timestamp": datetime.now(),
            "original_text": getattr(event, 'summary', ''),
            "geotag": {}  # Would extract from event
        }
        
        # Keep cache size manageable
        if len(self.processed_events_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.processed_events_cache.keys())[:100]
            for key in oldest_keys:
                del self.processed_events_cache[key]
    
    def _update_stats(self, total_submissions: int, duplicates: int, processing_time: float):
        """Update processing statistics"""
        self.stats["total_processed"] += total_submissions
        self.stats["duplicates_detected"] += duplicates
        
        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        new_avg = (current_avg + processing_time) / 2
        self.stats["avg_processing_time"] = new_avg
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of loaded models"""
        models = {
            "miner_version": "1.0.0"
        }
        
        # Text processing models
        if self.openai_client:
            models["text_model"] = f"openai_{self.openai_config['model']}"
        else:
            models["text_model"] = "rule_based_v1.0"
        
        # Visual processing models
        if self.openai_client:
            models["visual_model"] = f"openai_{self.openai_config['vision_model']}"
        elif getattr(self, 'blip_model', None):
            models["visual_model"] = "blip_base"
        else:
            models["visual_model"] = "mock"
        
        # OpenAI status
        models["openai_available"] = "yes" if self.openai_client else "no"
        
        return models

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        Enhanced for Luminar subnet with media processing considerations.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # Standard Bittensor blacklist logic
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        # Luminar-specific blacklist logic
        if hasattr(synapse, 'user_submissions'):
            # Check for reasonable submission count
            if len(synapse.user_submissions) > 100:
                bt.logging.warning(f"Blacklisting request with too many submissions: {len(synapse.user_submissions)}")
                return True, "Too many submissions"

        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        
        Enhanced for Luminar subnet with media processing priority.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # Get caller stake
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake_priority = float(self.metagraph.S[caller_uid])
        
        # Luminar-specific priority adjustments
        if hasattr(synapse, 'user_submissions'):
            # Higher priority for submissions with media
            media_bonus = 0.0
            for submission in synapse.user_submissions:
                if hasattr(submission, 'media_files') and submission.media_files:
                    media_bonus += 0.1  # Small bonus for media submissions
            
            final_priority = stake_priority + media_bonus
            bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {final_priority} (stake: {stake_priority}, media bonus: {media_bonus})")
            return final_priority
        
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {stake_priority}")
        return stake_priority


# This is the main function, which runs the Luminar miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"🚀 Luminar Miner running... {time.time()}")
            bt.logging.info(f"📊 Stats: {miner.stats}")
            time.sleep(5)
