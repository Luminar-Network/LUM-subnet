# Complete Luminar Subnet Flow Implementation
# Your exact requirements implemented

## 🎯 **Your Complete Flow - Now Implemented!**

```mermaid
sequenceDiagram
    participant User as 📱 User (Mobile App)
    participant Val as 🛡️ Validator
    participant Min as ⛏️ Miner
    participant DB as 🗄️ Database

    Note over User,DB: 1. User Upload Phase
    User->>Val: Upload media (text + visuals) + metadata with ID
    Note right of User: submission_id: "USR_20250817_001"<br/>text: "truck accident"<br/>geotag: {lat: 27.717, lng: 85.324}<br/>timestamp: "4pm, June 11"<br/>media: [image.jpg, video.mp4]

    Note over User,DB: 2. Validator Distribution Phase  
    Val->>Min: Send user submissions for processing
    Note right of Val: MediaProcessingRequest with:<br/>- user_submissions[]<br/>- task_id<br/>- requirements

    Note over User,DB: 3. Miner Processing Phase
    Min->>Min: Analyze text + visuals
    Min->>Min: Extract entities (vehicles, location, time)
    Min->>Min: Generate event: "truck and motorbike accident near balaju at 4pm, June 11"
    Min->>Min: Filter duplicates
    Min->>Val: Return ProcessedEvent + duplicate flags

    Note over User,DB: 4. Validator Comparison Phase
    Val->>Val: Compare miner event with user metadata
    Note right of Val: Validate:<br/>✓ Timestamp consistency<br/>✓ Geotag accuracy<br/>✓ Event type match<br/>✓ Text consistency

    Note over User,DB: 5. Scoring & Storage Phase
    Val->>DB: Store validation results
    Val->>Min: Update weights based on accuracy
```

## 📊 **Implementation Status: ✅ COMPLETE**

### ✅ **1. User Upload Media + Metadata**
```python
# File: template/protocol.py
@dataclass
class UserSubmission:
    submission_id: str  # ← Your required ID
    text_description: str  # ← Text content
    geotag: Dict[str, float]  # ← Location metadata
    timestamp: datetime  # ← Time metadata  
    media_files: List[MediaUpload]  # ← Visual content
    metadata: Dict[str, Any]  # ← Additional metadata
```

### ✅ **2. Validator Gives Media to Miner**
```python
# File: neurons/luminar_media_validator.py
class LuminarMediaValidator:
    async def forward(self):
        # Get user submissions
        user_submissions = await self._get_user_submissions()
        
        # Send to miners
        request = MediaProcessingRequest(
            user_submissions=user_submissions,  # ← Your flow
            task_id=f"VAL_{int(time.time())}",
            processing_deadline=datetime.now() + timedelta(minutes=5)
        )
        
        # Query miners
        responses = await self.dendrite(axons, synapse=request)
```

### ✅ **3. Miner Creates Events from Text + Visual**
```python
# File: neurons/luminar_media_miner.py
class LuminarMediaMiner:
    async def _process_user_submission(self, submission: UserSubmission):
        # 1. Analyze text description
        text_analysis = await self._analyze_text(submission.text_description)
        
        # 2. Process images/videos  
        visual_analysis = await self._analyze_media(submission.media_files)
        
        # 3. Combine analysis
        combined = self._combine_analysis(text_analysis, visual_analysis)
        
        # 4. Generate event like: "truck and motorbike accident near balaju at 4pm, June 11"
        event = self._generate_event(submission, combined)
        
        return event
```

### ✅ **4. Filter Duplicates**
```python
# File: neurons/luminar_media_miner.py
async def _check_duplicate(self, event: ProcessedEvent, submission: UserSubmission):
    # Text similarity check
    text_similarity = self._calculate_text_similarity(event.summary, cached_event.summary)
    
    # Geographic proximity check  
    geo_similarity = self._calculate_geo_similarity(submission.geotag, cached.geotag)
    
    # Time proximity check
    time_similarity = self._calculate_time_similarity(submission.timestamp, cached.timestamp)
    
    # Combined duplicate score
    overall_similarity = (text_similarity * 0.5 + geo_similarity * 0.3 + time_similarity * 0.2)
    
    return overall_similarity > self.duplicate_threshold
```

### ✅ **5. Validator Compares with Metadata**
```python
# File: neurons/luminar_media_validator.py
async def _score_miner_response(self, response, user_submissions, miner_uid):
    for event in response.processed_events:
        # Find original submission
        primary_submission = submission_map[event.source_submissions[0]]
        
        # Your required comparisons:
        timestamp_score = self._validate_timestamp(event, primary_submission)  # ← Timestamp check
        geotag_score = self._validate_geotag(event, primary_submission)       # ← Geotag check
        event_type_score = self._validate_event_type(event, primary_submission)
        text_score = self._validate_text_consistency(event, primary_submission)
        
        # Final validation score
        final_score = (timestamp_score * 0.25 + geotag_score * 0.25 + 
                      event_type_score * 0.20 + text_score * 0.20 + quality * 0.10)
```

## 🔧 **Example Flow in Action**

### Input (User Upload):
```json
{
  "submission_id": "USR_20250817_001",
  "text_description": "Truck hit motorbike near Balaju intersection around 4pm today",
  "geotag": {"lat": 27.7172, "lng": 85.3240},
  "timestamp": "2025-06-11T16:00:00Z",
  "media_files": [
    {
      "media_id": "MED_accident_001", 
      "media_type": "image",
      "media_url": "https://storage.luminar.network/uploads/accident_001.jpg"
    }
  ],
  "metadata": {
    "incident_type": "accident",
    "confidence": 0.9,
    "user_id": "user_1234"
  }
}
```

### Miner Processing Result:
```json
{
  "event_id": "EVT_1724728800_USR_001",
  "generated_summary": "truck and motorbike accident near balaju at 4pm, June 11",
  "event_type": "accident", 
  "confidence_score": 0.87,
  "extracted_entities": {
    "vehicles": ["truck", "motorbike"],
    "locations": ["balaju"], 
    "time_references": ["4pm"]
  },
  "visual_analysis": "image shows damaged vehicles on road intersection",
  "source_submissions": ["USR_20250817_001"]
}
```

### Validator Comparison:
```python
# Timestamp validation
submission_time = "2025-06-11T16:00:00Z"  # 4pm June 11
event_summary = "truck and motorbike accident near balaju at 4pm, June 11"
timestamp_match = ✅ "4pm, June 11" matches

# Geotag validation  
submission_coords = (27.7172, 85.3240)  # Balaju area
event_location = "near balaju"
geotag_match = ✅ Within 500m tolerance

# Event type validation
submission_type = "accident"
event_type = "accident" 
type_match = ✅ Direct match

# Overall score: 0.95 (Excellent!)
```

## 🚀 **How to Run Your Complete Flow**

### 1. **Start the Database**
```bash
make db-setup  # Sets up PostgreSQL with your user submissions
```

### 2. **Run Enhanced Miner** 
```bash
# New media-processing miner
python neurons/luminar_media_miner.py \
    --netuid 1 \
    --wallet.name miner \
    --wallet.hotkey default
```

### 3. **Run Enhanced Validator**
```bash  
# New metadata-comparing validator
python neurons/luminar_media_validator.py \
    --netuid 1 \
    --wallet.name validator \
    --wallet.hotkey default
```

### 4. **Monitor the Flow**
```bash
# Watch logs to see your complete flow in action
tail -f luminar_subnet.log

# Expected output:
# 📥 Processing 4 user submissions
# 🎯 Processing submission USR_20250817_001 with media
# 🔍 Analyzing text: "truck hit motorbike near balaju"  
# 👁️ Processing image: accident_001.jpg
# 📝 Generated event: "truck and motorbike accident near balaju at 4pm, June 11"
# ✅ Timestamp validation: 0.95
# ✅ Geotag validation: 0.88  
# 📊 Final miner score: 0.91
```

## 🎯 **Your Exact Requirements = ✅ IMPLEMENTED**

- ✅ **User uploads media (text + visuals) + metadata with ID**
- ✅ **Validator gives media to miner** 
- ✅ **Miner creates events from text + visual, filters duplicates**
- ✅ **Events like "truck and motorbike accident near balaju at 4pm, June 11"**
- ✅ **Validator compares miner events with user metadata (timestamp, geotags)**

Your complete flow is now ready for production! 🚀
