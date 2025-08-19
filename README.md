# **Luminar Subnet** <!-- omit in toc -->

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## The Decentralized Crime Intelligence Platform <!-- omit in toc -->

![Luminar Network Platform](https://luminar.network/images/analytics.jpg)

[Luminar](https://luminar.network/) is a revolutionary Bittensor subnet that transforms how crime data is collected, processed, and utilized. By leveraging decentralized intelligence and community-driven reporting, Luminar creates the world's first incentivized crime intelligence network.

### **Core Mission**

Luminar solves the critical problem of **fragmented, unstructured, and underutilized crime data** by creating a decentralized intelligence platform that:

- **Crowdsources** crime incident reports from community members
- **Verifies** data authenticity through multi-layered consensus
- **Processes** raw reports into structured, actionable intelligence
- **Incentivizes** quality contributions with $TAO rewards

---

## Table of Contents <!-- omit in toc -->

- [🎯 Problem Statement](#-problem-statement)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [👥 Network Participants](#-network-participants)
- [🔄 Data Flow](#-data-flow)
- [🏆 Incentive Mechanism](#-incentive-mechanism)
- [📋 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [📊 Monitoring](#-monitoring)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🎯 Problem Statement

Traditional crime data systems suffer from:

- **Fragmentation**: Data scattered across multiple agencies and platforms
- **Delays**: Incident reporting and processing can take days or weeks
- **Incompleteness**: Many crimes go unreported or lack sufficient detail
- **Accessibility**: Limited public access to actionable crime intelligence
- **Quality**: Inconsistent data standards and verification processes

### Luminar's Solution

Luminar creates a **real-time, community-driven crime intelligence network** that:

1. **Democratizes** crime reporting through mobile-first applications
2. **Incentivizes** quality data submission with cryptocurrency rewards
3. **Processes** reports using advanced AI for clustering and verification
4. **Delivers** structured intelligence for risk assessment and prevention

---

## 🏗️ Architecture Overview

Luminar operates as a **four-layer decentralized network** with complete user-to-intelligence flow:


```mermaid
graph TB
    subgraph "Layer 1: User Submission"
        A[Mobile App Users] --> B[Media Upload + Metadata]
        B --> C[Text + Visuals + Geotag + Timestamp]
        C --> D[User Submission with ID]
    end
    
    subgraph "Layer 2: Validator Distribution"
        D --> E[Luminar Validators]
        E --> F[Distribute to Miners]
        F --> G[MediaProcessingRequest]
    end
    
    subgraph "Layer 3: AI Processing (Miners)"
        G --> H[Text Analysis + Visual Processing]
        H --> I[CLIP + BLIP + NLP Models]
        I --> J[Event Generation + Duplicate Filtering]
        J --> K[Structured Events]
    end
    
    subgraph "Layer 4: Validation & Consensus"
        K --> L[Metadata Comparison]
        L --> M[Timestamp + Geotag + Type Validation]
        M --> N[Consensus Scoring + Weight Updates]
    end
    
    subgraph "Layer 5: Intelligence Database"
        N --> O[PostgreSQL + PostGIS]
        O --> P[Crime Intelligence API]
        P --> Q[Predictive Analytics + Risk Heatmaps]
    end
```

### Key Components

- **2-File Architecture**: Simplified `neurons/miner.py` and `neurons/validator.py`
- **OpenAI Integration**: Enhanced AI processing with GPT-4 and Vision API
- **PostgreSQL + PostGIS Database**: Geospatial crime data with advanced querying
- **AI Processing Pipeline**: OpenAI GPT, CLIP, BLIP, and transformer models for multimodal analysis
- **Metadata Validation**: Timestamp, geotag, and content consistency checking
- **Consensus Engine**: Bittensor's Yuma Consensus with custom scoring mechanisms
- **Real-time API**: Live crime intelligence for applications and services
- **Graceful Fallbacks**: CPU-only operation with rule-based processing

### Complete Flow Implementation

1. **User Uploads**: Media (images/videos) + text description + metadata with unique ID
2. **Validator Distribution**: Sends user submissions to miners for processing
3. **Miner Processing**: AI analysis using OpenAI GPT models or local fallbacks → structured events like "truck and motorbike accident near balaju at 4pm, June 11"
4. **Duplicate Detection**: Filters redundant reports using similarity algorithms
5. **Metadata Validation**: Compares miner events with original user metadata (timestamps, geotags)
6. **Consensus & Scoring**: Updates miner weights based on validation accuracy
7. **Intelligence Storage**: Verified events stored in geospatial database for analytics

### AI Processing Capabilities

**OpenAI Integration (Primary):**
- **Text Analysis**: GPT-3.5-turbo/GPT-4 for enhanced incident classification
- **Image Analysis**: GPT-4 Vision for scene understanding and object detection
- **Structured Output**: JSON-formatted responses with confidence scoring

**Local Fallbacks (Secondary):**
- **BLIP Models**: Image captioning and object detection
- **CLIP Models**: Visual-text similarity and embeddings
- **Rule-based Processing**: Regex pattern matching for basic analysis

**Graceful Degradation:**
- Works without GPU (CPU-only operation)
- Functions without OpenAI API key (uses local models)
- Falls back to rule-based processing if all AI models unavailable

---

## 👥 Network Participants

### 🔸 **Data Contributors (Users)**
- **Role**: Submit incident reports via mobile/web applications
- **Rewards**: Micro-bounties in $TAO for valid, non-duplicate reports
- **Requirements**: Geolocation, timestamp, incident description
- **Verification**: Account verification, media authenticity checks

### ⛏️ **Miners**
- **Role**: Process raw reports into structured crime events
- **Technology**: OpenAI GPT models, CLIP, BLIP, and local AI models for multimodal processing
- **Output**: Clean, machine-readable crime event objects with confidence scores
- **Evaluation**: Event accuracy, metadata consistency, processing quality
- **Requirements**: No GPU required (CPU-only operation supported)

### ✅ **Validators**
- **Role**: Verify miner outputs and maintain data quality
- **Responsibilities**:
  - Cross-check timestamps and geotags
  - Verify media authenticity
  - Score clustering accuracy
  - Detect duplicates and fraud
- **Rewards**: Based on consensus accuracy and diligence

---

## 🔄 Data Flow

### 1. **Incident Reporting**
```
User App → Raw Report → Metadata Tagging → Queue for Processing
```

### 2. **AI Processing**
```
Raw Reports → Semantic Analysis → Clustering → Event Generation
```

### 3. **Validation & Consensus**
```
Crime Events → Integrity Checks → Quality Scoring → Weight Updates
```

### 4. **Intelligence Output**
```
Verified Events → Database Storage → API Access → Risk Analytics
```

---

## 🏆 Incentive Mechanism

Luminar uses a **multi-tiered reward system** that incentivizes quality at every level:

### Reward Distribution

| Participant | Reward Basis | Weight |
|-------------|--------------|---------|
| **Data Contributors** | Valid report submissions | Micro-bounties |
| **Miners** | Processing accuracy & speed | 40% Clustering + 30% Integrity |
| **Validators** | Consensus accuracy | Verification quality |

### Scoring Criteria for Miners

1. **Data Integrity (30%)**
   - Timestamp/geotag consistency
   - Media authenticity verification
   - Source report validation

2. **Clustering Accuracy (40%)**
   - Semantic similarity of grouped reports
   - Event type classification accuracy
   - Clustering completeness

3. **Novelty & Redundancy (20%)**
   - Duplicate detection
   - Novel event identification
   - Quality over quantity

4. **Processing Quality (10%)**
   - Summary completeness
   - Metadata richness
   - Processing efficiency

### Anti-Sybil Measures

- **Economic Staking**: $TAO cost for network participation
- **Slashing Mechanism**: Penalties for false or misleading data
- **Dynamic Reputation**: Continuous performance-based weighting
- **Media Verification**: Hash-based authenticity checks

---

## 📋 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 13+ (optional for local development)
- **GPU NOT required** - CPU-only operation supported
- 8GB+ RAM (16GB+ recommended for local AI models)
- 50GB+ storage
- OpenAI API key (optional for enhanced processing)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib python3-pip git

# macOS
brew install postgresql python git
```

### Install Luminar Subnet

```bash
# Clone the repository
git clone https://github.com/khemrajregmi/Luminar-subnet.git
cd Luminar-subnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Database Setup

```bash
# Quick testnet deployment (automated)
./scripts/deploy_testnet.sh

# Manual database setup (optional)
python scripts/setup_test_database.py

# Test database connectivity
python scripts/test_database.py
```

### OpenAI Integration (Optional)

```bash
# Get your API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Test OpenAI integration
python -c "
import os
from neurons.miner import Miner
miner = Miner()
print('✅ OpenAI available:', bool(miner.openai_client))
"
```

---

## 🚀 Quick Start

### Running a Miner

```bash
# Basic miner setup
python neurons/miner.py \
    --netuid 414 \
    --wallet.name miner \
    --wallet.hotkey default \
    --logging.debug

# With OpenAI integration
OPENAI_API_KEY=sk-your-key python neurons/miner.py \
    --netuid 414 \
    --wallet.name miner \
    --wallet.hotkey default \
    --blacklist.min_stake_threshold 1000 \
    --neuron.name luminar_miner \
    --logging.debug
```

### Running a Validator

```bash
# Basic validator setup
python neurons/validator.py \
    --netuid 414 \
    --wallet.name validator \
    --wallet.hotkey default \
    --logging.debug

# With custom parameters
python neurons/validator.py \
    --netuid 414 \
    --wallet.name validator \
    --wallet.hotkey default \
    --neuron.sample_size 20 \
    --neuron.timeout 300 \
    --logging.debug
```

### Development Mode (Local Testing)

```bash
# Run with mock subtensor
python neurons/miner.py \
    --subtensor.network mock \
    --wallet.name test_miner \
    --logging.debug

python neurons/validator.py \
    --subtensor.network mock \
    --wallet.name test_validator \
    --logging.debug

# Test with database scripts
python scripts/test_database.py
python scripts/setup_test_database.py
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/luminar_subnet
POSTGRES_DB=luminar_subnet
POSTGRES_USER=luminar
POSTGRES_PASSWORD=secure_password

# API Configuration
API_BASE_URL=https://api.luminar.network/
API_KEY=your_api_key_here

# AI Model Configuration
HUGGINGFACE_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key_here  # Enhanced AI processing
OPENAI_MODEL=gpt-3.5-turbo          # Text analysis model
OPENAI_VISION_MODEL=gpt-4-vision-preview  # Image analysis
OPENAI_MAX_TOKENS=500               # Response length limit
OPENAI_TEMPERATURE=0.3              # Creativity level

# Network Configuration
SUBNET_NETUID=414
MIN_STAKE_THRESHOLD=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=luminar_subnet.log
```

### Advanced Configuration

```python
# config/luminar_config.py
LUMINAR_CONFIG = {
    "processing": {
        "max_batch_size": 50,
        "clustering_threshold": 0.7,
        "confidence_threshold": 0.3,
        "max_processing_time": 300
    },
    "validation": {
        "integrity_weight": 0.30,
        "clustering_weight": 0.40,
        "novelty_weight": 0.20,
        "quality_weight": 0.10
    },
    "database": {
        "connection_pool_size": 20,
        "query_timeout": 30,
        "backup_interval": 3600
    }
}
```

### File Structure

```
luminar-subnet/
├── neurons/
│   ├── miner.py           # Consolidated Luminar miner
│   └── validator.py       # Consolidated Luminar validator
├── scripts/
│   ├── deploy_testnet.sh  # Automated testnet deployment
│   ├── setup_database.py # Production database setup
│   ├── test_database.py  # Database testing
│   └── README.md         # Scripts documentation
├── template/
│   ├── protocol.py       # Luminar protocol definitions
│   ├── base/             # Base miner/validator classes
│   └── utils/            # Utility functions
├── docs/
│   ├── COMPLETE_FLOW.md      # Complete implementation guide
│   ├── OPENAI_INTEGRATION.md # OpenAI setup and usage
│   ├── DATABASE.md           # Database architecture
│   └── TESTNET_DEPLOYMENT.md # Deployment guide
├── requirements.txt      # Python dependencies
├── .env.testnet         # Testnet configuration
├── .env.production      # Production configuration
└── README.md           # This file
```

---

## 🧪 Testing

### Quick Testing

```bash
# Test database setup
python scripts/test_database.py

# Test miner functionality
python -c "
from neurons.miner import Miner
miner = Miner()
print('✅ Miner initializes successfully')
print('📊 OpenAI available:', bool(getattr(miner, 'openai_client', None)))
"

# Test validator functionality  
python -c "
from neurons.validator import Validator
validator = Validator()
print('✅ Validator initializes successfully')
"
```

### Unit Tests

```bash
# Run all tests (when available)
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_protocol.py -v
python -m pytest tests/test_miner.py -v
python -m pytest tests/test_validator.py -v
```

### Integration Tests

```bash
# Test complete subnet flow
python scripts/deploy_testnet.sh test

# Test with database
python scripts/setup_test_database.py
python scripts/test_database.py
```

### Performance Testing

```bash
# Benchmark processing speed
python scripts/benchmark_processing.py

# Load testing
python scripts/load_test.py --miners 10 --reports 1000
```

---

## 📊 Monitoring

### Key Metrics

- **Processing Throughput**: Reports processed per hour
- **Clustering Accuracy**: Semantic similarity scores
- **Validation Accuracy**: Consensus quality metrics
- **Network Health**: Active miners/validators
- **Data Quality**: Verified events percentage

### Monitoring Tools

```bash
# View real-time metrics
python scripts/monitor_subnet.py

# Export metrics to Prometheus
python scripts/prometheus_exporter.py

# Generate performance reports
python scripts/generate_report.py --timeframe 24h
```

### Dashboard Access

- **Grafana Dashboard**: http://localhost:3000/luminar
- **Subnet Stats**: https://stats.luminar.network/
- **Real-time Feed**: https://live.luminar.network/

---

## 🤝 Contributing

We welcome contributions from the community! Here's how to get involved:

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Areas for Contribution

- **AI Models**: Improve clustering and classification algorithms
- **Data Verification**: Enhance authenticity detection
- **Mobile Apps**: Build user-friendly reporting interfaces
- **Analytics**: Develop predictive crime models
- **Documentation**: Improve guides and tutorials

### Code Standards

- Follow PEP 8 for Python code
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation as needed

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Additional Resources

### 📚 **Documentation**
- **Complete Flow Guide**: [docs/COMPLETE_FLOW.md](docs/COMPLETE_FLOW.md)
- **OpenAI Integration**: [docs/OPENAI_INTEGRATION.md](docs/OPENAI_INTEGRATION.md)  
- **Database Architecture**: [docs/DATABASE.md](docs/DATABASE.md)
- **Testnet Deployment**: [docs/TESTNET_DEPLOYMENT.md](docs/TESTNET_DEPLOYMENT.md)
- **Scripts Reference**: [scripts/README.md](scripts/README.md)

### 🔗 **Links**
- **Website**: [https://luminar.network/](https://luminar.network/)
- **API Reference**: [https://api.luminar.network/docs](https://api.luminar.network/docs)
- **Discord**: [Join our community](https://discord.gg/luminar)
- **Twitter**: [@LuminarAI](https://twitter.com/LuminarAI)

### 🛠️ **Development Resources**
- **Bittensor Docs**: [https://bittensor.com/documentation/](https://bittensor.com/documentation/)
- **Yuma Consensus**: [https://bittensor.com/documentation/validating/yuma-consensus](https://bittensor.com/documentation/validating/yuma-consensus)
- **OpenAI API**: [https://platform.openai.com/docs](https://platform.openai.com/docs)

---

*Built with ❤️ for a safer world through decentralized intelligence*

**Architecture:** 2-file consolidated subnet with OpenAI integration  
**Status:** Production ready with CPU-only operation  
**License:** MIT - see [LICENSE](LICENSE) file for details

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Quickstarter template](#quickstarter-template)
- [Introduction](#introduction)
  - [Example](#example)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
  - [Install](#install)
- [Writing your own incentive mechanism](#writing-your-own-incentive-mechanism)
- [Writing your own subnet API](#writing-your-own-subnet-api)
- [Subnet Links](#subnet-links)
- [License](#license)

---
## Quickstarter template

This template contains all the required installation instructions, scripts, and files and functions for:
- Building Bittensor subnets.
- Creating custom incentive mechanisms and running these mechanisms on the subnets. 

In order to simplify the building of subnets, this template abstracts away the complexity of the underlying blockchain and other boilerplate code. While the default behavior of the template is sufficient for a simple subnet, you should customize the template in order to meet your specific requirements.
---

## Introduction

**IMPORTANT**: If you are new to Bittensor subnets, read this section before proceeding to [Installation](#installation) section. 

The Bittensor blockchain hosts multiple self-contained incentive mechanisms called **subnets**. Subnets are playing fields in which:
- Subnet miners who produce value, and
- Subnet validators who produce consensus

determine together the proper distribution of TAO for the purpose of incentivizing the creation of value, i.e., generating digital commodities, such as intelligence or data. 

Each subnet consists of:
- Subnet miners and subnet validators.
- A protocol using which the subnet miners and subnet validators interact with one another. This protocol is part of the incentive mechanism.
- The Bittensor API using which the subnet miners and subnet validators interact with Bittensor's onchain consensus engine [Yuma Consensus](https://bittensor.com/documentation/validating/yuma-consensus). The Yuma Consensus is designed to drive these actors: subnet validators and subnet miners, into agreement on who is creating value and what that value is worth. 

This starter template is split into three primary files. To write your own incentive mechanism, you should edit these files. These files are:
1. `template/protocol.py`: Contains the definition of the protocol used by subnet miners and subnet validators.
2. `neurons/miner.py`: Script that defines the subnet miner's behavior, i.e., how the subnet miner responds to requests from subnet validators.
3. `neurons/validator.py`: This script defines the subnet validator's behavior, i.e., how the subnet validator requests information from the subnet miners and determines the scores.

### Example

The Bittensor Subnet 1 for Text Prompting is built using this template. See [prompting](https://github.com/macrocosm-os/prompting) for how to configure the files and how to add monitoring and telemetry and support multiple miner types. Also see this Subnet 1 in action on [Taostats](https://taostats.io/subnets/netuid-1/) explorer.

---

## Installation

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- Use these instructions to run your subnet locally for your development and testing, or on Bittensor testnet or on Bittensor mainnet. 
- **IMPORTANT**: We **strongly recommend** that you first run your subnet locally and complete your development and testing before running the subnet on Bittensor testnet. Furthermore, make sure that you next run your subnet on Bittensor testnet before running it on the Bittensor mainnet.
- You can run your subnet either as a subnet owner, or as a subnet validator or as a subnet miner. 
- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that installation instructions differ based on your situation: For example, installing for local development and testing will require a few additional steps compared to installing for testnet. Similarly, installation instructions differ for a subnet owner vs a validator or a miner. 

### Install

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/running_on_staging.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).

---

## Writing your own incentive mechanism

As described in [Quickstarter template](#quickstarter-template) section above, when you are ready to write your own incentive mechanism, update this template repository by editing the following files. The code in these files contains detailed documentation on how to update the template. Read the documentation in each of the files to understand how to update the template. There are multiple **TODO**s in each of the files identifying sections you should update. These files are:
- `template/protocol.py`: Contains the definition of the wire-protocol used by miners and validators.
- `neurons/miner.py`: Script that defines the miner's behavior, i.e., how the miner responds to requests from validators.
- `neurons/validator.py`: This script defines the validator's behavior, i.e., how the validator requests information from the miners and determines the scores.
- `template/forward.py`: Contains the definition of the validator's forward pass.
- `template/reward.py`: Contains the definition of how validators reward miner responses.

In addition to the above files, you should also update the following files:
- `README.md`: This file contains the documentation for your project. Update this file to reflect your project's documentation.
- `CONTRIBUTING.md`: This file contains the instructions for contributing to your project. Update this file to reflect your project's contribution guidelines.
- `template/__init__.py`: This file contains the version of your project.
- `setup.py`: This file contains the metadata about your project. Update this file to reflect your project's metadata.
- `docs/`: This directory contains the documentation for your project. Update this directory to reflect your project's documentation.

__Note__
The `template` directory should also be renamed to your project name.
---

# Writing your own subnet API
To leverage the abstract `SubnetsAPI` in Bittensor, you can implement a standardized interface. This interface is used to interact with the Bittensor network and can be used by a client to interact with the subnet through its exposed axons.

What does Bittensor communication entail? Typically two processes, (1) preparing data for transit (creating and filling `synapse`s) and (2), processing the responses received from the `axon`(s).

This protocol uses a handler registry system to associate bespoke interfaces for subnets by implementing two simple abstract functions:
- `prepare_synapse`
- `process_responses`

These can be implemented as extensions of the generic `SubnetsAPI` interface.  E.g.:


This is abstract, generic, and takes(`*args`, `**kwargs`) for flexibility. See the extremely simple base class:
```python
class SubnetsAPI(ABC):
    def __init__(self, wallet: "bt.wallet"):
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=wallet)

    async def __call__(self, *args, **kwargs):
        return await self.query_api(*args, **kwargs)

    @abstractmethod
    def prepare_synapse(self, *args, **kwargs) -> Any:
        """
        Prepare the synapse-specific payload.
        """
        ...

    @abstractmethod
    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> Any:
        """
        Process the responses from the network.
        """
        ...

```


Here is a toy example:

```python
from bittensor.subnets import SubnetsAPI
from MySubnet import MySynapse

class MySynapseAPI(SubnetsAPI):
    def __init__(self, wallet: "bt.wallet"):
        super().__init__(wallet)
        self.netuid = 414

    def prepare_synapse(self, prompt: str) -> MySynapse:
        # Do any preparatory work to fill the synapse
        data = do_prompt_injection(prompt)

        # Fill the synapse for transit
        synapse = StoreUser(
            messages=[data],
        )
        # Send it along
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> str:
        # Look through the responses for information required by your application
        for response in responses:
            if response.dendrite.status_code != 200:
                continue
            # potentially apply post processing
            result_data = postprocess_data_from_response(response)
        # return data to the client
        return result_data
```

You can use a subnet API to the registry by doing the following:
1. Download and install the specific repo you want
1. Import the appropriate API handler from bespoke subnets
1. Make the query given the subnet specific API



# Subnet Links
In order to see real-world examples of subnets in-action, see the `subnet_links.py` document or access them from inside the `template` package by:
```python
import template
template.SUBNET_LINKS
[{'name': 'sn0', 'url': ''},
 {'name': 'sn1', 'url': 'https://github.com/opentensor/prompting/'},
 {'name': 'sn2', 'url': 'https://github.com/bittranslateio/bittranslate/'},
 {'name': 'sn3', 'url': 'https://github.com/gitphantomman/scraping_subnet/'},
 {'name': 'sn4', 'url': 'https://github.com/manifold-inc/targon/'},
...
]
```

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
# Luminar-subnet
