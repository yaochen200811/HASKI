# HASKI Framework

This repository contains the implementation of the **HASKI framework**, presented in the paper:

> **HASKI: Hallucination-Aware Agentic Self-Adaptive Knowledge Injection via Reinforcement Learning in Large Language Models**

The system implements a three-layer architecture for timing-aware, hallucination-reducing knowledge injection during LLM inference.

---

# System Overview

<img width="2193" height="1656" alt="flow color (1)" src="https://github.com/user-attachments/assets/7f7870bc-fb41-4929-867f-a462a22629eb" />

The framework is composed of three independent services:

## 1. Generation Service
Responsible for primary reasoning and response generation using a large language model.

## 2. Knowledge Service
Handles asynchronous external knowledge retrieval using:
- LLM-based topic extraction
- Wikidata queries

## 3. Timing Prediction Service (IPM)
Implements the IPM, which determines the optimal point to inject external knowledge during generation.

---

Each service is fully independent and can be deployed on separate machines if required.


# Quick Start

## 1. Install Python dependencies

We recommend using **Python 3.11**.

Clone the repository and install the required packages:

```bash
git clone <repository_url>
cd HASKI

pip install -r requirements.txt
```


## 2. Set up the Primary Reasoning Model

The Generation Service uses a GGUF reasoning model.

1. Select a supported reasoning model. HASKI supports both:
   - Online models hosted on Hugging Face (loaded automatically using the model repository ID).
   - Local GGUF models (downloaded and stored locally).

   For example, a local GGUF model can be downloaded as:
   **DeepSeek-R1-Distill-Qwen-1.5B-GGUF**

2. If using a local GGUF model, place the GGUF file in your preferred model directory.

3. Update the following fields in `config.yaml`:

```yaml
generation_model:
  name: <Hugging Face model repository ID or local model directory>
  gguf_file: <GGUF filename>
  device: cuda
```

Set `device` to the appropriate PyTorch device (`cuda`, `cpu`, or `mps`).


## 3. Set up the Knowledge Service

The Knowledge Service requires an OpenAI-compatible endpoint for topic extraction.

We recommend using **LM Studio** with **Llama-3.2-1B-Instruct**.

1. Install [**LM Studio**](https://lmstudio.ai/)
2. Download **Llama-3.2-1B-Instruct**.
3. Start the OpenAI-compatible server in LM Studio.
4. Update the following configuration:

```yaml
knowledge:
  client_url: http://localhost:1234/v1
  client_key: lm-studio
  model: llama-3.2-1b-instruct
```

Since LM Studio does not require authentication by default, `client_key` may be any non-empty string.


## 4. Set up the Timing Prediction Service

The Timing Prediction Service automatically downloads the base model from Hugging Face.

Before running the service:

1. Create a Hugging Face account if you do not already have one.
2. Accept the license for [**meta-llama/Llama-3.2-1B**](https://huggingface.co/meta-llama/Llama-3.2-1B) on Hugging Face.
3. Generate a Hugging Face access token.
4. Update `config.yaml`:

```yaml
timing:
  huggingface_token: hf_xxxxxxxxxxxxxxxxx
  mode: KI-RL-AS
  device: cuda
```

The repository already contains the fine-tuned IPM checkpoints. The Hugging Face token is only required to download the base Llama model used to load these checkpoints.

## 5. Configure the system

Review and update the remaining entries in `config.yaml`, including:

- API endpoints
- Model locations
- Execution devices
- Cache directory


## 6. Start the services

Launch the three services separately:

1. Knowledge Service
2. Timing Prediction Service (IPM)
3. Generation Service


## 7. Generate responses

After all services are running, send a GET request to:

```
/HASKI_generate
```

### Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `question` | string | The input question to be answered |
| `smart` | bool | If `true`, uses IPM to determine optimal knowledge injection timing. If `false`, injects knowledge immediately |


### Example Request

```text
GET /HASKI_generate?question=Who discovered penicillin?&smart=true
```

returns the final generated response from the Primary Reasoning Model.


# Configuration

Before running the framework, edit `config.yaml` to match your environment.

## API Configuration

```yaml
api:
  knowledge: URL of Knowledge Service <http://<knowledge-service>:8080>
  timing: URL of Timing Prediction Service (IPM) <http://<timing-service>:8082>
```

If all services are running on the same machine, these should point to the local service addresses.

## Generation Model Configuration

```yaml
generation_model:
  name: HuggingFace model name or path
  gguf_file: model file name (if applicable)
  device: torch device (cuda / cpu / mps)
  max_token: maximum generation length
```

You must update:

- `name`
- `gguf_file`
- `device`

## Knowledge Service Configuration

```yaml
knowledge:
  cache_dir: directory for caching retrieved data
  email: email used for Wikidata API requests
  client_url: OpenAI-compatible endpoint (e.g., LM Studio)
  client_key: API key for model access
  model: LLM used for topic extraction
```

You must update:

- `email` (required by the Wikidata API User-Agent)
- `client_url`
- `client_key`
- `model`

Requires access to an OpenAI-compatible API endpoint for the language model used in topic extraction.

We recommend using **LM Studio** with the **Llama-3.2-1B-Instruct** model.

If using LM Studio, load the model and set `client_url` to the OpenAI-compatible server endpoint (e.g., `http://localhost:1234/v1`). Set `client_key` to any non-empty string, as LM Studio does not require authentication by default.

## Timing (IPM) Configuration

```yaml
timing:
  mode: IPM variant (KI-RL-AS, KI-RL-AS-B)
  device: torch device (cuda / cpu / mps)
  huggingface_token: Hugging Face access token
```

You must update:

- `device`
- `huggingface_token`

# Running the Services

Start the three services in any order.

## 1. Knowledge Service

```bash
python knowledge_service.py
```

This starts a REST service responsible for:

- extracting search topics from user questions,
- retrieving relevant facts from Wikidata,
- serving asynchronous knowledge requests.

Available endpoints:

- `POST /fetch-knowledge`
- `GET /check-knowledge`

---

## 2. Timing Prediction Service

```bash
python timing_service.py
```

This starts the Injection Prediction Model (IPM), which predicts whether the current reasoning state is an appropriate point for knowledge injection.

Available endpoints:

- `POST /timing`
- `POST /reload-model`

---

## 3. Generation Service

```bash
python generation_service.py
```

This starts the main HASKI inference server.

The Generation Service coordinates the entire pipeline by:

1. receiving user questions,
2. starting asynchronous knowledge retrieval,
3. generating responses using the Primary Reasoning Model,
4. querying the Timing Prediction Service,
5. injecting external knowledge when appropriate,
6. returning the final response.

Available endpoint:

- `GET /HASKI_generate`


# System Workflow

1. A user submits a question to the Generation Service.
2. The Generation Service immediately starts asynchronous knowledge retrieval through the Knowledge Service.
3. The Primary Reasoning Model begins generating the response without waiting for retrieval to finish.
4. Once knowledge becomes available, the Timing Prediction Service evaluates whether the current reasoning state is suitable for knowledge injection.
5. If the prediction is **True**, the retrieved knowledge is injected into the model context.
6. The Primary Reasoning Model continues generation using the updated context until the response is complete.

---

# Features

* Asynchronous external knowledge retrieval
* Timing-aware knowledge injection using a learned Injection Prediction Model
* Distributed three-service architecture
* Modular design allowing each service to run on separate machines

---

# Citation

If you use this repository in your research, please cite the accompanying HASKI paper.

