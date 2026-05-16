# Local Agent Stack Demo

A high-performance, local-first agent stack demonstrating quantized LLM serving, RAG, and intelligent task routing.

## 🎯 Goals
- ✅ Run a local OpenAI-compatible inference server
- ✅ Build an air-gapped RAG pipeline with ChromaDB
- ✅ Benchmark tokens-per-second (TPS) across local backends
- ✅ Implement intelligent hybrid routing (local/cloud)
- ✅ Demonstrate privacy-preserving inference

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Application Layer                           │
│  (Your existing OpenAI-compatible code)             │
└──────────────────┬──────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │ Hybrid Router    │◄─── Complexity Analysis
          │ (Route Decision) │
          └────────┬────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼─────┐        ┌─────▼────┐
   │ Local LLM │        │ Cloud API │
   │ (Ollama)  │        │ (OpenAI)  │
   └────┬─────┘        └──────────┘
        │
   ┌────▼───────────┐
   │ RAG Pipeline   │
   │ (ChromaDB)     │
   └────────────────┘
```

##  Components

**Commands:**
```bash

# Pulls quantized version (recommended)
ollama pull mistral
```

### 2. **serve_local_model.py**
FastAPI server with OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints.

**Features:**
- OpenAI API-compatible response format
- Health check endpoint
- Proper error handling
- Direct Ollama backend routing

**Start server:**
```bash
python serve_local_model.py
# Server runs on http://127.0.0.1:8000
```

### 3. **rag_pipeline.py**
Air-gapped RAG with local vector store (ChromaDB) + embeddings (sentence-transformers).

**Features:**
- Local document ingestion
- Embedding-based retrieval
- Context augmentation for LLM
- Similarity scoring

**Usage:**
```python
from rag_pipeline import LocalTextStore, RAGAgent

store = LocalTextStore()
store.ingest_documents(your_documents)

agent = RAGAgent(store)
answer = agent.answer("Your question?")
```

### 4. **hybrid_router.py**
Intelligent routing between local and cloud backends.

**Routing Logic:**
- **Simple tasks** (summarize, translate, format): → **Local** (fast, free)
- **Complex tasks** (design, analyze, reason): → **Cloud** (capable, fallback)
- Configurable complexity thresholds
- Automatic fallback support

**Usage:**
```python
from hybrid_router import HybridRouter

router = HybridRouter(prefer_local=True)
result = router.route("Your prompt here")
```

### 5. **benchmark_tps.py**
Measure inference throughput across different prompt types.

**Metrics:**
- Tokens-per-second (TPS)
- Latency statistics
- Hardware utilization insights

**Run benchmarks:**
```bash
python benchmark_tps.py
```

### 6. **example_usage.py**
End-to-end demonstration of all components working together.

```bash
python example_usage.py
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed ([ollama.ai](https://ollama.ai))

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull a quantized model with Ollama:**
   ```bash
   ollama pull mistral
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   # Runs on http://127.0.0.1:11434
   ```

4. **Start the local inference server (in new terminal):**
   ```bash
   python serve_local_model.py
   # API server at http://127.0.0.1:8000
   ```

5. **Initialize RAG and test:**
   ```bash
   python rag_pipeline.py
   ```

6. **Run the full demo:**
   ```bash
   python example_usage.py
   ```

## 🔧 Configuration

Set environment variables to customize behavior:

```bash
# Local inference
export LOCAL_SERVER_HOST="0.0.0.0"
export LOCAL_SERVER_PORT=8000
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_MODEL="mistral:latest"

# Hybrid routing
export PREFER_LOCAL="true"
export LOCAL_COMPLEXITY_THRESHOLD=3

# Cloud fallback (optional)
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"
export CLOUD_MODEL="gpt-3.5-turbo"
```

## Benchmarking

### Expected Performance (Mistral-7B Quantized)
- **Throughput**: 10-50 TPS (hardware dependent)
- **Latency**: 50-200ms per request
- **Memory**: 4-6 GB
- **Model Size**: ~3-4 GB (GGUF quantized)

### Run Benchmarks
```bash
python benchmark_tps.py
```

## API Compatibility

The local server is OpenAI-compatible. Drop-in replacement:

**Before (Cloud):**
```python
import openai
openai.api_key = "sk-..."
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After (Local):**
```python
import openai
openai.api_base = "http://127.0.0.1:8000/v1"
openai.api_key = "not-needed"  # Local, no key required
response = openai.ChatCompletion.create(
    model="mistral",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Health Checks

```bash
# Check if local server is running
curl http://127.0.0.1:8000/health

# List available models
curl http://127.0.0.1:8000/v1/models

# Test inference
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
``
