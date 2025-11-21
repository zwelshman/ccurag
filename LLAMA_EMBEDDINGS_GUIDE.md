# Guide: Integrating Llama Embeddings into CCuRAG

## Overview

This guide explains how to integrate Llama-based embeddings into your RAG (Retrieval Augmented Generation) system. Currently, the system uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). This guide provides multiple options for upgrading to Llama-based embeddings.

---

## Current Architecture

**Files Involved:**
- `config.py` - Configuration (line 51, 55)
- `vector_store_pinecone.py` - Embedding generation (lines 34-39, 169, 291, 340)

**Current Flow:**
```
Documents → SentenceTransformer (all-MiniLM-L6-v2) → 384-dim vectors → Pinecone
Query → SentenceTransformer (same model) → 384-dim vector → Similarity Search
```

---

## Option 1: Llama-Compatible SentenceTransformers (Recommended)

### Best Models

| Model | Dimensions | Size | Best For |
|-------|-----------|------|----------|
| `BAAI/llm-embedder` | 768 | 1.5GB | Llama-2 based, optimized for RAG |
| `thenlper/gte-large` | 1024 | 670MB | High quality, balanced |
| `jinaai/jina-embeddings-v2-base-en` | 768 | 560MB | Fast, code + docs |
| `hkunlp/instructor-large` | 768 | 1.3GB | Task-specific instructions |

### Implementation Steps

#### Step 1: Update `config.py`

Change lines 51 and 55:

```python
# Before:
PINECONE_DIMENSION = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# After (for llm-embedder):
PINECONE_DIMENSION = 768
EMBEDDING_MODEL = "BAAI/llm-embedder"
```

**That's it!** The existing `SentenceTransformer` code in `vector_store_pinecone.py` will automatically work with these models.

#### Step 2: Re-index Your Data

After changing the model:
1. Delete the existing Pinecone index (or create a new index name)
2. Re-run the indexing process from the Setup page in the Streamlit app
3. The new embeddings will be 768-dimensional

---

## Option 2: Direct Llama Model with Custom Pooling

For using base Llama models directly (e.g., `meta-llama/Llama-2-7b-hf`), you need custom pooling logic.

### Implementation

Create a new file `llama_embedder.py`:

```python
"""Llama-based embedding generation."""
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List

logger = logging.getLogger(__name__)


class LlamaEmbedder:
    """Generate embeddings using Llama models."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = None):
        """
        Initialize Llama embedder.

        Args:
            model_name: Hugging Face model ID
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading Llama model '{model_name}' on {self.device}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device
        )
        self.model.eval()

        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"✓ Llama model loaded successfully")

    def encode(self, texts: List[str] | str, batch_size: int = 8) -> List[List[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors (as lists of floats)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state with mean pooling
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings[0] if single_input else all_embeddings

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.

        Args:
            token_embeddings: Token-level embeddings [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Sentence embeddings [batch, hidden_dim]
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask
```

### Update `config.py`:

```python
# Model Settings
ANTHROPIC_MODEL = "claude-opus-4-1"
EMBEDDING_MODEL = "meta-llama/Llama-2-7b-hf"  # Or meta-llama/Llama-3-8b
EMBEDDING_TYPE = "llama"  # New: "llama" or "sentence-transformer"

# Pinecone Settings
PINECONE_DIMENSION = 4096  # Llama-2-7b hidden size
```

### Update `vector_store_pinecone.py`:

Modify the `__init__` method (around line 34):

```python
# OLD CODE:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Loading SentenceTransformer model '{Config.EMBEDDING_MODEL}' on device: {device}")
logger.info("This may take 1-5 minutes on first run (downloading model from HuggingFace)...")
self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
logger.info("✓ SentenceTransformer model loaded successfully")

# NEW CODE:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_type = getattr(Config, 'EMBEDDING_TYPE', 'sentence-transformer')

if embedding_type == 'llama':
    from llama_embedder import LlamaEmbedder
    logger.info(f"Loading Llama model '{Config.EMBEDDING_MODEL}' on device: {device}")
    logger.info("This may take 5-10 minutes on first run (downloading large model)...")
    self.embedding_model = LlamaEmbedder(Config.EMBEDDING_MODEL, device=device)
    logger.info("✓ Llama model loaded successfully")
else:
    logger.info(f"Loading SentenceTransformer model '{Config.EMBEDDING_MODEL}' on device: {device}")
    logger.info("This may take 1-5 minutes on first run (downloading model from HuggingFace)...")
    self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
    logger.info("✓ SentenceTransformer model loaded successfully")
```

### Update `requirements.txt`:

Add:
```
transformers>=4.36.0
accelerate>=0.25.0
```

---

## Option 3: llamafile (Local Embeddings Server)

Run Llama embeddings as a local API server.

### Setup Steps

1. **Download llamafile:**
```bash
# Download llama.cpp embeddings server
wget https://github.com/Mozilla-Ocho/llamafile/releases/latest/download/llamafile
chmod +x llamafile
```

2. **Create embedding API wrapper** (`llamafile_embedder.py`):

```python
"""Llamafile embedding client."""
import requests
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class LlamafileEmbedder:
    """Client for llamafile embedding server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize client.

        Args:
            base_url: llamafile server URL
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized llamafile client: {self.base_url}")

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Encode texts using llamafile.

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings as list(s) of floats
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/embedding",
                json={"content": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])

        return embeddings[0] if single_input else embeddings

    def tolist(self):
        """Compatibility method for SentenceTransformer API."""
        return self
```

3. **Update `config.py`:**
```python
EMBEDDING_MODEL = "llamafile"
EMBEDDING_TYPE = "llamafile"
PINECONE_DIMENSION = 4096  # Depends on Llama model used
LLAMAFILE_URL = "http://localhost:8080"
```

4. **Update `vector_store_pinecone.py` (add to __init__):**

```python
elif embedding_type == 'llamafile':
    from llamafile_embedder import LlamafileEmbedder
    llamafile_url = getattr(Config, 'LLAMAFILE_URL', 'http://localhost:8080')
    logger.info(f"Connecting to llamafile server: {llamafile_url}")
    self.embedding_model = LlamafileEmbedder(llamafile_url)
    logger.info("✓ Llamafile client initialized")
```

5. **Start llamafile server:**
```bash
./llamafile --model llama-2-7b.gguf --embedding --port 8080
```

---

## Option 4: OpenAI-Compatible API (Llama.cpp Server)

Use llama.cpp server with OpenAI-compatible endpoint.

### Setup

1. **Install llama.cpp server:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

2. **Start server:**
```bash
./server -m models/llama-2-7b.gguf --embedding --port 8080
```

3. **Create API client** (`llama_api_embedder.py`):

```python
"""Llama.cpp API embedding client."""
import requests
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class LlamaAPIEmbedder:
    """Client for llama.cpp server (OpenAI-compatible)."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized llama.cpp API client: {self.base_url}")

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode texts via llama.cpp API."""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={
                "input": texts,
                "model": "llama-2"
            }
        )
        response.raise_for_status()

        embeddings = [item["embedding"] for item in response.json()["data"]]
        return embeddings[0] if single_input else embeddings
```

---

## Comparison Table

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **SentenceTransformer-compatible** | Easy (2-line change), fast | Limited to pre-trained models | Quick upgrade, production |
| **Direct Llama (Transformers)** | Full control, latest models | Requires GPU, slower, more code | Research, customization |
| **llamafile** | Self-contained, portable | Extra process, network overhead | Local deployment |
| **llama.cpp API** | Fast inference, quantized | Separate server, setup complexity | High-performance production |

---

## Migration Checklist

When switching embedding models:

- [ ] **Update `config.py`** with new model name and dimensions
- [ ] **Update embedding dimension** (`PINECONE_DIMENSION`)
- [ ] **Delete old Pinecone index** OR create new index name
- [ ] **Re-index all documents** (embeddings are not compatible across models)
- [ ] **Test query performance** on sample questions
- [ ] **Monitor GPU/CPU usage** and inference speed
- [ ] **Update `requirements.txt`** if using new libraries

---

## Recommended Configuration

For most users, **Option 1 with `BAAI/llm-embedder`** is recommended:

```python
# config.py
PINECONE_DIMENSION = 768
EMBEDDING_MODEL = "BAAI/llm-embedder"
```

**Why?**
- Llama-2 based architecture
- Optimized for retrieval tasks
- Drop-in replacement (no code changes in vector_store_pinecone.py)
- Good balance of quality and speed
- Works with existing SentenceTransformer code

---

## Performance Considerations

### Memory Requirements

| Model Type | VRAM (GPU) | RAM (CPU) |
|------------|-----------|-----------|
| all-MiniLM-L6-v2 | 0.5 GB | 1 GB |
| llm-embedder | 2 GB | 4 GB |
| Llama-2-7b (full) | 14 GB | 28 GB |
| Llama-2-7b (quantized) | 4 GB | 8 GB |

### Inference Speed (approximate)

| Model | Embeddings/sec (GPU) | Embeddings/sec (CPU) |
|-------|---------------------|---------------------|
| all-MiniLM-L6-v2 | 500-1000 | 100-200 |
| llm-embedder | 100-300 | 20-50 |
| Llama-2-7b | 20-50 | 2-5 |

---

## Testing

After changing the embedding model:

```python
# Test script (test_embeddings.py)
from config import Config
from vector_store_pinecone import PineconeVectorStore

# Initialize
vs = PineconeVectorStore()

# Test embedding generation
test_text = "What is machine learning?"
embedding = vs.embedding_model.encode(test_text)

print(f"Model: {Config.EMBEDDING_MODEL}")
print(f"Embedding dimension: {len(embedding)}")
print(f"Expected dimension: {Config.PINECONE_DIMENSION}")
assert len(embedding) == Config.PINECONE_DIMENSION, "Dimension mismatch!"
print("✓ Test passed!")
```

---

## Troubleshooting

### Issue: "Dimension mismatch"
**Solution:** Ensure `PINECONE_DIMENSION` matches the model's output dimension.

### Issue: "CUDA out of memory"
**Solutions:**
- Use CPU: Set `device='cpu'` in the embedder
- Use smaller model: Switch to `BAAI/bge-small-en-v1.5` (384 dims)
- Use quantization: Load with `torch_dtype=torch.float16`

### Issue: "Slow inference on CPU"
**Solutions:**
- Enable ONNX runtime: `pip install onnxruntime`
- Use lighter model: `sentence-transformers/all-MiniLM-L12-v2`
- Increase batch size for bulk encoding

---

## Further Reading

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [BAAI/llm-embedder on Hugging Face](https://huggingface.co/BAAI/llm-embedder)
- [Llama 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Pinecone Embeddings Guide](https://docs.pinecone.io/docs/embeddings)

---

## Summary: Quick Start

**For immediate upgrade with minimal changes:**

1. Edit `config.py`:
   ```python
   PINECONE_DIMENSION = 768
   EMBEDDING_MODEL = "BAAI/llm-embedder"
   ```

2. Delete old Pinecone index from the Streamlit app

3. Re-index your GitHub repositories

4. Done! You're now using Llama-based embeddings.

No other code changes required.
