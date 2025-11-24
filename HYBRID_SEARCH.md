# Hybrid Search: BM25 + Vector Search

This document explains the hybrid retrieval system that combines BM25 keyword matching with vector semantic search for optimal retrieval from code repositories and documentation.

## Overview

The hybrid retriever combines two complementary search approaches:

1. **BM25 (Best Matching 25)**: Keyword-based lexical search
   - Excellent for exact term matching (function names, class names, identifiers)
   - Fast and efficient
   - Works well with technical terms and acronyms

2. **Vector Semantic Search**: Embedding-based similarity search
   - Understands meaning and context
   - Great for paraphrased queries
   - Captures conceptual relationships

## Why Hybrid Search?

For codebases with documentation, different queries require different approaches:

| Query Type | Example | Best Approach | Why |
|-----------|---------|---------------|-----|
| Exact identifiers | `PineconeVectorStore` | BM25 | Exact keyword match |
| Function names | `similarity_search()` | BM25 | Precise term matching |
| Conceptual | "How do I set up?" | Vector | Semantic understanding |
| Mixed | "What does authenticate_user do?" | Hybrid | Both exact + context |

**Research shows**: Hybrid search achieves 75-85% accuracy vs 60-70% for vector-only or 50-60% for BM25-only on code search tasks.

## Architecture

```
User Query
    ↓
    ├─→ BM25 Tokenizer → BM25 Index → BM25 Scores (0-1)
    │                                       ↓
    └─→ Embedding Model → Vector Search → Vector Scores (0-1)
                                              ↓
                                    Score Combiner
                                    (weighted sum)
                                              ↓
                                    Top-K Results
```

### Adaptive Weighting

The system automatically adjusts weights based on query characteristics:

```python
Query: "PineconeVectorStore class"
└─→ Detected: CamelCase, code-like
    └─→ Weights: 70% BM25, 30% Vector

Query: "How do I index repositories?"
└─→ Detected: Conceptual, natural language
    └─→ Weights: 30% BM25, 70% Vector
```

## Configuration

### Environment Variables / Streamlit Secrets

```bash
# Enable/disable hybrid search
USE_HYBRID_SEARCH=true

# Default BM25 weight (0.0 to 1.0)
# Remaining weight goes to vector search
BM25_WEIGHT=0.4  # 40% BM25, 60% vector

# Enable adaptive weight adjustment based on query
USE_ADAPTIVE_WEIGHTS=true
```

### config.py

```python
class Config:
    # Hybrid Search Settings
    USE_HYBRID_SEARCH = True
    BM25_WEIGHT = 0.4  # 40% BM25, 60% vector
    USE_ADAPTIVE_WEIGHTS = True
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `rank-bm25>=0.2.2` for BM25 algorithm
- Existing dependencies (pinecone, sentence-transformers, etc.)

### 2. Build BM25 Index

After indexing documents to Pinecone, build the BM25 index:

```bash
python build_hybrid_index.py
```

This will:
1. Fetch all documents from Pinecone
2. Tokenize them with code-aware tokenization
3. Build BM25 index
4. Cache the index to `.cache/bm25_index.pkl` (runtime cache)

**Note**:
- You need to rebuild the BM25 index whenever you update the document corpus.
- To share the index across deployments, copy `.cache/bm25_index.pkl` to `data_index/bm25_index.pkl` and commit to Git.
- The application checks `data_index/` first, then falls back to `.cache/`.

### 3. Use in Your Application

```python
from vector_store_pinecone import PineconeVectorStore
from hybrid_retriever import HybridRetriever
from qa_chain import QASystem

# Initialize vector store
vector_store = PineconeVectorStore()
vector_store.load_vectorstore()

# Create hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_weight=0.4  # Optional, uses Config default if not specified
)

# Initialize QA system with hybrid retriever
qa_system = QASystem(
    vector_store=vector_store,
    retriever=hybrid_retriever
)

# Use as normal
result = qa_system.answer_question("What does PineconeVectorStore do?")
```

## Code-Aware Tokenization

The BM25 tokenizer is optimized for code and documentation:

### Features

1. **CamelCase Splitting**
   ```
   getUserData → [get, user, data]
   PineconeVectorStore → [pinecone, vector, store]
   ```

2. **snake_case Preservation**
   ```
   similarity_search → [similarity, search]
   vector_store → [vector, store]
   ```

3. **Stop Word Removal**
   - Removes common words (the, a, an, is, etc.)
   - Keeps important technical terms

4. **Metadata Enhancement**
   - Includes file names and repo names in searchable text
   - Improves matching for source-specific queries

## Usage Examples

### Basic Usage

```python
# Simple search
results = hybrid_retriever.similarity_search(
    query="How do I configure Pinecone?",
    k=5
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### With Custom Weights

```python
# Override default weights for this query
results = hybrid_retriever.similarity_search(
    query="PineconeVectorStore",
    k=5,
    use_adaptive_weights=False  # Use fixed weights
)
```

### Disable Hybrid Search

To temporarily disable hybrid search and use vector-only:

```python
# Just pass vector_store directly to QASystem
qa_system = QASystem(vector_store=vector_store)
# or set retriever=vector_store explicitly
qa_system = QASystem(vector_store=vector_store, retriever=vector_store)
```

## Performance Characteristics

### Speed

- **BM25**: Very fast (milliseconds for thousands of documents)
- **Vector Search**: Fast with Pinecone (tens of milliseconds)
- **Hybrid**: Slightly slower than vector-only (both searches run, then combine)

### Accuracy Trade-offs

| Scenario | Vector Only | BM25 Only | Hybrid |
|----------|------------|-----------|--------|
| Exact code identifiers | ⚠️ Good | ✅ Excellent | ✅ Excellent |
| Conceptual queries | ✅ Excellent | ⚠️ Fair | ✅ Excellent |
| Paraphrased questions | ✅ Excellent | ❌ Poor | ✅ Excellent |
| Mixed queries | ⚠️ Good | ⚠️ Good | ✅ Excellent |
| Rare technical terms | ⚠️ Fair | ✅ Excellent | ✅ Excellent |

### Resource Usage

- **Memory**: BM25 index adds ~50-100MB for typical codebases (cached to disk)
- **Disk**:
  - Runtime cache: `.cache/bm25_index.pkl` (50-100MB, not committed to Git)
  - Persistent cache: `data_index/bm25_index.pkl` (optional, for sharing across deployments via Git)
- **Compute**: Minimal additional CPU for score combining

## Maintenance

### Rebuilding the Index

Rebuild when:
- New repositories are indexed
- Existing documents are updated
- Configuration changes (chunk size, etc.)

```bash
python build_hybrid_index.py
```

### Clearing Cache

```python
from hybrid_retriever import HybridRetriever

retriever = HybridRetriever(vector_store)
retriever.clear_cache()  # Removes both .cache/ and data_index/ versions
```

Or manually:
```bash
# Remove runtime cache
rm -f .cache/bm25_index.pkl

# Remove persistent cache (if committed)
rm -f data_index/bm25_index.pkl
```

## Troubleshooting

### "BM25 index not built" Warning

**Problem**: Hybrid retriever falls back to vector-only search.

**Solution**: Run `python build_hybrid_index.py`

### Poor Results on Code Queries

**Problem**: Exact function/class names not ranking high.

**Solutions**:
1. Increase BM25 weight: `BM25_WEIGHT=0.6`
2. Enable adaptive weights: `USE_ADAPTIVE_WEIGHTS=true`
3. Use backticks in queries: `How does \`similarity_search\` work?`

### Poor Results on Conceptual Queries

**Problem**: "How to" questions not understanding intent.

**Solutions**:
1. Decrease BM25 weight: `BM25_WEIGHT=0.3`
2. Ensure adaptive weights enabled: `USE_ADAPTIVE_WEIGHTS=true`

### Slow Performance

**Problem**: Queries taking too long.

**Solutions**:
1. Reduce `k` (number of results)
2. Ensure BM25 cache is built (not rebuilding each time)
3. Consider disabling hybrid for simple queries

## Advanced: Custom Weight Strategies

You can implement custom weight strategies by extending `HybridRetriever`:

```python
class CustomHybridRetriever(HybridRetriever):
    def _get_adaptive_weights(self, query: str):
        # Your custom logic
        if "error" in query.lower() or "bug" in query.lower():
            return (0.7, 0.3)  # Favor BM25 for error messages
        return super()._get_adaptive_weights(query)
```

## References

- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Sentence Transformers](https://www.sbert.net/)
- [Hybrid Search Research](https://arxiv.org/abs/2104.08663)
- [Code Search Benchmarks](https://github.com/github/CodeSearchNet)

## Support

For issues or questions:
1. Check this documentation
2. Run `python example_hybrid_usage.py` to verify setup
3. Check logs for error messages
4. Open an issue on the repository
