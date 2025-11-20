# Vector Database Implementation Examples

This directory contains example implementations for different persistent vector database backends.

## Overview

The main application currently uses **ChromaDB** with local file persistence. While this works for local development, it has limitations for cloud deployments (like Streamlit Cloud). These examples show how to migrate to cloud-native vector database solutions.

## Available Implementations

### 1. Cloud Storage + ChromaDB (`cloud_storage_sync.py`)
**Quick migration path - minimal code changes**

Syncs the local ChromaDB directory to/from cloud storage (AWS S3, Google Cloud Storage, or Azure Blob Storage).

- **Complexity**: Low-Medium
- **Cost**: Very Low (~$1-5/month)
- **Setup Time**: 1-2 hours
- **Best For**: Quick deployment to Streamlit Cloud

**Pros:**
- Keeps existing ChromaDB code
- Minimal changes required
- Works with S3, GCS, or Azure

**Cons:**
- Download/upload overhead on startup
- Manual sync management

### 2. Pinecone (`pinecone_backend.py`)
**Production-ready managed solution**

Fully managed cloud vector database with excellent performance and scalability.

- **Complexity**: Medium
- **Cost**: Free tier (100K vectors), then $70+/month
- **Setup Time**: 4-6 hours
- **Best For**: Production applications

**Pros:**
- Zero infrastructure management
- Excellent performance
- Built-in monitoring
- Simple API

**Cons:**
- Vendor lock-in
- Cost at scale

### 3. Supabase + pgvector (`supabase_backend.py`)
**Budget-friendly PostgreSQL solution**

PostgreSQL with vector extension, hosted on Supabase with generous free tier.

- **Complexity**: Medium-High
- **Cost**: Free tier (500MB), then $25+/month
- **Setup Time**: 6-8 hours
- **Best For**: Projects needing both structured and vector data

**Pros:**
- Generous free tier
- Combines relational + vector data
- Open source (self-hostable)
- PostgreSQL ecosystem

**Cons:**
- More complex setup
- Requires SQL knowledge

## Quick Comparison

| Feature | Cloud Storage + ChromaDB | Pinecone | Supabase |
|---------|-------------------------|----------|----------|
| **Setup** | ⭐⭐ Easy | ⭐⭐ Easy | ⭐⭐⭐⭐ Complex |
| **Cost** | $ Very Low | $$ Medium | $ Low |
| **Free Tier** | ✅ Yes | ⚠️ Limited | ✅ Generous |
| **Managed** | ❌ No | ✅ Yes | ✅ Yes |
| **Scalability** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |
| **Code Changes** | Minimal | Medium | Medium-High |

## Getting Started

1. **Read the proposal**: Start with `../PERSISTENT_VECTOR_DB_PROPOSAL.md` for a detailed comparison
2. **Choose your backend**: Based on your requirements (cost, complexity, features)
3. **Follow the setup instructions**: Each file contains detailed setup instructions in its docstrings
4. **Test locally**: Verify the implementation works before deploying
5. **Deploy**: Push to Streamlit Cloud or your preferred platform

## Installation

Each implementation requires specific dependencies:

### For Cloud Storage + ChromaDB:
```bash
# AWS S3
pip install boto3

# Google Cloud Storage
pip install google-cloud-storage

# Azure Blob Storage
pip install azure-storage-blob
```

### For Pinecone:
```bash
pip install pinecone[grpc]
```

### For Supabase:
```bash
pip install supabase
```

## Configuration

Add the appropriate credentials to your `.env` file or Streamlit secrets:

### Cloud Storage + ChromaDB
```env
# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# GCS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Azure
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

### Pinecone
```env
PINECONE_API_KEY=your_api_key
```

### Supabase
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
```

## Integration Guide

### Step 1: Update config.py
Add the necessary configuration for your chosen backend:

```python
# In config.py
class Config:
    # ... existing config ...

    # For Pinecone
    PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

    # For Supabase
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_KEY = get_secret("SUPABASE_KEY")
```

### Step 2: Replace VectorStoreManager
In your app files (`app.py`, `qa_chain.py`), replace the import:

```python
# OLD:
from vector_store import VectorStoreManager

# NEW (for Pinecone):
from examples.pinecone_backend import PineconeVectorStore as VectorStoreManager

# NEW (for Supabase):
from examples.supabase_backend import SupabaseVectorStore as VectorStoreManager
```

### Step 3: Update Initialization
Update how you initialize the vector store:

```python
# Pinecone
vector_store = PineconeVectorStore(
    api_key=Config.PINECONE_API_KEY,
    index_name="bhfdsc-repos"
)

# Supabase
vector_store = SupabaseVectorStore(
    supabase_url=Config.SUPABASE_URL,
    supabase_key=Config.SUPABASE_KEY
)
```

## Testing

Before deploying, test your implementation locally:

```python
# Test script
from config import Config
from examples.pinecone_backend import PineconeVectorStore

# Initialize
store = PineconeVectorStore(api_key=Config.PINECONE_API_KEY)

# Test indexing with sample documents
docs = [
    {"content": "Test document", "metadata": {"source": "test"}}
]
store.create_vectorstore(docs)

# Test search
results = store.similarity_search("test query", k=5)
print(f"Found {len(results)} results")
```

## Troubleshooting

### Cloud Storage + ChromaDB
- **Issue**: Download fails on startup
- **Solution**: Check cloud credentials and bucket permissions

### Pinecone
- **Issue**: Index creation fails
- **Solution**: Verify API key and region availability

### Supabase
- **Issue**: RPC function not found
- **Solution**: Ensure you've run the SQL setup script in Supabase

## Performance Considerations

- **ChromaDB + Cloud Storage**: Slow startup (download time)
- **Pinecone**: Fast, optimized for vector search
- **Supabase**: Good for <100K vectors, slower at scale

## Migration Path

### Recommended Approach:

1. **Phase 1** (Week 1): Implement Cloud Storage + ChromaDB for immediate deployment
2. **Phase 2** (Week 2-3): Evaluate and implement Pinecone or Supabase
3. **Phase 3** (Week 4): Full migration and cleanup

## Support

For detailed information, see:
- Main proposal: `../PERSISTENT_VECTOR_DB_PROPOSAL.md`
- Code examples: Each `.py` file contains detailed docstrings
- Setup instructions: Run `python filename.py` to see setup instructions

## Contributing

These are example implementations. You may need to adapt them to your specific use case:
- Adjust batch sizes for your dataset
- Modify chunking parameters
- Add error handling
- Implement monitoring
- Add retry logic

## License

These examples are provided as-is for educational and implementation purposes.
