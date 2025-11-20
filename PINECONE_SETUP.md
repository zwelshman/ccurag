# Pinecone Integration Setup Guide

This application now uses **Pinecone** as the default vector database backend, providing persistent cloud storage for your vector embeddings.

## Quick Start

### 1. Get Your Pinecone API Key

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up or log in to your account
3. Navigate to **API Keys** in the console
4. Copy your API key

### 2. Configure Environment Variables

Add your Pinecone API key to your `.env` file:

```bash
# Copy the example file if you haven't already
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional (these are the defaults)
VECTOR_STORE_BACKEND=pinecone
PINECONE_INDEX_NAME=ccuindex
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIMENSION=384
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will automatically:
- Connect to your Pinecone account
- Create the index if it doesn't exist (first run)
- Use the existing index if already created

## Index Configuration

Your Pinecone index is configured with:
- **Name**: `ccuindex` (or whatever you set in PINECONE_INDEX_NAME)
- **Dimensions**: 384 (matches `sentence-transformers/all-MiniLM-L6-v2` model)
- **Metric**: Cosine similarity
- **Type**: Serverless (free tier available)
- **Cloud**: AWS us-east-1

## Advantages of Pinecone

✅ **Persistent Storage**: Data persists across app restarts
✅ **Cloud-Native**: No local storage needed
✅ **Scalable**: Handles millions of vectors
✅ **Managed**: No infrastructure to maintain
✅ **Fast**: Optimized for vector similarity search
✅ **Free Tier**: 100K vectors included

## Switching Between Backends

The application supports both Pinecone and ChromaDB. To switch:

### Use Pinecone (Default)
```env
VECTOR_STORE_BACKEND=pinecone
```

### Use ChromaDB (Local)
```env
VECTOR_STORE_BACKEND=chroma
```

## Deployment to Streamlit Cloud

### Add Secrets

In Streamlit Cloud, go to **App Settings** → **Secrets** and add:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"
VECTOR_STORE_BACKEND = "pinecone"
PINECONE_INDEX_NAME = "ccuindex"
```

### Deploy

1. Push your code to GitHub
2. Deploy via [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in the app settings
4. Index your repositories once via the Setup page
5. Data persists across deployments! 🎉

## Troubleshooting

### "Index does not exist" Error

The app will automatically create the index on first use. If you see this error:

1. Go to the **Setup** page
2. Click **"Index/Re-index Repositories"**
3. The app will create the index and populate it

### Dimension Mismatch Error

If you see a dimension mismatch error:

1. Delete your existing Pinecone index in the console
2. Let the app recreate it with correct dimensions (384)
3. Re-index your repositories

### API Key Not Found

Make sure your `.env` file contains:
```env
PINECONE_API_KEY=your_actual_api_key
```

Or for Streamlit Cloud, add it to Secrets.

## Cost Considerations

**Free Tier** (Starter):
- 100,000 vectors
- 1 project
- 5 indexes per project
- Serverless deployment

This is sufficient for the BHFDSC repository indexing (~50-100K vectors depending on settings).

**Paid Plans**: Starting at $70/month for larger usage.

## Architecture

```
┌─────────────────┐
│   Streamlit     │
│   Frontend      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ VectorStoreMan- │────▶│   Pinecone   │
│    ager         │     │    Cloud     │
│  (Factory)      │     │   Index      │
└─────────────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│  Pinecone       │
│  Vector Store   │
│  Implementation │
└─────────────────┘
```

## Performance Tips

1. **Batch Size**: Default is 100 vectors per batch (optimal for most use cases)
2. **Index Type**: Serverless is recommended for variable workloads
3. **Region**: Choose region closest to your users (default: us-east-1)
4. **Namespaces**: Currently not used, but can segment data if needed in future

## Security

- API keys are stored in environment variables or Streamlit secrets
- Never commit `.env` files to version control
- Pinecone uses TLS encryption in transit
- Data is encrypted at rest

## Support

For Pinecone-specific issues:
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Support](https://support.pinecone.io/)

For application issues:
- Check the Streamlit app logs
- Review the application logs in the console
- Open an issue on GitHub

## Migration from ChromaDB

If you previously used ChromaDB:

1. Update your `.env`: `VECTOR_STORE_BACKEND=pinecone`
2. Add your `PINECONE_API_KEY`
3. Run the app - the abstraction layer handles the rest!
4. Re-index your repositories (one-time only)
5. Your old `chroma_db/` directory can be deleted

## Next Steps

Once configured:
1. ✅ Start the Streamlit app
2. ✅ Go to the **Setup** page
3. ✅ Click **"Index/Re-index Repositories"**
4. ✅ Wait for indexing to complete (5-30 minutes)
5. ✅ Use the **Q&A** page to ask questions!

Your vector database is now persistent and will survive app restarts! 🚀
