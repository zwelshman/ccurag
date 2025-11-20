# Deployment Guide for Streamlit Cloud

This guide explains how to deploy the BHFDSC Q&A application to Streamlit Cloud.

## Important Notes on Streamlit Cloud Deployment

### Persistent Storage Limitation

**Critical:** Streamlit Cloud does not provide persistent file storage. This means:

- The vector database (`chroma_db/`) will be lost when the app restarts
- You'll need to re-index repositories after each deployment or restart
- Indexing takes 10-30 minutes each time

### Solutions for Production

For production deployments, consider these alternatives:

1. **Cloud Vector Database**: Use Pinecone, Weaviate, or Qdrant instead of ChromaDB
2. **Pre-built Index**: Build the index locally and commit it to your repository (if size permits)
3. **External Storage**: Use cloud storage (S3, GCS) to persist the vector database
4. **Self-hosting**: Deploy on a platform with persistent storage (Heroku, DigitalOcean, AWS, etc.)

## Deployment Steps

### 1. Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Anthropic API key

### 2. Push Code to GitHub

```bash
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `app.py`
5. Click "Deploy"

### 4. Configure Secrets

In the Streamlit Cloud dashboard:

1. Click on your app
2. Go to "Settings" → "Secrets"
3. Add your secrets in TOML format:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
GITHUB_TOKEN = "your_github_token_here"  # Optional but recommended
GITHUB_ORG = "BHFDSC"
```

4. Click "Save"

### 5. Index Repositories

After deployment:

1. Open your deployed app
2. Go to the "Setup" page (it should open automatically if no index exists)
3. Click "Index/Re-index Repositories"
4. Confirm and wait 10-30 minutes for indexing to complete
5. Once complete, go to the "Q&A" page to start asking questions

## Configuration Options

### Python Version

Create a `.python-version` file to specify the Python version:

```
3.11
```

### System Dependencies

If you need additional system packages, create a `packages.txt` file:

```
build-essential
```

### Python Dependencies

The `requirements.txt` file is already configured with all necessary dependencies.

## Limitations on Streamlit Cloud

1. **Memory**: 1 GB RAM (may need to reduce `MAX_FILES_PER_REPO` in config.py)
2. **Storage**: No persistent storage (vector DB is lost on restart)
3. **CPU**: Limited CPU resources (indexing may be slower)
4. **Timeout**: Long-running operations may timeout

## Optimizations for Streamlit Cloud

To improve performance and reduce resource usage on Streamlit Cloud:

### 1. Reduce Files Per Repository

Edit `config.py`:

```python
MAX_FILES_PER_REPO = 20  # Reduced from 50
```

### 2. Adjust Chunk Size

Edit `config.py`:

```python
CHUNK_SIZE = 500  # Reduced from 1000
CHUNK_OVERLAP = 100  # Reduced from 200
```

### 3. Limit File Types

Edit `config.py` to index only essential file types:

```python
INDEXED_FILE_EXTENSIONS = [".md", ".py", ".ipynb"]  # Only critical files
```

## Monitoring and Troubleshooting

### View Logs

In Streamlit Cloud:
1. Click on your app
2. Click "Manage app"
3. Click "Logs" to view application logs

### Common Issues

**Out of Memory Error:**
- Reduce `MAX_FILES_PER_REPO` in config.py
- Reduce `CHUNK_SIZE` in config.py
- Index fewer file types

**Timeout During Indexing:**
- Reduce `MAX_FILES_PER_REPO`
- Consider indexing specific repositories instead of all

**API Rate Limiting:**
- Add a `GITHUB_TOKEN` to your secrets
- Reduce number of repositories processed

## Alternative: Local Development

For development and testing, you can run the app locally:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Index repositories (one-time, takes 10-30 minutes)
python index_repos.py

# Run the app
streamlit run app.py
```

## Support

For issues or questions:
- Check the [Streamlit documentation](https://docs.streamlit.io)
- Review the main [README.md](README.md)
- Open an issue on GitHub
