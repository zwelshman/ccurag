# Cloud Persistence Setup for Streamlit Cloud

This guide explains how to persist the hybrid retriever (BM25 index) and code intelligence metadata when deploying to Streamlit Cloud or other ephemeral hosting environments.

## Problem

Streamlit Cloud has **ephemeral storage** - any files written to disk are lost when the app restarts. This affects:
- **BM25 Index** (`.cache/bm25_index.pkl`) - ~10-50MB pickle file
- **Code Metadata** (`.cache/code_metadata.json`) - JSON file with code intelligence data

Without cloud persistence, these files need to be rebuilt on every app restart (can take 5-10 minutes).

## Solution

The application now supports **cloud-based persistence** with automatic fallback:

1. **Google Drive** (Recommended) - Free 15GB with Google account
2. **AWS S3** (Alternative) - Pay-as-you-go cloud storage
3. **Local Storage** (Fallback) - For development

## Setup Guide: Google Drive (Recommended)

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID

### Step 2: Enable Google Drive API

1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Google Drive API"
3. Click **Enable**

### Step 3: Create Service Account

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **Service Account**
3. Fill in details:
   - **Service account name**: `ccurag-storage` (or any name)
   - **Service account ID**: Auto-generated
   - **Description**: "Service account for CCURAG cache storage"
4. Click **Create and Continue**
5. Skip optional steps, click **Done**

### Step 4: Generate Service Account Key

1. Click on the service account you just created
2. Go to **Keys** tab
3. Click **Add Key** → **Create new key**
4. Choose **JSON** format
5. Click **Create** - a JSON file will download

### Step 5: Configure Streamlit Secrets

#### For Streamlit Cloud:

1. Go to your app on [Streamlit Cloud](https://share.streamlit.io/)
2. Click **Settings** → **Secrets**
3. Add the following (paste the entire JSON file content):

```toml
# Copy the ENTIRE content of your service account JSON file
GDRIVE_CREDENTIALS_JSON = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "ccurag-storage@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}
'''

# Also add your other secrets
ANTHROPIC_API_KEY = "sk-ant-..."
PINECONE_API_KEY = "..."
GITHUB_TOKEN = "ghp_..."
```

#### For Local Development:

Add to `.env` file:

```bash
GDRIVE_CREDENTIALS_JSON='{"type":"service_account",...}'
```

Or use the JSON file path:

```bash
GDRIVE_CREDENTIALS_JSON="/path/to/service-account-key.json"
```

### Step 6: Deploy and Verify

1. Push your code to GitHub
2. Streamlit Cloud will automatically redeploy
3. Check logs for:
   ```
   ✓ Google Drive storage initialized (folder: ccurag-cache)
   ```

## How It Works

### Automatic Storage Selection

The `CloudStorage` class automatically chooses the best available backend:

1. **Google Drive** - If `GDRIVE_CREDENTIALS_JSON` is set
2. **AWS S3** - If `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set
3. **Local Files** - Fallback for development

### Storage Structure

Google Drive will contain:
```
ccurag-cache/           (folder created automatically)
├── bm25_index.pkl      (BM25 search index)
└── code_metadata.json  (Code intelligence metadata)
```

### Cache Loading Flow

```mermaid
graph LR
    A[App Starts] --> B{Cloud<br/>Creds?}
    B -->|Yes| C[Load from<br/>Google Drive]
    B -->|No| D[Check Local<br/>Cache]
    C --> E[✓ Ready]
    D -->|Found| E
    D -->|Not Found| F[Rebuild Indices]
    F --> G[Save to Cloud]
    G --> E
```

## Building and Uploading Initial Cache

You need to build the cache files once before deploying to Streamlit Cloud:

### Option 1: Build Locally and Upload

1. **Build indices locally**:
   ```bash
   # Build vector store (Pinecone - already cloud-hosted)
   python build_index.py

   # Build code metadata
   python build_metadata_index.py

   # Build BM25 index
   python build_hybrid_index.py
   ```

2. **Upload to Google Drive manually**:
   - Files are in `.cache/` directory
   - Run with Google Drive credentials in `.env`
   - Indices will auto-upload on next run

### Option 2: Build on Streamlit Cloud (First Run)

1. Deploy app with cloud credentials configured
2. First startup will be slow (10-15 minutes) as it builds indices
3. Subsequent startups will load from Google Drive (~10 seconds)

### Force Rebuild

To force rebuilding indices from scratch:

```bash
python build_hybrid_index.py --force-rebuild
python build_metadata_index.py --force-rebuild
```

This will rebuild and re-upload to cloud storage.

## Alternative: AWS S3 Setup

If you prefer S3 over Google Drive:

### Step 1: Create S3 Bucket

1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com/)
2. Create a new bucket (e.g., `ccurag-cache-yourusername`)
3. Keep default settings (private bucket)

### Step 2: Create IAM User

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Create user with programmatic access
3. Attach policy: `AmazonS3FullAccess` (or create custom policy)
4. Save **Access Key ID** and **Secret Access Key**

### Step 3: Configure Secrets

Add to Streamlit secrets or `.env`:

```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "ccurag-cache-yourusername"
```

## Cost Comparison

| Storage | Free Tier | Storage Cost | Transfer Cost | Total (est.) |
|---------|-----------|--------------|---------------|--------------|
| **Google Drive** | 15 GB free | Free | Free | **$0/month** |
| **AWS S3** | 5 GB, 12 months | $0.023/GB | $0.09/GB | **~$0.50-2/month** |

**Recommendation**: Use Google Drive for the free tier.

## Troubleshooting

### "Failed to initialize Google Drive"

Check:
- JSON credentials are valid
- Google Drive API is enabled in your project
- No extra quotes or formatting issues in secrets

### "File not found in Google Drive"

The cache hasn't been uploaded yet. Run:
```bash
python build_hybrid_index.py
python build_metadata_index.py
```

### "Permission denied" errors

Ensure the service account has the Drive API scope:
```python
scopes=['https://www.googleapis.com/auth/drive.file']
```

### App is slow to start

First startup builds indices (~10-15 min). Subsequent starts load from cloud (~10 sec).

### Local development works, but Streamlit Cloud fails

Check:
1. Secrets are properly configured in Streamlit Cloud settings
2. JSON formatting in secrets (use triple quotes `'''`)
3. All required packages are in `requirements.txt`

## Monitoring

View logs to confirm cloud storage is working:

```
✓ Google Drive storage initialized (folder: ccurag-cache)
✓ Loaded BM25 index with 1234 documents from cache
✓ Loaded metadata for 42 repos from cache
```

If you see local storage warnings, cloud credentials aren't configured:

```
⚠ Google Drive credentials not found
⚠ S3 credentials not found, using local storage
```

## Security Best Practices

1. **Service Account Permissions**: Only grant `drive.file` scope (not full Drive access)
2. **Rotate Keys**: Regenerate service account keys periodically
3. **Private Bucket**: Never make your S3 bucket or Drive folder public
4. **Secrets Management**: Use Streamlit secrets, not hardcoded values

## Performance

| Operation | Local | Google Drive | AWS S3 |
|-----------|-------|--------------|--------|
| **Cache Load (BM25)** | 200ms | 2-3s | 1-2s |
| **Cache Load (Metadata)** | 50ms | 500ms | 300ms |
| **Cache Save** | 100ms | 3-5s | 2-3s |

**Note**: Slightly slower than local, but enables persistence in cloud deployments.

## Migration from Local to Cloud

If you have existing local caches:

1. Configure cloud storage credentials
2. Restart the app
3. Caches will be auto-uploaded on first save
4. Verify in Google Drive web interface

## Need Help?

- Check logs for detailed error messages
- Ensure all environment variables are set
- Verify Google Drive API is enabled
- Test credentials locally first before deploying
