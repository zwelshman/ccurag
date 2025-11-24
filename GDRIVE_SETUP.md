# Google Drive Setup Guide

This guide explains how to set up Google Drive storage for the CCURAG application.

## ⚠️ Important: Service Account Limitation

**Service accounts CANNOT write to personal Google Drive.** Google changed this policy to prevent abuse.

You have two options:

### Option 1: OAuth2 (FREE ✅ Recommended for Development)

Use your **personal Google Drive** (free 15GB storage).

**Pros:**
- ✅ Free (uses your existing Google Drive)
- ✅ Easy setup for local development
- ✅ 15GB storage included

**Cons:**
- ⚠️ Requires manual OAuth flow (browser authentication)
- ⚠️ Token expires (needs refresh)
- ⚠️ Not ideal for automated deployments

### Option 2: Service Account with Shared Drive (Requires Google Workspace)

Use a **Google Workspace Shared Drive** (requires paid account).

**Pros:**
- ✅ No user interaction needed (automated)
- ✅ Good for production deployments
- ✅ Can be shared with team

**Cons:**
- ❌ Requires Google Workspace subscription ($6+/user/month)
- ❌ Must create and configure Shared Drive
- ❌ More complex setup

---

## Setup Instructions

### Option 1: OAuth2 Setup (Personal Google Drive - FREE)

#### Step 1: Create OAuth2 Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable **Google Drive API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"
4. Configure OAuth consent screen (if not already done):
   - Navigate to "APIs & Services" > "OAuth consent screen"
   - Choose "External" user type (unless you have Google Workspace)
   - Fill in required fields (app name, user support email, developer email)
   - Click "Save and Continue"
   - Skip scopes (click "Save and Continue")
   - **IMPORTANT:** Add test users:
     - Scroll to "Test users" section
     - Click "+ ADD USERS"
     - Add your Google email address
     - Click "SAVE"
5. Create OAuth2 credentials:
   - Navigate to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" (or "Web application" if you prefer)
   - Give it a name (e.g., "CCURAG Local Dev")
   - If using "Web application", add these redirect URIs:
     - `http://localhost:8080/`
     - `http://localhost:8081/`
     - `http://localhost:8082/`
   - Click "Create"
6. Download the credentials JSON file
   - Click the download icon next to your OAuth client
   - Save as `oauth_client.json` in your project directory

#### Step 2: Run the Setup Script

```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Option A: Easy setup (accepts JSON content OR file path)
python setup_gdrive_oauth_easy.py

# Option B: Original setup (file path only)
python setup_gdrive_oauth.py
```

This will:
1. Prompt you for the OAuth client JSON (file path or paste content)
2. Open a browser window for you to authorize the app
3. Create `.gdrive_token.pickle` in your project directory
4. Test the connection

**If you see "App hasn't been verified":**
- Make sure you added yourself as a test user (Step 1)
- Or click "Advanced" → "Go to [App Name] (unsafe)" → "Continue"

#### Step 3: Run Your App

The app will automatically detect and use `.gdrive_token.pickle`.

You should see:
```
INFO:cloud_storage:✓ Using OAuth2 credentials (personal Google Drive)
INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)
```

When saving files:
```
INFO:cloud_storage:✓ Saved pickle to Google Drive: bm25_index.pkl
```

#### Important Notes

- **Keep `.gdrive_token.pickle` secure** - it provides access to your Google Drive
- **Add to `.gitignore`** - don't commit this file to version control
- **Token expires** - you may need to re-run `setup_gdrive_oauth.py` periodically
- **For Streamlit Cloud** - OAuth is tricky; consider using service account with shared drive instead

---

### Option 2: Service Account Setup (Shared Drive - Requires Workspace)

#### Prerequisites

- Google Workspace account ($6+/user/month)
- Admin access to create Shared Drives

#### Step 1: Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable **Google Drive API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"
4. Create service account:
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and description
   - Click "Create and Continue"
   - Skip role assignment (click "Continue")
   - Click "Done"
5. Create and download key:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Click "Create"
   - Save the downloaded JSON file

#### Step 2: Create Shared Drive

1. Go to [Google Drive](https://drive.google.com)
2. Click "Shared drives" in the left sidebar
3. Click "New" (+ icon)
4. Name it (e.g., "CCURAG Cache")
5. Click "Create"

#### Step 3: Share with Service Account

1. Open the Shared Drive you just created
2. Click the gear icon (settings)
3. Click "Manage members"
4. Add the service account email (found in the JSON file: `client_email`)
5. Give it "Manager" or "Content manager" access
6. Click "Send"

#### Step 4: Configure .env File

Your service account JSON must be on **ONE SINGLE LINE** in the `.env` file.

**Convert the JSON:**

```bash
# Use the converter script
python convert_gdrive_creds.py

# Enter the path to your service account JSON file
# Copy the output line to your .env file
```

**Or manually:**

Open the JSON file, remove all line breaks, and paste as a single line:

```bash
GDRIVE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"your-service-account@your-project.iam.gserviceaccount.com","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}
```

**Important:**
- The `\n` in the private key should be literal `\n` (not actual newlines)
- No quotes around the entire JSON
- Everything on ONE line

#### Step 5: Update Folder Name (Optional)

In your `.env` file, set the Shared Drive folder name:

```bash
GDRIVE_FOLDER_NAME=ccurag-cache
```

This folder will be created automatically in the Shared Drive.

#### Step 6: Run Your App

The app will use the service account credentials.

You should see:
```
INFO:cloud_storage:✓ Using service account credentials (requires shared drive)
INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)
```

---

## Verification

After setup, run your app and check the logs:

### Success Messages

```
INFO:cloud_storage:✓ Using OAuth2 credentials (personal Google Drive)
# OR
INFO:cloud_storage:✓ Using service account credentials (requires shared drive)

INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)
INFO:cloud_storage:✓ Saved pickle to Google Drive: bm25_index.pkl
```

### Error Messages

#### "Google Drive credentials not found"
```
INFO:cloud_storage:⚠ Google Drive credentials not found
```
**Fix:** Run `setup_gdrive_oauth.py` or set `GDRIVE_CREDENTIALS_JSON` in `.env`

#### "Service Accounts do not have storage quota"
```
ERROR:cloud_storage:Failed to save pickle to Google Drive: <HttpError 403...storageQuotaExceeded...
```
**Fix:** Service accounts need a Shared Drive. Either:
- Use OAuth2 instead (free): run `setup_gdrive_oauth.py`
- Create a Shared Drive and share it with the service account

#### "python-dotenv could not parse statement"
```
python-dotenv could not parse statement starting at line 29
```
**Fix:** Your `GDRIVE_CREDENTIALS_JSON` has line breaks. Convert to single line using `convert_gdrive_creds.py`

#### "google-api-python-client not installed"
```
INFO:cloud_storage:⚠ google-api-python-client not installed
```
**Fix:** `pip install -r requirements.txt`

---

## Troubleshooting

See [GDRIVE_TROUBLESHOOTING.md](GDRIVE_TROUBLESHOOTING.md) for detailed troubleshooting steps.

### Quick Diagnostics

```bash
# Check if OAuth token exists
ls -la .gdrive_token.pickle

# Check .env file (don't print sensitive data!)
grep "GDRIVE_CREDENTIALS_JSON" .env | head -c 50

# Test your setup
python diagnose_gdrive.py
```

---

## For Streamlit Cloud Deployment

### Recommended: Service Account + Shared Drive

OAuth2 is difficult to use with Streamlit Cloud because it requires browser interaction.

**Steps:**
1. Follow "Option 2: Service Account Setup" above
2. Convert your service account JSON to single line
3. Add to Streamlit secrets:
   - Go to your app settings on Streamlit Cloud
   - Add to secrets:
     ```toml
     GDRIVE_CREDENTIALS_JSON = '{"type":"service_account",...}'
     GDRIVE_FOLDER_NAME = "ccurag-cache"
     ```
4. Make sure your Shared Drive is shared with the service account email

---

## Security Best Practices

1. **Never commit credentials to git:**
   ```bash
   # Add to .gitignore
   .gdrive_token.pickle
   oauth_client.json
   *_credentials.json
   .env
   ```

2. **Limit OAuth scopes:**
   - The app only needs `https://www.googleapis.com/auth/drive.file`
   - This limits access to files created by the app

3. **Service account permissions:**
   - Only share the specific Shared Drive (not entire domain)
   - Use minimum required permissions (Content Manager)

4. **Rotate credentials periodically:**
   - Regenerate OAuth tokens every few months
   - Rotate service account keys annually

---

## Cost Comparison

| Option | Storage | Cost | Best For |
|--------|---------|------|----------|
| OAuth2 (Personal Drive) | 15 GB | **Free** | Development, personal use |
| Workspace Shared Drive | 30 GB+ | $6-18/user/month | Production, teams |

---

## Support

If you encounter issues:

1. Check [GDRIVE_TROUBLESHOOTING.md](GDRIVE_TROUBLESHOOTING.md)
2. Run `python diagnose_gdrive.py`
3. Check application logs for error messages
4. Open an issue with:
   - Error messages (remove sensitive data)
   - Which option you're using (OAuth2 or service account)
   - Operating system and Python version
