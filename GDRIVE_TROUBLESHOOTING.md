# Google Drive Setup Troubleshooting

## Problem: Files saving locally instead of to Google Drive

If you see `✓ Saved pickle locally: .cache\bm25_index.pkl` instead of `✓ Saved pickle to Google Drive`, Google Drive was not properly initialized.

## Diagnostic Steps

### 1. Check for initialization messages

Look at the **very beginning** of your application logs. You should see ONE of these:

- ✅ `INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)`
  - **Success!** Google Drive is working

- ⚠️ `INFO:cloud_storage:⚠ Google Drive credentials not found`
  - **Issue:** The `GDRIVE_CREDENTIALS_JSON` environment variable is not set or empty
  - **Fix:** Check your `.env` file (see below)

- ⚠️ `INFO:cloud_storage:⚠ google-api-python-client not installed`
  - **Issue:** Missing required library
  - **Fix:** Run `pip install google-api-python-client google-auth`

- ⚠️ `INFO:cloud_storage:⚠ Failed to initialize Google Drive: [error message]`
  - **Issue:** Invalid credentials or API error
  - **Fix:** Check the error message for details

### 2. Verify your .env file format

Open your `.env` file and check the `GDRIVE_CREDENTIALS_JSON` line:

**Option A: JSON as a single line (recommended)**
```bash
GDRIVE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"your-service-account@your-project.iam.gserviceaccount.com","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}
```

**Important:**
- The entire JSON must be on ONE line
- NO spaces around the `=` sign
- The `\n` in the private key should be literal `\n` characters, not actual newlines

**Option B: File path**
```bash
GDRIVE_CREDENTIALS_JSON=/path/to/your/credentials.json
```

### 3. Verify your Google Service Account credentials

Your JSON should contain these required fields:
- `type` (should be "service_account")
- `project_id`
- `private_key_id`
- `private_key`
- `client_email`
- `client_id`

### 4. Check required library is installed

```bash
pip install google-api-python-client google-auth
```

### 5. Verify Google Drive API is enabled

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project
3. Navigate to "APIs & Services" > "Library"
4. Search for "Google Drive API"
5. Make sure it's **enabled**

## Quick Test

After fixing your setup, restart your application and look for the initialization message at the **very top** of the logs:

```
INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)
```

Then when you build/index, you should see:
```
INFO:cloud_storage:✓ Saved pickle to Google Drive: bm25_index.pkl
```

## Still not working?

Check the following:

1. **Restart required**: Changes to `.env` file require restarting the application
2. **Quotes**: Don't use quotes around the JSON in .env (e.g., `GDRIVE_CREDENTIALS_JSON={"type":...}` not `GDRIVE_CREDENTIALS_JSON="{"type":...}"`)
3. **Escaping**: In the JSON, newlines in the private key should be `\n` not actual newline characters
4. **Permissions**: The service account must have "Google Drive API" access scope

## Manual verification

Run this Python code to test your credentials manually:

```python
import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

# Load credentials
creds_json = os.getenv("GDRIVE_CREDENTIALS_JSON")
print(f"Credentials found: {bool(creds_json)}")
print(f"Credentials length: {len(creds_json) if creds_json else 0}")

if creds_json:
    # Parse JSON
    creds_dict = json.loads(creds_json)
    print(f"Service account: {creds_dict.get('client_email')}")

    # Test authentication
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )

    service = build('drive', 'v3', credentials=credentials)
    results = service.files().list(pageSize=1).execute()
    print("✓ Successfully authenticated with Google Drive!")
```
