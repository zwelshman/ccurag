#!/usr/bin/env python3
"""Diagnostic script to verify Google Drive credentials setup."""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def diagnose():
    """Run diagnostics for Google Drive setup."""
    print("=" * 60)
    print("Google Drive Setup Diagnostics")
    print("=" * 60)
    print()

    # Check 1: Environment variable exists
    print("1. Checking GDRIVE_CREDENTIALS_JSON environment variable...")
    gdrive_creds = os.getenv("GDRIVE_CREDENTIALS_JSON")

    if not gdrive_creds:
        print("   ❌ GDRIVE_CREDENTIALS_JSON not found in .env file")
        print("   → Make sure you have this line in your .env file:")
        print('     GDRIVE_CREDENTIALS_JSON={"type":"service_account",...}')
        return False
    else:
        print(f"   ✓ Found GDRIVE_CREDENTIALS_JSON ({len(gdrive_creds)} characters)")

    print()

    # Check 2: Valid JSON format
    print("2. Validating JSON format...")
    try:
        # Try parsing as JSON
        creds_dict = json.loads(gdrive_creds)
        print("   ✓ Valid JSON format")

        # Check required fields
        required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        missing_fields = [f for f in required_fields if f not in creds_dict]

        if missing_fields:
            print(f"   ⚠️  Missing required fields: {', '.join(missing_fields)}")
        else:
            print(f"   ✓ All required fields present")
            print(f"   - Service account: {creds_dict.get('client_email', 'N/A')}")
            print(f"   - Project ID: {creds_dict.get('project_id', 'N/A')}")

    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON format: {e}")
        print("   → If it's a file path, check if the file exists")

        # Check if it might be a file path
        if os.path.exists(gdrive_creds):
            print(f"   → Found file at path: {gdrive_creds}")
            try:
                with open(gdrive_creds, 'r') as f:
                    creds_dict = json.load(f)
                print("   ✓ File contains valid JSON")
            except Exception as e2:
                print(f"   ❌ Error reading file: {e2}")
                return False
        else:
            print("   → Make sure the JSON is on a single line and properly escaped")
            return False

    print()

    # Check 3: Google API client library
    print("3. Checking google-api-python-client installation...")
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        print("   ✓ google-api-python-client is installed")
    except ImportError:
        print("   ❌ google-api-python-client not installed")
        print("   → Run: pip install google-api-python-client google-auth")
        return False

    print()

    # Check 4: Try to authenticate
    print("4. Testing Google Drive authentication...")
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )

        service = build('drive', 'v3', credentials=credentials)
        print("   ✓ Successfully created Drive service")

        # Try to list files to verify access
        results = service.files().list(pageSize=1, fields="files(id, name)").execute()
        print("   ✓ Successfully authenticated with Google Drive API")

    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("   → Check that the service account has Drive API enabled")
        print("   → Verify the private key is correct")
        return False

    print()
    print("=" * 60)
    print("✓ All checks passed! Google Drive should work correctly.")
    print("=" * 60)
    print()
    print("When you run the app, you should see:")
    print('  INFO:cloud_storage:✓ Google Drive storage initialized (folder: ccurag-cache)')
    print('  INFO:cloud_storage:✓ Saved pickle to Google Drive: bm25_index.pkl')
    print()

    return True

if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)
