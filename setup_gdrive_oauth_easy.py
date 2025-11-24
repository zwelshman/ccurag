#!/usr/bin/env python3
"""Setup Google Drive OAuth2 credentials - accepts JSON content or file path.

This script helps you authenticate with your personal Google account to use
your free 15GB Google Drive storage.
"""

import os
import json
import pickle
import tempfile
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
except ImportError:
    print("❌ Required packages not installed")
    print("Run: pip install google-auth-oauthlib google-auth google-api-python-client")
    exit(1)

# Scopes required for Drive access
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def setup_oauth_credentials():
    """Set up OAuth2 credentials for Google Drive."""

    print("=" * 70)
    print("Google Drive OAuth2 Setup for Personal Account")
    print("=" * 70)
    print()
    print("This will allow the app to access YOUR Google Drive (free 15GB).")
    print()
    print("Prerequisites:")
    print("1. You need OAuth2 Client ID credentials (Desktop OR Web Application)")
    print("2. Get them from: https://console.cloud.google.com/apis/credentials")
    print("3. Create OAuth 2.0 Client ID -> Desktop or Web application")
    print("4. Download the JSON file (or copy the JSON content)")
    print()
    print("=" * 70)
    print()
    print("You can provide EITHER:")
    print("  1. Path to the JSON file (e.g., oauth_client.json)")
    print("  2. Paste the JSON content directly (entire JSON object)")
    print()

    user_input = input("Enter file path OR paste JSON content: ").strip()

    # Determine if input is a file path or JSON content
    oauth_client_file = None
    temp_file = None

    if user_input.startswith('{'):
        # User pasted JSON content
        try:
            # Validate JSON
            json_data = json.loads(user_input)

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(json_data, temp_file)
            temp_file.close()
            oauth_client_file = temp_file.name
            print("✓ JSON content validated and saved to temporary file")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON content: {e}")
            return False
    else:
        # User provided file path
        oauth_client_file = user_input
        if not os.path.exists(oauth_client_file):
            print(f"❌ File not found: {oauth_client_file}")
            print()
            print("Tip: If you copied JSON content, make sure it starts with '{'")
            print("     Or save the JSON to a file first and provide the path")
            return False

    try:
        creds = None
        token_file = ".gdrive_token.pickle"

        # Check if we have saved credentials
        if os.path.exists(token_file):
            print("Found existing token, loading...")
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired token...")
                creds.refresh(Request())
            else:
                print()
                print("Starting OAuth2 flow...")
                print("A browser window will open for you to authorize the app.")
                print()
                flow = InstalledAppFlow.from_client_secrets_file(
                    oauth_client_file, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for future use
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
            print(f"✓ Credentials saved to {token_file}")

        # Test the credentials
        print()
        print("Testing Google Drive access...")
        try:
            service = build('drive', 'v3', credentials=creds)
            results = service.files().list(pageSize=1, fields="files(id, name)").execute()
            print("✓ Successfully authenticated with Google Drive!")

            # Get user info
            about = service.about().get(fields="user,storageQuota").execute()
            user_email = about['user']['emailAddress']
            print(f"✓ Authenticated as: {user_email}")

            if 'storageQuota' in about:
                quota = about['storageQuota']
                limit = int(quota.get('limit', 0))
                usage = int(quota.get('usage', 0))
                if limit > 0:
                    used_gb = usage / (1024**3)
                    limit_gb = limit / (1024**3)
                    print(f"✓ Storage: {used_gb:.2f} GB / {limit_gb:.2f} GB used")

        except Exception as e:
            print(f"❌ Failed to access Google Drive: {e}")
            return False

        print()
        print("=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print()
        print("The app will now use OAuth2 credentials from:")
        print(f"  {os.path.abspath(token_file)}")
        print()
        print("Make sure this file is in the same directory when running the app.")
        print()

        return True

    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

if __name__ == "__main__":
    import sys
    success = setup_oauth_credentials()
    sys.exit(0 if success else 1)
