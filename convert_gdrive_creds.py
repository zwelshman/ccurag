#!/usr/bin/env python3
"""Convert Google Drive credentials JSON to single-line format for .env file."""

import json
import sys

print("=" * 70)
print("Google Drive Credentials Converter for .env file")
print("=" * 70)
print()
print("This script converts your multi-line Google service account JSON")
print("to a single-line format suitable for the .env file.")
print()

# Ask user for the credentials file path
creds_file = input("Enter path to your Google service account JSON file: ").strip()

try:
    # Read the JSON file
    with open(creds_file, 'r') as f:
        creds_data = json.load(f)

    # Convert to single-line JSON (compact format)
    single_line_json = json.dumps(creds_data, separators=(',', ':'))

    print()
    print("✓ Successfully converted JSON to single-line format")
    print()
    print("=" * 70)
    print("Add this line to your .env file:")
    print("=" * 70)
    print()
    print(f"GDRIVE_CREDENTIALS_JSON={single_line_json}")
    print()
    print("=" * 70)
    print("IMPORTANT:")
    print("1. Copy the ENTIRE line above (including GDRIVE_CREDENTIALS_JSON=)")
    print("2. Replace the existing GDRIVE_CREDENTIALS_JSON entry in your .env file")
    print("3. Make sure it's all on ONE line with NO line breaks")
    print("4. Restart your application")
    print("=" * 70)

except FileNotFoundError:
    print(f"❌ Error: File not found: {creds_file}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON format: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
