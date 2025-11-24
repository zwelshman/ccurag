"""Cloud storage abstraction for persistent caching in Streamlit Cloud.

This module provides a unified interface for storing cache files either locally
(for development) or in Google Drive (recommended for Streamlit Cloud).
"""

import os
import pickle
import json
import logging
import io
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CloudStorage:
    """Abstraction layer for cloud-based file storage.

    Priority order:
    1. Google Drive (if credentials available)
    2. Local storage (fallback for development)
    """

    def __init__(self, folder_name: Optional[str] = None):
        """Initialize cloud storage.

        Args:
            folder_name: Folder/bucket name for cloud storage
        """
        self.folder_name = folder_name or "ccurag-cache"
        self.backend = "local"  # Options: "gdrive", "local"
        self.service = None

        # Try to initialize cloud storage (Google Drive)
        if folder_name:
            self._init_gdrive()

    def _init_gdrive(self):
        """Initialize Google Drive client if credentials are available."""
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
            from google.oauth2 import service_account
            import json

            # Import here to avoid circular dependency
            from config import Config

            # Check if Google Drive credentials are available
            gdrive_creds_json = Config.GDRIVE_CREDENTIALS_JSON

            if gdrive_creds_json:
                # Parse credentials from JSON string
                try:
                    creds_dict = json.loads(gdrive_creds_json)
                except json.JSONDecodeError:
                    # If it's a file path instead of JSON string
                    if os.path.exists(gdrive_creds_json):
                        with open(gdrive_creds_json, 'r') as f:
                            creds_dict = json.load(f)
                    else:
                        raise ValueError("Invalid Google Drive credentials format")

                # Create credentials from service account
                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=['https://www.googleapis.com/auth/drive.file']
                )

                # Build the Drive service
                self.service = build('drive', 'v3', credentials=credentials)
                self.backend = "gdrive"

                # Get or create cache folder
                self.folder_id = self._get_or_create_folder(self.folder_name)

                logger.info(f"✓ Google Drive storage initialized (folder: {self.folder_name})")
            else:
                logger.info("⚠ Google Drive credentials not found")
        except ImportError:
            logger.info("⚠ google-api-python-client not installed")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize Google Drive: {e}")

    def _get_or_create_folder(self, folder_name: str) -> str:
        """Get or create a folder in Google Drive.

        Args:
            folder_name: Name of the folder

        Returns:
            Folder ID
        """
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])

        if folders:
            return folders[0]['id']

        # Create new folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.service.files().create(body=file_metadata, fields='id').execute()
        return folder['id']

    def _get_file_id(self, file_path: str) -> Optional[str]:
        """Get the ID of a file in Google Drive.

        Args:
            file_path: Path to the file (relative to cache folder)

        Returns:
            File ID if found, None otherwise
        """
        # Extract just the filename from the path
        filename = os.path.basename(file_path)

        query = f"name='{filename}' and '{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        return files[0]['id'] if files else None

    def exists(self, file_path: str) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to the file (relative to cache directory)

        Returns:
            True if file exists, False otherwise
        """
        if self.backend == "gdrive":
            try:
                file_id = self._get_file_id(file_path)
                return file_id is not None
            except:
                return False
        else:
            return os.path.exists(file_path)

    def save_pickle(self, data: Any, file_path: str):
        """Save data as pickle file.

        Args:
            data: Data to save
            file_path: Path to save to (relative to cache directory)
        """
        if self.backend == "gdrive":
            try:
                from googleapiclient.http import MediaIoBaseUpload

                # Serialize to bytes
                pickled_data = pickle.dumps(data)
                filename = os.path.basename(file_path)

                # Check if file already exists
                file_id = self._get_file_id(file_path)

                media = MediaIoBaseUpload(
                    io.BytesIO(pickled_data),
                    mimetype='application/octet-stream',
                    resumable=True
                )

                if file_id:
                    # Update existing file
                    self.service.files().update(
                        fileId=file_id,
                        media_body=media
                    ).execute()
                else:
                    # Create new file
                    file_metadata = {
                        'name': filename,
                        'parents': [self.folder_id]
                    }
                    self.service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()

                logger.info(f"✓ Saved pickle to Google Drive: {filename}")
            except Exception as e:
                logger.error(f"Failed to save pickle to Google Drive: {e}")
                raise
        else:
            # Local storage
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"✓ Saved pickle locally: {file_path}")

    def load_pickle(self, file_path: str) -> Any:
        """Load data from pickle file.

        Args:
            file_path: Path to load from (relative to cache directory)

        Returns:
            Loaded data
        """
        if self.backend == "gdrive":
            try:
                from googleapiclient.http import MediaIoBaseDownload

                file_id = self._get_file_id(file_path)
                if not file_id:
                    raise FileNotFoundError(f"File not found in Google Drive: {file_path}")

                request = self.service.files().get_media(fileId=file_id)
                file_buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(file_buffer, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()

                file_buffer.seek(0)
                data = pickle.loads(file_buffer.read())
                logger.info(f"✓ Loaded pickle from Google Drive: {os.path.basename(file_path)}")
                return data
            except Exception as e:
                logger.error(f"Failed to load pickle from Google Drive: {e}")
                raise
        else:
            # Local storage
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✓ Loaded pickle locally: {file_path}")
            return data

    def save_json(self, data: Any, file_path: str):
        """Save data as JSON file.

        Args:
            data: Data to save (must be JSON-serializable)
            file_path: Path to save to (relative to cache directory)
        """
        if self.backend == "gdrive":
            try:
                from googleapiclient.http import MediaIoBaseUpload

                # Serialize to JSON string
                json_data = json.dumps(data, indent=2)
                filename = os.path.basename(file_path)

                # Check if file already exists
                file_id = self._get_file_id(file_path)

                media = MediaIoBaseUpload(
                    io.BytesIO(json_data.encode('utf-8')),
                    mimetype='application/json',
                    resumable=True
                )

                if file_id:
                    # Update existing file
                    self.service.files().update(
                        fileId=file_id,
                        media_body=media
                    ).execute()
                else:
                    # Create new file
                    file_metadata = {
                        'name': filename,
                        'parents': [self.folder_id]
                    }
                    self.service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()

                logger.info(f"✓ Saved JSON to Google Drive: {filename}")
            except Exception as e:
                logger.error(f"Failed to save JSON to Google Drive: {e}")
                raise
        else:
            # Local storage
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✓ Saved JSON locally: {file_path}")

    def load_json(self, file_path: str) -> Any:
        """Load data from JSON file.

        Args:
            file_path: Path to load from (relative to cache directory)

        Returns:
            Loaded data
        """
        if self.backend == "gdrive":
            try:
                from googleapiclient.http import MediaIoBaseDownload

                file_id = self._get_file_id(file_path)
                if not file_id:
                    raise FileNotFoundError(f"File not found in Google Drive: {file_path}")

                request = self.service.files().get_media(fileId=file_id)
                file_buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(file_buffer, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()

                file_buffer.seek(0)
                json_data = file_buffer.read().decode('utf-8')
                data = json.loads(json_data)
                logger.info(f"✓ Loaded JSON from Google Drive: {os.path.basename(file_path)}")
                return data
            except Exception as e:
                logger.error(f"Failed to load JSON from Google Drive: {e}")
                raise
        else:
            # Local storage
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded JSON locally: {file_path}")
            return data

    def delete(self, file_path: str):
        """Delete a file.

        Args:
            file_path: Path to delete (relative to cache directory)
        """
        if self.backend == "gdrive":
            try:
                file_id = self._get_file_id(file_path)
                if file_id:
                    self.service.files().delete(fileId=file_id).execute()
                    logger.info(f"✓ Deleted from Google Drive: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"File not found in Google Drive: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete from Google Drive: {e}")
        else:
            # Local storage
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"✓ Deleted locally: {file_path}")
