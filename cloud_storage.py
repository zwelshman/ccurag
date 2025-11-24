"""Local storage abstraction for persistent caching.

This module provides a simple interface for storing cache files locally.
Cache files should be committed to the Git repository for sharing.
"""

import os
import pickle
import json
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CloudStorage:
    """Local file storage for caching.

    All cache files are stored locally and should be committed to Git
    for sharing across deployments.
    """

    def __init__(self, folder_name: str = None):
        """Initialize local storage.

        Args:
            folder_name: Folder name for cache storage (not used, kept for compatibility)
        """
        logger.info("✓ Using local storage (cache files will be saved to Git)")

    def exists(self, file_path: str) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(file_path)

    def save_pickle(self, data: Any, file_path: str):
        """Save data as pickle file.

        Args:
            data: Data to save
            file_path: Path to save to
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Saved pickle locally: {file_path}")

    def load_pickle(self, file_path: str) -> Any:
        """Load data from pickle file.

        Args:
            file_path: Path to load from

        Returns:
            Loaded data
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"✓ Loaded pickle locally: {file_path}")
        return data

    def save_json(self, data: Any, file_path: str):
        """Save data as JSON file.

        Args:
            data: Data to save (must be JSON-serializable)
            file_path: Path to save to
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✓ Saved JSON locally: {file_path}")

    def load_json(self, file_path: str) -> Any:
        """Load data from JSON file.

        Args:
            file_path: Path to load from

        Returns:
            Loaded data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded JSON locally: {file_path}")
        return data

    def delete(self, file_path: str):
        """Delete a file.

        Args:
            file_path: Path to delete
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"✓ Deleted locally: {file_path}")
