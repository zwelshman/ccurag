"""Configuration management for the BHFDSC Q&A application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Streamlit for secrets support (only available when running in Streamlit)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def get_secret(key: str, default=None):
    """Get a secret from Streamlit secrets or environment variables."""
    # First try Streamlit secrets (for Streamlit Cloud deployment)
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except (AttributeError, FileNotFoundError):
            # Streamlit secrets not available, fall back to environment
            pass

    # Fall back to environment variables
    return os.getenv(key, default)


class Config:
    """Application configuration."""

    # API Keys
    ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
    GITHUB_TOKEN = get_secret("GITHUB_TOKEN")

    # GitHub Organization
    GITHUB_ORG = get_secret("GITHUB_ORG", "BHFDSC")

    # Vector Store Settings
    CHROMA_DB_DIR = "chroma_db"
    COLLECTION_NAME = "bhfdsc_repos"

    # Model Settings
    ANTHROPIC_MODEL = "claude-haiku-4-5"

    # Chunking Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # File types to index
    INDEXED_FILE_EXTENSIONS = [
        ".md", ".py", ".r", ".R", ".ipynb",
        ".txt", ".rst", ".html", ".yaml", ".yml", ".json"
    ]

    # Max files per repo (to avoid rate limiting)
    MAX_FILES_PER_REPO = 50

    # Repository sampling (set to None to index all repos)
    SAMPLE_REPOS = None  # Set to a number (e.g., 20) to sample random repos

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Set it in .env file (local) or Streamlit secrets (cloud)."
            )
        return True
