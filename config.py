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
            # Check if key exists in secrets
            if key in st.secrets:
                value = st.secrets[key]
                # Return the secret value if it's not None
                if value is not None:
                    return value
        except (AttributeError, FileNotFoundError, KeyError):
            # Streamlit secrets not available, fall back to environment
            pass

    # Fall back to environment variables
    return os.getenv(key, default)


class Config:
    """Application configuration."""

    # API Keys
    GITHUB_TOKEN = get_secret("GITHUB_TOKEN")

    # GitHub Settings
    GITHUB_ORG = get_secret("GITHUB_ORG", "BHFDSC")

    # File types to index (code files only for static analysis)
    INDEXED_FILE_EXTENSIONS = [
        ".py", ".r", ".R", ".sql"
    ]
