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
    ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
    GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
    PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

    # GitHub Settings
    GITHUB_ORG = get_secret("GITHUB_ORG", "BHFDSC")

    # Pinecone Settings
    PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "ccuindex")
    PINECONE_CLOUD = get_secret("PINECONE_CLOUD", "aws")
    PINECONE_REGION = get_secret("PINECONE_REGION", "us-east-1")
    PINECONE_DIMENSION = 768

    # Model Settings
    ANTHROPIC_MODEL = "claude-sonnet-4-5"
    EMBEDDING_MODEL = "BAAI/llm-embedder" 

    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_FILES_PER_REPO = 75

    # File types to index
    INDEXED_FILE_EXTENSIONS = [
        ".md", ".py", ".r", ".R", ".ipynb", ".csv"
        ".txt", ".rst", ".html", ".yaml", ".yml", ".json", ".sql"
    ]

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Set it in .env file or Streamlit secrets."
            )
        if not cls.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Set it in .env file or Streamlit secrets."
            )
        return True
