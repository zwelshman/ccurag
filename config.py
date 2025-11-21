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

    # GitHub Organization
    GITHUB_ORG = get_secret("GITHUB_ORG", "BHFDSC")

    # Vector Store Settings
    VECTOR_STORE_BACKEND = get_secret("VECTOR_STORE_BACKEND", "pinecone")  # "pinecone" or "chroma"

    # ChromaDB Settings (legacy)
    CHROMA_DB_DIR = "chroma_db"
    COLLECTION_NAME = "bhfdsc_repos"

    # Pinecone Settings
    PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "ccuindex")
    PINECONE_CLOUD = get_secret("PINECONE_CLOUD", "aws")
    PINECONE_REGION = get_secret("PINECONE_REGION", "us-east-1")
    PINECONE_DIMENSION = int(get_secret("PINECONE_DIMENSION", "384"))

    # Model Settings
    ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions

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

    # Test repository subset (for quick testing and development)
    # Set USE_TEST_REPOS=true in .env to enable test mode
    USE_TEST_REPOS = get_secret("USE_TEST_REPOS", "false").lower() in ["true", "1", "yes"]
    TEST_REPOS = [
        "hds_curated_assets","documentation"
    ]

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Set it in .env file (local) or Streamlit secrets (cloud)."
            )

        # Validate vector store configuration
        if cls.VECTOR_STORE_BACKEND == "pinecone":
            if not cls.PINECONE_API_KEY:
                raise ValueError(
                    "PINECONE_API_KEY is required when using Pinecone backend. "
                    "Set it in .env file (local) or Streamlit secrets (cloud)."
                )

        return True
