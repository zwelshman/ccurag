"""Configuration management for the BHFDSC Q&A application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    # GitHub Organization
    GITHUB_ORG = os.getenv("GITHUB_ORG", "BHFDSC")

    # Vector Store Settings
    CHROMA_DB_DIR = "chroma_db"
    COLLECTION_NAME = "bhfdsc_repos"

    # Model Settings
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

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

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Set it in .env file.")
        return True
