"""Vector store abstraction layer - factory for different backends."""

import logging
from typing import List, Dict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class to replace langchain Document."""
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


class VectorStoreManager:
    """
    Factory class that creates the appropriate vector store backend.

    This provides a unified interface regardless of which backend is configured.
    """

    def __new__(cls):
        """
        Create and return the appropriate vector store backend.

        Returns:
            An instance of either PineconeVectorStore or ChromaVectorStore
        """
        backend = Config.VECTOR_STORE_BACKEND.lower()

        if backend == "pinecone":
            logger.info("Using Pinecone vector store backend")
            from vector_store_pinecone import PineconeVectorStore
            return PineconeVectorStore()
        elif backend == "chroma":
            logger.info("Using ChromaDB vector store backend")
            from vector_store_chroma import ChromaVectorStore
            return ChromaVectorStore()
        else:
            raise ValueError(
                f"Unknown vector store backend: {backend}. "
                f"Valid options are 'pinecone' or 'chroma'"
            )


def get_vector_store():
    """
    Convenience function to get a vector store instance.

    Returns:
        A vector store instance based on configuration
    """
    return VectorStoreManager()


# Export Document class for backward compatibility
__all__ = ['VectorStoreManager', 'Document', 'get_vector_store']
