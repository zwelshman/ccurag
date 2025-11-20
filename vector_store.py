"""Vector store factory - creates the appropriate backend based on configuration."""

import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_vector_store():
    """
    Get a vector store instance based on configuration.

    Returns:
        Vector store instance (PineconeVectorStore or ChromaVectorStore)

    Raises:
        ValueError: If an unknown backend is configured
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


# Legacy alias for backward compatibility
VectorStoreManager = get_vector_store
