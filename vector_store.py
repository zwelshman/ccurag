"""Vector store using Pinecone."""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class."""
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def get_vector_store():
    """Get Pinecone vector store instance."""
    from vector_store_pinecone import PineconeVectorStore
    return PineconeVectorStore()


# Export for backward compatibility
__all__ = ['Document', 'get_vector_store']
