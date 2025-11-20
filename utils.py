"""Shared utilities for the BHFDSC Q&A application."""

import logging
from typing import List, Dict, Callable, Optional
from config import Config

logger = logging.getLogger(__name__)


class Document:
    """Simple document class for storing text content with metadata."""

    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def split_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Size of each chunk (defaults to Config.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (defaults to Config.CHUNK_OVERLAP)

    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def update_progress(msg: str, pct: float, callback: Optional[Callable] = None):
    """
    Update progress with logging and optional callback.

    Args:
        msg: Progress message
        pct: Progress percentage (0.0 to 1.0)
        callback: Optional callback function(message: str, progress_pct: float)
    """
    logger.info(msg)
    if callback:
        try:
            callback(msg, pct)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
