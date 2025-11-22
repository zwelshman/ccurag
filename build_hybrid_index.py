"""Build BM25 index from existing Pinecone vector store."""

import logging
from vector_store_pinecone import PineconeVectorStore, Document
from hybrid_retriever import HybridRetriever
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_all_documents_from_pinecone(vector_store: PineconeVectorStore,
                                     batch_size: int = 100) -> list:
    """Fetch all documents from Pinecone index.

    Since Pinecone doesn't have a direct "get all" method, we'll use a
    workaround by querying with random vectors and collecting unique docs.

    Args:
        vector_store: Initialized Pinecone vector store
        batch_size: Number of results per query

    Returns:
        List of Document objects
    """
    import numpy as np

    logger.info("Fetching documents from Pinecone...")

    # Get index stats to know how many vectors we have
    stats = vector_store.get_stats()
    total_vectors = stats.get('total_vector_count', 0)

    if total_vectors == 0:
        logger.warning("No vectors found in Pinecone index")
        return []

    logger.info(f"Found {total_vectors} vectors in Pinecone index")

    # Create a dummy query to fetch documents
    # We'll query with a zero vector to get random documents
    dummy_embedding = [0.0] * Config.PINECONE_DIMENSION

    # Fetch documents in batches
    all_docs = []
    seen_content = set()

    # Fetch more than needed to ensure we get everything
    fetch_k = min(10000, total_vectors)  # Pinecone limit

    logger.info(f"Fetching up to {fetch_k} documents...")

    results = vector_store.index.query(
        vector=dummy_embedding,
        top_k=fetch_k,
        include_metadata=True
    )

    for match in results.get('matches', []):
        metadata = match.get('metadata', {})
        text = metadata.pop('text', '')

        # Deduplicate by content
        content_hash = hash(text[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            all_docs.append(Document(page_content=text, metadata=metadata))

    logger.info(f"✓ Fetched {len(all_docs)} unique documents from Pinecone")

    return all_docs


def build_bm25_index(force_rebuild=True):
    """Build BM25 index from existing Pinecone vector store.

    Args:
        force_rebuild: If True, rebuild even if cache exists
    """
    logger.info("=" * 60)
    logger.info("BUILDING HYBRID INDEX (BM25 + Vector)")
    logger.info("=" * 60)

    # Validate config
    Config.validate()

    # Initialize vector store
    logger.info("Initializing Pinecone vector store...")
    vector_store = PineconeVectorStore()
    vector_store.load_vectorstore()

    # Fetch all documents
    documents = fetch_all_documents_from_pinecone(vector_store)

    if not documents:
        logger.error("No documents found in vector store. Please run indexing first.")
        return

    # Initialize hybrid retriever
    logger.info("Initializing hybrid retriever...")
    hybrid_retriever = HybridRetriever(vector_store)

    # Build BM25 index
    logger.info("Building BM25 index...")
    hybrid_retriever.build_bm25_index(documents, force_rebuild=force_rebuild)

    # Print stats
    stats = hybrid_retriever.get_index_stats()
    logger.info("=" * 60)
    logger.info("HYBRID INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"BM25 documents indexed: {stats['bm25_documents']}")
    logger.info(f"Vector store vectors: {stats['vector_store_vectors']}")
    logger.info(f"BM25 weight: {stats['bm25_weight']:.2f}")
    logger.info(f"Vector weight: {stats['vector_weight']:.2f}")
    logger.info(f"Adaptive weights: {stats['adaptive_weights_enabled']}")
    logger.info("=" * 60)

    # Test the hybrid search
    logger.info("\nTesting hybrid search with sample query...")
    test_query = "PineconeVectorStore"
    results = hybrid_retriever.similarity_search(test_query, k=3)

    logger.info(f"\nTop 3 results for '{test_query}':")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:100].replace('\n', ' ')
        logger.info(f"  {i}. {source}")
        logger.info(f"     Preview: {preview}...")

    logger.info("\n✓ Hybrid index is ready to use!")


def main(force_rebuild=True):
    """Main function for calling from other modules."""
    build_bm25_index(force_rebuild=force_rebuild)


if __name__ == "__main__":
    build_bm25_index()
