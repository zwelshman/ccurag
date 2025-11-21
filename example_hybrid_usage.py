"""Example script demonstrating hybrid retriever usage."""

import logging
from vector_store_pinecone import PineconeVectorStore
from hybrid_retriever import HybridRetriever
from qa_chain import QASystem
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate hybrid retriever usage."""
    print("=" * 60)
    print("HYBRID RETRIEVER EXAMPLE")
    print("=" * 60)

    # Validate configuration
    Config.validate()

    # Step 1: Initialize Pinecone vector store
    logger.info("Step 1: Initializing Pinecone vector store...")
    vector_store = PineconeVectorStore()
    vector_store.load_vectorstore()
    print("✓ Vector store loaded")

    # Step 2: Check if using hybrid search
    if Config.USE_HYBRID_SEARCH:
        logger.info("Step 2: Initializing hybrid retriever...")
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_weight=Config.BM25_WEIGHT
        )

        # The BM25 index should already be built using build_hybrid_index.py
        # If not built yet, it will fall back to vector search only
        stats = hybrid_retriever.get_index_stats()

        if not stats['bm25_indexed']:
            print("⚠ BM25 index not found. Please run: python build_hybrid_index.py")
            print("  Falling back to vector search only for this session.")
            retriever = vector_store
        else:
            print(f"✓ Hybrid retriever initialized")
            print(f"  - BM25 documents: {stats['bm25_documents']}")
            print(f"  - BM25 weight: {stats['bm25_weight']:.2f}")
            print(f"  - Vector weight: {stats['vector_weight']:.2f}")
            print(f"  - Adaptive weights: {stats['adaptive_weights_enabled']}")
            retriever = hybrid_retriever
    else:
        logger.info("Hybrid search disabled, using vector search only")
        retriever = vector_store
        print("✓ Using vector search only")

    # Step 3: Initialize QA system with the retriever
    logger.info("Step 3: Initializing QA system...")
    qa_system = QASystem(vector_store=vector_store, retriever=retriever)
    print("✓ QA system ready")

    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)

    # Example queries that showcase hybrid search benefits
    test_queries = [
        # Code-specific queries (favor BM25)
        ("What does the PineconeVectorStore class do?", "Code class name - exact match"),

        # Conceptual queries (favor vector search)
        ("How do I set up the environment?", "Conceptual - semantic understanding"),

        # Mixed queries
        ("What embedding model is used and why?", "Mixed - concept + specifics"),
    ]

    for query, description in test_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Type: {description}")
        print("-" * 60)

        # Answer the question
        result = qa_system.answer_question(
            question=query,
            num_docs=5,
            include_follow_ups=False,  # Disable for brevity
            include_metadata=True
        )

        # Display results
        print(f"Answer: {result['answer'][:300]}...")

        if 'metadata' in result:
            meta = result['metadata']
            print(f"\nMetadata:")
            print(f"  - Confidence: {meta.get('confidence', 'N/A')}")
            print(f"  - Answer type: {meta.get('answer_type', 'N/A')}")
            print(f"  - Sources cited: {meta.get('num_sources_cited', 0)}/{meta.get('num_sources_retrieved', 0)}")

        print(f"\nTop sources:")
        for i, doc in enumerate(result['source_documents'][:2], 1):
            source = doc['metadata']['source']
            print(f"  {i}. {source}")

    print("\n" + "=" * 60)
    print("COMPARISON: Vector vs Hybrid Search")
    print("=" * 60)

    # Compare results for a code-specific query
    if isinstance(retriever, HybridRetriever):
        code_query = "similarity_search function"

        print(f"\nQuery: '{code_query}'")
        print("This query contains exact function name - should favor BM25\n")

        # Get results with hybrid search
        hybrid_results = retriever.similarity_search(code_query, k=3)
        print("Hybrid Search Results:")
        for i, doc in enumerate(hybrid_results, 1):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  {i}. {source}")
            print(f"     {preview}...")

        # Get results with vector search only
        vector_results = vector_store.similarity_search(code_query, k=3)
        print("\nVector-Only Search Results:")
        for i, doc in enumerate(vector_results, 1):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  {i}. {source}")
            print(f"     {preview}...")

        print("\n→ Notice how hybrid search may rank exact matches higher!")

    print("\n" + "=" * 60)
    print("✓ Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
