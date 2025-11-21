"""Force a complete re-index of all repositories.

This script bypasses the checkpoint system to ensure all documents
are fetched and indexed into Pinecone, even if they were previously processed.
"""

import logging
import sys
from pathlib import Path
from github_indexer import GitHubIndexer
from vector_store import VectorStoreManager
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Force complete re-indexing."""
    try:
        # Validate configuration
        Config.validate()

        # Delete checkpoint file if it exists
        checkpoint_file = Path(".checkpoint.json")
        if checkpoint_file.exists():
            logger.info("Removing existing checkpoint to force fresh indexing...")
            checkpoint_file.unlink()
            logger.info("Checkpoint removed")

        logger.info(f"Starting FORCED re-indexing for organization: {Config.GITHUB_ORG}")
        logger.info("This will fetch and index all repositories from scratch.")

        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Fetch and index all repositories (with resume disabled)
        logger.info("Fetching repositories and their contents...")
        sample_size = Config.SAMPLE_REPOS
        if sample_size:
            logger.info(f"Sampling {sample_size} random repositories")

        # This will start fresh since we deleted the checkpoint
        documents, changed_repos = indexer.index_all_repos(sample_size=sample_size, resume=False)

        if not documents:
            logger.error("No documents were fetched! Check GitHub token and organization name.")
            sys.exit(1)

        logger.info(f"Successfully fetched {len(documents)} documents")

        # Create vector store
        vector_store_manager = VectorStoreManager()

        # For Pinecone, check if index exists
        if Config.VECTOR_STORE_BACKEND == "pinecone":
            from vector_store_pinecone import PineconeVectorStore
            pinecone_store = PineconeVectorStore()
            existing_indexes = [index.name for index in pinecone_store.pc.list_indexes()]

            if Config.PINECONE_INDEX_NAME in existing_indexes:
                logger.warning(f"Index {Config.PINECONE_INDEX_NAME} already exists")
                response = input("Delete and recreate? (yes/no): ")
                if response.lower() == 'yes':
                    logger.info("Deleting existing index...")
                    pinecone_store.delete_vectorstore()
                    logger.info("Index deleted")
                else:
                    logger.info("Using existing index with upsert...")
                    pinecone_store.upsert_documents(documents)
                    logger.info("Documents upserted successfully!")
                    sys.exit(0)

        # Create new vector store
        logger.info("Creating vector store...")
        vector_store_manager.create_vectorstore(documents)
        logger.info("Vector store created successfully!")

        # Verify the index
        if Config.VECTOR_STORE_BACKEND == "pinecone":
            pinecone_store = PineconeVectorStore()
            pinecone_store.load_vectorstore()
            stats = pinecone_store.get_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"✅ Verification: Index now contains {vector_count} vectors")

            if vector_count == 0:
                logger.error("❌ WARNING: Index still has 0 vectors after indexing!")
                logger.error("This indicates a problem with the embedding/upsert process.")
                sys.exit(1)

        logger.info("✅ Forced re-indexing completed successfully!")
        logger.info("You can now run the Streamlit app: streamlit run app.py")

    except KeyboardInterrupt:
        logger.info("\n\nIndexing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during forced re-indexing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
