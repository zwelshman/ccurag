"""Script to index GitHub repositories into the vector store."""

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
    """Index all repositories from the GitHub organization."""
    try:
        # Validate configuration
        Config.validate()

        logger.info(f"Starting indexing process for organization: {Config.GITHUB_ORG}")
        logger.info("Note: Progress is automatically saved. If interrupted, simply re-run to resume.")

        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Fetch and index all repositories (with automatic resume support)
        logger.info("Fetching repositories and their contents...")
        sample_size = Config.SAMPLE_REPOS
        if sample_size:
            logger.info(f"Sampling {sample_size} random repositories (Config.SAMPLE_REPOS={sample_size})")
        documents, changed_repos = indexer.index_all_repos(sample_size=sample_size, resume=True)

        if not documents:
            logger.info("No new or changed documents found. All repositories are up to date!")

            # Verify vector store actually has data
            if Config.VECTOR_STORE_BACKEND == "pinecone":
                from vector_store_pinecone import PineconeVectorStore
                pinecone_store = PineconeVectorStore()
                try:
                    pinecone_store.load_vectorstore()
                    stats = pinecone_store.get_stats()
                    vector_count = stats.get('total_vector_count', 0)
                    if vector_count > 0:
                        logger.info(f"Vector store verified: {vector_count} vectors present")
                        sys.exit(0)
                    else:
                        logger.warning("⚠️  Vector store has 0 vectors despite completed checkpoint!")
                        logger.warning("Checkpoint may be corrupted. Deleting and re-indexing...")
                        checkpoint_file = Path(".checkpoint.json")
                        if checkpoint_file.exists():
                            checkpoint_file.unlink()
                        # Re-run indexing
                        documents, changed_repos = indexer.index_all_repos(sample_size=sample_size, resume=False)
                        if not documents:
                            logger.error("Still no documents after checkpoint reset!")
                            sys.exit(1)
                except Exception as e:
                    logger.error(f"Error verifying vector store: {e}")
                    sys.exit(1)
            else:
                sys.exit(0)

        logger.info(f"Successfully fetched {len(documents)} documents")
        logger.info(f"Repositories with changes: {len(changed_repos)}")

        # Create or update vector store
        vector_store_manager = VectorStoreManager()

        # Check if this is an incremental update or initial indexing
        # For Pinecone, check if index exists
        if Config.VECTOR_STORE_BACKEND == "pinecone":
            from vector_store_pinecone import PineconeVectorStore
            pinecone_store = PineconeVectorStore()
            existing_indexes = [index.name for index in pinecone_store.pc.list_indexes()]

            if Config.PINECONE_INDEX_NAME in existing_indexes:
                # Incremental update: upsert documents
                logger.info("Performing incremental update to existing vector store...")
                pinecone_store.upsert_documents(documents, repos_to_update=changed_repos)
                logger.info("Incremental update completed successfully!")
            else:
                # Initial indexing: create new vector store
                logger.info("Creating new vector store...")
                vector_store_manager.create_vectorstore(documents)
                logger.info("Vector store created successfully!")
        else:
            # For ChromaDB, check if database directory exists
            db_path = Path(Config.CHROMA_DB_DIR)
            if db_path.exists():
                response = input(f"Vector store already exists at {Config.CHROMA_DB_DIR}. Overwrite? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Indexing cancelled.")
                    sys.exit(0)

                # Remove existing database
                import shutil
                shutil.rmtree(Config.CHROMA_DB_DIR)
                logger.info("Removed existing vector store")

            vector_store_manager.create_vectorstore(documents)
            logger.info("Vector store created successfully!")

        logger.info("Indexing completed successfully!")
        logger.info("You can now run the Streamlit app: streamlit run app.py")

    except KeyboardInterrupt:
        logger.info("\n\nIndexing interrupted by user.")
        logger.info("Progress has been saved to .checkpoint.json")
        logger.info("Run this script again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        logger.info("\nProgress has been saved to .checkpoint.json")
        logger.info("Run this script again to resume from where you left off.")
        sys.exit(1)


if __name__ == "__main__":
    main()
