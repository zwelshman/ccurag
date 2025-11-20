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

        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Fetch and index all repositories
        logger.info("Fetching repositories and their contents...")
        sample_size = Config.SAMPLE_REPOS
        if sample_size:
            logger.info(f"Sampling {sample_size} random repositories (Config.SAMPLE_REPOS={sample_size})")
        documents = indexer.index_all_repos(sample_size=sample_size)

        if not documents:
            logger.error("No documents found to index!")
            sys.exit(1)

        logger.info(f"Successfully fetched {len(documents)} documents")

        # Create vector store
        logger.info("Creating vector store...")
        vector_store_manager = VectorStoreManager()

        # Check if vector store already exists
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

        logger.info("Indexing completed successfully!")
        logger.info(f"Vector store saved to: {Config.CHROMA_DB_DIR}")
        logger.info("You can now run the Streamlit app: streamlit run app.py")

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
