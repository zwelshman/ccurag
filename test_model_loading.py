"""Test script to verify SentenceTransformer loading works without meta tensor errors."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test that the model loads without meta tensor errors."""
    try:
        logger.info("Testing PineconeVectorStore initialization...")
        from vector_store_pinecone import PineconeVectorStore

        # This will attempt to load the SentenceTransformer model
        # If there's a meta tensor issue, it should be caught and handled
        store = PineconeVectorStore()

        logger.info("✓ PineconeVectorStore initialized successfully!")
        logger.info("✓ No meta tensor errors encountered!")

        # Test encoding a simple query
        logger.info("Testing model encoding...")
        test_query = "Hello world"
        embedding = store.embedding_model.encode(test_query)
        logger.info(f"✓ Successfully encoded test query. Embedding shape: {embedding.shape}")

        return True

    except NotImplementedError as e:
        if "meta tensor" in str(e).lower():
            logger.error(f"❌ Meta tensor error still occurring: {e}")
            logger.error("The fix did not resolve the issue.")
            return False
        else:
            logger.error(f"❌ Unexpected NotImplementedError: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
