"""
Example implementation: Pinecone Vector Database Backend

This shows how to migrate from ChromaDB to Pinecone for production deployment.

Pinecone is a managed vector database service that's excellent for production use.
"""

import logging
import time
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class Document:
    """Simple document class compatible with existing code."""

    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


class PineconeVectorStore:
    """
    Pinecone-based vector store manager.

    Drop-in replacement for VectorStoreManager using Pinecone instead of ChromaDB.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "bhfdsc-repos",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,  # Dimension for all-MiniLM-L6-v2
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: SentenceTransformer model name
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            cloud: Cloud provider ('aws', 'gcp', or 'azure')
            region: Cloud region
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = None

    def _split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Simple text splitter."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        return chunks

    def create_vectorstore(self, documents: List[Dict], progress_callback=None):
        """
        Create a vector store from documents.

        Args:
            documents: List of documents to index
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info(f"Processing {len(documents)} documents for Pinecone...")

        def update_progress(msg: str, pct: float):
            """Helper to update progress."""
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except Exception:
                    pass

        update_progress(f"Processing {len(documents)} documents...", 0.0)

        # Create or recreate index
        try:
            # Delete existing index if it exists
            if self.index_name in [index.name for index in self.pc.list_indexes()]:
                logger.info(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(1)  # Wait for deletion to complete

            # Create new index with serverless spec
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)

            update_progress("Index created successfully", 0.05)

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

        # Connect to index
        self.index = self.pc.Index(self.index_name)
        update_progress("Connected to index", 0.1)

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        update_progress("Splitting documents into chunks...", 0.15)
        chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc["metadata"]

            # Split text into chunks
            chunks = self._split_text(content)

            for chunk in chunks:
                if chunk.strip():
                    all_texts.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(f"doc_{chunk_id}")
                    chunk_id += 1

            # Update progress during chunking (15% to 30%)
            if doc_idx % 10 == 0:
                pct = 0.15 + (doc_idx / len(documents)) * 0.15
                update_progress(f"Chunking: {doc_idx}/{len(documents)} documents", pct)

        update_progress(f"Created {len(all_texts)} chunks", 0.3)

        # Create embeddings and upsert in batches
        update_progress("Creating embeddings and uploading to Pinecone...", 0.35)
        batch_size = 100
        total_batches = (len(all_texts) + batch_size - 1) // batch_size

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Create embeddings
            embeddings = self.embedding_model.encode(batch_texts).tolist()

            # Prepare vectors for Pinecone
            # Pinecone requires (id, embedding, metadata) tuples
            vectors = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, embeddings, batch_metadata)):
                # Add the text content to metadata (Pinecone doesn't store it separately)
                metadata_with_text = metadata.copy()
                metadata_with_text['text'] = batch_texts[idx]
                vectors.append((doc_id, embedding, metadata_with_text))

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

            # Update progress (35% to 95%)
            pct = 0.35 + (batch_num / total_batches) * 0.6
            update_progress(f"Uploading batch {batch_num}/{total_batches} to Pinecone", pct)

        update_progress("Vector store created successfully!", 1.0)
        logger.info(f"Successfully indexed {len(all_texts)} chunks to Pinecone")
        return self.index

    def load_vectorstore(self):
        """Load an existing vector store."""
        logger.info(f"Loading existing Pinecone index: {self.index_name}")

        # Check if index exists
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            raise ValueError(f"Index '{self.index_name}' does not exist")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Loaded index with {stats['total_vector_count']} vectors")

        return self.index

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of Document objects
        """
        if self.index is None:
            raise ValueError("Vector store not initialized. Call load_vectorstore() first.")

        # Create query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        # Convert to Document objects
        documents = []
        for match in results['matches']:
            metadata = match['metadata']
            # Extract text from metadata (we stored it there)
            text = metadata.pop('text', '')
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def delete_vectorstore(self):
        """Delete the vector store index."""
        if self.index_name in [index.name for index in self.pc.list_indexes()]:
            logger.info(f"Deleting Pinecone index: {self.index_name}")
            self.pc.delete_index(self.index_name)
            self.index = None
            logger.info("Index deleted successfully")
        else:
            logger.warning(f"Index '{self.index_name}' does not exist")

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            raise ValueError("Vector store not initialized")

        return self.index.describe_index_stats()


# ==================== Configuration Example ====================

class PineconeConfig:
    """Configuration for Pinecone integration."""

    # Add these to your existing Config class
    PINECONE_API_KEY = None  # Set via environment variable or Streamlit secrets
    PINECONE_INDEX_NAME = "bhfdsc-repos"
    PINECONE_CLOUD = "aws"  # or 'gcp', 'azure'
    PINECONE_REGION = "us-east-1"  # Depends on cloud provider

    @classmethod
    def validate(cls):
        """Validate Pinecone configuration."""
        if not cls.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Get your API key from https://app.pinecone.io/"
            )


# ==================== Integration Example ====================

def example_integration():
    """
    Example of how to integrate Pinecone into your existing code.

    Replace VectorStoreManager with PineconeVectorStore in your app.
    """

    # In config.py, add:
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    class Config:
        # ... existing config ...

        # Pinecone settings
        PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = "bhfdsc-repos"
        PINECONE_CLOUD = "aws"
        PINECONE_REGION = "us-east-1"
    """

    # In app.py or qa_chain.py, replace:
    # OLD:
    # from vector_store import VectorStoreManager
    # vector_store_manager = VectorStoreManager()

    # NEW:
    # from pinecone_backend import PineconeVectorStore
    # from config import Config
    #
    # vector_store_manager = PineconeVectorStore(
    #     api_key=Config.PINECONE_API_KEY,
    #     index_name=Config.PINECONE_INDEX_NAME,
    #     cloud=Config.PINECONE_CLOUD,
    #     region=Config.PINECONE_REGION
    # )

    # The rest of your code remains the same!
    # vector_store_manager.create_vectorstore(documents)
    # vector_store_manager.load_vectorstore()
    # results = vector_store_manager.similarity_search(query, k=5)


# ==================== Setup Instructions ====================

def setup_instructions():
    """Setup instructions for Pinecone."""
    print("""
    # Pinecone Setup Instructions

    ## 1. Create Pinecone Account
    - Go to https://app.pinecone.io/
    - Sign up for a free account (100K vectors, 1 index included)

    ## 2. Get API Key
    - Navigate to "API Keys" in the Pinecone console
    - Create a new API key or copy your default key
    - Save this securely

    ## 3. Install Dependencies
    ```bash
    pip install pinecone[grpc]
    ```

    ## 4. Configure Environment Variables
    Add to your .env file:
    ```
    PINECONE_API_KEY=your-api-key-here
    ```

    Or for Streamlit Cloud, add to .streamlit/secrets.toml:
    ```toml
    PINECONE_API_KEY = "your-api-key-here"
    ```

    ## 5. Update requirements.txt
    ```
    pinecone[grpc]>=3.0.0
    ```

    ## 6. Choose Region
    Free tier regions (as of 2024):
    - AWS: us-east-1
    - GCP: us-central1
    - Azure: eastus2

    ## 7. Migration Steps
    1. Update config.py with Pinecone settings
    2. Replace VectorStoreManager import with PineconeVectorStore
    3. Update initialization code
    4. Test locally before deploying
    5. Re-index your documents to Pinecone

    ## Cost Considerations
    - Free tier: 100K vectors, 1 index
    - Paid tier: Starting at $70/month for 2M vectors
    - No charges for queries on free tier

    ## Advantages over ChromaDB + Cloud Storage
    ✅ No download/upload delays
    ✅ Always up-to-date across all instances
    ✅ Better performance at scale
    ✅ Built-in monitoring and analytics
    ✅ Managed infrastructure
    ✅ Automatic backups
    """)


if __name__ == "__main__":
    setup_instructions()
