"""Pinecone vector store implementation."""

import os
import logging
import time
from typing import List, Dict

# Set environment variables BEFORE importing torch to prevent meta tensor issues
# This tells PyTorch/Transformers to avoid using meta device during model loading
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')  # Ensure we can download if needed

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class to replace langchain Document."""
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


class PineconeVectorStore:
    """Pinecone-based vector store manager."""

    def __init__(self):
        """Initialize Pinecone vector store."""
        logger.info("Initializing Pinecone vector store...")
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimension = Config.PINECONE_DIMENSION
        self.cloud = Config.PINECONE_CLOUD
        self.region = Config.PINECONE_REGION

        # Initialize embedding model with explicit device to avoid meta tensor issues
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading SentenceTransformer model '{Config.EMBEDDING_MODEL}' on device: {device}")
        logger.info("This may take 1-5 minutes on first run (downloading model from HuggingFace)...")

        # Load model with multi-fallback approach to handle meta tensor issues in PyTorch 2.0+
        # Environment variables set at module level help prevent meta device usage
        try:
            # Attempt 1: Load model directly on target device
            self.embedding_model = SentenceTransformer(
                Config.EMBEDDING_MODEL,
                device=device
            )
            logger.info(f"✓ Model loaded successfully on {device}")

        except (NotImplementedError, RuntimeError) as e:
            error_msg = str(e).lower()

            if "meta tensor" in error_msg or "cannot copy out of meta" in error_msg:
                # Attempt 2: Load on CPU first, then move to target device
                logger.warning("Meta tensor error detected, using CPU-first loading strategy...")
                try:
                    self.embedding_model = SentenceTransformer(
                        Config.EMBEDDING_MODEL,
                        device='cpu'
                    )
                    # Move to target device (should work from CPU -> GPU/other)
                    self.embedding_model = self.embedding_model.to(device)
                    logger.info(f"✓ Model loaded on CPU and successfully moved to {device}")

                except Exception as e2:
                    # Attempt 3: Keep on CPU if move fails
                    logger.warning(f"Could not move to {device}, keeping model on CPU: {e2}")
                    self.embedding_model = SentenceTransformer(
                        Config.EMBEDDING_MODEL,
                        device='cpu'
                    )
                    logger.info("✓ Model loaded and running on CPU")
            else:
                # Re-raise if it's a different error
                raise

        logger.info("✓ SentenceTransformer model initialization complete")

        # Initialize Pinecone client
        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        logger.info("✓ Pinecone client initialized")

    def _split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Simple text splitter."""
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

    def create_vectorstore(self, documents: List[Dict], progress_callback=None):
        """
        Create a vector store from documents.

        Args:
            documents: List of documents to index
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info("=" * 60)
        logger.info("STARTING VECTOR STORE CREATION")
        logger.info("=" * 60)
        logger.info(f"Processing {len(documents)} documents for Pinecone...")
        logger.info(f"Estimated chunks: ~{len(documents) * 3} (varies by document size)")
        logger.info(f"This process may take 10-30 minutes depending on document count...")

        def update_progress(msg: str, pct: float):
            """Helper to update progress."""
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except Exception:
                    pass

        update_progress(f"Processing {len(documents)} documents...", 0.0)

        # Create index only if it doesn't exist
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
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
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                update_progress("Connected to existing index", 0.05)

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
        logger.info(f"Will process {total_batches} batches of ~{batch_size} chunks each")

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Create embeddings
            logger.info(f"[Batch {batch_num}/{total_batches}] Creating embeddings for {len(batch_texts)} chunks...")
            embeddings = self.embedding_model.encode(batch_texts).tolist()
            logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Embeddings created")

            # Prepare vectors for Pinecone
            vectors = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, embeddings, batch_metadata)):
                # Add the text content to metadata
                metadata_with_text = metadata.copy()
                metadata_with_text['text'] = batch_texts[idx]
                vectors.append((doc_id, embedding, metadata_with_text))

            # Upsert to Pinecone
            logger.info(f"[Batch {batch_num}/{total_batches}] Uploading to Pinecone...")
            self.index.upsert(vectors=vectors)
            logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Upload complete")

            # Update progress (35% to 95%)
            pct = 0.35 + (batch_num / total_batches) * 0.6
            update_progress(f"Completed batch {batch_num}/{total_batches}", pct)

        update_progress("Vector store created successfully!", 1.0)
        logger.info(f"Successfully indexed {len(all_texts)} chunks to Pinecone")
        return self.index

    def ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if necessary."""
        if self.index is not None:
            return  # Already connected

        logger.info(f"Ensuring Pinecone index exists: {self.index_name}")

        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
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

                logger.info(f"✓ Index '{self.index_name}' created successfully")
            else:
                logger.info(f"✓ Using existing Pinecone index: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise

    def upsert_documents(self, documents: List[Dict], repo_name: str = None):
        """
        Incrementally upsert documents to Pinecone (streaming mode).

        This method is used for processing repositories one at a time,
        inserting into Pinecone as we go rather than batching everything.

        Args:
            documents: List of documents to upsert (typically from one repo)
            repo_name: Optional repo name for logging purposes
        """
        if not documents:
            logger.warning("No documents to upsert")
            return

        # Ensure index exists
        self.ensure_index_exists()

        repo_label = f" from {repo_name}" if repo_name else ""
        logger.info(f"Upserting {len(documents)} documents{repo_label} to Pinecone...")

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        # Get current max chunk ID from index stats to avoid collisions
        stats = self.index.describe_index_stats()
        current_vector_count = stats.get('total_vector_count', 0)
        chunk_id_offset = current_vector_count

        logger.info(f"Splitting {len(documents)} documents into chunks...")
        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc["metadata"]

            # Split text into chunks
            chunks = self._split_text(content)

            for chunk in chunks:
                if chunk.strip():
                    all_texts.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(f"doc_{chunk_id_offset}")
                    chunk_id_offset += 1

        logger.info(f"✓ Created {len(all_texts)} chunks from {len(documents)} documents")

        # Create embeddings and upsert in batches
        batch_size = 100
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        logger.info(f"Upserting in {total_batches} batch(es) of ~{batch_size} chunks each")

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Create embeddings
            logger.info(f"  [Batch {batch_num}/{total_batches}] Creating embeddings for {len(batch_texts)} chunks...")
            embeddings = self.embedding_model.encode(batch_texts).tolist()

            # Prepare vectors for Pinecone
            vectors = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, embeddings, batch_metadata)):
                metadata_with_text = metadata.copy()
                metadata_with_text['text'] = batch_texts[idx]
                vectors.append((doc_id, embedding, metadata_with_text))

            # Upsert to Pinecone
            logger.info(f"  [Batch {batch_num}/{total_batches}] Uploading to Pinecone...")
            self.index.upsert(vectors=vectors)
            logger.info(f"  [Batch {batch_num}/{total_batches}] ✓ Upload complete")

        logger.info(f"✓ Successfully upserted {len(all_texts)} chunks{repo_label}")

    def load_vectorstore(self):
        """Load an existing vector store."""
        logger.info(f"Loading existing Pinecone index: {self.index_name}")

        # Check if index exists
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Loaded index with {stats.get('total_vector_count', 0)} vectors")

        return self.index

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
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
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            # Extract text from metadata
            text = metadata.pop('text', '')
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def delete_vectorstore(self):
        """Delete the vector store index."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name in existing_indexes:
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

    def delete_repo_vectors(self, repo_full_name: str):
        """Delete all vectors for a specific repository.

        Args:
            repo_full_name: Full name of repository (e.g., "owner/repo")
        """
        if self.index is None:
            # Try to load the index
            self.load_vectorstore()

        logger.info(f"Deleting vectors for repository: {repo_full_name}")
        try:
            # Delete by metadata filter
            self.index.delete(filter={"repo": {"$eq": repo_full_name}})
            logger.info(f"Successfully deleted vectors for {repo_full_name}")
        except Exception as e:
            logger.error(f"Error deleting vectors for {repo_full_name}: {e}")
            raise

