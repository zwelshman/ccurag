"""Pinecone vector store implementation."""

import logging
import time
from typing import List, Dict
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
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimension = Config.PINECONE_DIMENSION
        self.cloud = Config.PINECONE_CLOUD
        self.region = Config.PINECONE_REGION

        # Initialize embedding model with explicit device to avoid meta tensor issues
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading SentenceTransformer on device: {device}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None

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

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Create embeddings
            embeddings = self.embedding_model.encode(batch_texts).tolist()

            # Prepare vectors for Pinecone
            vectors = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, embeddings, batch_metadata)):
                # Add the text content to metadata
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
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Loaded index with {stats.get('total_vector_count', 0)} vectors")

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

    def upsert_documents(self, documents: List[Dict], repos_to_update: List[str] = None, progress_callback=None):
        """
        Upsert documents to an existing vector store, optionally deleting old vectors for specific repos first.

        Args:
            documents: List of documents to index
            repos_to_update: List of repo full names to update (will delete old vectors first)
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info(f"Upserting {len(documents)} documents to Pinecone...")

        def update_progress(msg: str, pct: float):
            """Helper to update progress."""
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except Exception:
                    pass

        update_progress(f"Upserting {len(documents)} documents...", 0.0)

        # Load existing index
        if self.index is None:
            self.load_vectorstore()
        update_progress("Connected to index", 0.05)

        # Delete old vectors for repos that are being updated
        if repos_to_update:
            update_progress(f"Deleting old vectors for {len(repos_to_update)} repos...", 0.1)
            for repo_name in repos_to_update:
                self.delete_repo_vectors(repo_name)
            update_progress("Old vectors deleted", 0.2)

        # Process and split documents (similar to create_vectorstore)
        all_texts = []
        all_metadatas = []
        all_ids = []

        update_progress("Splitting documents into chunks...", 0.25)
        chunk_id = int(time.time() * 1000)  # Use timestamp to avoid ID collisions

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

            # Update progress during chunking (25% to 40%)
            if doc_idx % 10 == 0:
                pct = 0.25 + (doc_idx / len(documents)) * 0.15
                update_progress(f"Chunking: {doc_idx}/{len(documents)} documents", pct)

        update_progress(f"Created {len(all_texts)} chunks", 0.4)

        # Create embeddings and upsert in batches
        update_progress("Creating embeddings and uploading to Pinecone...", 0.45)
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
            vectors = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, embeddings, batch_metadata)):
                # Add the text content to metadata
                metadata_with_text = metadata.copy()
                metadata_with_text['text'] = batch_texts[idx]
                vectors.append((doc_id, embedding, metadata_with_text))

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

            # Update progress (45% to 95%)
            pct = 0.45 + (batch_num / total_batches) * 0.5
            update_progress(f"Uploading batch {batch_num}/{total_batches} to Pinecone", pct)

        update_progress("Documents upserted successfully!", 1.0)
        logger.info(f"Successfully upserted {len(all_texts)} chunks to Pinecone")
        return self.index
