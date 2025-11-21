"""ChromaDB vector store implementation (legacy)."""

import logging
from typing import List, Dict
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class to replace langchain Document."""
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


class ChromaVectorStore:
    """ChromaDB-based vector store manager (legacy)."""

    def __init__(self):
        """Initialize vector store components."""
        logger.info("Initializing ChromaDB vector store...")
        # Initialize embedding model with explicit device to avoid meta tensor issues
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading SentenceTransformer model '{Config.EMBEDDING_MODEL}' on device: {device}")
        logger.info("This may take 1-5 minutes on first run (downloading model from HuggingFace)...")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
        logger.info("✓ SentenceTransformer model loaded successfully")
        self.client = None
        self.collection = None

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
        """Create a vector store from documents.

        Args:
            documents: List of documents to index
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info("=" * 60)
        logger.info("STARTING VECTOR STORE CREATION")
        logger.info("=" * 60)
        logger.info(f"Processing {len(documents)} documents for ChromaDB...")
        logger.info(f"Estimated chunks: ~{len(documents) * 3} (varies by document size)")
        logger.info(f"This process may take 10-30 minutes depending on document count...")

        def update_progress(msg: str, pct: float):
            """Helper to update progress via callback or logging."""
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except:
                    pass  # Ignore callback errors

        update_progress(f"Processing {len(documents)} documents...", 0.0)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        try:
            self.client.delete_collection(name=Config.COLLECTION_NAME)
        except:
            pass

        self.collection = self.client.create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        update_progress("Initialized database", 0.05)

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        update_progress("Splitting documents into chunks...", 0.1)
        chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc["metadata"]

            # Split text into chunks
            chunks = self._split_text(content)

            for chunk in chunks:
                if chunk.strip():  # Only add non-empty chunks
                    all_texts.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(f"doc_{chunk_id}")
                    chunk_id += 1

            # Update progress during chunking (10% to 30%)
            if doc_idx % 10 == 0:
                pct = 0.1 + (doc_idx / len(documents)) * 0.2
                update_progress(f"Chunking: {doc_idx}/{len(documents)} documents", pct)

        update_progress(f"Created {len(all_texts)} chunks", 0.3)

        # Add to collection in batches
        update_progress("Creating embeddings and adding to vector store...", 0.35)
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

            # Add to collection
            logger.info(f"[Batch {batch_num}/{total_batches}] Adding to ChromaDB...")
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Added to database")

            # Update progress (35% to 95%)
            pct = 0.35 + (batch_num / total_batches) * 0.6
            update_progress(f"Completed batch {batch_num}/{total_batches}", pct)

        update_progress("Vector store created successfully!", 1.0)
        return self.collection

    def ensure_collection_exists(self):
        """Ensure the ChromaDB collection exists, create if necessary."""
        if self.collection is not None:
            return  # Already connected

        logger.info(f"Ensuring ChromaDB collection exists: {Config.COLLECTION_NAME}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # Try to get existing collection, create if doesn't exist
        try:
            self.collection = self.client.get_collection(name=Config.COLLECTION_NAME)
            logger.info(f"✓ Using existing ChromaDB collection: {Config.COLLECTION_NAME}")
        except Exception:
            logger.info(f"Creating new ChromaDB collection: {Config.COLLECTION_NAME}")
            self.collection = self.client.create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✓ Collection '{Config.COLLECTION_NAME}' created successfully")

    def upsert_documents(self, documents: List[Dict], repo_name: str = None):
        """
        Incrementally upsert documents to ChromaDB (streaming mode).

        This method is used for processing repositories one at a time,
        inserting into ChromaDB as we go rather than batching everything.

        Args:
            documents: List of documents to upsert (typically from one repo)
            repo_name: Optional repo name for logging purposes
        """
        if not documents:
            logger.warning("No documents to upsert")
            return

        # Ensure collection exists
        self.ensure_collection_exists()

        repo_label = f" from {repo_name}" if repo_name else ""
        logger.info(f"Upserting {len(documents)} documents{repo_label} to ChromaDB...")

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        # Get current count to avoid ID collisions
        collection_count = self.collection.count()
        chunk_id_offset = collection_count

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

            # Add to ChromaDB
            logger.info(f"  [Batch {batch_num}/{total_batches}] Adding to ChromaDB...")
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            logger.info(f"  [Batch {batch_num}/{total_batches}] ✓ Added to database")

        logger.info(f"✓ Successfully upserted {len(all_texts)} chunks{repo_label}")

    def load_vectorstore(self):
        """Load an existing vector store."""
        logger.info("Loading existing vector store...")

        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection(name=Config.COLLECTION_NAME)

        logger.info("Vector store loaded successfully")
        return self.collection

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.collection is None:
            raise ValueError("Vector store not initialized.")

        # Create query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Convert to Document objects
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents
