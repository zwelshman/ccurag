"""ChromaDB vector store implementation (legacy)."""

import logging
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import Config
from utils import Document, split_text, update_progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store manager (legacy)."""

    def __init__(self):
        """Initialize vector store components."""
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = None
        self.collection = None

    def create_vectorstore(self, documents: List[Dict], progress_callback=None):
        """Create a vector store from documents.

        Args:
            documents: List of documents to index
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info(f"Processing {len(documents)} documents...")
        update_progress(f"Processing {len(documents)} documents...", 0.0, progress_callback)

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

        update_progress("Initialized database", 0.05, progress_callback)

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        update_progress("Splitting documents into chunks...", 0.1, progress_callback)
        chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc["metadata"]

            # Split text into chunks
            chunks = split_text(content)

            for chunk in chunks:
                if chunk.strip():  # Only add non-empty chunks
                    all_texts.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(f"doc_{chunk_id}")
                    chunk_id += 1

            # Update progress during chunking (10% to 30%)
            if doc_idx % 10 == 0:
                pct = 0.1 + (doc_idx / len(documents)) * 0.2
                update_progress(f"Chunking: {doc_idx}/{len(documents)} documents", pct, progress_callback)

        update_progress(f"Created {len(all_texts)} chunks", 0.3, progress_callback)

        # Add to collection in batches
        update_progress("Creating embeddings and adding to vector store...", 0.35, progress_callback)
        batch_size = 100
        total_batches = (len(all_texts) + batch_size - 1) // batch_size

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Create embeddings
            embeddings = self.embedding_model.encode(batch_texts).tolist()

            # Add to collection
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadata,
                ids=batch_ids
            )

            # Update progress (35% to 95%)
            pct = 0.35 + (batch_num / total_batches) * 0.6
            update_progress(f"Adding batch {batch_num}/{total_batches} to vector store", pct, progress_callback)

        update_progress("Vector store created successfully!", 1.0, progress_callback)
        return self.collection

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
