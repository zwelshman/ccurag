"""Vector store management using ChromaDB directly."""

import logging
from typing import List, Dict
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


class VectorStoreManager:
    """Manages the vector store for document retrieval."""

    def __init__(self):
        """Initialize vector store components."""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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

    def create_vectorstore(self, documents: List[Dict]):
        """Create a vector store from documents."""
        logger.info(f"Processing {len(documents)} documents...")

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

        # Process and split documents
        all_texts = []
        all_metadatas = []
        all_ids = []

        logger.info("Splitting documents into chunks...")
        chunk_id = 0

        for doc in documents:
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

        logger.info(f"Created {len(all_texts)} chunks")

        # Add to collection in batches
        logger.info("Creating embeddings and adding to vector store...")
        batch_size = 100

        for i in range(0, len(all_texts), batch_size):
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

            logger.info(f"Added batch {i // batch_size + 1}/{(len(all_texts) + batch_size - 1) // batch_size}")

        logger.info("Vector store created successfully")
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
