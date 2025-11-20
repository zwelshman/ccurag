"""Vector store management using ChromaDB."""

import logging
from typing import List, Dict
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector store for document retrieval."""

    def __init__(self):
        """Initialize vector store components."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Dict]) -> Chroma:
        """Create a vector store from documents."""
        logger.info(f"Processing {len(documents)} documents...")

        # Convert to LangChain documents
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(langchain_docs)
        logger.info(f"Created {len(split_docs)} chunks")

        # Create vector store
        logger.info("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.CHROMA_DB_DIR,
        )

        logger.info("Vector store created successfully")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """Load an existing vector store."""
        logger.info("Loading existing vector store...")
        self.vectorstore = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_DIR,
        )
        logger.info("Vector store loaded successfully")
        return self.vectorstore

    def get_retriever(self, k: int = 5):
        """Get a retriever from the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(query, k=k)
