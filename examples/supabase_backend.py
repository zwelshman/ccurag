"""
Example implementation: Supabase + pgvector Backend

This shows how to use PostgreSQL with pgvector extension via Supabase.

Supabase offers a generous free tier (500MB database) and combines
vector search with traditional relational database capabilities.
"""

import logging
from typing import List, Dict, Optional
import os
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import numpy as np

logger = logging.getLogger(__name__)


class Document:
    """Simple document class compatible with existing code."""

    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


class SupabaseVectorStore:
    """
    Supabase + pgvector based vector store manager.

    Drop-in replacement for VectorStoreManager using Supabase.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize Supabase vector store.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (anon/service key)
            table_name: Name of the database table
            embedding_model: SentenceTransformer model name
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.table_name = table_name

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize Supabase client
        self.client: Client = create_client(supabase_url, supabase_key)

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

    def _setup_database(self):
        """
        Set up the database schema.

        This needs to be run once to create the necessary tables and functions.
        Note: This requires database admin privileges.
        """
        logger.info("Setting up Supabase database schema...")

        # SQL to create tables and enable pgvector
        # This should be run in the Supabase SQL editor manually
        setup_sql = f"""
        -- Enable the pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Create documents table
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(384), -- 384 dimensions for all-MiniLM-L6-v2
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create index for faster similarity search
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
        ON {self.table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Create function for similarity search
        CREATE OR REPLACE FUNCTION match_documents(
            query_embedding vector(384),
            match_threshold float,
            match_count int
        )
        RETURNS TABLE (
            id int,
            content text,
            metadata jsonb,
            similarity float
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                {self.table_name}.id,
                {self.table_name}.content,
                {self.table_name}.metadata,
                1 - ({self.table_name}.embedding <=> query_embedding) AS similarity
            FROM {self.table_name}
            WHERE 1 - ({self.table_name}.embedding <=> query_embedding) > match_threshold
            ORDER BY similarity DESC
            LIMIT match_count;
        $$;
        """

        logger.info(
            "\nPlease run the following SQL in your Supabase SQL Editor:\n"
            "----------------------------------------\n"
            f"{setup_sql}\n"
            "----------------------------------------"
        )

        return setup_sql

    def create_vectorstore(self, documents: List[Dict], progress_callback=None):
        """
        Create a vector store from documents.

        Args:
            documents: List of documents to index
            progress_callback: Optional callback function(message: str, progress_pct: float)
        """
        logger.info(f"Processing {len(documents)} documents for Supabase...")

        def update_progress(msg: str, pct: float):
            """Helper to update progress."""
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except Exception:
                    pass

        update_progress(f"Processing {len(documents)} documents...", 0.0)

        # Clear existing data
        try:
            logger.info(f"Clearing existing data from {self.table_name}...")
            self.client.table(self.table_name).delete().neq('id', 0).execute()
            update_progress("Cleared existing data", 0.05)
        except Exception as e:
            logger.warning(f"Could not clear existing data: {e}")

        # Process and split documents
        all_texts = []
        all_metadatas = []

        update_progress("Splitting documents into chunks...", 0.1)

        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc["metadata"]

            # Split text into chunks
            chunks = self._split_text(content)

            for chunk in chunks:
                if chunk.strip():
                    all_texts.append(chunk)
                    all_metadatas.append(metadata)

            # Update progress during chunking (10% to 30%)
            if doc_idx % 10 == 0:
                pct = 0.1 + (doc_idx / len(documents)) * 0.2
                update_progress(f"Chunking: {doc_idx}/{len(documents)} documents", pct)

        update_progress(f"Created {len(all_texts)} chunks", 0.3)

        # Create embeddings and insert in batches
        update_progress("Creating embeddings and uploading to Supabase...", 0.35)
        batch_size = 100
        total_batches = (len(all_texts) + batch_size - 1) // batch_size

        for i in range(0, len(all_texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = all_texts[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]

            # Create embeddings
            embeddings = self.embedding_model.encode(batch_texts)

            # Prepare data for insertion
            rows = []
            for text, embedding, metadata in zip(batch_texts, embeddings, batch_metadata):
                rows.append({
                    'content': text,
                    'embedding': embedding.tolist(),
                    'metadata': metadata
                })

            # Insert into Supabase
            try:
                self.client.table(self.table_name).insert(rows).execute()
            except Exception as e:
                logger.error(f"Error inserting batch {batch_num}: {e}")
                # Continue with other batches
                continue

            # Update progress (35% to 95%)
            pct = 0.35 + (batch_num / total_batches) * 0.6
            update_progress(f"Uploading batch {batch_num}/{total_batches} to Supabase", pct)

        update_progress("Vector store created successfully!", 1.0)
        logger.info(f"Successfully indexed {len(all_texts)} chunks to Supabase")

    def load_vectorstore(self):
        """
        Load an existing vector store (check if table exists and has data).
        """
        logger.info(f"Checking Supabase table: {self.table_name}")

        try:
            # Try to count rows
            result = self.client.table(self.table_name).select('id', count='exact').limit(1).execute()
            count = result.count if hasattr(result, 'count') else 0
            logger.info(f"Table exists with {count} rows")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise ValueError(f"Table '{self.table_name}' does not exist or is not accessible")

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of Document objects
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Call the match_documents function
        try:
            result = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': k
                }
            ).execute()

            # Convert to Document objects
            documents = []
            for row in result.data:
                documents.append(Document(
                    page_content=row['content'],
                    metadata=row['metadata']
                ))

            return documents

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            # Fallback to manual calculation if RPC fails
            return self._fallback_similarity_search(query_embedding, k)

    def _fallback_similarity_search(self, query_embedding: List[float], k: int) -> List[Document]:
        """
        Fallback similarity search using client-side calculation.

        This is slower but works if the RPC function is not available.
        """
        logger.info("Using fallback similarity search...")

        # Fetch all documents (not recommended for large datasets)
        result = self.client.table(self.table_name).select('*').execute()

        # Calculate similarities
        documents_with_scores = []
        query_emb = np.array(query_embedding)

        for row in result.data:
            doc_emb = np.array(row['embedding'])
            # Cosine similarity
            similarity = 1 - np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            documents_with_scores.append((similarity, row))

        # Sort by similarity and take top k
        documents_with_scores.sort(key=lambda x: x[0], reverse=True)

        # Convert to Document objects
        documents = []
        for _, row in documents_with_scores[:k]:
            documents.append(Document(
                page_content=row['content'],
                metadata=row['metadata']
            ))

        return documents

    def delete_vectorstore(self):
        """Delete all data from the vector store table."""
        logger.info(f"Deleting all data from {self.table_name}...")
        try:
            self.client.table(self.table_name).delete().neq('id', 0).execute()
            logger.info("All data deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting data: {e}")

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        try:
            result = self.client.table(self.table_name).select('id', count='exact').execute()
            count = result.count if hasattr(result, 'count') else 0
            return {
                'total_documents': count,
                'table_name': self.table_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# ==================== Configuration Example ====================

class SupabaseConfig:
    """Configuration for Supabase integration."""

    # Add these to your existing Config class
    SUPABASE_URL = None  # e.g., "https://xxxxx.supabase.co"
    SUPABASE_KEY = None  # Your anon or service role key
    SUPABASE_TABLE_NAME = "documents"

    @classmethod
    def validate(cls):
        """Validate Supabase configuration."""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY are required. "
                "Get them from https://app.supabase.com/"
            )


# ==================== Setup Instructions ====================

def setup_instructions():
    """Setup instructions for Supabase + pgvector."""
    print("""
    # Supabase + pgvector Setup Instructions

    ## 1. Create Supabase Project
    - Go to https://app.supabase.com/
    - Sign up and create a new project
    - Wait for the project to be provisioned (~2 minutes)

    ## 2. Get API Credentials
    - In your project dashboard, go to Settings > API
    - Copy the "Project URL" (SUPABASE_URL)
    - Copy the "anon/public" key (SUPABASE_KEY)
      - For read-only operations, use anon key
      - For write operations, you may need the service_role key

    ## 3. Enable pgvector Extension
    - Go to Database > Extensions in Supabase dashboard
    - Search for "vector"
    - Enable the "vector" extension

    ## 4. Create Database Schema
    - Go to SQL Editor in Supabase dashboard
    - Create a new query
    - Run the SQL provided by _setup_database() method:

    ```sql
    -- Enable the pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Create documents table
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB,
        embedding vector(384),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Create index
    CREATE INDEX documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

    -- Create search function
    CREATE OR REPLACE FUNCTION match_documents(
        query_embedding vector(384),
        match_threshold float,
        match_count int
    )
    RETURNS TABLE (
        id int,
        content text,
        metadata jsonb,
        similarity float
    )
    LANGUAGE sql STABLE
    AS $$
        SELECT
            documents.id,
            documents.content,
            documents.metadata,
            1 - (documents.embedding <=> query_embedding) AS similarity
        FROM documents
        WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
        ORDER BY similarity DESC
        LIMIT match_count;
    $$;
    ```

    ## 5. Install Dependencies
    ```bash
    pip install supabase
    ```

    ## 6. Configure Environment Variables
    Add to your .env file:
    ```
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-anon-or-service-key
    ```

    Or for Streamlit Cloud, add to .streamlit/secrets.toml:
    ```toml
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-or-service-key"
    ```

    ## 7. Update requirements.txt
    ```
    supabase>=2.0.0
    ```

    ## 8. Integration Example
    In config.py:
    ```python
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_KEY = get_secret("SUPABASE_KEY")
    ```

    In your app:
    ```python
    from supabase_backend import SupabaseVectorStore
    from config import Config

    vector_store = SupabaseVectorStore(
        supabase_url=Config.SUPABASE_URL,
        supabase_key=Config.SUPABASE_KEY
    )
    ```

    ## Cost Considerations
    - Free tier: 500MB database, 2GB bandwidth, 50MB file storage
    - Paid tier: Starting at $25/month for more resources
    - No additional charges for vector operations

    ## Advantages
    ✅ Generous free tier (500MB database)
    ✅ Combines vector search with PostgreSQL
    ✅ Real-time subscriptions available
    ✅ Built-in authentication and storage
    ✅ Open source (can self-host)
    ✅ Row-level security for multi-tenant apps
    ✅ Automatic backups (paid plans)

    ## Performance Tips
    1. Use IVFFlat index for better search performance
    2. Adjust 'lists' parameter based on dataset size:
       - Small (<10K vectors): lists = 10-50
       - Medium (10K-100K): lists = 100-500
       - Large (>100K): lists = 1000+
    3. Consider using service_role key for indexing operations
    4. Use connection pooling for production apps
    """)


if __name__ == "__main__":
    setup_instructions()
