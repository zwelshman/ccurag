"""Hybrid retriever combining BM25 and vector search for optimal code+docs retrieval."""

import logging
import re
import pickle
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from vector_store_pinecone import PineconeVectorStore, Document
from config import Config
from cloud_storage import CloudStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining BM25 keyword search and vector semantic search."""

    def __init__(self, vector_store: PineconeVectorStore, bm25_weight: float = None):
        """Initialize hybrid retriever.

        Args:
            vector_store: Initialized Pinecone vector store
            bm25_weight: Weight for BM25 scores (0-1). If None, uses Config value.
                        Remaining weight goes to vector search.
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight if bm25_weight is not None else Config.BM25_WEIGHT
        self.vector_weight = 1.0 - self.bm25_weight

        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_ids: List[str] = []

        # Cache configuration
        self.cache_dir = ".cache"
        self.data_index_dir = "data_index"
        self.bm25_cache_file = os.path.join(self.cache_dir, "bm25_index.pkl")
        self.bm25_data_index_file = os.path.join(self.data_index_dir, "bm25_index.pkl")

        # Initialize cloud storage
        self.storage = CloudStorage()

        logger.info(f"Initialized HybridRetriever with BM25 weight: {self.bm25_weight:.2f}, "
                   f"Vector weight: {self.vector_weight:.2f}")

    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization for code and documentation.

        Handles:
        - CamelCase splitting (getUserData -> get, user, data)
        - snake_case preservation
        - Special characters in code
        - Common stop words removal for better relevance

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Split CamelCase: getUserData -> get User Data
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Replace special chars with spaces (but preserve underscores in words)
        text = re.sub(r'[^\w\s_]', ' ', text)

        # Split on whitespace and underscores
        tokens = []
        for word in text.split():
            # Split on underscores but keep the parts
            parts = word.split('_')
            tokens.extend([p for p in parts if p])

        # Remove very short tokens (less than 2 chars) and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'this',
                     'that', 'it', 'if'}

        tokens = [t for t in tokens if len(t) >= 2 and t not in stop_words]

        return tokens

    def build_bm25_index(self, documents: List[Document], force_rebuild: bool = False):
        """Build BM25 index from documents.

        Args:
            documents: List of Document objects to index
            force_rebuild: If True, rebuild even if cache exists
        """
        # Try to load from cache first - check data_index, then .cache
        if not force_rebuild:
            # Try data_index first (committed files)
            if self.storage.exists(self.bm25_data_index_file):
                logger.info("Loading BM25 index from data_index...")
                try:
                    cache_data = self.storage.load_pickle(self.bm25_data_index_file)
                    self.bm25 = cache_data['bm25']
                    self.documents = cache_data['documents']
                    self.tokenized_corpus = cache_data['tokenized_corpus']
                    self.doc_ids = cache_data['doc_ids']
                    logger.info(f"✓ Loaded BM25 index with {len(self.documents)} documents from data_index")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load from data_index, trying .cache: {e}")

            # Fallback to .cache directory
            if self.storage.exists(self.bm25_cache_file):
                logger.info("Loading BM25 index from cache...")
                try:
                    cache_data = self.storage.load_pickle(self.bm25_cache_file)
                    self.bm25 = cache_data['bm25']
                    self.documents = cache_data['documents']
                    self.tokenized_corpus = cache_data['tokenized_corpus']
                    self.doc_ids = cache_data['doc_ids']
                    logger.info(f"✓ Loaded BM25 index with {len(self.documents)} documents from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cache, rebuilding: {e}")

        logger.info(f"Building BM25 index for {len(documents)} documents...")

        self.documents = documents
        self.tokenized_corpus = []
        self.doc_ids = []

        # Tokenize all documents
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                logger.info(f"  Tokenizing document {i}/{len(documents)}...")

            # Combine content and important metadata for better matching
            searchable_text = doc.page_content

            # Add file name and repo for better code matching
            if 'source' in doc.metadata:
                searchable_text += " " + doc.metadata['source']
            if 'repo' in doc.metadata:
                searchable_text += " " + doc.metadata['repo']

            tokens = self._tokenize(searchable_text)
            self.tokenized_corpus.append(tokens)

            # Generate doc ID from metadata or position
            doc_id = doc.metadata.get('doc_id', f'doc_{i}')
            self.doc_ids.append(doc_id)

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"✓ BM25 index built successfully with {len(self.documents)} documents")

        # Save to cache
        self._save_cache()

    def _save_cache(self):
        """Save BM25 index to cache file."""
        try:
            cache_data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'tokenized_corpus': self.tokenized_corpus,
                'doc_ids': self.doc_ids
            }
            self.storage.save_pickle(cache_data, self.bm25_cache_file)
        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")

    def clear_cache(self):
        """Clear the BM25 index cache from both .cache and data_index."""
        cleared = False
        if self.storage.exists(self.bm25_cache_file):
            self.storage.delete(self.bm25_cache_file)
            cleared = True
        if self.storage.exists(self.bm25_data_index_file):
            self.storage.delete(self.bm25_data_index_file)
            cleared = True
        if cleared:
            logger.info("✓ BM25 cache cleared")

    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        """Adaptively adjust weights based on query characteristics.

        Code-like queries get higher BM25 weight for exact matching.
        Conceptual queries get higher vector weight for semantic understanding.

        Args:
            query: User's query text

        Returns:
            Tuple of (bm25_weight, vector_weight)
        """
        if not Config.USE_ADAPTIVE_WEIGHTS:
            return (self.bm25_weight, self.vector_weight)

        # Check for code-like patterns
        has_camel_case = bool(re.search(r'[a-z][A-Z]', query))
        has_snake_case = '_' in query
        has_parentheses = '(' in query or ')' in query
        has_backticks = '`' in query
        has_dots = '.' in query and not query.endswith('.')
        has_quotes = '"' in query or "'" in query

        code_indicators = sum([
            has_camel_case, has_snake_case, has_parentheses,
            has_backticks, has_dots, has_quotes
        ])

        # Adjust weights based on code indicators
        if code_indicators >= 3:
            # Very code-like: favor BM25 heavily
            return (0.7, 0.3)
        elif code_indicators >= 2:
            # Somewhat code-like: favor BM25 moderately
            return (0.6, 0.4)
        elif code_indicators == 1:
            # Slightly code-like: balanced with slight BM25 preference
            return (0.5, 0.5)
        else:
            # Conceptual query: favor vector search
            return (0.3, 0.7)

    def similarity_search(self, query: str, k: int = 20,
                         use_adaptive_weights: bool = None,
                         return_metadata: bool = False) -> List[Document]:
        """Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Query text
            k: Number of results to return
            use_adaptive_weights: Override config to enable/disable adaptive weights
            return_metadata: If True, attach search metadata to documents

        Returns:
            List of top-k Document objects (with optional search_metadata)
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built, falling back to vector search only")
            return self.vector_store.similarity_search(query, k=k)

        if not self.documents:
            logger.warning("No documents in BM25 index, falling back to vector search")
            return self.vector_store.similarity_search(query, k=k)

        # Determine weights
        if use_adaptive_weights or (use_adaptive_weights is None and Config.USE_ADAPTIVE_WEIGHTS):
            bm25_w, vector_w = self._get_adaptive_weights(query)
            logger.info(f"Using adaptive weights: BM25={bm25_w:.2f}, Vector={vector_w:.2f}")
        else:
            bm25_w, vector_w = self.bm25_weight, self.vector_weight

        # Get BM25 scores
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        if bm25_scores.max() > 0:
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores_norm = np.zeros_like(bm25_scores)

        # Get vector search results - retrieve more to ensure good coverage
        vector_k = min(len(self.documents), k * 3)
        try:
            vector_results = self.vector_store.similarity_search(query, k=vector_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}, falling back to BM25 only")
            # Fall back to BM25 only
            top_k_indices = np.argsort(bm25_scores)[::-1][:k]
            return [self.documents[i] for i in top_k_indices if i < len(self.documents)]

        # Create mapping from document content to BM25 index
        # This handles the case where vector store and BM25 might have different ordering
        content_to_bm25_idx = {}
        for i, doc in enumerate(self.documents):
            # Use a hash of content as key for matching
            content_hash = hash(doc.page_content[:200])  # First 200 chars for matching
            content_to_bm25_idx[content_hash] = i

        # Score vector results and combine with BM25
        combined_scores = {}

        # Add vector results with their scores
        for rank, doc in enumerate(vector_results):
            content_hash = hash(doc.page_content[:200])

            # Vector score: decreases with rank (1.0 for first result, approaching 0 for last)
            vector_score = 1.0 - (rank / len(vector_results))

            # Get BM25 score if document exists in BM25 index
            if content_hash in content_to_bm25_idx:
                bm25_idx = content_to_bm25_idx[content_hash]
                bm25_score = bm25_scores_norm[bm25_idx]
            else:
                bm25_score = 0.0

            # Combine scores
            combined_score = (bm25_w * bm25_score) + (vector_w * vector_score)
            combined_scores[content_hash] = {
                'score': combined_score,
                'doc': doc,
                'bm25_score': bm25_score,
                'vector_score': vector_score
            }

        # Add high-scoring BM25 results that might not be in vector results
        top_bm25_threshold = 0.5  # Only add BM25 results with normalized score > 0.5
        for i, score in enumerate(bm25_scores_norm):
            if score > top_bm25_threshold:
                doc = self.documents[i]
                content_hash = hash(doc.page_content[:200])

                if content_hash not in combined_scores:
                    # This doc was highly ranked by BM25 but not in vector results
                    combined_score = bm25_w * score
                    combined_scores[content_hash] = {
                        'score': combined_score,
                        'doc': doc,
                        'bm25_score': score,
                        'vector_score': 0.0
                    }

        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]

        # Log scoring details for debugging
        logger.info(f"Hybrid search returned {len(sorted_results)} results")
        logger.info(f"Used weights: BM25={bm25_w:.2f}, Vector={vector_w:.2f}")
        for i, result in enumerate(sorted_results[:3]):  # Log top 3
            logger.debug(f"  Result {i+1}: combined={result['score']:.3f}, "
                        f"bm25={result['bm25_score']:.3f}, "
                        f"vector={result['vector_score']:.3f}")

        # Attach search metadata if requested
        results = []
        for r in sorted_results:
            doc = r['doc']
            if return_metadata:
                # Store metadata as document attribute
                doc.search_metadata = {
                    'bm25_weight_used': bm25_w,
                    'vector_weight_used': vector_w,
                    'bm25_score': r['bm25_score'],
                    'vector_score': r['vector_score'],
                    'combined_score': r['score'],
                    'adaptive_weights_used': use_adaptive_weights or (use_adaptive_weights is None and Config.USE_ADAPTIVE_WEIGHTS)
                }
            results.append(doc)

        return results

    def get_index_stats(self) -> Dict:
        """Get statistics about the hybrid index.

        Returns:
            Dictionary with index statistics
        """
        vector_stats = self.vector_store.get_stats()

        cache_exists = (
            self.storage.exists(self.bm25_cache_file) or
            self.storage.exists(self.bm25_data_index_file)
        )

        return {
            'bm25_documents': len(self.documents),
            'bm25_indexed': self.bm25 is not None,
            'vector_store_vectors': vector_stats.get('total_vector_count', 0),
            'bm25_weight': self.bm25_weight,
            'vector_weight': self.vector_weight,
            'adaptive_weights_enabled': Config.USE_ADAPTIVE_WEIGHTS,
            'cache_exists': cache_exists
        }
