"""Question-answering system using Anthropic SDK directly."""

import logging
from typing import Dict, List
from anthropic import Anthropic
from config import Config
from vector_store import VectorStoreManager
from query_router import QueryRouter, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    """Question-answering system using RAG with Anthropic Claude."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize QA system."""
        Config.validate()
        self.vector_store_manager = vector_store_manager
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.query_router = QueryRouter()

        # Custom prompt template
        self.system_prompt = """You are an AI assistant helping researchers understand the BHF Data Science Centre (BHFDSC) GitHub organization. This organization focuses on cardiovascular health research, particularly studying COVID-19's impact on cardiovascular disease.

Use the following pieces of context from the organization's repositories to answer the question. If you don't know the answer based on the context, say so - don't make up information.

When referencing specific repositories or code, include the source information."""

    def answer_question(self, question: str, k: int = 20) -> Dict:
        """Answer a question using RAG with query routing.

        Args:
            question: The user's question
            k: Number of documents to retrieve (may be adjusted based on query type)

        Returns:
            Dictionary containing answer and source documents
        """
        logger.info(f"Processing question: {question}")

        # Classify the query
        query_type = self.query_router.classify_query(question)
        query_metadata = self.query_router.get_query_metadata(question, query_type)

        # Route to appropriate handler
        if query_type == QueryType.REPOSITORY_LIST:
            # Use suggested k for repository queries
            k_to_use = query_metadata.get("suggested_k", k)
            return self._handle_repository_list_query(question, k_to_use)

        elif query_type == QueryType.LATEST_PROJECTS:
            # Use suggested k for temporal queries
            k_to_use = query_metadata.get("suggested_k", k)
            return self._handle_temporal_query(question, k_to_use)

        else:
            # Use standard RAG flow for general Q&A and code explanation
            return self._handle_general_qa(question, k)

    def _handle_repository_list_query(self, question: str, k: int = 10) -> Dict:
        """Handle queries asking for list of repositories.

        Args:
            question: The user's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with repository list as answer
        """
        logger.info(f"Handling repository list query with k={k}")
        repos = self.get_relevant_repos(question, k=k)

        # Format as a clean list response
        if repos:
            repo_lines = []
            for i, repo in enumerate(repos, 1):
                repo_name = repo['name']
                repo_url = repo['url']
                mention_count = repo.get('mention_count', 1)
                repo_lines.append(f"{i}. **{repo_name}** - {repo_url}")
                if mention_count > 1:
                    repo_lines.append(f"   _(Found in {mention_count} file(s))_")

            repo_list = "\n".join(repo_lines)
            answer = f"Here are the repositories that match your query:\n\n{repo_list}"
        else:
            answer = "I couldn't find any repositories matching your query in the indexed data."

        return {
            "answer": answer,
            "repositories": repos,  # Structured data
            "source_documents": [],  # No need to show source docs for repo list
            "query_type": "repository_list"
        }

    def _handle_temporal_query(self, question: str, k: int = 10) -> Dict:
        """Handle queries asking for latest/recent projects.

        Args:
            question: The user's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with temporally-sorted answer
        """
        logger.info(f"Handling temporal query with k={k}")

        # Get more candidates than needed for better temporal sorting
        search_k = k * 3
        docs = self.vector_store_manager.similarity_search(question, k=search_k)

        # Re-rank by timestamp (newest first)
        docs_with_time = []
        for doc in docs:
            updated_at = doc.metadata.get('updated_at', '1900-01-01')
            docs_with_time.append((doc, updated_at))

        docs_sorted = sorted(docs_with_time, key=lambda x: x[1], reverse=True)[:k]

        # Extract just the documents
        recent_docs = [doc for doc, _ in docs_sorted]

        logger.info(f"Re-ranked {search_k} documents by timestamp, returning top {k}")

        # Continue with normal RAG flow using recent_docs
        return self._generate_answer_from_docs(question, recent_docs, query_type="latest_projects")

    def _handle_general_qa(self, question: str, k: int = 5) -> Dict:
        """Handle general question-answering queries.

        Args:
            question: The user's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Handling general Q&A query with k={k}")

        # Retrieve relevant documents
        docs = self.vector_store_manager.similarity_search(question, k=k)

        return self._generate_answer_from_docs(question, docs, query_type="general_qa")

    def _generate_answer_from_docs(self, question: str, docs: List, query_type: str = "general_qa") -> Dict:
        """Generate answer from retrieved documents using Claude.

        Args:
            question: The user's question
            docs: Retrieved documents
            query_type: Type of query for metadata

        Returns:
            Dictionary with answer and source documents
        """
        # Format context from retrieved documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            updated_at = doc.metadata.get("updated_at", "")

            # Include timestamp in context for temporal queries
            if updated_at and query_type == "latest_projects":
                context_parts.append(f"[Source {i}: {source} (Updated: {updated_at[:10]})]\n{content}\n")
            else:
                context_parts.append(f"[Source {i}: {source}]\n{content}\n")

        context = "\n".join(context_parts)

        # Create the user message with context and question
        user_message = f"""Context from BHFDSC repositories:

{context}

Question: {question}

Answer: Let me help you with that based on the BHFDSC repositories."""

        # Call Claude API
        try:
            message = self.client.messages.create(
                model=Config.ANTHROPIC_MODEL,
                max_tokens=20000,
                temperature=0.0,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            answer = message.content[0].text

            # Format response
            response = {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "url": doc.metadata.get("url", ""),
                        "repo": doc.metadata.get("repo", ""),
                        "updated_at": doc.metadata.get("updated_at", ""),
                    }
                    for doc in docs
                ],
                "query_type": query_type
            }

            return response

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}", exc_info=True)
            raise

    def get_relevant_repos(self, question: str, k: int = 10, include_stats: bool = True) -> List[Dict]:
        """Get relevant repositories with usage statistics.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            include_stats: Whether to include statistics about mentions

        Returns:
            List of repository dictionaries with statistics
        """
        # Search more documents to get better repo coverage
        search_k = k * 2
        docs = self.vector_store_manager.similarity_search(question, k=search_k)

        # Aggregate by repository with statistics
        repos = {}
        for doc in docs:
            repo_name = doc.metadata.get("repo", "")
            if not repo_name:
                continue

            if repo_name not in repos:
                url = doc.metadata.get("url", "")
                # Remove /blob/ path from URL to get repo URL
                if "/blob/" in url:
                    url = url.split("/blob/")[0]

                repos[repo_name] = {
                    "name": repo_name,
                    "url": url,
                    "mention_count": 0,
                    "file_paths": [],
                    "latest_update": doc.metadata.get("updated_at", ""),
                    "relevance_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }

            # Update statistics
            repos[repo_name]["mention_count"] += 1

            file_path = doc.metadata.get("path", "")
            if file_path and file_path not in repos[repo_name]["file_paths"]:
                repos[repo_name]["file_paths"].append(file_path)

            # Keep the most recent update date
            doc_updated = doc.metadata.get("updated_at", "")
            if doc_updated > repos[repo_name]["latest_update"]:
                repos[repo_name]["latest_update"] = doc_updated

        # Sort by mention count (relevance) and latest update
        repo_list = sorted(
            repos.values(),
            key=lambda x: (x["mention_count"], x["latest_update"]),
            reverse=True
        )[:k]

        logger.info(f"Found {len(repo_list)} unique repositories from {search_k} documents")

        return repo_list
