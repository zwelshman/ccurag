"""Question-answering system using Anthropic SDK directly."""

import logging
from typing import Dict, List
from anthropic import Anthropic
from config import Config
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    """Question-answering system using RAG with Anthropic Claude."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize QA system."""
        Config.validate()
        self.vector_store_manager = vector_store_manager
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        # Custom prompt template
        self.system_prompt = """You are an AI assistant helping researchers understand the BHF Data Science Centre (BHFDSC) GitHub organization. This organization focuses on cardiovascular health research, particularly studying COVID-19's impact on cardiovascular disease.

Use the following pieces of context from the organization's repositories to answer the question. If you don't know the answer based on the context, say so - don't make up information.

When referencing specific repositories or code, include the source information."""

    def answer_question(self, question: str, k: int = 5) -> Dict:
        """Answer a question using RAG."""
        logger.info(f"Processing question: {question}")

        # Retrieve relevant documents
        docs = self.vector_store_manager.similarity_search(question, k=k)

        # Format context from retrieved documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
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
                    }
                    for doc in docs
                ]
            }

            return response

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}", exc_info=True)
            raise

    def get_relevant_repos(self, question: str, k: int = 10) -> List[Dict]:
        """Get relevant repositories for a question."""
        docs = self.vector_store_manager.similarity_search(question, k=k)

        # Extract unique repos
        repos = {}
        for doc in docs:
            repo_name = doc.metadata.get("repo", "")
            if repo_name and repo_name not in repos:
                url = doc.metadata.get("url", "")
                # Remove /blob/ path from URL to get repo URL
                if "/blob/" in url:
                    url = url.split("/blob/")[0]

                repos[repo_name] = {
                    "name": repo_name,
                    "url": url,
                    "relevance_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }

        return list(repos.values())
