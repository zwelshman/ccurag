"""Simple question-answering system using RAG."""

import logging
from typing import Dict, List
from anthropic import Anthropic
from config import Config
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    """Simple RAG-based question-answering system."""

    def __init__(self, vector_store: VectorStoreManager):
        """Initialize QA system.

        Args:
            vector_store: Vector store for document retrieval
        """
        Config.validate()
        self.vector_store = vector_store
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        self.system_prompt = """You are an AI assistant helping researchers understand the BHF Data Science Centre (BHFDSC) GitHub organization.

Use the context provided to answer the question. If you don't know the answer based on the context, say so - don't make up information."""

    def answer_question(self, question: str, num_docs: int = 5) -> Dict:
        """Answer a question using RAG.

        Args:
            question: User's question
            num_docs: Number of documents to retrieve

        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Answering question: {question}")

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=num_docs)

        # Build context from documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")

        context = "\n".join(context_parts)

        # Generate answer using Claude
        user_message = f"""Context:

{context}

Question: {question}

Answer:"""

        try:
            message = self.client.messages.create(
                model=Config.ANTHROPIC_MODEL,
                max_tokens=2000,
                temperature=0.0,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )

            answer = message.content[0].text

            # Format response
            return {
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

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}", exc_info=True)
            raise
