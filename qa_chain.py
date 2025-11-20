"""Question-answering chain using LangChain and Anthropic."""

import logging
from typing import Dict, List
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import Config
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    """Question-answering system using RAG."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize QA system."""
        Config.validate()
        self.vector_store_manager = vector_store_manager
        self.llm = ChatAnthropic(
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            model=Config.ANTHROPIC_MODEL,
            temperature=0.0,
        )

        # Custom prompt template
        self.prompt_template = """You are an AI assistant helping researchers understand the BHF Data Science Centre (BHFDSC) GitHub organization. This organization focuses on cardiovascular health research, particularly studying COVID-19's impact on cardiovascular disease.

Use the following pieces of context from the organization's repositories to answer the question. If you don't know the answer based on the context, say so - don't make up information.

When referencing specific repositories or code, include the source information.

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the BHFDSC repositories.

"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def create_qa_chain(self):
        """Create the QA chain."""
        retriever = self.vector_store_manager.get_retriever(k=5)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

        return qa_chain

    def answer_question(self, question: str) -> Dict:
        """Answer a question using the QA chain."""
        logger.info(f"Processing question: {question}")

        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})

        # Format response
        response = {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "repo": doc.metadata.get("repo", ""),
                }
                for doc in result["source_documents"]
            ]
        }

        return response

    def get_relevant_repos(self, question: str, k: int = 10) -> List[Dict]:
        """Get relevant repositories for a question."""
        docs = self.vector_store_manager.similarity_search(question, k=k)

        # Extract unique repos
        repos = {}
        for doc in docs:
            repo_name = doc.metadata.get("repo", "")
            if repo_name and repo_name not in repos:
                repos[repo_name] = {
                    "name": repo_name,
                    "url": doc.metadata.get("url", "").split("/blob/")[0] if "/blob/" in doc.metadata.get("url", "") else doc.metadata.get("url", ""),
                    "relevance_snippet": doc.page_content[:200] + "..."
                }

        return list(repos.values())
