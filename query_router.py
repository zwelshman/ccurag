"""Query routing and classification for optimized query handling."""

import logging
from enum import Enum
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be handled by the system."""

    REPOSITORY_LIST = "repository_list"  # Queries asking for list of repositories
    LATEST_PROJECTS = "latest_projects"   # Queries asking for latest/recent projects
    CODE_EXPLANATION = "code_explanation" # Queries about how code works
    GENERAL_QA = "general_qa"            # General question-answering


class QueryRouter:
    """Routes queries to appropriate handlers based on query intent."""

    def __init__(self):
        """Initialize query router."""
        self.repository_keywords = [
            "which repositories",
            "what repositories",
            "list repositories",
            "repos that use",
            "repositories that use",
            "projects that use",
            "which projects",
            "what projects",
            "repos with",
            "repositories with",
            "show me repositories",
            "show me repos",
            "find repositories",
            "find repos",
        ]

        self.temporal_keywords = [
            "latest",
            "recent",
            "newest",
            "most recent",
            "last",
            "current",
            "up-to-date",
            "updated",
        ]

        self.code_explanation_keywords = [
            "how does",
            "how do",
            "explain",
            "what does",
            "walk through",
            "describe how",
            "show me how",
        ]

    def classify_query(self, question: str) -> QueryType:
        """Classify query intent.

        Args:
            question: The user's question

        Returns:
            QueryType enum indicating the query type
        """
        question_lower = question.lower()

        # Check for repository listing queries
        is_repo_query = any(keyword in question_lower for keyword in self.repository_keywords)

        # Check for temporal queries
        is_temporal_query = any(keyword in question_lower for keyword in self.temporal_keywords)

        # Check for code explanation queries
        is_code_query = any(keyword in question_lower for keyword in self.code_explanation_keywords)

        # Determine query type based on keywords
        if is_repo_query and is_temporal_query:
            # Combined query: "latest repos that use X"
            logger.info(f"Query classified as: LATEST_PROJECTS (combined repo + temporal)")
            return QueryType.LATEST_PROJECTS

        elif is_repo_query:
            # Pure repository list query: "which repos use X"
            logger.info(f"Query classified as: REPOSITORY_LIST")
            return QueryType.REPOSITORY_LIST

        elif is_temporal_query:
            # Pure temporal query: "latest projects with X"
            logger.info(f"Query classified as: LATEST_PROJECTS")
            return QueryType.LATEST_PROJECTS

        elif is_code_query:
            # Code explanation query
            logger.info(f"Query classified as: CODE_EXPLANATION")
            return QueryType.CODE_EXPLANATION

        else:
            # Default to general Q&A
            logger.info(f"Query classified as: GENERAL_QA (default)")
            return QueryType.GENERAL_QA

    def get_query_metadata(self, question: str, query_type: QueryType) -> Dict[str, Any]:
        """Extract additional metadata about the query.

        Args:
            question: The user's question
            query_type: The classified query type

        Returns:
            Dictionary with query metadata
        """
        metadata = {
            "query_type": query_type.value,
            "original_question": question,
        }

        # For repository queries, suggest higher k value
        if query_type in [QueryType.REPOSITORY_LIST, QueryType.LATEST_PROJECTS]:
            metadata["suggested_k"] = 10
        else:
            metadata["suggested_k"] = 5

        # For temporal queries, note that sorting is required
        if query_type == QueryType.LATEST_PROJECTS:
            metadata["requires_temporal_sort"] = True
        else:
            metadata["requires_temporal_sort"] = False

        return metadata
