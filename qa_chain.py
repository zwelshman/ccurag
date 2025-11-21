"""Simple question-answering system using RAG."""

import logging
from typing import Dict
from anthropic import Anthropic
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''class QASystem:
    """Simple RAG-based question-answering system."""

    def __init__(self, vector_store):
        """Initialize QA system.

        Args:
            vector_store: Pinecone vector store for document retrieval
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
            raise'''

import logging
from typing import Dict, List, Optional
from anthropic import Anthropic
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    """Enhanced RAG-based question-answering system for BHFDSC GitHub organization."""

    def __init__(self, vector_store):
        """Initialize QA system with enhanced prompting.

        Args:
            vector_store: Pinecone vector store for document retrieval
        """
        Config.validate()
        self.vector_store = vector_store
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        self.system_prompt = """You are an AI assistant specializing in the BHF Data Science Centre (BHFDSC) GitHub organization. Your role is to help researchers understand the codebase, documentation, and best practices.

Guidelines:
- Answer based ONLY on the provided context documents
- Cite specific sources using [Source N] notation when making claims
- If information is unclear or contradictory across sources, acknowledge this
- If the context doesn't contain relevant information, clearly state this limitation
- Provide structured, actionable answers when appropriate
- For technical questions, include code examples if present in the context
- Distinguish between definitive answers and inferences based on partial information
- When discussing code or configurations, preserve exact naming conventions
- Be concise but comprehensive - avoid unnecessary verbosity"""

        # Confidence indicators for answer quality assessment
        self.confidence_indicators = {
            "high": ["according to", "[source", "specifically states", "clearly shows", 
                    "documentation states", "explicitly mentions"],
            "medium": ["suggests", "appears to", "likely", "based on", "indicates",
                      "seems to", "typically"],
            "low": ["might", "possibly", "unclear", "not found", "insufficient",
                   "cannot determine", "no information"]
        }

    def answer_question(self, 
                       question: str, 
                       num_docs: int = 5,
                       use_cot: Optional[bool] = None,
                       include_follow_ups: bool = True,
                       include_metadata: bool = True) -> Dict:
        """Answer a question using enhanced RAG with better prompting.

        Args:
            question: User's question
            num_docs: Number of documents to retrieve
            use_cot: Whether to use chain-of-thought reasoning (auto-detected if None)
            include_follow_ups: Whether to generate follow-up questions
            include_metadata: Whether to include answer metadata (confidence, type, etc.)

        Returns:
            Dictionary with answer, sources, metadata, and optional follow-ups
        """
        logger.info(f"Answering question: {question}")
        
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=num_docs)
            
            if not docs:
                return self._create_no_context_response(question)
            
            # Format context with enhanced structure
            context = self._format_context(docs)
            
            # Determine if chain-of-thought would help
            if use_cot is None:
                use_cot = self._should_use_cot(question)
            
            # Build the user message
            user_message = self._build_user_message(question, context, use_cot)
            
            # Generate answer with Claude
            answer = self._generate_answer(user_message)
            
            # Build response
            response = {
                "answer": answer,
                "source_documents": self._format_source_documents(docs)
            }
            
            # Add metadata if requested
            if include_metadata:
                metadata = self._extract_answer_metadata(answer, docs, question)
                response.update(metadata)
            
            # Generate follow-up questions if requested
            if include_follow_ups:
                response["suggested_questions"] = self._generate_follow_ups(
                    question, answer
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in QA system: {e}", exc_info=True)
            return self._create_error_response(e)

    def _format_context(self, docs: List) -> str:
        """Format retrieved documents with enhanced structure.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            repo = metadata.get("repo", "")
            last_modified = metadata.get("last_modified", "")
            doc_type = metadata.get("type", "document")
            
            # Build metadata header
            header_parts = [f"[Source {i}]"]
            if repo:
                header_parts.append(f"Repo: {repo}")
            if doc_type != "document":
                header_parts.append(f"Type: {doc_type}")
            if last_modified:
                header_parts.append(f"Updated: {last_modified}")
            
            header = " | ".join(header_parts)
            header += f"\nFile: {source}"
            
            context_parts.append(
                f"{header}\n{'='*60}\n{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)

    def _should_use_cot(self, question: str) -> bool:
        """Determine if chain-of-thought reasoning would be beneficial.
        
        Args:
            question: User's question
            
        Returns:
            Boolean indicating whether to use CoT
        """
        complex_indicators = [
            "how", "why", "explain", "compare", "difference between",
            "relationship", "architecture", "design", "implement",
            "best practice", "when should", "trade-off"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in complex_indicators)

    def _build_user_message(self, question: str, context: str, use_cot: bool) -> str:
        """Build the user message with appropriate prompting strategy.
        
        Args:
            question: User's question
            context: Formatted context from retrieved documents
            use_cot: Whether to use chain-of-thought reasoning
            
        Returns:
            Formatted user message for Claude
        """
        if use_cot:
            return f"""Based on the following context documents from the BHFDSC GitHub organization, please answer the question.

<context>
{context}
</context>

<question>
{question}
</question>

Please think through this step-by-step:

1. First, identify which sources are most relevant to the question
2. Extract the key information from each relevant source  
3. Note any gaps, contradictions, or limitations in the available information
4. Synthesize the information to form a comprehensive answer
5. Provide your final answer with appropriate [Source N] citations

Step-by-step analysis:"""
        
        else:
            return f"""Based on the following context documents from the BHFDSC GitHub organization, please answer the question.

<context>
{context}
</context>

<question>
{question}
</question>

Instructions:
- Base your answer ONLY on the provided context
- Cite relevant sources using [Source N] notation
- If the answer requires information from multiple sources, synthesize them coherently
- If the context lacks sufficient information, explicitly state what's missing
- For code-related questions, include relevant snippets if available in the context
- Be direct and concise while being comprehensive

Answer:"""

    def _generate_answer(self, user_message: str) -> str:
        """Generate answer using Claude API.
        
        Args:
            user_message: Formatted prompt for Claude
            
        Returns:
            Generated answer text
        """
        message = self.client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=2000,
            temperature=0.1,  # Slight randomness for natural language
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
            stop_sequences=["Human:", "Question:"]  # Prevent hallucination
        )
        
        return message.content[0].text.strip()

    def _extract_answer_metadata(self, answer: str, docs: List, question: str) -> Dict:
        """Extract metadata about the answer quality and characteristics.
        
        Args:
            answer: Generated answer text
            docs: Retrieved documents
            question: Original question
            
        Returns:
            Dictionary with answer metadata
        """
        answer_lower = answer.lower()
        
        # Determine confidence level
        confidence = "medium"  # default
        for level, indicators in self.confidence_indicators.items():
            if any(indicator in answer_lower for indicator in indicators):
                confidence = level
                break
        
        # Count source citations
        num_sources_cited = sum(
            1 for i in range(1, len(docs) + 1) 
            if f"[source {i}]" in answer_lower
        )
        
        # Classify answer type
        answer_type = self._classify_answer_type(answer)
        
        # Check if answer indicates missing information
        has_limitations = any(
            phrase in answer_lower 
            for phrase in ["not found", "insufficient", "not available", "unclear"]
        )
        
        return {
            "metadata": {
                "confidence": confidence,
                "answer_type": answer_type,
                "num_sources_retrieved": len(docs),
                "num_sources_cited": num_sources_cited,
                "has_limitations": has_limitations,
                "used_chain_of_thought": "step" in answer_lower and "analysis" in answer_lower,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _classify_answer_type(self, answer: str) -> str:
        """Classify the type of answer provided.
        
        Args:
            answer: Generated answer text
            
        Returns:
            Answer type classification
        """
        answer_lower = answer.lower()
        
        if "```" in answer:
            return "code_example"
        elif "not found" in answer_lower or "insufficient" in answer_lower:
            return "insufficient_context"
        elif any(word in answer_lower for word in ["step", "first", "then", "finally", "1.", "2."]):
            return "procedural"
        elif "?" in answer and ("yes" in answer_lower or "no" in answer_lower):
            return "direct_answer"
        elif any(word in answer_lower for word in ["because", "since", "therefore", "thus"]):
            return "explanatory"
        else:
            return "informational"

    def _generate_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            List of follow-up questions
        """
        try:
            # Truncate answer for context
            answer_summary = answer[:500] + "..." if len(answer) > 500 else answer
            
            follow_up_prompt = f"""Based on this Q&A about the BHFDSC GitHub organization, suggest 2-3 relevant follow-up questions that would help the user dive deeper.

Original Question: {question}

Answer Summary: {answer_summary}

Generate follow-up questions that:
1. Explore related technical details
2. Clarify implementation specifics
3. Connect to broader system architecture

Return only the questions, one per line, without numbering or bullets."""
            
            message = self.client.messages.create(
                model=Config.ANTHROPIC_MODEL,
                max_tokens=200,
                temperature=0.5,  # Higher temperature for variety
                messages=[{"role": "user", "content": follow_up_prompt}]
            )
            
            # Parse and clean follow-up questions
            follow_ups = [
                q.strip().lstrip("- ").lstrip("• ")
                for q in message.content[0].text.strip().split('\n')
                if q.strip() and not q.strip().isdigit()
            ]
            
            return follow_ups[:3]  # Limit Limit to 3 questions
            
        except Exception as e:
            logger.warning(f"Failed to generate follow-ups: {e}")
            return []

    def _format_source_documents(self, docs: List) -> List[Dict]:
        """Format source documents for response.
        
        Args:
            docs: Retrieved documents
            
        Returns:
            List of formatted source document dictionaries
        """
        formatted_docs = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            preview = content[:300] + "..." if len(content) > 300 else content
            
            formatted_doc = {
                "source_number": i,
                "content_preview": preview,
                "full_content": content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "repo": doc.metadata.get("repo", ""),
                    "type": doc.metadata.get("type", "document"),
                    "last_modified": doc.metadata.get("last_modified", "")
                }
            }
            
            # Add relevance score if available
            if hasattr(doc, "score"):
                formatted_doc["relevance_score"] = doc.score
            
            formatted_docs.append(formatted_doc)
        
        return formatted_docs

    def _create_no_context_response(self, question: str) -> Dict:
        """Create response when no documents are retrieved.
        
        Args:
            question: User's question
            
        Returns:
            Response dictionary indicating no context found
        """
        return {
            "answer": (
                "I couldn't find any relevant documentation in the BHFDSC GitHub "
                "organization to answer your question. This might be because:\n\n"
                "1. The information doesn't exist in the indexed repositories\n"
                "2. The search terms didn't match the available documentation\n"
                "3. The topic might be covered under different terminology\n\n"
                "You might want to try rephrasing your question or checking the "
                "repository directly."
            ),
            "source_documents": [],
            "metadata": {
                "confidence": "none",
                "answer_type": "no_context",
                "num_sources_retrieved": 0,
                "num_sources_cited": 0,
                "has_limitations": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _create_error_response(self, error: Exception) -> Dict:
        """Create response for error cases.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error response dictionary
        """
        return {
            "answer": (
                "I encountered an error while processing your question. "
                "Please try again or rephrase your question."
            ),
            "error": str(error),
            "source_documents": [],
            "metadata": {
                "confidence": "none",
                "answer_type": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def batch_answer(self, questions: List[str], **kwargs) -> List[Dict]:
        """Answer multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            **kwargs: Additional arguments passed to answer_question
            
        Returns:
            List of response dictionaries
        """
        responses = []
        for question in questions:
            logger.info(f"Processing batch question {len(responses) + 1}/{len(questions)}")
            response = self.answer_question(question, **kwargs)
            responses.append(response)
        
        return responses
