"""Streamlit app for BHFDSC Q&A system."""

import streamlit as st
import logging
from pathlib import Path
from vector_store import VectorStoreManager
from qa_chain import QASystem
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BHFDSC Repository Q&A",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_qa_system():
    """Load the QA system (cached)."""
    try:
        Config.validate()

        # Check if vector store exists
        db_path = Path(Config.CHROMA_DB_DIR)
        if not db_path.exists():
            st.error(
                f"Vector store not found at {Config.CHROMA_DB_DIR}. "
                "Please run 'python index_repos.py' first to index the repositories."
            )
            st.stop()

        vector_store_manager = VectorStoreManager()
        vector_store_manager.load_vectorstore()

        qa_system = QASystem(vector_store_manager)
        return qa_system

    except Exception as e:
        st.error(f"Error loading QA system: {e}")
        st.stop()


def main():
    """Main Streamlit app."""
    st.title("❤️ BHF Data Science Centre Repository Q&A")
    st.markdown(
        """
        Ask questions about the [BHFDSC GitHub organization](https://github.com/BHFDSC) repositories.
        This system uses AI to search through ~100 repositories focused on cardiovascular health research
        and COVID-19 impacts.
        """
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This application uses:
            - **Anthropic Claude** for AI responses
            - **LangChain** for RAG pipeline
            - **ChromaDB** for vector storage

            **Indexed Organization:** BHFDSC

            **Topics Covered:**
            - CVD-COVID-UK research
            - Cardiovascular disease analysis
            - Electronic health records
            - Data analysis pipelines
            """
        )

        st.header("Settings")
        k_docs = st.slider(
            "Number of source documents",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of relevant documents to retrieve"
        )

    # Load QA system
    qa_system = load_qa_system()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. [{source['source']}]({source['url']})**")
                        if source.get("repo"):
                            st.caption(f"Repository: {source['repo']}")
                        st.text(source['content'])
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about BHFDSC repositories..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Get response from QA system
        with st.chat_message("assistant"):
            with st.spinner("Searching repositories and generating answer..."):
                try:
                    result = qa_system.answer_question(question)

                    answer = result["answer"]
                    sources = result["source_documents"]

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. [{source['source']}]({source['url']})**")
                                if source.get("repo"):
                                    st.caption(f"Repository: {source['repo']}")
                                st.text(source['content'])
                                st.divider()

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error generating answer: {e}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)

    # Example questions
    st.divider()
    st.subheader("💡 Example Questions")

    example_questions = [
        "What COVID-19 and cardiovascular research projects are in this organization?",
        "Which repositories contain Python code for data analysis?",
        "What phenotyping algorithms are used in the CCU projects?",
        "Show me repositories that work with linked electronic health records",
        "What machine learning methods are used in these projects?",
    ]

    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                # Trigger the question
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


if __name__ == "__main__":
    main()
