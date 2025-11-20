"""Streamlit app for BHFDSC Q&A system."""

import streamlit as st
import logging
from pathlib import Path
import shutil
import zipfile
import io
from vector_store import VectorStoreManager
from qa_chain import QASystem
from config import Config
from github_indexer import GitHubIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BHFDSC Repository Q&A",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_vector_store_exists():
    """Check if vector store exists."""
    db_path = Path(Config.CHROMA_DB_DIR)
    return db_path.exists() and any(db_path.iterdir())


@st.cache_resource
def load_qa_system():
    """Load the QA system (cached)."""
    try:
        Config.validate()

        # Check if vector store exists
        if not check_vector_store_exists():
            return None

        vector_store_manager = VectorStoreManager()
        vector_store_manager.load_vectorstore()

        qa_system = QASystem(vector_store_manager)
        return qa_system

    except Exception as e:
        st.error(f"Error loading QA system: {e}")
        logger.error(f"Error loading QA system: {e}", exc_info=True)
        return None


def run_indexing():
    """Run the repository indexing process."""
    try:
        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Fetch and index all repositories
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Fetching repositories from BHFDSC organization...")
        documents = indexer.index_all_repos()
        progress_bar.progress(50)

        if not documents:
            st.error("No documents found to index!")
            return False

        status_text.text(f"Successfully fetched {len(documents)} documents. Creating vector store...")

        # Remove existing database if it exists
        db_path = Path(Config.CHROMA_DB_DIR)
        if db_path.exists():
            shutil.rmtree(Config.CHROMA_DB_DIR)

        # Create vector store
        vector_store_manager = VectorStoreManager()
        vector_store_manager.create_vectorstore(documents)

        progress_bar.progress(100)
        status_text.text("✅ Indexing completed successfully!")

        st.success(f"Indexed {len(documents)} documents from BHFDSC repositories!")
        return True

    except Exception as e:
        st.error(f"Error during indexing: {e}")
        logger.error(f"Error during indexing: {e}", exc_info=True)
        return False


def render_admin_page():
    """Render the admin/setup page."""
    st.title("⚙️ Setup & Administration")

    st.info(
        """
        **Note for Streamlit Cloud Users:**

        Streamlit Cloud doesn't have persistent storage, so the vector database will be lost when the app restarts.
        You'll need to re-index the repositories after each deployment or app restart.

        For production use, consider:
        - Using a cloud vector database (Pinecone, Weaviate, etc.)
        - Pre-building the index and uploading it
        - Using a persistent storage solution
        """
    )

    st.divider()

    # Check current status
    db_exists = check_vector_store_exists()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Current Status")
        if db_exists:
            st.success("✅ Vector store is ready")
            db_path = Path(Config.CHROMA_DB_DIR)
            try:
                # Try to get some stats
                from vector_store import VectorStoreManager
                vsm = VectorStoreManager()
                vsm.load_vectorstore()
                st.metric("Database Location", Config.CHROMA_DB_DIR)
            except Exception as e:
                st.warning(f"Vector store exists but couldn't load stats: {e}")
        else:
            st.warning("⚠️ Vector store not found")
            st.info("You need to index the repositories before using the Q&A system.")

    with col2:
        st.subheader("🔧 Actions")

        if st.button("🔄 Index/Re-index Repositories", type="primary", use_container_width=True):
            st.warning(
                """
                **This will:**
                - Fetch all repositories from BHFDSC GitHub organization
                - Download README files and code files
                - Create embeddings and store them in ChromaDB
                - Take approximately 10-30 minutes

                **Proceed?**
                """
            )
            if st.button("✅ Yes, Start Indexing", type="secondary"):
                with st.spinner("Indexing repositories... This may take 10-30 minutes."):
                    success = run_indexing()
                    if success:
                        st.balloons()
                        st.success("Indexing complete! You can now use the Q&A system.")
                        st.info("Go to the 'Q&A' tab in the sidebar to start asking questions.")
                        # Clear the cache to reload the QA system
                        st.cache_resource.clear()

        if db_exists:
            st.divider()
            if st.button("🗑️ Delete Vector Store", type="secondary", use_container_width=True):
                db_path = Path(Config.CHROMA_DB_DIR)
                shutil.rmtree(db_path)
                st.success("Vector store deleted.")
                st.cache_resource.clear()
                st.rerun()

    st.divider()
    st.subheader("📋 Configuration")

    st.code(f"""
GitHub Organization: {Config.GITHUB_ORG}
Anthropic Model: {Config.ANTHROPIC_MODEL}
Vector Store: {Config.CHROMA_DB_DIR}
Chunk Size: {Config.CHUNK_SIZE}
Chunk Overlap: {Config.CHUNK_OVERLAP}
Max Files per Repo: {Config.MAX_FILES_PER_REPO}
    """, language="text")


def render_qa_page():
    """Render the Q&A page."""
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
            - **ChromaDB** for vector storage
            - **Sentence Transformers** for embeddings

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

    # Check if vector store exists
    if not check_vector_store_exists():
        st.error("⚠️ Vector store not found!")
        st.info(
            """
            The repository index hasn't been created yet.

            Please go to the **Setup** page in the sidebar and click **Index Repositories** to get started.
            """
        )
        st.stop()

    # Load QA system
    qa_system = load_qa_system()

    if qa_system is None:
        st.error("Failed to load QA system. Please check the logs or re-index the repositories.")
        st.stop()

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


def main():
    """Main application with page navigation."""
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["Q&A", "Setup"],
            index=0 if check_vector_store_exists() else 1,
        )

    # Render the selected page
    if page == "Q&A":
        render_qa_page()
    elif page == "Setup":
        render_admin_page()


if __name__ == "__main__":
    main()
