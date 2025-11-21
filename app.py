"""Streamlit app for BHFDSC Q&A system."""

import streamlit as st
import logging
from pathlib import Path
import shutil
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
    backend = Config.VECTOR_STORE_BACKEND.lower()

    if backend == "pinecone":
        # Check if Pinecone index exists
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            existing_indexes = [index.name for index in pc.list_indexes()]
            return Config.PINECONE_INDEX_NAME in existing_indexes
        except Exception as e:
            logger.error(f"Error checking Pinecone index: {e}")
            return False
    elif backend == "chroma":
        # Check if ChromaDB directory exists
        db_path = Path(Config.CHROMA_DB_DIR)
        return db_path.exists() and any(db_path.iterdir())
    else:
        return False


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


def run_indexing(sample_size=None):
    """Run the repository indexing process with checkpoint/resume support.

    Args:
        sample_size: If provided, randomly sample this many repositories.
                    If None, index all repositories.
    """
    try:
        # Create progress indicators
        progress_bar = st.progress(0, text="Starting indexing process...")
        status_text = st.empty()
        details_expander = st.expander("📋 Detailed Progress", expanded=True)

        with details_expander:
            repo_status = st.empty()
            doc_status = st.empty()
            embedding_status = st.empty()

        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Check for existing checkpoint
        from pathlib import Path as CheckPath
        checkpoint_file = CheckPath(".checkpoint.json")
        if checkpoint_file.exists():
            status_text.info("🔄 **Resuming from previous session...**")
            with details_expander:
                repo_status.info("📋 Found checkpoint file - resuming from where we left off")

        # Phase 1 & 2 Combined: Fetch and index repositories
        status_text.info("🔍 **Phase 1/2:** Fetching repository list from BHFDSC...")
        progress_bar.progress(10, text="Fetching repository list...")

        with details_expander:
            repo_status.info("📋 Retrieving repository information...")

        status_text.caption("💾 Progress is automatically saved - you can safely interrupt and resume later")

        # Use the checkpoint-enabled indexing method (fetches repos internally)
        all_documents, changed_repos = indexer.index_all_repos(sample_size=sample_size, resume=True)

        # Get total repos from the checkpoint
        total_repos = indexer.checkpoint_manager.checkpoint_data.get('total_repos', 0)

        with details_expander:
            repo_status.success(f"✅ Processed {total_repos} repositories")

        # Update progress bar to completion of phase 2
        progress_bar.progress(50, text="Repository indexing complete")

        if not all_documents:
            st.error("No documents found to index!")
            return False

        with details_expander:
            doc_status.success(f"✅ Collected {len(all_documents)} documents from {total_repos} repositories")

        # Phase 2: Create vector store
        status_text.info(f"🧮 **Phase 2/2:** Creating vector store from {len(all_documents)} documents...")
        progress_bar.progress(50, text="Creating vector embeddings...")

        # Remove existing database if it exists
        db_path = Path(Config.CHROMA_DB_DIR)
        if db_path.exists():
            shutil.rmtree(Config.CHROMA_DB_DIR)

        # Create vector store with progress updates
        vector_store_manager = VectorStoreManager()

        # We'll need to enhance vector_store to show progress, for now show intermediate steps
        with details_expander:
            embedding_status.info("🔄 Splitting documents into chunks...")

        progress_bar.progress(60, text="Splitting documents into chunks...")

        # This will internally handle the chunking and embedding
        vector_store_manager.create_vectorstore(all_documents, progress_callback=lambda msg, pct: (
            progress_bar.progress(60 + int(pct * 0.35), text=msg),
            embedding_status.info(f"🔄 {msg}")
        ))

        progress_bar.progress(100, text="✅ Indexing complete!")
        status_text.success("✅ **Indexing completed successfully!**")

        with details_expander:
            embedding_status.success(f"✅ Vector store created successfully!")

        st.balloons()
        st.success(f"🎉 Successfully indexed {len(all_documents)} documents from {total_repos} BHFDSC repositories!")
        return True

    except KeyboardInterrupt:
        st.warning("⚠️ Indexing interrupted by user")
        st.info("💾 Progress has been saved. Click 'Index/Re-index Repositories' to resume.")
        return False
    except Exception as e:
        st.error(f"❌ Error during indexing: {e}")
        st.info("💾 Progress has been saved to .checkpoint.json. Click 'Index/Re-index Repositories' to resume from where you left off.")
        logger.error(f"Error during indexing: {e}", exc_info=True)
        return False


def render_admin_page():
    """Render the admin/setup page."""
    st.title("⚙️ Setup & Administration")

    # Show info based on backend
    if Config.VECTOR_STORE_BACKEND == "pinecone":
        st.success(
            """
            **Using Pinecone Cloud Vector Database**

            Your vector database is stored in Pinecone cloud and persists across app restarts.
            You only need to index repositories once, and the data will remain available.
            """
        )
    else:
        st.info(
            """
            **Note for Streamlit Cloud Users:**

            Streamlit Cloud doesn't have persistent storage, so the vector database will be lost when the app restarts.
            You'll need to re-index the repositories after each deployment or app restart.

            For production use, consider switching to Pinecone backend (set VECTOR_STORE_BACKEND=pinecone).
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
            try:
                # Try to get some stats
                from vector_store import VectorStoreManager
                vsm = VectorStoreManager()
                vsm.load_vectorstore()

                if Config.VECTOR_STORE_BACKEND == "pinecone":
                    st.metric("Backend", "Pinecone Cloud")
                    st.metric("Index Name", Config.PINECONE_INDEX_NAME)
                    try:
                        stats = vsm.get_stats()
                        vector_count = stats.get('total_vector_count', 0)
                        st.metric("Total Vectors", f"{vector_count:,}")
                    except:
                        pass
                else:
                    st.metric("Backend", "ChromaDB (Local)")
                    st.metric("Database Location", Config.CHROMA_DB_DIR)
            except Exception as e:
                st.warning(f"Vector store exists but couldn't load stats: {e}")
        else:
            st.warning("⚠️ Vector store not found")
            st.info("You need to index the repositories before using the Q&A system.")

    with col2:
        st.subheader("🔧 Actions")

        # Initialize session state for indexing confirmation
        if "confirm_indexing" not in st.session_state:
            st.session_state.confirm_indexing = False
        if "indexing_started" not in st.session_state:
            st.session_state.indexing_started = False
        if "sample_repos" not in st.session_state:
            st.session_state.sample_repos = False

        # Show initial button or confirmation
        if not st.session_state.confirm_indexing and not st.session_state.indexing_started:
            if st.button("🔄 Index/Re-index Repositories", type="primary", use_container_width=True):
                st.session_state.confirm_indexing = True
                st.rerun()

        if st.session_state.confirm_indexing and not st.session_state.indexing_started:
            # Add sampling option checkbox
            st.session_state.sample_repos = st.checkbox(
                "Sample 20 random repositories (faster for testing)",
                value=st.session_state.sample_repos,
                help="Enable this to index only 20 random repositories instead of all repositories. Useful for quick testing."
            )

            backend_name = "Pinecone" if Config.VECTOR_STORE_BACKEND == "pinecone" else "ChromaDB"

            if st.session_state.sample_repos:
                st.warning(
                    f"""
                    **This will:**
                    - Fetch 20 random repositories from BHFDSC GitHub organization
                    - Download README files and code files
                    - Create embeddings and store them in {backend_name}
                    - Take approximately 2-5 minutes

                    **Proceed?**
                    """
                )
            else:
                st.warning(
                    f"""
                    **This will:**
                    - Fetch all repositories from BHFDSC GitHub organization
                    - Download README files and code files
                    - Create embeddings and store them in {backend_name}
                    - Take approximately 10-30 minutes

                    **Proceed?**
                    """
                )

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, Start Indexing", type="primary", use_container_width=True):
                    st.session_state.indexing_started = True
                    st.session_state.confirm_indexing = False
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", type="secondary", use_container_width=True):
                    st.session_state.confirm_indexing = False
                    st.rerun()

        # Run indexing if started
        if st.session_state.indexing_started:
            sample_size = 20 if st.session_state.sample_repos else None
            success = run_indexing(sample_size=sample_size)

            # Reset states
            st.session_state.indexing_started = False
            st.session_state.confirm_indexing = False

            if success:
                st.success("Indexing complete! You can now use the Q&A system.")
                st.info("Go to the 'Q&A' tab in the sidebar to start asking questions.")
                # Clear the cache to reload the QA system
                st.cache_resource.clear()
                # Add a button to go to Q&A page
                if st.button("Go to Q&A Page", type="primary"):
                    st.rerun()

        if db_exists:
            st.divider()
            if st.button("🗑️ Delete Vector Store", type="secondary", use_container_width=True):
                try:
                    if Config.VECTOR_STORE_BACKEND == "pinecone":
                        from vector_store import VectorStoreManager
                        vsm = VectorStoreManager()
                        vsm.delete_vectorstore()
                        st.success("Pinecone index deleted.")
                    else:
                        db_path = Path(Config.CHROMA_DB_DIR)
                        shutil.rmtree(db_path)
                        st.success("ChromaDB vector store deleted.")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting vector store: {e}")

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

    # Example questions (shown first)
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
            st.markdown(f"- {example}")

    # Q&A Chat (shown below example questions)
    st.divider()
    st.subheader("💬 Ask Your Question")

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
