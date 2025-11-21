"""Streamlit app for BHFDSC Q&A system."""

import streamlit as st
import logging
from vector_store import get_vector_store
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
    """Check if Pinecone vector store exists."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        existing_indexes = [index.name for index in pc.list_indexes()]
        return Config.PINECONE_INDEX_NAME in existing_indexes
    except Exception as e:
        logger.error(f"Error checking Pinecone index: {e}")
        return False


@st.cache_resource
def load_qa_system():
    """Load the QA system (cached)."""
    try:
        Config.validate()

        # Check if vector store exists
        if not check_vector_store_exists():
            return None

        vector_store = get_vector_store()
        vector_store.load_vectorstore()

        qa_system = QASystem(vector_store)
        return qa_system

    except Exception as e:
        st.error(f"Error loading QA system: {e}")
        logger.error(f"Error loading QA system: {e}", exc_info=True)
        return None


def run_indexing():
    """Run the repository indexing process."""
    try:
        # Create progress indicators
        progress_bar = st.progress(0, text="Starting indexing process...")
        status_text = st.empty()
        details_expander = st.expander("📋 Detailed Progress", expanded=True)

        with details_expander:
            repo_status = st.empty()
            doc_status = st.empty()

        # Initialize GitHub indexer
        indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

        # Fetch repositories
        status_text.info("🔍 Fetching repository list from BHFDSC...")
        repos = indexer.get_all_repos()
        total_repos = len(repos)

        with details_expander:
            repo_status.success(f"✅ Found {total_repos} repositories")

        progress_bar.progress(10, text=f"Found {total_repos} repositories")

        # Index repositories
        status_text.info(f"📥 Indexing and uploading repositories to Pinecone...")

        # Initialize vector store
        vector_store = get_vector_store()

        with details_expander:
            doc_status.info(f"🔄 Processing repositories sequentially...")

        # Index all repos
        total_documents = indexer.index_all_repos(vector_store=vector_store)

        if total_documents == 0:
            st.error("No documents found to index!")
            return False

        progress_bar.progress(100, text="✅ Indexing complete!")
        status_text.success("✅ **Indexing completed successfully!**")

        with details_expander:
            doc_status.success(f"✅ Processed {total_documents} documents from {total_repos} repositories")

        st.balloons()
        st.success(f"🎉 Successfully indexed {total_documents} documents from {total_repos} BHFDSC repositories!")
        return True

    except Exception as e:
        st.error(f"❌ Error during indexing: {e}")
        logger.error(f"Error during indexing: {e}", exc_info=True)
        return False


def render_admin_page():
    """Render the admin/setup page."""
    st.title("⚙️ Setup & Administration")

    st.success(
        """
        **Using Pinecone Cloud Vector Database**

        Your vector database is stored in Pinecone cloud and persists across app restarts.
        You only need to index repositories once, and the data will remain available.
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
                vsm = get_vector_store()
                vsm.load_vectorstore()

                st.metric("Backend", "Pinecone Cloud")
                st.metric("Index Name", Config.PINECONE_INDEX_NAME)
                try:
                    stats = vsm.get_stats()
                    vector_count = stats.get('total_vector_count', 0)
                    st.metric("Total Vectors", f"{vector_count:,}")
                except:
                    pass
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

        # Show initial button or confirmation
        if not st.session_state.confirm_indexing and not st.session_state.indexing_started:
            if st.button("🔄 Index Repositories", type="primary", use_container_width=True):
                st.session_state.confirm_indexing = True
                st.rerun()

        if st.session_state.confirm_indexing and not st.session_state.indexing_started:
            st.warning(
                """
                **This will:**
                - Fetch all repositories from BHFDSC GitHub organization
                - Download README files and code files
                - Create embeddings and store them in Pinecone
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
            success = run_indexing()

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
                    vsm = get_vector_store()
                    vsm.delete_vectorstore()
                    st.success("Pinecone index deleted.")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting vector store: {e}")

    st.divider()
    st.subheader("📋 Configuration")

    st.code(f"""
GitHub Organization: {Config.GITHUB_ORG}
Anthropic Model: {Config.ANTHROPIC_MODEL}
Vector Store: Pinecone ({Config.PINECONE_INDEX_NAME})
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
        This system uses AI to search through repositories focused on cardiovascular health research
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
            - **Pinecone** for vector storage
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
                        metadata = source.get('metadata', {})
                        st.markdown(f"**{i}. [{metadata.get('source', 'Unknown')}]({metadata.get('url', '')})**")
                        if metadata.get("repo"):
                            st.caption(f"Repository: {metadata['repo']}")
                        st.text(source.get('content_preview', source.get('full_content', '')))
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
                    result = qa_system.answer_question(question, num_docs=k_docs)

                    answer = result["answer"]
                    sources = result["source_documents"]

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                metadata = source.get('metadata', {})
                                st.markdown(f"**{i}. [{metadata.get('source', 'Unknown')}]({metadata.get('url', '')})**")
                                if metadata.get("repo"):
                                    st.caption(f"Repository: {metadata['repo']}")
                                st.text(source.get('content_preview', source.get('full_content', '')))
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
