"""Streamlit app for BHFDSC Q&A system."""

import streamlit as st
import logging
import os
from vector_store import get_vector_store
from qa_chain import QASystem
from config import Config
from github_indexer import GitHubIndexer
from hybrid_retriever import HybridRetriever

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


def check_bm25_index_exists():
    """Check if BM25 hybrid index exists in either .cache or data_index."""
    bm25_cache_file = os.path.join(".cache", "bm25_index.pkl")
    bm25_data_index_file = os.path.join("data_index", "bm25_index.pkl")
    return os.path.exists(bm25_cache_file) or os.path.exists(bm25_data_index_file)


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

        # Use hybrid retriever if enabled and BM25 index exists
        retriever = None
        if Config.USE_HYBRID_SEARCH:
            if check_bm25_index_exists():
                logger.info("Loading hybrid retriever with BM25...")
                try:
                    hybrid_retriever = HybridRetriever(vector_store)
                    # Load BM25 index from cache
                    hybrid_retriever.build_bm25_index([], force_rebuild=False)
                    retriever = hybrid_retriever
                    logger.info("✓ Hybrid retriever loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load hybrid retriever, using vector-only: {e}")
                    retriever = vector_store
            else:
                logger.warning("Hybrid search enabled but BM25 index not found. Using vector-only search.")
                logger.warning("Build the BM25 index in Setup page to enable hybrid search.")
                retriever = vector_store
        else:
            logger.info("Hybrid search disabled, using vector-only search")
            retriever = vector_store

        qa_system = QASystem(vector_store, retriever=retriever)
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
    """Render the admin/setup page - simplified to only Pinecone index management."""
    st.title("⚙️ Pinecone Vector Store Setup")

    st.info(
        """
        **Pinecone Cloud Vector Database**

        Your vector database is stored in Pinecone cloud and persists across app restarts.
        Index repositories once to enable the Q&A system.

        **Note**: BM25 index is managed in the Q&A tab.
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
Embedding Model: {Config.EMBEDDING_MODEL}
Chunk Size: {Config.CHUNK_SIZE}
Chunk Overlap: {Config.CHUNK_OVERLAP}
    """, language="text")


def render_qa_page():
    """Render the Q&A page with hybrid search management."""
    st.title("❤️ BHF Data Science Centre Repository Q&A")
    st.markdown(
        """
        Ask questions about the [BHFDSC GitHub organization](https://github.com/BHFDSC) repositories.
        This system uses AI to search through repositories focused on cardiovascular health research
        and COVID-19 impacts.
        """
    )

    # Hybrid Search Management Section (Collapsible)
    with st.expander("🔍 Hybrid Search Configuration", expanded=False):
        st.markdown(
            """
            **Hybrid search combines BM25 keyword matching with vector semantic search for better results.**

            - **BM25**: Excels at exact term matching (function names, identifiers)
            - **Vector Search**: Understands meaning and context
            - **Hybrid**: Combines both for optimal retrieval
            """
        )

        bm25_exists = check_bm25_index_exists()
        db_exists = check_vector_store_exists()

        col_bm25_1, col_bm25_2 = st.columns(2)

        with col_bm25_1:
            st.subheader("Status")
            if Config.USE_HYBRID_SEARCH:
                if bm25_exists:
                    st.success("✅ Hybrid search is active")
                    st.metric("BM25 Weight", f"{Config.BM25_WEIGHT:.0%}")
                    st.metric("Vector Weight", f"{1-Config.BM25_WEIGHT:.0%}")
                    st.metric("Adaptive Weights", "Enabled" if Config.USE_ADAPTIVE_WEIGHTS else "Disabled")
                else:
                    st.warning("⚠️ Hybrid search enabled but BM25 index not built")
                    st.info("Build the BM25 index below to enable hybrid search.")
            else:
                st.info("ℹ️ Hybrid search is disabled")
                st.caption("Using vector-only search")

        with col_bm25_2:
            st.subheader("Actions")
            if not db_exists:
                st.warning("⚠️ Vector store must be indexed first")
                st.info("Go to Setup tab to index repositories")
            else:
                if st.button("🔍 Build BM25 Index", type="primary" if not bm25_exists else "secondary", use_container_width=True, key="qa_build_bm25"):
                    with st.spinner("Building BM25 index... This may take a few minutes."):
                        try:
                            from build_hybrid_index import main as build_hybrid_main
                            import sys
                            from io import StringIO

                            old_stdout = sys.stdout
                            sys.stdout = StringIO()

                            try:
                                build_hybrid_main(force_rebuild=True)
                                output = sys.stdout.getvalue()
                            finally:
                                sys.stdout = old_stdout

                            st.success("✅ BM25 index built successfully!")
                            st.cache_resource.clear()

                            with st.expander("📋 Build Log"):
                                st.text(output)

                        except Exception as e:
                            st.error(f"Error building BM25 index: {e}")
                            logger.error(f"Error building BM25 index: {e}", exc_info=True)

                if bm25_exists:
                    if st.button("🗑️ Clear BM25 Cache", type="secondary", use_container_width=True, key="qa_clear_bm25"):
                        try:
                            retriever = HybridRetriever(get_vector_store())
                            retriever.clear_cache()
                            st.success("BM25 cache cleared.")
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing BM25 cache: {e}")

        st.divider()
        st.caption("""
        **Storage**: BM25 index is cached locally in `.cache/bm25_index.pkl`.
        For committed storage, copy to `data_index/bm25_index.pkl` and commit to Git.
        The app automatically checks both locations.
        """)

    st.divider()

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

        st.divider()

        # Retrieval method indicator
        st.header("Retrieval Method")
        bm25_exists = check_bm25_index_exists()

        if Config.USE_HYBRID_SEARCH and bm25_exists:
            st.success("🔍 **Hybrid Search Active**")
            st.caption("Using BM25 + Vector search")
            with st.expander("ℹ️ Details"):
                st.write(f"**BM25 Weight:** {Config.BM25_WEIGHT:.0%}")
                st.write(f"**Vector Weight:** {1-Config.BM25_WEIGHT:.0%}")
                if Config.USE_ADAPTIVE_WEIGHTS:
                    st.write("**Adaptive weights:** Enabled")
                    st.caption("Weights adjust based on query type")
        elif Config.USE_HYBRID_SEARCH and not bm25_exists:
            st.warning("⚠️ **Vector Search Only**")
            st.caption("BM25 index not built")
            st.info("Build BM25 index using the configuration panel above")
        else:
            st.info("ℹ️ **Vector Search Only**")
            st.caption("Hybrid search disabled")

        st.divider()

        st.header("Settings")
        k_docs = st.slider(
            "Number of source documents",
            min_value=3,
            max_value=20,
            value=20,
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
        'What is the standard pipeline project?',
        'what is in the health data science team documentation?',
        'What is the hds_curated_asset for patient demograhics, what is the methodology behind it?',
        'Which projects are investigating the impact of diabetes on covid-19?,'
        'What projects have defined an MI phenotype?',
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

                        # Show search metadata if available (from hybrid search)
                        if 'search_metadata' in source:
                            sm = source['search_metadata']
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("BM25 Score", f"{sm.get('bm25_score', 0):.3f}")
                            with cols[1]:
                                st.metric("Vector Score", f"{sm.get('vector_score', 0):.3f}")
                            with cols[2]:
                                st.metric("Combined", f"{sm.get('combined_score', 0):.3f}")

                            if i == 1 and sm.get('adaptive_weights_used'):
                                st.caption(f"⚙️ Adaptive weights: BM25={sm.get('bm25_weight_used', 0):.0%}, Vector={sm.get('vector_weight_used', 0):.0%}")

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

                                # Show search metadata if available (from hybrid search)
                                if 'search_metadata' in source:
                                    sm = source['search_metadata']
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric("BM25 Score", f"{sm.get('bm25_score', 0):.3f}")
                                    with cols[1]:
                                        st.metric("Vector Score", f"{sm.get('vector_score', 0):.3f}")
                                    with cols[2]:
                                        st.metric("Combined", f"{sm.get('combined_score', 0):.3f}")

                                    if i == 1 and sm.get('adaptive_weights_used'):
                                        st.caption(f"⚙️ Adaptive weights: BM25={sm.get('bm25_weight_used', 0):.0%}, Vector={sm.get('vector_weight_used', 0):.0%}")

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


def render_documentation_page():
    """Render the comprehensive documentation page."""
    st.title("📚 Project Documentation")

    st.markdown("""
    This page provides a comprehensive overview of how the BHFDSC Repository Q&A System works,
    the technologies used, and potential future enhancements.
    """)

    # Create tabs for different documentation sections
    doc_tabs = st.tabs([
        "🏗️ Architecture",
        "🔧 How It Works",
        "🛠️ Technologies",
        "🚀 Future Enhancements"
    ])

    # Tab 1: Architecture
    with doc_tabs[0]:
        st.header("System Architecture")

        st.markdown("""
        ### Overview

        This is a **hybrid RAG (Retrieval-Augmented Generation) system** designed specifically for code repositories. It provides:

        **Conversational Q&A**: Ask natural language questions about repositories and get AI-generated answers based on actual code and documentation using hybrid search combining BM25 keyword matching and vector semantic search.
        """)

        st.subheader("High-Level Architecture")
        st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (Streamlit Web App)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │   Q&A System   │
                  │  (RAG-based)   │
                  └────────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Hybrid Retriever│
                  │  BM25 + Vector  │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
    ┌──────────────┐  ┌─────────┐  ┌──────────────┐
    │   BM25 Index │  │ Pinecone│  │  Anthropic   │
    │  (keyword)   │  │ (vector)│  │Claude (LLM)  │
    └──────────────┘  └─────────┘  └──────────────┘
        """, language="text")

        st.subheader("Data Flow")

        st.markdown("""
        #### 1. Indexing Phase (One-time setup)

        ```
        GitHub Repos → Fetch Files → Process Documents → Create Embeddings → Pinecone
                                                              ↓
                                                         Build BM25 Index
                                                              ↓
        ```

        #### 2. Q&A Query Phase (Runtime)

        ```
        User Question
              ↓
        Hybrid Search
        ├─→ BM25: Keyword matching (exact terms, identifiers)
        └─→ Vector: Semantic search (meaning, context)
              ↓
        Combine & Rank Results (adaptive weights)
              ↓
        Top-K Documents (typically 20)
              ↓
        Format Context + Question → Anthropic Claude API
              ↓
        Generated Answer + Source Citations
        ```

