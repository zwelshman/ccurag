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
        """)

    # Tab 2: How It Works
    with doc_tabs[1]:
        st.header("How The System Works")

        st.subheader("1. RAG (Retrieval-Augmented Generation)")
        st.markdown("""
        **RAG** solves the problem of LLMs not knowing about your specific codebase by:

        1. **Retrieval**: Finding relevant documents from your repositories
        2. **Augmentation**: Adding those documents as context to the prompt
        3. **Generation**: Having the LLM generate an answer based on that context

        **Why RAG?**
        - LLMs have a knowledge cutoff and don't know about your private repos
        - RAG provides factual, source-cited answers from your actual code
        - Updates are easy: re-index repositories, no need to retrain models
        """)

        st.subheader("2. Hybrid Search (BM25 + Vector)")
        st.markdown("""
        This system uses **hybrid retrieval** combining two search methods:

        #### BM25 (Best Matching 25)
        - **Type**: Keyword-based lexical search
        - **Strengths**:
          - Exact term matching (function names like `get_user_data`)
          - Acronyms and identifiers (`HES`, `COVID_19`)
          - Code-specific tokens
        - **How it works**: TF-IDF scoring with document length normalization

        #### Vector Semantic Search
        - **Type**: Embedding-based similarity search
        - **Strengths**:
          - Understanding meaning and context
          - Paraphrased queries ("find authentication code" → matches "login functions")
          - Conceptual relationships
        - **How it works**:
          1. Convert text to 768-dimensional embeddings using `BAAI/llm-embedder`
          2. Store embeddings in Pinecone vector database
          3. Find most similar embeddings using cosine similarity

        #### Hybrid Combination
        ```python
        # Adaptive weight calculation based on query type
        if query_has_code_patterns(query):  # CamelCase, snake_case, ()
            weights = (0.7, 0.3)  # Favor BM25
        else:
            weights = (0.3, 0.7)  # Favor vector search

        # Combine scores
        final_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
        ```

        **Result**: Best of both worlds! Code queries get exact matches, conceptual queries get semantic understanding.
        """)

        st.subheader("3. Data Storage & Persistence")
        st.markdown("""
        #### Pinecone (Vector Database)
        - **Purpose**: Store document embeddings in the cloud
        - **Persistence**: Data remains across app restarts
        - **Advantages**:
          - Managed service (no infrastructure)
          - Fast similarity search (<50ms)
          - Scalable (handles millions of vectors)

        #### Local Cache Storage
        - `.cache/bm25_index.pkl`: BM25 index (50-100MB) - runtime cache
        - `data_index/bm25_index.pkl`: Committed BM25 index
        - **Persistence**: Cache files can be committed to Git via `data_index/` folder
        - **How it works**:
          1. Build indices locally - saved to `.cache/`
          2. For sharing: copy files to `data_index/` and commit to Git
          3. App checks `data_index/` first (committed), then `.cache/` (runtime)
          4. Subsequent runs load from available location instantly
        """)

    # Tab 3: Technologies
    with doc_tabs[2]:
        st.header("Technologies & Tools")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Core Technologies")
            st.markdown("""
            #### Anthropic Claude API
            - **Model**: `claude-sonnet-4-5`
            - **Purpose**: Generate natural language answers from context
            - **Features**:
              - 200K token context window (fits ~500 pages of code)
              - High accuracy for technical content
              - Strong instruction following

            #### Pinecone Vector Database
            - **Type**: Managed vector database
            - **Index**: Serverless, auto-scaling
            - **Dimensions**: 768 (matching embedding model)
            - **Metric**: Cosine similarity
            - **Region**: AWS us-east-1

            #### Sentence Transformers
            - **Model**: `BAAI/llm-embedder`
            - **Output**: 768-dimensional dense vectors
            - **Optimization**: Fine-tuned for retrieval tasks
            - **Speed**: ~100 docs/second on CPU

            #### BM25 (rank-bm25)
            - **Algorithm**: Okapi BM25
            - **Parameters**: k1=1.5, b=0.75 (default)
            - **Tokenization**: Custom code-aware tokenizer
              - CamelCase splitting
              - snake_case preservation
              - Stop word removal
            """)

        with col2:
            st.subheader("Supporting Technologies")
            st.markdown("""
            #### Streamlit
            - **Purpose**: Web interface
            - **Features**:
              - Reactive UI components
              - Session state management
              - Resource caching (`@st.cache_resource`)

            #### File Processing
            - **Text parsing**: Python, R, SQL, Markdown files
            - **Pattern matching**: Efficient content extraction

            #### GitHub API (PyGithub)
            - **Purpose**: Fetch repository contents
            - **Rate limits**: 60/hour (unauth), 5000/hour (with token)
            - **Indexed file types**:
              - Documentation: `.md`, `.rst`, `.txt`
              - Code: `.py`, `.r`, `.R`, `.sql`
              - Data: `.csv`, `.json`, `.yaml`
              - Notebooks: `.ipynb`

            #### Git Version Control
            - **Purpose**: Persist caches and share across deployments
            - **Storage**: Cache files committed to Git repository
            """)

        st.divider()

        st.subheader("Why These Choices?")
        st.markdown("""
        | Component | Why This Choice | Alternatives Considered |
        |-----------|----------------|------------------------|
        | **Claude** | Best accuracy for technical content, large context window | GPT-4, open-source LLMs |
        | **Pinecone** | Managed, no infrastructure, excellent docs | Weaviate, Qdrant, ChromaDB |
        | **BM25** | Fast, proven for code search, no training needed | Elasticsearch, custom scoring |
        | **Streamlit** | Rapid prototyping, Python-native, easy deployment | Gradio, Flask, FastAPI |
        | **llm-embedder** | Optimized for retrieval, good quality/speed balance | OpenAI embeddings, E5, BGE |
        """)

    # Tab 4: Future Enhancements
    with doc_tabs[3]:
        st.header("Future Enhancements")

        st.markdown("""
        Based on this proof-of-concept, here are potential improvements and extensions:
        """)

        st.subheader("🎯 Short-term Enhancements (1-3 months)")

        with st.expander("1. Enhanced Code Understanding"):
            st.markdown("""
            - **Call graph analysis**: Map function dependencies across repos
            - **Data flow tracking**: Trace how data moves through pipelines
            - **Import resolution**: Understand cross-repo dependencies
            - **Complexity metrics**: Calculate cyclomatic complexity, LOC
            - **Dead code detection**: Find unused functions/variables

            **Impact**: Deeper insights into code architecture and maintainability
            """)

        with st.expander("2. Improved Search Quality"):
            st.markdown("""
            - **Query expansion**: Add synonyms and related terms automatically
            - **Re-ranking**: Use cross-encoder for final result reordering
            - **Negative examples**: Learn from poorly ranked results
            - **Query classification**: Route different query types to optimal retrievers
            - **Result deduplication**: Remove near-duplicate code snippets

            **Impact**: Higher accuracy, fewer irrelevant results, better user experience
            """)

        with st.expander("3. User Experience Improvements"):
            st.markdown("""
            - **Conversation history**: Maintain context across multiple questions
            - **Follow-up suggestions**: Auto-generate relevant next questions
            - **Filters**: Filter by repo, date, file type, language
            - **Syntax highlighting**: Pretty code display in results
            - **Direct GitHub links**: Jump to exact line in GitHub UI
            - **Export results**: Download answers as PDF/Markdown

            **Impact**: More efficient workflows, better productivity
            """)

        st.subheader("🚀 Medium-term Enhancements (3-6 months)")

        with st.expander("4. Advanced Analytics"):
            st.markdown("""
            - **Temporal analysis**: Track how code evolves over time
            - **Contributor insights**: Who writes what type of code?
            - **Pattern detection**: Automatically find common patterns/anti-patterns
            - **Impact analysis**: Predict affected repos from proposed changes
            - **Test coverage mapping**: Link tests to source code

            **Impact**: Strategic insights for technical leadership
            """)

        with st.expander("5. Automated Code Quality"):
            st.markdown("""
            - **Style consistency checker**: Flag deviations from org standards
            - **Security scanning**: Detect potential vulnerabilities
            - **Dependency audit**: Track outdated or risky dependencies
            - **Documentation coverage**: Find undocumented code
            - **Best practice suggestions**: Recommend improvements based on org patterns

            **Impact**: Higher code quality, reduced technical debt
            """)

        with st.expander("6. Multi-modal Capabilities"):
            st.markdown("""
            - **Diagram understanding**: Parse architecture diagrams, flowcharts
            - **Screenshot analysis**: Extract info from images in README files
            - **Notebook support**: Deep analysis of Jupyter notebooks
            - **API spec parsing**: Index OpenAPI/Swagger specs
            - **Database schema visualization**: Auto-generate ERDs

            **Impact**: Handle more diverse documentation formats
            """)

        st.subheader("🌟 Long-term Vision (6-12 months)")

        with st.expander("7. AI-Powered Code Generation"):
            st.markdown("""
            - **Template generation**: Create boilerplate based on org patterns
            - **Code completion**: Suggest context-aware code snippets
            - **Refactoring suggestions**: Propose improvements with diffs
            - **Test generation**: Auto-create tests for uncovered code
            - **Documentation generation**: Auto-write docstrings, READMEs

            **Impact**: Accelerate development, ensure consistency
            """)

        with st.expander("8. Collaborative Features"):
            st.markdown("""
            - **Shared annotations**: Team members add notes to code
            - **Question history**: Searchable log of all Q&A sessions
            - **Expert routing**: Auto-tag questions for relevant team members
            - **Learning paths**: Curated sequences for onboarding
            - **Code review insights**: Surface relevant context during review

            **Impact**: Better knowledge sharing, faster onboarding
            """)

        with st.expander("9. Integration Ecosystem"):
            st.markdown("""
            - **VS Code extension**: Search codebase from IDE
            - **Slack/Teams bot**: Answer questions in chat
            - **GitHub Actions integration**: Auto-update indices on push
            - **JIRA/Linear integration**: Link tickets to related code
            - **CI/CD integration**: Validate changes against org patterns
            - **API endpoints**: Programmatic access to all features

            **Impact**: Seamless workflow integration
            """)

        with st.expander("10. Advanced RAG Techniques"):
            st.markdown("""
            - **Hierarchical retrieval**: Coarse-to-fine document selection
            - **Multi-hop reasoning**: Chain multiple lookups for complex questions
            - **Agentic workflows**: LLM decides which tools to use
            - **Active learning**: Learn from user feedback to improve ranking
            - **Personalization**: Adapt to individual user preferences

            **Impact**: Handle more complex queries, better accuracy
            """)

        st.divider()

        st.subheader("📊 Success Metrics")
        st.markdown("""
        To evaluate these enhancements, track:

        - **User Engagement**: Daily active users, questions per session
        - **Answer Quality**: Thumbs up/down ratings, citation accuracy
        - **Time Saved**: Onboarding time, time to find info
        - **Code Quality**: Reduction in bugs, style violations, duplicates
        - **Adoption**: % of team using tool, % of repos indexed
        """)


def main():
    """Main application with page navigation."""
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["Q&A", "Documentation", "Setup"],
            index=0 if check_vector_store_exists() else 2,
        )

    # Render the selected page
    if page == "Q&A":
        render_qa_page()
    elif page == "Documentation":
        render_documentation_page()
    elif page == "Setup":
        render_admin_page()


if __name__ == "__main__":
    main()
