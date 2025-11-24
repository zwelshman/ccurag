"""Streamlit app for BHFDSC Q&A system with Code Intelligence."""

import streamlit as st
import logging
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vector_store import get_vector_store
from qa_chain import QASystem
from config import Config
from github_indexer import GitHubIndexer
from code_analyzer import CodeAnalyzer
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


def check_metadata_exists():
    """Check if code metadata cache exists."""
    cache_file = os.path.join(".cache", "code_metadata.json")
    return os.path.exists(cache_file)


def check_bm25_index_exists():
    """Check if BM25 hybrid index exists."""
    bm25_cache_file = os.path.join(".cache", "bm25_index.pkl")
    return os.path.exists(bm25_cache_file)


@st.cache_resource
def load_code_analyzer():
    """Load the code analyzer (cached)."""
    try:
        analyzer = CodeAnalyzer()
        stats = analyzer.get_stats()

        if stats['total_files'] == 0:
            return None

        return analyzer
    except Exception as e:
        logger.error(f"Error loading code analyzer: {e}", exc_info=True)
        return None


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

        **Note**: BM25 hybrid search and code metadata are managed within their respective tabs (Q&A and Code Intelligence).
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
Max Files per Repo: {Config.MAX_FILES_PER_REPO}
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
        **Storage**: BM25 index is cached locally or in cloud storage (Google Drive/S3) for fast loading.
        Configure cloud storage credentials in Streamlit secrets or .env file for persistence across sessions.
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


def render_code_intelligence_page():
    """Render the Code Intelligence page with metadata management."""
    st.title("🧠 Code Intelligence Dashboard")
    st.markdown(
        """
        Explore organizational intelligence about the BHFDSC codebase through static code analysis.
        Track table usage, function dependencies, and discover similar projects.
        """
    )

    # Code Metadata Management Section (Collapsible)
    with st.expander("⚙️ Code Metadata Configuration", expanded=False):
        st.markdown(
            """
            **Code Intelligence analyzes Python, R, and SQL files to extract structured metadata.**

            - **Table Usage**: Track HDS curated assets usage across projects
            - **Function Analysis**: Identify standardized function usage
            - **Module Imports**: See which libraries are used where
            - **Semantic Clustering**: Find similar algorithms and code patterns
            """
        )

        metadata_exists = check_metadata_exists()

        col_meta_1, col_meta_2 = st.columns(2)

        with col_meta_1:
            st.subheader("Status")
            if metadata_exists:
                st.success("✅ Metadata index exists")
                try:
                    analyzer = load_code_analyzer()
                    if analyzer:
                        stats = analyzer.get_stats()
                        st.metric("Files Analyzed", stats['total_files'])
                        st.metric("Repos Analyzed", stats['total_repos'])
                        st.metric("Tables Found", stats['total_unique_tables'])
                        st.metric("Functions Found", stats['total_unique_functions'])
                except:
                    pass
            else:
                st.warning("⚠️ Metadata index not found")
                st.info("Build the metadata index below to enable Code Intelligence features.")

        with col_meta_2:
            st.subheader("Actions")
            if st.button("🧠 Build Metadata Index", type="primary" if not metadata_exists else "secondary", use_container_width=True, key="ci_build_metadata"):
                with st.spinner("Building metadata index... This may take 10-30 minutes."):
                    try:
                        from build_metadata_index import main as build_metadata_main
                        import sys
                        from io import StringIO

                        old_stdout = sys.stdout
                        sys.stdout = StringIO()

                        try:
                            build_metadata_main(force_rebuild=True)
                            output = sys.stdout.getvalue()
                        finally:
                            sys.stdout = old_stdout

                        st.success("✅ Metadata index built successfully!")
                        st.cache_resource.clear()

                        with st.expander("📋 Build Log"):
                            st.text(output)

                    except Exception as e:
                        st.error(f"Error building metadata index: {e}")
                        logger.error(f"Error building metadata index: {e}", exc_info=True)

            if metadata_exists:
                if st.button("🗑️ Clear Metadata Cache", type="secondary", use_container_width=True, key="ci_clear_metadata"):
                    try:
                        analyzer = CodeAnalyzer()
                        analyzer.clear_cache()
                        st.success("Metadata cache cleared.")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")

        st.divider()
        st.caption("""
        **Storage**: Metadata is cached locally or in cloud storage (Google Drive/S3) for fast loading.
        Configure cloud storage credentials in Streamlit secrets or .env file for persistence across sessions.
        """)

    st.divider()

    # Check if metadata exists
    if not check_metadata_exists():
        st.error("⚠️ Metadata index not found!")
        st.info(
            """
            The code metadata hasn't been built yet.

            Use the **Code Metadata Configuration** panel above to build the metadata index.
            This will parse all Python, R, and SQL files to extract structured metadata.
            """
        )
        st.stop()

    # Load analyzer
    analyzer = load_code_analyzer()

    if analyzer is None:
        st.error("Failed to load code analyzer. Please rebuild the metadata index using the panel above.")
        st.stop()

    # Get statistics
    stats = analyzer.get_stats()

    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📁 Table Usage",
        "⚙️ Function Usage",
        "📦 Module Usage",
        "🔍 Similar Projects",
        "🔗 Cross-Analysis"
    ])

    # TAB 1: Dashboard
    with tab1:
        st.header("Overview Statistics")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Repositories", stats['total_repos'])
        with col2:
            st.metric("Files Analyzed", stats['total_files'])
        with col3:
            st.metric("Unique Tables", stats['total_unique_tables'])
        with col4:
            st.metric("Unique Functions", stats['total_unique_functions'])

        st.divider()

        # Tracked tables with usage
        st.subheader("HDS Curated Assets Usage")

        if stats['tracked_tables_found']:
            tables_df = pd.DataFrame([
                {"Table": k, "Repos Using": v}
                for k, v in stats['tracked_tables_found'].items()
            ]).sort_values('Repos Using', ascending=False)

            fig2 = px.bar(
                tables_df.head(10),
                y='Table',
                x='Repos Using',
                orientation='h',
                title="Top 10 Most Used HDS Tables",
                color='Repos Using',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("📋 View All Tracked Tables"):
                st.dataframe(tables_df, use_container_width=True)
        else:
            st.info("No HDS curated assets found in the analyzed code.")

    # TAB 2: Table Usage Explorer
    with tab2:
        st.header("Table Dependency Tracker")
        st.markdown("Find which repositories use specific HDS curated asset tables.")

        # Group tables by category
        demographics_tables = [
            "hds_curated_assets__demographics",
            "hds_curated_assets__date_of_birth_individual",
            "hds_curated_assets__date_of_birth_multisource",
            "hds_curated_assets__ethnicity_individual",
            "hds_curated_assets__ethnicity_multisource",
            "hds_curated_assets__sex_individual",
            "hds_curated_assets__sex_multisource",
            "hds_curated_assets__lsoa_individual",
            "hds_curated_assets__lsoa_multisource",
        ]

        covid_tables = ["hds_curated_assets__covid_positive"]

        deaths_tables = [
            "hds_curated_assets__deaths_single",
            "hds_curated_assets__deaths_cause_of_death",
        ]

        hes_tables = [
            "hds_curated_assets__hes_apc_cips_cips",
            "hds_curated_assets__hes_apc_cips_episodes",
            "hds_curated_assets__hes_apc_cips_provider_spells",
            "hds_curated_assets__hes_apc_diagnosis",
            "hds_curated_assets__hes_apc_procedure",
            "hds_curated_assets__hes_apc_provider_spells",
        ]

        # Category selector
        category = st.radio(
            "Select Category",
            ["Demographics", "COVID", "Deaths", "HES Assets", "All Tables"],
            horizontal=True
        )

        # Get table list based on category
        if category == "Demographics":
            table_list = demographics_tables
        elif category == "COVID":
            table_list = covid_tables
        elif category == "Deaths":
            table_list = deaths_tables
        elif category == "HES Assets":
            table_list = hes_tables
        else:
            table_list = sorted(analyzer.TRACKED_TABLES)

        # Table selector
        selected_table = st.selectbox("Select Table", table_list)

        if selected_table:
            usage = analyzer.get_table_usage(selected_table)

            if usage['total_repos'] == 0:
                st.warning(f"❌ No usage found for `{selected_table}`")
            else:
                st.success(f"✅ **{selected_table}** is used in **{usage['total_repos']}** repositories")

                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Repositories", usage['total_repos'])
                with col2:
                    st.metric("Total Files", usage['total_files'])

                st.divider()

                # Repositories using this table
                st.subheader("Repositories")
                for repo in usage['repos']:
                    st.markdown(f"- `{repo}`")

                st.divider()

                # Files by type
                st.subheader("Files by Type")
                for file_type, files in usage['files_by_type'].items():
                    with st.expander(f"{file_type.capitalize()} ({len(files)} files)"):
                        for file_info in files:
                            st.markdown(f"**{file_info['repo']}**")
                            st.code(file_info['file'], language="text")

    # TAB 3: Function Usage Tracker
    with tab3:
        st.header("Function Usage Tracker")
        st.markdown("Track which repositories use HDS functions and standardized code.")

        # Search box
        function_pattern = st.text_input(
            "Search for function pattern",
            value="hds",
            help="Enter a pattern to search for (e.g., 'hds', 'curate', 'phenotype')"
        )

        if function_pattern:
            usage = analyzer.get_function_usage(function_pattern)

            st.metric("Functions Found", usage['total_functions_found'])

            if usage['total_functions_found'] == 0:
                st.info(f"No functions matching '{function_pattern}' found in the codebase.")
            else:
                st.divider()

                # Sort by usage
                sorted_functions = sorted(
                    usage['functions'].items(),
                    key=lambda x: x[1]['total_repos'],
                    reverse=True
                )

                for func_name, func_data in sorted_functions:
                    with st.expander(f"📦 **{func_name}** — {func_data['total_repos']} repos, {func_data['total_files']} files"):
                        # Repos
                        st.markdown("**Repositories:**")
                        for repo in func_data['repos']:
                            st.markdown(f"- `{repo}`")

                        st.divider()

                        # Files by type
                        st.markdown("**Files by Type:**")
                        for file_type, files in func_data['files_by_type'].items():
                            st.markdown(f"- {file_type}: {len(files)} files")

                        st.divider()

                        # Sample files
                        st.markdown("**Example Files:**")
                        for file_info in func_data['all_files'][:5]:
                            st.code(f"{file_info['repo']}/{file_info['file']}", language="text")

    # TAB 4: Module Usage Tracker
    with tab4:
        st.header("Module Usage Tracker")
        st.markdown("Track which repositories import specific Python modules (e.g., hds_functions).")

        # Search box
        module_name = st.text_input(
            "Search for module name",
            value="hds_functions",
            help="Enter a module name to search for (e.g., 'hds_functions', 'pandas', 'numpy')"
        )

        if module_name:
            usage = analyzer.get_module_usage(module_name)

            st.metric("Total Repositories", usage['total_repos'])
            st.metric("Total Files", usage['total_files'])

            if usage['total_repos'] == 0:
                st.info(f"No imports of module '{module_name}' found in the codebase.")
            else:
                st.divider()

                # Repos
                st.subheader("Repositories")
                for repo in usage['repos']:
                    st.markdown(f"- `{repo}`")

                st.divider()

                # Files by type
                st.subheader("Files by Type")
                for file_type, files in usage['files_by_type'].items():
                    with st.expander(f"{file_type}: {len(files)} files"):
                        for file_info in files[:10]:
                            st.code(f"{file_info['repo']}/{file_info['file']}", language="text")
                            if 'import' in file_info:
                                st.caption(f"Import: {file_info['import']}")

    # TAB 5: Similar Projects Finder
    with tab5:
        st.header("Similar Projects Discovery")
        st.markdown("Use semantic search to find similar algorithms and potential code duplication.")

        # Check if vector store exists
        if not check_vector_store_exists():
            st.warning("⚠️ Vector store not found. Please index repositories first.")
            st.stop()

        # Predefined queries
        st.subheader("Quick Searches")
        col1, col2, col3 = st.columns(3)

        query = None
        with col1:
            if st.button("🚬 Smoking Algorithm", use_container_width=True):
                query = "smoking algorithm"
        with col2:
            if st.button("🩺 Diabetes Algorithm", use_container_width=True):
                query = "diabetes algorithm"
        with col3:
            if st.button("❤️ MI Algorithm", use_container_width=True):
                query = "myocardial infarction algorithm"

        # Custom search
        custom_query = st.text_input("Or enter custom search", placeholder="e.g., phenotyping algorithm, data curation...")

        if custom_query:
            query = custom_query

        if query:
            with st.spinner(f"Searching for: {query}..."):
                try:
                    # Load hybrid retriever
                    vector_store = get_vector_store()
                    vector_store.load_vectorstore()

                    # Load documents for hybrid search
                    # For now, we'll skip loading all docs and just note this limitation
                    st.info("Note: Similar project search requires the hybrid index. Run `python build_hybrid_index.py` first for best results.")

                    # Try to use just analyzer without hybrid retriever
                    # This will return empty since we need the retriever
                    results = analyzer.find_similar_projects(query, hybrid_retriever=None, k=10)

                    if not results:
                        st.warning("No results found. Make sure the hybrid index is built: `python build_hybrid_index.py`")
                    else:
                        st.success(f"Found {len(results)} similar projects")

                        for i, project in enumerate(results, 1):
                            with st.expander(f"{i}. **{project['repo']}** — {project['relevance_score']} matching files"):
                                # Tables used
                                if project['tables_used']:
                                    st.markdown(f"**Tables Used:** {', '.join(list(project['tables_used'])[:10])}")

                                # Functions used
                                if project['functions_used']:
                                    st.markdown(f"**Functions Used:** {', '.join(list(project['functions_used'])[:5])}")

                                st.divider()

                                # Sample files
                                if project['matched_files']:
                                    st.markdown("**Sample Files:**")
                                    for file_info in project['matched_files'][:3]:
                                        st.markdown(f"- `{file_info['file']}` ({file_info['type']})")
                                        with st.container():
                                            st.caption(file_info['snippet'][:200] + "...")

                except Exception as e:
                    st.error(f"Error during search: {e}")

    # TAB 6: Cross-Analysis
    with tab6:
        st.header("Cross-Dataset Analysis")
        st.markdown("Find repositories that use multiple data sources together.")

        # Predefined cross-analyses
        st.subheader("Common Combinations")

        if st.button("🔍 COVID + Deaths", use_container_width=True):
            covid_usage = analyzer.get_table_usage("hds_curated_assets__covid_positive")
            deaths_usage = analyzer.get_table_usage("hds_curated_assets__deaths_single")

            covid_repos = set(covid_usage['repos'])
            deaths_repos = set(deaths_usage['repos'])
            both = covid_repos & deaths_repos

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("COVID Only", len(covid_repos - deaths_repos))
            with col2:
                st.metric("Deaths Only", len(deaths_repos - covid_repos))
            with col3:
                st.metric("Both", len(both))

            if both:
                st.success(f"**Repositories using both COVID and Deaths data:**")
                for repo in sorted(both):
                    st.markdown(f"- `{repo}`")
            else:
                st.info("No repositories found using both datasets.")

        st.divider()

        # Custom cross-analysis
        st.subheader("Custom Analysis")

        # Multi-select for tables
        all_tracked_tables = sorted(analyzer.TRACKED_TABLES)
        selected_tables = st.multiselect(
            "Select tables to cross-analyze",
            all_tracked_tables,
            help="Select 2 or more tables to find repos using all of them"
        )

        if len(selected_tables) >= 2:
            if st.button("🔍 Find Repos Using All Selected Tables"):
                # Get repos for each table
                repos_per_table = []
                for table in selected_tables:
                    usage = analyzer.get_table_usage(table)
                    repos_per_table.append(set(usage['repos']))

                # Find intersection
                common_repos = repos_per_table[0]
                for repo_set in repos_per_table[1:]:
                    common_repos = common_repos & repo_set

                st.metric("Repositories Using All Selected Tables", len(common_repos))

                if common_repos:
                    st.success("**Repositories:**")
                    for repo in sorted(common_repos):
                        st.markdown(f"- `{repo}`")

                    # Show what tables each repo uses
                    with st.expander("📊 Detailed Breakdown"):
                        for repo in sorted(common_repos):
                            st.markdown(f"**{repo}**")
                            for table in selected_tables:
                                usage = analyzer.get_table_usage(table)
                                files = [f for f in usage['all_files'] if f['repo'] == repo]
                                st.markdown(f"  - `{table}`: {len(files)} files")
                else:
                    st.info("No repositories found using all selected tables.")


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

        This is a **hybrid RAG (Retrieval-Augmented Generation) system** combined with **static code analysis**
        for organizational intelligence. It provides two complementary capabilities:

        1. **Conversational Q&A**: Ask natural language questions about repositories
        2. **Code Intelligence**: Precise queries about code structure and dependencies
        """)

        st.subheader("High-Level Architecture")
        st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (Streamlit Web App)                        │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             ▼                                    ▼
    ┌────────────────┐                  ┌─────────────────┐
    │   Q&A System   │                  │ Code Intelligence│
    │  (RAG-based)   │                  │   (Static AST)  │
    └────────┬───────┘                  └────────┬────────┘
             │                                    │
             ▼                                    ▼
    ┌─────────────────┐                 ┌──────────────────┐
    │ Hybrid Retriever│                 │  Code Metadata   │
    │  BM25 + Vector  │                 │  Cache (.json)   │
    └────────┬────────┘                 └──────────────────┘
             │
             ├──────────────┬─────────────┐
             ▼              ▼             ▼
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
                                                         Parse Code (AST)
                                                              ↓
                                                       Extract Metadata
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

        #### 3. Code Intelligence Query Phase (Runtime)

        ```
        Structured Query (e.g., "Which repos use table X?")
              ↓
        Load Cached Metadata
              ↓
        Direct Lookup (no LLM needed)
              ↓
        Exact Results (repos, files, line numbers)
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

        st.subheader("3. Code Intelligence (Static Analysis)")
        st.markdown("""
        Beyond LLM-based Q&A, the system provides **deterministic code analysis**:

        #### What It Does
        - **Parses** Python/R/SQL files using AST (Abstract Syntax Tree)
        - **Extracts** structured metadata:
          - Table references (e.g., `hds_curated_assets__demographics`)
          - Function calls (e.g., `hds_functions.curate_data()`)
          - Module imports (e.g., `import pandas as pd`)
          - File classifications (curation, analysis, phenotyping)
        - **Builds** reverse indices for fast lookup

        #### Why It's Useful
        - **Instant queries**: No LLM delay, no API costs
        - **Exact results**: Line numbers, file paths, exact matches
        - **Cross-referencing**: Which projects use multiple data sources?
        - **Organizational insights**: Track standardized function usage

        #### Example Queries
        ```python
        # Find all repos using a specific HDS table
        usage = analyzer.get_table_usage("hds_curated_assets__deaths_single")
        → Returns: repos, files, line numbers

        # Find similar algorithms
        projects = analyzer.find_similar_projects("smoking phenotype")
        → Returns: repos with similar code patterns

        # Track function usage
        usage = analyzer.get_function_usage("hds_functions")
        → Returns: which repos import/call HDS functions
        ```
        """)

        st.subheader("4. Data Storage & Persistence")
        st.markdown("""
        #### Pinecone (Vector Database)
        - **Purpose**: Store document embeddings in the cloud
        - **Persistence**: Data remains across app restarts
        - **Advantages**:
          - Managed service (no infrastructure)
          - Fast similarity search (<50ms)
          - Scalable (handles millions of vectors)

        #### Cloud Storage (Google Drive / S3)
        - **Purpose**: Cache BM25 index and code metadata
        - **Why needed**: Streamlit Cloud has ephemeral storage
        - **How it works**:
          1. Build indices locally or on first cloud run
          2. Upload to Google Drive or S3
          3. Subsequent runs download from cloud (10s vs 10min rebuild)

        #### Local Cache (Development)
        - `.cache/bm25_index.pkl`: BM25 index (50-100MB)
        - `.cache/code_metadata.json`: Code intelligence data (5-20MB)
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

            #### Python Static Analysis
            - **ast module**: Parse Python to AST
            - **re module**: Regex for SQL table extraction
            - **File parsing**: Python, R, SQL, Markdown

            #### GitHub API (PyGithub)
            - **Purpose**: Fetch repository contents
            - **Rate limits**: 60/hour (unauth), 5000/hour (with token)
            - **Indexed file types**:
              - Documentation: `.md`, `.rst`, `.txt`
              - Code: `.py`, `.r`, `.R`, `.sql`
              - Data: `.csv`, `.json`, `.yaml`
              - Notebooks: `.ipynb`

            #### Cloud Storage SDKs
            - **Google Drive**: `google-auth`, `googleapiclient`
            - **Purpose**: Persist caches across ephemeral instances
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
            ["Q&A", "Code Intelligence", "Documentation", "Setup"],
            index=0 if check_vector_store_exists() else 3,
        )

    # Render the selected page
    if page == "Q&A":
        render_qa_page()
    elif page == "Code Intelligence":
        render_code_intelligence_page()
    elif page == "Documentation":
        render_documentation_page()
    elif page == "Setup":
        render_admin_page()


if __name__ == "__main__":
    main()
