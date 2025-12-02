"""Streamlit app dedicated to Code Intelligence - analyzing BHFDSC codebase structure."""

import streamlit as st
import logging
import os
import pandas as pd
import plotly.express as px
from code_analyzer import CodeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BHFDSC Code Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_metadata_exists():
    """Check if code metadata cache exists in either .cache or data_index."""
    cache_file = os.path.join(".cache", "code_metadata.json")
    data_index_file = os.path.join("data_index", "code_metadata.json")
    return os.path.exists(cache_file) or os.path.exists(data_index_file)


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


def render_metadata_management():
    """Render the metadata management section."""
    with st.expander("⚙️ Code Metadata Configuration", expanded=False):
        st.markdown(
            """
            Analyzes Python, R, SQL, and Jupyter notebook files to extract structured metadata:
            table usage, function calls, and module imports.
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
            if st.button("🧠 Build Metadata Index", type="primary" if not metadata_exists else "secondary", use_container_width=True):
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
                if st.button("🗑️ Clear Metadata Cache", type="secondary", use_container_width=True):
                    try:
                        analyzer = CodeAnalyzer()
                        analyzer.clear_cache()
                        st.success("Metadata cache cleared.")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")

        st.divider()
        st.caption("Metadata cached in `.cache/code_metadata.json` or `data_index/code_metadata.json`")


def render_dashboard_tab(analyzer, stats):
    """Render the dashboard overview tab."""
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

    # Temporal Analysis Section
    st.subheader("⏰ Temporal Analysis")

    from datetime import datetime, timedelta

    col_period1, col_period2, col_period3 = st.columns([1, 1, 1])

    with col_period1:
        enable_dashboard_temporal = st.checkbox("Enable temporal filtering", value=False, key="dashboard_temporal")

    dashboard_start_date = None
    dashboard_end_date = None

    if enable_dashboard_temporal:
        with col_period2:
            # Default: last year
            default_start = datetime.now() - timedelta(days=365)
            dashboard_start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="Show activity from this date onwards",
                key="dashboard_start"
            )
            dashboard_start_date = dashboard_start_date.isoformat()

        with col_period3:
            dashboard_end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Show activity before this date",
                key="dashboard_end"
            )
            dashboard_end_date = dashboard_end_date.isoformat()

        # Show temporal metrics
        active_repos = analyzer.get_repos_active_in_period(dashboard_start_date, dashboard_end_date)

        st.divider()

        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            st.metric(
                "Active Repositories in Period",
                len(active_repos),
                help="Repositories with files modified during selected period"
            )
        with col_temp2:
            percentage = (len(active_repos) / stats['total_repos'] * 100) if stats['total_repos'] > 0 else 0
            st.metric(
                "Activity Rate",
                f"{percentage:.1f}%",
                help="Percentage of repositories active in period"
            )

        if active_repos:
            with st.expander(f"📋 View Active Repositories ({len(active_repos)})"):
                for repo in active_repos:
                    st.markdown(f"- `{repo}`")

        st.divider()

        # Show table usage in period
        st.subheader("📊 Most Used Tables in Period")

        # Get usage for tracked tables in the period
        tracked_tables_in_period = {}
        for table in analyzer.TRACKED_TABLES:
            usage = analyzer.get_table_usage_in_period(table, dashboard_start_date, dashboard_end_date)
            if usage['total_repos'] > 0:
                tracked_tables_in_period[table] = usage['total_repos']

        if tracked_tables_in_period:
            period_df = pd.DataFrame([
                {"Table": k, "Repos Using": v}
                for k, v in tracked_tables_in_period.items()
            ]).sort_values('Repos Using', ascending=False)

            fig = px.bar(
                period_df.head(10),
                y='Table',
                x='Repos Using',
                orientation='h',
                title=f"Top Tables Used in Selected Period",
                color='Repos Using',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 View All Tables in Period"):
                st.dataframe(period_df, use_container_width=True)
        else:
            st.info("No tracked table usage found in the selected period.")
    else:
        st.info("💡 Enable temporal filtering to see repository activity and table usage over time.")

    st.divider()

    # Tracked tables with usage (all-time)
    st.subheader("HDS Curated Assets Usage (All-Time)")

    if stats['tracked_tables_found']:
        tables_df = pd.DataFrame([
            {"Table": k, "Repos Using": v}
            for k, v in stats['tracked_tables_found'].items()
        ]).sort_values('Repos Using', ascending=False)

        fig = px.bar(
            tables_df.head(10),
            y='Table',
            x='Repos Using',
            orientation='h',
            title="Top 10 Most Used HDS Tables",
            color='Repos Using',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View All Tracked Tables"):
            st.dataframe(tables_df, use_container_width=True)
    else:
        st.info("No HDS curated assets found in the analyzed code.")

    st.divider()

    # Table Usage Explorer
    st.subheader("📁 Table Usage Explorer")

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
        ["Demographics", "COVID", "Deaths", "HES Assets", "All Extracted Tables"],
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
        # Show all tables that were actually extracted from the code
        table_list = analyzer.get_all_tables()

    # Table selector
    selected_table = st.selectbox("Select Table", table_list)

    if selected_table:
        # Temporal filtering options
        st.divider()
        st.subheader("⏰ Temporal Filters")

        col_temporal1, col_temporal2, col_temporal3 = st.columns([1, 1, 1])

        with col_temporal1:
            enable_temporal = st.checkbox("Enable date range filtering", value=False, key="table_explorer_temporal")

        start_date = None
        end_date = None

        if enable_temporal:
            with col_temporal2:
                # Default: last year
                default_start = datetime.now() - timedelta(days=365)
                start_date = st.date_input(
                    "Start Date",
                    value=default_start,
                    help="Files modified on or after this date",
                    key="table_explorer_start"
                )
                start_date = start_date.isoformat()

            with col_temporal3:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    help="Files modified before this date",
                    key="table_explorer_end"
                )
                end_date = end_date.isoformat()

        st.divider()

        # Get usage data (with or without temporal filters)
        if enable_temporal:
            usage = analyzer.get_table_usage_in_period(selected_table, start_date, end_date)
        else:
            usage = analyzer.get_table_usage(selected_table)

        if usage['total_repos'] == 0:
            if enable_temporal:
                st.warning(f"❌ No usage found for `{selected_table}` in the selected time period")
            else:
                st.warning(f"❌ No usage found for `{selected_table}`")
        else:
            if enable_temporal:
                st.success(f"✅ **{selected_table}** was used in **{usage['total_repos']}** repositories during the selected period")
            else:
                st.success(f"✅ **{selected_table}** is used in **{usage['total_repos']}** repositories")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Repositories", usage['total_repos'])
            with col2:
                st.metric("Total Files", usage['total_files'])

            st.divider()

            # Adoption timeline chart
            if not enable_temporal:
                st.subheader("📈 Adoption Timeline")
                timeline_data = analyzer.get_adoption_timeline(selected_table)

                if timeline_data:
                    timeline_df = pd.DataFrame([
                        {"Period": period, "Repositories": count}
                        for period, count in timeline_data.items()
                    ]).sort_values("Period")

                    fig = px.line(
                        timeline_df,
                        x="Period",
                        y="Repositories",
                        title=f"Table Usage Over Time: {selected_table}",
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title="Year-Month",
                        yaxis_title="Number of Repositories",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No temporal data available for this table.")

                st.divider()

            # Repositories using this table
            st.subheader("Repositories")
            for repo in usage['repos']:
                st.markdown(f"- `{repo}`")


def render_function_usage_tab(analyzer):
    """Render the function usage tracker tab."""
    st.header("Function Usage Tracker")

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


def render_module_usage_tab(analyzer):
    """Render the module usage tracker tab."""
    st.header("Module Usage Tracker")

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


def render_documentation_tab():
    """Render the documentation tab explaining architecture and processing."""
    st.header("📚 How It Works")

    st.markdown("""
    This app uses **static code analysis** to extract organizational intelligence from the BHFDSC codebase.
    It provides instant, deterministic queries without requiring LLMs or API calls.
    """)

    # Architecture section
    st.subheader("🏗️ Architecture")

    st.markdown("""
    ### System Overview

    ```
    GitHub Repositories
           ↓
    [1] Fetch Code Files (.py, .R, .sql)
           ↓
    [2] Parse with AST/Regex
           ↓
    [3] Extract Metadata
           ↓
    [4] Build Indices
           ↓
    [5] Cache to JSON
           ↓
    [6] Query Interface (this app)
    ```
    """)

    st.divider()

    # Processing steps
    st.subheader("⚙️ Processing Steps")

    with st.expander("**Step 1: Repository Fetching**", expanded=True):
        st.markdown("""
        - Connects to GitHub API using personal access token
        - Fetches all repositories from BHFDSC organization
        - Downloads relevant code files:
          - Python files (`.py`)
          - Jupyter notebooks (`.ipynb`)
          - R files (`.R`)
          - SQL files (`.sql`)
        - Filters out non-code files (images, binaries, etc.)
        """)

    with st.expander("**Step 2: Static Code Parsing**"):
        st.markdown("""
        **Python Analysis** (using AST - Abstract Syntax Tree):
        - Parse files into syntax tree using Python's `ast` module
        - Walk through all nodes to find:
          - Function calls (e.g., `analyzer.get_table_usage()`)
          - Import statements (e.g., `import pandas as pd`)
          - String literals containing SQL queries
          - Table references in SQL strings

        **Jupyter Notebook Analysis** (JSON parsing + AST):
        - Parse `.ipynb` JSON structure
        - Extract code from all code cells
        - Combine cells and analyze as Python code
        - Preserves all imports, function calls, and table references

        **R Analysis** (using regex patterns):
        - Extract function calls from R code
        - Identify `library()` and `require()` calls
        - Find dplyr/dbplyr table references (e.g., `tbl(con, "table_name")`)

        **SQL Analysis** (using regex patterns):
        - Extract table names from FROM clauses
        - Extract table names from JOIN clauses
        - Handle schema-qualified names (e.g., `schema.table`)
        """)

    with st.expander("**Step 3: Metadata Extraction**"):
        st.markdown("""
        For each file, we extract:

        **Tables Referenced:**
        - All HDS curated assets (e.g., `hds_curated_assets__demographics`)
        - Custom table patterns
        - Tracks which files/repos use which tables

        **Function Calls:**
        - All function invocations
        - Special tracking for HDS functions
        - Patterns like `hds_*`, `curate_*`, `phenotype_*`

        **Module Imports:**
        - Python: `import` and `from ... import` statements
        - R: `library()` and `require()` calls
        - Tracks dependency usage across projects

        **File Classification:**
        - Curation scripts (data processing)
        - Analysis scripts (statistical analysis)
        - Phenotyping scripts (disease definition)
        - Utility scripts (helper functions)
        """)

    with st.expander("**Step 4: Index Building**"):
        st.markdown("""
        Creates reverse indices for fast lookups:

        ```python
        # Table → Repos mapping
        {
            "hds_curated_assets__demographics": [
                "repo1", "repo2", ...
            ]
        }

        # Function → Files mapping
        {
            "hds_functions.curate_data": [
                {"repo": "repo1", "file": "script.py"},
                ...
            ]
        }

        # Module → Usage mapping
        {
            "pandas": {
                "repos": ["repo1", "repo2"],
                "files": [...]
            }
        }
        ```

        These indices enable O(1) lookup time for queries.
        """)

    with st.expander("**Step 5: Caching**"):
        st.markdown("""
        **Storage Locations:**
        - Runtime cache: `.cache/code_metadata.json` (not committed to Git)
        - Persistent cache: `data_index/code_metadata.json` (committed to Git)

        **Cache Format:** JSON with structure:
        ```json
        {
            "files": [...],
            "repos": [...],
            "tables": {...},
            "functions": {...},
            "modules": {...},
            "metadata": {
                "last_updated": "2024-01-15",
                "total_files": 450
            }
        }
        ```

        **Benefits:**
        - No need to re-parse on every app restart
        - Instant loading (<1 second)
        - Can be shared via Git
        - Typical size: 5-20 MB
        """)

    st.divider()

    # Technical details
    st.subheader("🔧 Technical Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Python AST Example:**
        ```python
        import ast

        code = '''
        df = spark.table("hds_curated_assets__demographics")
        '''

        tree = ast.parse(code)
        # Walk nodes to find table name
        # → Extracts "hds_curated_assets__demographics"
        ```

        **Benefits:**
        - Accurate parsing (no false positives)
        - Handles complex code structures
        - Extracts context (line numbers, function scope)
        """)

    with col2:
        st.markdown("""
        **SQL Regex Example:**
        ```sql
        SELECT *
        FROM hds_curated_assets__deaths_single d
        JOIN hds_curated_assets__demographics demo
          ON d.person_id = demo.person_id
        ```

        **Extraction:**
        - FROM clause → `hds_curated_assets__deaths_single`
        - JOIN clause → `hds_curated_assets__demographics`
        - Handles aliases (d, demo)
        """)

    st.divider()

    # Performance metrics
    st.subheader("⚡ Performance")

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Metadata Build Time", "10-30 min", help="One-time indexing of all repos")

    with metrics_col2:
        st.metric("Query Response Time", "<100 ms", help="Dictionary lookup, no API calls")

    with metrics_col3:
        st.metric("Cache Size", "5-20 MB", help="Compressed JSON with all metadata")

    st.divider()

    # Tracked assets
    st.subheader("📋 Tracked HDS Curated Assets")

    st.markdown("""
    The system specifically tracks these HDS curated asset tables:

    **Demographics:**
    - `hds_curated_assets__demographics`
    - `hds_curated_assets__date_of_birth_*`
    - `hds_curated_assets__ethnicity_*`
    - `hds_curated_assets__sex_*`
    - `hds_curated_assets__lsoa_*`

    **Health Outcomes:**
    - `hds_curated_assets__covid_positive`
    - `hds_curated_assets__deaths_single`
    - `hds_curated_assets__deaths_cause_of_death`

    **Hospital Episodes (HES):**
    - `hds_curated_assets__hes_apc_*` (multiple variants)

    *See `code_analyzer.py` for the complete list of tracked tables.*
    """)

    st.divider()

    # Use cases
    st.subheader("💡 Use Cases")

    use_case_col1, use_case_col2 = st.columns(2)

    with use_case_col1:
        st.markdown("""
        **For Data Scientists:**
        - Find repos using specific data tables
        - Discover existing phenotyping algorithms
        - See how others handle similar analyses
        - Identify code reuse opportunities
        """)

    with use_case_col2:
        st.markdown("""
        **For Team Leads:**
        - Track adoption of standardized assets
        - Identify code duplication
        - Understand cross-project dependencies
        - Plan data governance strategies
        """)

    st.divider()

    # Limitations
    st.subheader("⚠️ Limitations")

    st.markdown("""
    **What it CAN do:**
    - ✅ Find exact table/function references in code
    - ✅ Track imports and dependencies
    - ✅ Provide instant, deterministic results
    - ✅ Work offline (after initial build)

    **What it CANNOT do:**
    - ❌ Understand runtime behavior or dynamic queries
    - ❌ Answer semantic questions (use Q&A app for that)
    - ❌ Detect tables built from variables (e.g., `table_name = f"prefix_{suffix}"`)
    - ❌ Analyze code execution or data flow

    **Complementary Tool:** For semantic questions like "What is the methodology behind X?",
    use the main Q&A app which uses LLMs and vector search.
    """)


def render_cross_analysis_tab(analyzer):
    """Render the cross-analysis tab."""
    st.header("Cross-Dataset Analysis")

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


def main():
    """Main application."""
    st.title("🧠 BHFDSC Code Intelligence Dashboard")

    # Metadata management section
    render_metadata_management()

    st.divider()

    # Check if metadata exists
    if not check_metadata_exists():
        st.error("⚠️ Metadata index not found!")
        st.info("Use the **Code Metadata Configuration** panel above to build the metadata index.")
        st.stop()

    # Load analyzer
    analyzer = load_code_analyzer()

    if analyzer is None:
        st.error("Failed to load code analyzer. Please rebuild the metadata index using the panel above.")
        st.stop()

    # Get statistics
    stats = analyzer.get_stats()

    # Sidebar with summary info
    with st.sidebar:
        st.header("Quick Stats")
        st.metric("Repositories", stats['total_repos'])
        st.metric("Files Analyzed", stats['total_files'])
        st.metric("Unique Tables", stats['total_unique_tables'])
        st.metric("Unique Functions", stats['total_unique_functions'])

    # Tabs for different features
    tabs = st.tabs([
        "📊 Dashboard",
        "⚙️ Function Usage",
        "📦 Module Usage",
        "🔗 Cross-Analysis",
        "📚 How It Works"
    ])

    with tabs[0]:
        render_dashboard_tab(analyzer, stats)

    with tabs[1]:
        render_function_usage_tab(analyzer)

    with tabs[2]:
        render_module_usage_tab(analyzer)

    with tabs[3]:
        render_cross_analysis_tab(analyzer)

    with tabs[4]:
        render_documentation_tab()


if __name__ == "__main__":
    main()
