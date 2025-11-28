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
        st.caption("""
        **Storage**: Metadata is cached locally in `.cache/code_metadata.json`.
        For committed storage, copy to `data_index/code_metadata.json` and commit to Git.
        The app automatically checks both locations.
        """)


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

    # Tracked tables with usage
    st.subheader("HDS Curated Assets Usage")

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


def render_table_usage_tab(analyzer):
    """Render the table usage explorer tab."""
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


def render_function_usage_tab(analyzer):
    """Render the function usage tracker tab."""
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


def render_module_usage_tab(analyzer):
    """Render the module usage tracker tab."""
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


def render_cross_analysis_tab(analyzer):
    """Render the cross-analysis tab."""
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


def main():
    """Main application."""
    st.title("🧠 BHFDSC Code Intelligence Dashboard")

    st.markdown(
        """
        Explore organizational intelligence about the BHFDSC codebase through static code analysis.
        Track table usage, function dependencies, and discover similar projects.
        """
    )

    # Metadata management section
    render_metadata_management()

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

    # Sidebar with summary info
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **Code Intelligence** provides static analysis of BHFDSC repositories:

            - Track HDS curated asset usage
            - Find function dependencies
            - Analyze module imports
            - Discover cross-dataset usage
            """
        )

        st.divider()

        st.header("Quick Stats")
        st.metric("Repositories", stats['total_repos'])
        st.metric("Files Analyzed", stats['total_files'])
        st.metric("Unique Tables", stats['total_unique_tables'])
        st.metric("Unique Functions", stats['total_unique_functions'])

    # Tabs for different features
    tabs = st.tabs([
        "📊 Dashboard",
        "📁 Table Usage",
        "⚙️ Function Usage",
        "📦 Module Usage",
        "🔗 Cross-Analysis"
    ])

    with tabs[0]:
        render_dashboard_tab(analyzer, stats)

    with tabs[1]:
        render_table_usage_tab(analyzer)

    with tabs[2]:
        render_function_usage_tab(analyzer)

    with tabs[3]:
        render_module_usage_tab(analyzer)

    with tabs[4]:
        render_cross_analysis_tab(analyzer)


if __name__ == "__main__":
    main()
