"""Example usage of the CodeAnalyzer for organizational intelligence queries.

Demonstrates:
1. analyzer.get_table_usage() - Which repos use specific HDS curated asset tables?
2. analyzer.get_function_usage() - Who's using hds_functions?
3. analyzer.find_similar_projects() - RAG-based clustering of similar algorithms
"""

import logging
import json
from config import Config
from code_analyzer import CodeAnalyzer
from vector_store_pinecone import PineconeVectorStore
from hybrid_retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_json(data, indent=2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent))


def example_1_table_usage(analyzer: CodeAnalyzer):
    """Example 1: Query table usage for HDS curated assets."""
    print_section("EXAMPLE 1: Table Usage Analysis")

    # Define tables to query
    tables_to_query = [
        # Demographics
        "hds_curated_assets__demographics",
        "hds_curated_assets__sex_multisource",
        "hds_curated_assets__ethnicity_multisource",

        # COVID
        "hds_curated_assets__covid_positive",

        # Deaths
        "hds_curated_assets__deaths_single",
        "hds_curated_assets__deaths_cause_of_death",

        # HES
        "hds_curated_assets__hes_apc_diagnosis",
        "hds_curated_assets__hes_apc_procedure",
    ]

    for table in tables_to_query:
        print(f"\n📊 Table: {table}")
        print("-" * 80)

        usage = analyzer.get_table_usage(table)

        if usage['total_repos'] == 0:
            print("  ❌ No usage found")
        else:
            print(f"  ✓ Used in {usage['total_repos']} repositories")
            print(f"  ✓ Found in {usage['total_files']} files")

            # Show repos
            print(f"\n  Repositories:")
            for repo in usage['repos'][:5]:  # Show first 5
                print(f"    - {repo}")
            if len(usage['repos']) > 5:
                print(f"    ... and {len(usage['repos']) - 5} more")

            # Show breakdown by file type
            print(f"\n  Files by type:")
            for file_type, files in usage['files_by_type'].items():
                print(f"    {file_type}: {len(files)} files")

            # Show example files
            print(f"\n  Example files:")
            for file_info in usage['all_files'][:3]:  # Show first 3
                print(f"    - {file_info['repo']}/{file_info['file']} ({file_info['type']})")
            if len(usage['all_files']) > 3:
                print(f"    ... and {len(usage['all_files']) - 3} more")


def example_2_function_usage(analyzer: CodeAnalyzer):
    """Example 2: Query hds_functions usage."""
    print_section("EXAMPLE 2: HDS Functions Usage Analysis")

    print("🔍 Searching for all hds_functions usage...\n")

    usage = analyzer.get_function_usage("hds")

    print(f"Found {usage['total_functions_found']} unique HDS functions\n")

    if usage['total_functions_found'] == 0:
        print("❌ No HDS functions found in the codebase")
        print("\nNote: This is expected if:")
        print("  1. The metadata index hasn't been built yet")
        print("  2. The BHFDSC repos don't use standardized HDS function libraries")
        return

    # Show top functions by usage
    functions_by_usage = sorted(
        usage['functions'].items(),
        key=lambda x: x[1]['total_repos'],
        reverse=True
    )

    print("Top HDS functions by repository count:\n")
    for func_name, func_data in functions_by_usage[:10]:  # Top 10
        print(f"📦 {func_name}")
        print(f"   Used in {func_data['total_repos']} repos, {func_data['total_files']} files")
        print(f"   Repos: {', '.join(func_data['repos'][:3])}")
        if len(func_data['repos']) > 3:
            print(f"         ... and {len(func_data['repos']) - 3} more")

        # Show file type breakdown
        if func_data['files_by_type']:
            types = [f"{k}({len(v)})" for k, v in func_data['files_by_type'].items()]
            print(f"   Types: {', '.join(types)}")
        print()


def example_3_similar_projects(analyzer: CodeAnalyzer, hybrid_retriever: HybridRetriever):
    """Example 3: Find similar projects using RAG-based clustering."""
    print_section("EXAMPLE 3: Similar Projects Discovery")

    algorithms_to_search = [
        "smoking algorithm",
        "diabetes algorithm",
        "myocardial infarction algorithm",
        "MI algorithm"
    ]

    for query in algorithms_to_search:
        print(f"\n🔎 Searching for: '{query}'")
        print("-" * 80)

        results = analyzer.find_similar_projects(
            query=query,
            hybrid_retriever=hybrid_retriever,
            k=10
        )

        if not results:
            print("  ❌ No similar projects found")
            continue

        print(f"  ✓ Found {len(results)} similar projects\n")

        # Show top 5 results
        for i, project in enumerate(results[:5], 1):
            print(f"  {i}. {project['repo']}")
            print(f"     Relevance: {project['relevance_score']:.1f} matching files")
            print(f"     File types: {', '.join(project['file_types'])}")

            if project['tables_used']:
                print(f"     Tables: {', '.join(list(project['tables_used'])[:5])}")

            if project['functions_used']:
                print(f"     Functions: {', '.join(list(project['functions_used'])[:3])}")

            # Show a snippet from one file
            if project['matched_files']:
                sample_file = project['matched_files'][0]
                print(f"\n     Sample file: {sample_file['file']}")
                print(f"     Snippet: {sample_file['snippet'][:150]}...")

            print()


def example_4_cross_analysis(analyzer: CodeAnalyzer):
    """Example 4: Cross-analysis - Which projects use both COVID and deaths data?"""
    print_section("EXAMPLE 4: Cross-Analysis - COVID + Deaths Studies")

    covid_usage = analyzer.get_table_usage("hds_curated_assets__covid_positive")
    deaths_usage = analyzer.get_table_usage("hds_curated_assets__deaths_single")

    covid_repos = set(covid_usage['repos'])
    deaths_repos = set(deaths_usage['repos'])

    # Find intersection
    both = covid_repos & deaths_repos

    print(f"📊 COVID data used in: {len(covid_repos)} repos")
    print(f"📊 Deaths data used in: {len(deaths_repos)} repos")
    print(f"📊 Both COVID + Deaths: {len(both)} repos\n")

    if both:
        print("Repositories using both COVID and Deaths data:")
        for repo in sorted(both):
            print(f"  - {repo}")
    else:
        print("No repositories found using both datasets")


def main():
    """Run all examples."""
    logger.info("Starting CodeAnalyzer examples...")

    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Initialize analyzer (loads from cache if available)
    logger.info("Loading CodeAnalyzer...")
    analyzer = CodeAnalyzer()

    # Check if metadata is available
    stats = analyzer.get_stats()
    if stats['total_files'] == 0:
        print("\n" + "!"*80)
        print("⚠️  WARNING: No metadata found!")
        print("!"*80)
        print("\nPlease build the metadata index first by running:")
        print("  python build_metadata_index.py")
        print("\nThis will analyze all repositories and cache the metadata.")
        print("!"*80 + "\n")
        return

    logger.info(f"Loaded metadata for {stats['total_repos']} repos, {stats['total_files']} files")

    # Initialize hybrid retriever for semantic search (Example 3)
    logger.info("Initializing hybrid retriever for semantic search...")
    try:
        vector_store = PineconeVectorStore(
            api_key=Config.PINECONE_API_KEY,
            index_name=Config.PINECONE_INDEX_NAME,
            cloud=Config.PINECONE_CLOUD,
            region=Config.PINECONE_REGION
        )

        # Try to load from existing vector store
        from build_hybrid_index import load_documents_from_vector_store
        documents = load_documents_from_vector_store(vector_store)

        if not documents:
            logger.warning("No documents found in vector store. Example 3 will be skipped.")
            hybrid_retriever = None
        else:
            hybrid_retriever = HybridRetriever(vector_store)
            hybrid_retriever.build_bm25_index(documents)
            logger.info(f"Hybrid retriever ready with {len(documents)} documents")
    except Exception as e:
        logger.warning(f"Could not initialize hybrid retriever: {e}")
        logger.warning("Example 3 (similar projects) will be skipped")
        hybrid_retriever = None

    # Run examples
    try:
        example_1_table_usage(analyzer)
        example_2_function_usage(analyzer)

        if hybrid_retriever:
            example_3_similar_projects(analyzer, hybrid_retriever)
        else:
            print_section("EXAMPLE 3: Similar Projects Discovery")
            print("⚠️  Skipped - hybrid retriever not available")
            print("Build the hybrid index first: python build_hybrid_index.py")

        example_4_cross_analysis(analyzer)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

    print("\n" + "="*80)
    print("✓ Examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
