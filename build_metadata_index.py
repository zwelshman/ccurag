"""Build metadata index by analyzing all documents in the vector store.

This script:
1. Fetches documents from the GitHub indexer or vector store
2. Runs code analysis to extract table/function usage
3. Caches the metadata for fast querying
"""

import logging
import sys
from config import Config
from github_indexer import GitHubIndexer
from code_analyzer import CodeAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(force_rebuild: bool = False):
    """Build metadata index from GitHub repositories.

    Args:
        force_rebuild: Force rebuild even if cache exists
    """
    logger.info("Starting metadata index build...")

    # Validate config
    Config.validate()

    # Initialize GitHub indexer
    logger.info(f"Fetching repositories from {Config.GITHUB_ORG}...")
    indexer = GitHubIndexer(github_token=Config.GITHUB_TOKEN)

    # Get all repos
    repos = indexer.get_all_repos()
    logger.info(f"Found {len(repos)} repositories")

    # Collect all documents
    all_documents = []

    for i, repo in enumerate(repos, 1):
        repo_full_name = repo['full_name']
        logger.info(f"Processing repo {i}/{len(repos)}: {repo_full_name}")

        try:
            # Get file contents
            repo_contents = indexer.get_repo_contents(repo_full_name)

            # Convert to document format
            for doc_dict in repo_contents:
                all_documents.append(doc_dict)

        except Exception as e:
            logger.error(f"Failed to process repo {repo_full_name}: {e}")
            continue

    logger.info(f"Collected {len(all_documents)} documents from {len(repos)} repositories")

    # Initialize analyzer and build index
    logger.info("Building metadata index...")
    analyzer = CodeAnalyzer()
    analyzer.index_documents(all_documents, force_rebuild=force_rebuild)

    # Print statistics
    stats = analyzer.get_stats()
    logger.info("\n" + "="*60)
    logger.info("METADATA INDEX STATISTICS")
    logger.info("="*60)
    logger.info(f"Total repositories: {stats['total_repos']}")
    logger.info(f"Total files analyzed: {stats['total_files']}")
    logger.info(f"Unique tables found: {stats['total_unique_tables']}")
    logger.info(f"Unique functions found: {stats['total_unique_functions']}")
    logger.info(f"Tracked HDS tables found: {stats['tracked_tables_count']}/{len(analyzer.TRACKED_TABLES)}")

    logger.info("\nFile types breakdown:")
    for file_type, count in sorted(stats['file_types'].items(), key=lambda x: -x[1]):
        logger.info(f"  {file_type}: {count} files")

    logger.info("\nTracked tables with usage:")
    for table, repo_count in sorted(stats['tracked_tables_found'].items(), key=lambda x: -x[1]):
        if repo_count > 0:
            logger.info(f"  {table}: {repo_count} repos")

    logger.info("="*60)
    logger.info("✓ Metadata index build complete!")
    logger.info(f"Cache saved to: {analyzer.metadata_cache_file}")


if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv

    if force:
        logger.info("Force rebuild enabled")

    main(force_rebuild=force)
