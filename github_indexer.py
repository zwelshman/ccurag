"""GitHub repository indexing functionality."""

import logging
from typing import List, Dict
from github import Github, GithubException
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIndexer:
    """Fetches and indexes repositories from a GitHub organization."""

    def __init__(self, github_token: str = None):
        """Initialize GitHub client.

        Args:
            github_token: GitHub API token
        """
        self.github = Github(github_token) if github_token else Github()
        self.org_name = Config.GITHUB_ORG

    def get_all_repos(self) -> List[Dict]:
        """Fetch all repositories from the organization."""
        try:
            org = self.github.get_organization(self.org_name)
            repos = org.get_repos()

            repo_list = []
            for repo in repos:
                repo_info = {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description or "",
                    "url": repo.html_url,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "topics": repo.get_topics(),
                }
                repo_list.append(repo_info)
                logger.info(f"Found repo: {repo.name}")

            logger.info(f"Total repositories found: {len(repo_list)}")
            return repo_list

        except GithubException as e:
            logger.error(f"Error fetching repositories: {e}")
            raise

    def get_repo_contents(self, repo_full_name: str) -> List[Dict]:
        """Fetch file contents from a repository.

        Args:
            repo_full_name: Full name of repository (e.g., "owner/repo")

        Returns:
            List of document dictionaries
        """
        documents = []

        try:
            repo = self.github.get_repo(repo_full_name)

            # Get README
            try:
                readme = repo.get_readme()
                content = readme.decoded_content.decode('utf-8')
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": f"{repo_full_name}/README.md",
                        "repo": repo_full_name,
                        "type": "readme",
                        "url": readme.html_url,
                    }
                })
                logger.info(f"Fetched README for {repo_full_name}")
            except GithubException:
                logger.warning(f"No README found for {repo_full_name}")

            # Get all code files (no limit for complete analysis)
            contents = repo.get_contents("")

            while contents:
                try:
                    file_content = contents.pop(0)

                    if file_content.type == "dir":
                        # Add directory contents to the queue
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                        except GithubException:
                            continue
                    else:
                        # Check if file extension is in our indexed list
                        if any(file_content.name.endswith(ext) for ext in Config.INDEXED_FILE_EXTENSIONS):
                            try:
                                content = file_content.decoded_content.decode('utf-8')
                                documents.append({
                                    "content": content,
                                    "metadata": {
                                        "source": f"{repo_full_name}/{file_content.path}",
                                        "repo": repo_full_name,
                                        "type": "file",
                                        "url": file_content.html_url,
                                        "path": file_content.path,
                                    }
                                })
                            except (UnicodeDecodeError, GithubException):
                                continue

                except Exception as e:
                    logger.warning(f"Error processing file: {e}")
                    continue

            logger.info(f"Fetched {len(documents)} documents from {repo_full_name}")
            return documents

        except GithubException as e:
            logger.error(f"Error fetching repo contents for {repo_full_name}: {e}")
            return documents

    def index_all_repos(self, vector_store=None) -> int:
        """Index all repositories.

        Args:
            vector_store: Vector store instance to use for upserting

        Returns:
            Total count of documents processed
        """
        logger.info("Starting repository indexing")

        # Get all repos
        repos = self.get_all_repos()
        total_repos = len(repos)
        total_documents = 0

        # Process each repository sequentially
        for i, repo in enumerate(repos, 1):
            repo_full_name = repo['full_name']

            try:
                logger.info(f"Processing repo {i}/{total_repos}: {repo_full_name}")

                # Create repo metadata document
                repo_doc = {
                    "content": f"Repository: {repo['name']}\n\nDescription: {repo['description']}\n\nLanguage: {repo['language']}\n\nTopics: {', '.join(repo['topics'])}",
                    "metadata": {
                        "source": repo['full_name'],
                        "repo": repo['full_name'],
                        "type": "repo_info",
                        "url": repo['url'],
                    }
                }

                # Get file contents
                repo_contents = self.get_repo_contents(repo_full_name)

                # Combine all documents for this repo
                repo_documents = [repo_doc] + repo_contents
                doc_count = len(repo_documents)
                total_documents += doc_count

                logger.info(f"Indexed {doc_count} documents from {repo_full_name}")

                # Insert into vector store
                if vector_store:
                    logger.info(f"Inserting {doc_count} documents into Pinecone...")
                    vector_store.upsert_documents(repo_documents, repo_name=repo_full_name)
                    logger.info(f"Inserted {doc_count} documents")

            except Exception as e:
                logger.error(f"Failed to process repo {repo_full_name}: {e}")
                continue

        logger.info(f"Indexing complete: {total_documents} total documents from {total_repos} repositories")
        return total_documents
