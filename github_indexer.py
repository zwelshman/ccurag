"""GitHub repository indexing functionality."""

import logging
import random
from typing import List, Dict, Optional
from github import Github, GithubException
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIndexer:
    """Fetches and indexes repositories from a GitHub organization."""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize GitHub client."""
        self.github = Github(github_token) if github_token else Github()
        self.org_name = Config.GITHUB_ORG

    def get_all_repos(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Fetch all repositories from the organization.

        Args:
            sample_size: If provided, randomly sample this many repositories.
                        If None, fetch all repositories.
        """
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

            # Sample repositories if requested
            if sample_size is not None and sample_size < len(repo_list):
                original_count = len(repo_list)
                repo_list = random.sample(repo_list, sample_size)
                logger.info(f"Sampled {sample_size} repositories from {original_count} total")

            return repo_list

        except GithubException as e:
            logger.error(f"Error fetching repositories: {e}")
            raise

    def get_repo_contents(self, repo_full_name: str) -> List[Dict]:
        """Fetch file contents from a repository."""
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

            # Get other files
            file_count = 0
            contents = repo.get_contents("")

            while contents and file_count < Config.MAX_FILES_PER_REPO:
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
                                file_count += 1
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

    def index_all_repos(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Index all repositories in the organization.

        Args:
            sample_size: If provided, randomly sample this many repositories.
                        If None, index all repositories.
        """
        all_documents = []
        repos = self.get_all_repos(sample_size=sample_size)

        for i, repo in enumerate(repos, 1):
            logger.info(f"Indexing repo {i}/{len(repos)}: {repo['full_name']}")

            # Add repo metadata as a document
            repo_doc = {
                "content": f"Repository: {repo['name']}\n\nDescription: {repo['description']}\n\nLanguage: {repo['language']}\n\nTopics: {', '.join(repo['topics'])}",
                "metadata": {
                    "source": repo['full_name'],
                    "repo": repo['full_name'],
                    "type": "repo_info",
                    "url": repo['url'],
                }
            }
            all_documents.append(repo_doc)

            # Get file contents
            repo_contents = self.get_repo_contents(repo['full_name'])
            all_documents.extend(repo_contents)

        logger.info(f"Total documents indexed: {len(all_documents)}")
        return all_documents
