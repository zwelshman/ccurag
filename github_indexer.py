"""GitHub repository indexing functionality."""

import logging
import random
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from github import Github, GithubException
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for resumable repository indexing."""

    def __init__(self, checkpoint_file: str = ".checkpoint.json"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = None

    def load(self) -> Optional[Dict]:
        """Load checkpoint from file."""
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found, starting fresh")
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint: {self.checkpoint_data['completed_repos']}/{self.checkpoint_data['total_repos']} repos completed")
            return self.checkpoint_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return None

    def validate(self, org_name: str, sample_size: Optional[int], total_repos: int) -> bool:
        """Validate checkpoint matches current request.

        Args:
            org_name: GitHub organization name
            sample_size: Sample size (None for all repos)
            total_repos: Total number of repos to process

        Returns:
            True if checkpoint is valid and can be resumed
        """
        if not self.checkpoint_data:
            return False

        # Check if organization matches
        if self.checkpoint_data.get('organization') != org_name:
            logger.warning(f"Checkpoint organization mismatch. Expected {org_name}, got {self.checkpoint_data.get('organization')}")
            return False

        # Check if sample size matches
        if self.checkpoint_data.get('sample_size') != sample_size:
            logger.warning(f"Checkpoint sample_size mismatch. Expected {sample_size}, got {self.checkpoint_data.get('sample_size')}")
            return False

        # Check if total repos matches (within reason - some repos might have been added/removed)
        checkpoint_total = self.checkpoint_data.get('total_repos', 0)
        if abs(checkpoint_total - total_repos) > 5:  # Allow 5 repo difference
            logger.warning(f"Checkpoint total_repos mismatch. Expected {total_repos}, got {checkpoint_total}")
            return False

        return True

    def create(self, org_name: str, sample_size: Optional[int], total_repos: int) -> Dict:
        """Create new checkpoint.

        Args:
            org_name: GitHub organization name
            sample_size: Sample size (None for all repos)
            total_repos: Total number of repos to process

        Returns:
            New checkpoint data
        """
        self.checkpoint_data = {
            'organization': org_name,
            'sample_size': sample_size,
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'processed_repos': [],
            'repos_metadata': {},  # New: Track commit SHA and metadata per repo
            'total_repos': total_repos,
            'completed_repos': 0,
            'documents_collected': 0
        }
        self.save()
        logger.info(f"Created new checkpoint for {total_repos} repositories")
        return self.checkpoint_data

    def update(self, repo_full_name: str, doc_count: int, commit_sha: Optional[str] = None):
        """Update checkpoint with completed repository.

        Args:
            repo_full_name: Full name of completed repository
            doc_count: Number of documents collected from this repo
            commit_sha: Latest commit SHA of the repository
        """
        if not self.checkpoint_data:
            return

        # Add to processed repos list if not already there
        if repo_full_name not in self.checkpoint_data['processed_repos']:
            self.checkpoint_data['processed_repos'].append(repo_full_name)

        # Store detailed metadata for the repo
        if 'repos_metadata' not in self.checkpoint_data:
            self.checkpoint_data['repos_metadata'] = {}

        self.checkpoint_data['repos_metadata'][repo_full_name] = {
            'commit_sha': commit_sha,
            'last_indexed': datetime.now().isoformat(),
            'doc_count': doc_count
        }

        self.checkpoint_data['completed_repos'] = len(self.checkpoint_data['processed_repos'])
        self.checkpoint_data['documents_collected'] = self.checkpoint_data.get('documents_collected', 0) + doc_count
        self.checkpoint_data['last_updated'] = datetime.now().isoformat()
        self.save()

    def save(self):
        """Save checkpoint to file."""
        if not self.checkpoint_data:
            return

        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_processed_repos(self) -> List[str]:
        """Get list of already processed repository names."""
        if not self.checkpoint_data:
            return []
        return self.checkpoint_data.get('processed_repos', [])

    def get_repo_commit_sha(self, repo_full_name: str) -> Optional[str]:
        """Get the stored commit SHA for a repository.

        Args:
            repo_full_name: Full name of repository

        Returns:
            Commit SHA if found, None otherwise
        """
        if not self.checkpoint_data:
            return None
        repos_metadata = self.checkpoint_data.get('repos_metadata', {})
        repo_info = repos_metadata.get(repo_full_name, {})
        return repo_info.get('commit_sha')

    def has_repo_changed(self, repo_full_name: str, current_commit_sha: str) -> bool:
        """Check if a repository has changed since last indexing.

        Args:
            repo_full_name: Full name of repository
            current_commit_sha: Current commit SHA of the repository

        Returns:
            True if repo has changed or is new, False if unchanged
        """
        stored_sha = self.get_repo_commit_sha(repo_full_name)
        if stored_sha is None:
            return True  # New repo
        return stored_sha != current_commit_sha

    def delete(self):
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint file deleted")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
        self.checkpoint_data = None


class GitHubIndexer:
    """Fetches and indexes repositories from a GitHub organization."""

    def __init__(self, github_token: Optional[str] = None, checkpoint_file: str = ".checkpoint.json"):
        """Initialize GitHub client.

        Args:
            github_token: GitHub API token
            checkpoint_file: Path to checkpoint file for resume capability
        """
        self.github = Github(github_token) if github_token else Github()
        self.org_name = Config.GITHUB_ORG
        self.checkpoint_manager = CheckpointManager(checkpoint_file)

    def get_all_repos(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Fetch all repositories from the organization.

        Args:
            sample_size: If provided, randomly sample this many repositories.
                        If None, fetch all repositories.
        """
        try:
            org = self.github.get_organization(self.org_name)
            repos = org.get_repos()

            # Check if test mode is enabled
            use_test_repos = Config.USE_TEST_REPOS
            test_repos_filter = set(Config.TEST_REPOS) if use_test_repos else None

            if use_test_repos:
                if not test_repos_filter:
                    logger.warning("USE_TEST_REPOS is enabled but TEST_REPOS list is empty. No repos will be indexed.")
                else:
                    logger.info(f"Test mode enabled. Will filter for repos: {', '.join(test_repos_filter)}")

            repo_list = []
            for repo in repos:
                # Filter by test repos if enabled
                if use_test_repos:
                    if test_repos_filter and repo.name not in test_repos_filter:
                        continue  # Skip repos not in test list

                # Get the latest commit SHA from the default branch
                commit_sha = None
                try:
                    if repo.default_branch:
                        branch = repo.get_branch(repo.default_branch)
                        commit_sha = branch.commit.sha
                except GithubException:
                    logger.warning(f"Could not fetch commit SHA for {repo.full_name}")

                repo_info = {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description or "",
                    "url": repo.html_url,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "topics": repo.get_topics(),
                    "commit_sha": commit_sha,  # New: Store latest commit SHA
                }
                repo_list.append(repo_info)
                logger.info(f"Found repo: {repo.name} (SHA: {commit_sha[:7] if commit_sha else 'N/A'})")

            logger.info(f"Total repositories found: {len(repo_list)}")

            # Sample repositories if requested (and not in test mode)
            if sample_size is not None and sample_size < len(repo_list) and not use_test_repos:
                original_count = len(repo_list)
                repo_list = random.sample(repo_list, sample_size)
                logger.info(f"Sampled {sample_size} repositories from {original_count} total")

            return repo_list

        except GithubException as e:
            logger.error(f"Error fetching repositories: {e}")
            raise

    def get_repo_contents(self, repo_full_name: str, max_retries: int = 4) -> List[Dict]:
        """Fetch file contents from a repository with retry logic.

        Args:
            repo_full_name: Full name of repository (e.g., "owner/repo")
            max_retries: Maximum number of retry attempts for rate limit errors

        Returns:
            List of document dictionaries
        """
        documents = []
        retry_count = 0
        backoff_seconds = 2

        while retry_count <= max_retries:
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
                except GithubException as e:
                    if e.status == 403 and 'rate limit' in str(e).lower():
                        raise  # Re-raise rate limit errors to trigger retry
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
                            except GithubException as e:
                                if e.status == 403 and 'rate limit' in str(e).lower():
                                    raise  # Re-raise rate limit errors
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
                                except (UnicodeDecodeError, GithubException) as e:
                                    if isinstance(e, GithubException) and e.status == 403 and 'rate limit' in str(e).lower():
                                        raise  # Re-raise rate limit errors
                                    continue

                    except Exception as e:
                        if isinstance(e, GithubException) and e.status == 403 and 'rate limit' in str(e).lower():
                            raise  # Re-raise rate limit errors
                        logger.warning(f"Error processing file: {e}")
                        continue

                logger.info(f"Fetched {len(documents)} documents from {repo_full_name}")
                return documents

            except GithubException as e:
                # Check if it's a rate limit error
                if e.status == 403 and 'rate limit' in str(e).lower():
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"Rate limit hit for {repo_full_name}. Retrying in {backoff_seconds}s (attempt {retry_count}/{max_retries})")
                        time.sleep(backoff_seconds)
                        backoff_seconds *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Rate limit exceeded for {repo_full_name} after {max_retries} retries. Skipping.")
                        return documents
                else:
                    logger.error(f"Error fetching repo contents for {repo_full_name}: {e}")
                    return documents

        return documents

    def index_all_repos(self, sample_size: Optional[int] = None, resume: bool = True) -> Tuple[List[Dict], List[str]]:
        """Index all repositories in the organization with checkpoint/resume support.

        Args:
            sample_size: If provided, randomly sample this many repositories.
                        If None, index all repositories.
            resume: If True, attempt to resume from checkpoint if it exists.

        Returns:
            Tuple of (documents, changed_repos) where:
                - documents: List of all documents collected from repositories
                - changed_repos: List of repo names that had changes (not new repos)
        """
        all_documents = []
        repos = self.get_all_repos(sample_size=sample_size)
        total_repos = len(repos)

        # Load checkpoint if resume is enabled
        checkpoint = None
        processed_repos = set()
        if resume:
            checkpoint = self.checkpoint_manager.load()
            if checkpoint and self.checkpoint_manager.validate(self.org_name, sample_size, total_repos):
                processed_repos = set(self.checkpoint_manager.get_processed_repos())
                logger.info(f"Resuming from checkpoint: {len(processed_repos)} repos already processed")
                # Note: We can't restore documents from checkpoint, so we'll start with empty list
                # but skip already processed repos
            else:
                if checkpoint:
                    logger.info("Checkpoint invalid or outdated, starting fresh")
                checkpoint = None

        # Create new checkpoint if not resuming or checkpoint is invalid
        if not checkpoint:
            self.checkpoint_manager.create(self.org_name, sample_size, total_repos)

        # Filter out unchanged repos (check commit SHA)
        repos_to_process = []
        changed_repos = []  # Track repos that have changes (not new)
        repos_skipped = 0
        for repo in repos:
            repo_full_name = repo['full_name']
            commit_sha = repo.get('commit_sha')

            # Check if repo needs processing
            if commit_sha and not self.checkpoint_manager.has_repo_changed(repo_full_name, commit_sha):
                logger.info(f"Skipping {repo_full_name} - no changes detected")
                repos_skipped += 1
            else:
                repos_to_process.append(repo)
                if commit_sha:
                    is_new = repo_full_name not in processed_repos
                    if not is_new:
                        changed_repos.append(repo_full_name)  # Existing repo with changes
                    logger.info(f"Will process {repo_full_name} - {'new repo' if is_new else 'changes detected'}")

        if repos_skipped > 0:
            logger.info(f"Skipped {repos_skipped} unchanged repos")
            logger.info(f"Processing {len(repos_to_process)} new or changed repos (including {len(changed_repos)} with updates)")

        # Process each repository sequentially
        for i, repo in enumerate(repos_to_process, 1):
            overall_index = total_repos - len(repos_to_process) + i
            logger.info(f"Indexing repo {overall_index}/{total_repos}: {repo['full_name']}")

            try:
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

                # Update checkpoint after successful processing
                doc_count = len(repo_contents) + 1  # +1 for repo_doc
                commit_sha = repo.get('commit_sha')
                self.checkpoint_manager.update(repo['full_name'], doc_count, commit_sha)
                logger.info(f"Checkpoint updated: {overall_index}/{total_repos} repos completed")

            except Exception as e:
                logger.error(f"Failed to process repo {repo['full_name']}: {e}")
                logger.info("Progress has been saved. You can resume by running the script again.")
                # Don't update checkpoint for this repo - it will be retried on resume
                raise

        # All repos processed successfully - keep checkpoint for incremental updates
        logger.info(f"Total documents indexed: {len(all_documents)}")
        logger.info(f"Repositories with changes: {len(changed_repos)}")
        logger.info("All repositories processed successfully! Checkpoint saved for incremental updates.")

        return all_documents, changed_repos
