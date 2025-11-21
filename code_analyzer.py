"""Code analyzer for extracting structured metadata from repositories.

Provides organizational intelligence about codebase by parsing:
- SQL table references (HDS curated assets)
- Function imports and usage (hds_functions)
- Semantic clustering of similar projects
"""

import re
import ast
import logging
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeMetadata:
    """Metadata extracted from a code file."""
    file_path: str
    repo: str
    language: str

    # Table usage
    tables_used: Set[str] = field(default_factory=set)

    # Function usage
    imports: Set[str] = field(default_factory=set)
    function_calls: Set[str] = field(default_factory=set)

    # File classification
    file_type: str = "unknown"  # 'curation', 'analysis', 'notebook', etc.

    # Raw content for semantic search
    content: str = ""


class CodeAnalyzer:
    """Analyzes code repositories to extract structured metadata."""

    # HDS Curated Assets to track
    TRACKED_TABLES = {
        # Person-level demographics
        "hds_curated_assets__date_of_birth_individual",
        "hds_curated_assets__date_of_birth_multisource",
        "hds_curated_assets__ethnicity_individual",
        "hds_curated_assets__ethnicity_multisource",
        "hds_curated_assets__sex_individual",
        "hds_curated_assets__sex_multisource",
        "hds_curated_assets__lsoa_individual",
        "hds_curated_assets__lsoa_multisource",
        "hds_curated_assets__demographics",

        # COVID
        "hds_curated_assets__covid_positive",

        # Deaths
        "hds_curated_assets__deaths_single",
        "hds_curated_assets__deaths_cause_of_death",

        # HES assets
        "hds_curated_assets__hes_apc_cips_cips",
        "hds_curated_assets__hes_apc_cips_episodes",
        "hds_curated_assets__hes_apc_cips_provider_spells",
        "hds_curated_assets__hes_apc_diagnosis",
        "hds_curated_assets__hes_apc_procedure",
        "hds_curated_assets__hes_apc_provider_spells",
    }

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize code analyzer.

        Args:
            cache_dir: Directory to store metadata cache
        """
        self.cache_dir = cache_dir
        self.metadata_cache_file = os.path.join(cache_dir, "code_metadata.json")

        # Metadata storage: repo -> file -> CodeMetadata
        self.metadata: Dict[str, Dict[str, CodeMetadata]] = defaultdict(dict)

        # Reverse indices for fast lookup
        self.table_to_repos: Dict[str, Set[str]] = defaultdict(set)
        self.table_to_files: Dict[str, List[Dict]] = defaultdict(list)
        self.function_to_repos: Dict[str, Set[str]] = defaultdict(set)
        self.function_to_files: Dict[str, List[Dict]] = defaultdict(list)

        # Load cached metadata if available
        self._load_cache()

    def analyze_file(self, content: str, file_path: str, repo: str) -> CodeMetadata:
        """Analyze a single file and extract metadata.

        Args:
            content: File content
            file_path: Path to file in repo
            repo: Repository name

        Returns:
            CodeMetadata object with extracted information
        """
        # Determine language
        language = self._detect_language(file_path)

        metadata = CodeMetadata(
            file_path=file_path,
            repo=repo,
            language=language,
            content=content
        )

        # Classify file type
        metadata.file_type = self._classify_file(file_path, content)

        # Extract table references
        metadata.tables_used = self._extract_table_references(content, language)

        # Extract imports and function calls
        if language == "python":
            metadata.imports, metadata.function_calls = self._parse_python_code(content)
        elif language in ["r", "rmarkdown"]:
            metadata.imports, metadata.function_calls = self._parse_r_code(content)

        return metadata

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".py", ".ipynb"]:
            return "python"
        elif ext in [".r", ".rmd"]:
            return "r"
        elif ext == ".sql":
            return "sql"
        elif ext in [".md", ".markdown"]:
            return "markdown"
        else:
            return "unknown"

    def _classify_file(self, file_path: str, content: str) -> str:
        """Classify file as curation, analysis, notebook, etc."""
        path_lower = file_path.lower()
        content_lower = content.lower()

        # Check path patterns
        if "curation" in path_lower or "curate" in path_lower:
            return "curation"
        elif "analysis" in path_lower or "analyse" in path_lower or "analyze" in path_lower:
            return "analysis"
        elif file_path.endswith(".ipynb"):
            return "notebook"
        elif "phenotype" in path_lower or "phenotyping" in path_lower:
            return "phenotyping"
        elif "algorithm" in path_lower:
            return "algorithm"

        # Check content patterns
        if "create table" in content_lower or "create or replace" in content_lower:
            return "curation"
        elif "select" in content_lower and len(content_lower.split("select")) > 5:
            return "analysis"

        return "unknown"

    def _extract_table_references(self, content: str, language: str) -> Set[str]:
        """Extract table references from code.

        Looks for:
        - SQL: FROM table_name, JOIN table_name
        - Python: spark.table("table_name"), df = read_table("table_name")
        - R: dbGetQuery patterns
        """
        tables = set()

        # Pattern 1: SQL FROM/JOIN clauses
        # FROM table_name, FROM db.table_name, JOIN table_name
        sql_patterns = [
            r'\bFROM\s+[\w.]+\b',
            r'\bJOIN\s+[\w.]+\b',
            r'\bINTO\s+[\w.]+\b',
            r'\bUPDATE\s+[\w.]+\b',
            r'\bTABLE\s+[\w.]+\b',
        ]

        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract table name (last word)
                table_ref = match.group().split()[-1]
                # Remove quotes, backticks, semicolons
                table_ref = table_ref.strip('`"\';')
                tables.add(table_ref)

        # Pattern 2: Spark/Python table references
        # spark.table("table_name"), spark.sql("... FROM table_name ...")
        python_table_patterns = [
            r'spark\.table\(["\']([^"\']+)["\']\)',
            r'spark\.read\.table\(["\']([^"\']+)["\']\)',
            r'read_table\(["\']([^"\']+)["\']\)',
            r'sqlContext\.table\(["\']([^"\']+)["\']\)',
        ]

        for pattern in python_table_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                tables.add(match.group(1))

        # Pattern 3: R database query patterns
        # dbGetQuery(conn, "SELECT * FROM table_name")
        r_patterns = [
            r'dbGetQuery\([^)]+["\']SELECT[^"\']*FROM\s+([\w.]+)',
            r'dbReadTable\([^,]+,\s*["\']([^"\']+)["\']\)',
        ]

        for pattern in r_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                tables.add(match.group(1))

        # Filter to only tracked tables (or tables that look like HDS assets)
        filtered_tables = set()
        for table in tables:
            # Remove database prefixes (dars_nic_391419_j3w9t_collab.table -> table)
            table_name = table.split('.')[-1]

            # Check if it's a tracked table or looks like an HDS asset
            if table_name in self.TRACKED_TABLES:
                filtered_tables.add(table_name)
            elif table_name.startswith("hds_curated_assets__"):
                filtered_tables.add(table_name)
            elif table_name.startswith("hds_"):
                filtered_tables.add(table_name)

        return filtered_tables

    def _parse_python_code(self, content: str) -> tuple[Set[str], Set[str]]:
        """Parse Python code to extract imports and function calls.

        Returns:
            Tuple of (imports, function_calls)
        """
        imports = set()
        function_calls = set()

        try:
            # Parse AST
            tree = ast.parse(content)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_import = f"{module}.{alias.name}" if module else alias.name
                        imports.add(full_import)

                # Extract function calls (especially hds_functions)
                elif isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)
                    if func_name and "hds" in func_name.lower():
                        function_calls.add(func_name)

        except SyntaxError:
            # If AST parsing fails, use regex fallback
            import_patterns = [
                r'^import\s+([\w.]+)',
                r'^from\s+([\w.]+)\s+import',
            ]

            for pattern in import_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    imports.add(match.group(1))

            # Look for hds function calls
            hds_func_pattern = r'\b(hds[\w.]*)\s*\('
            matches = re.finditer(hds_func_pattern, content, re.IGNORECASE)
            for match in matches:
                function_calls.add(match.group(1))

        return imports, function_calls

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from AST Call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For calls like obj.method() or module.function()
            value_name = self._get_function_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        return None

    def _parse_r_code(self, content: str) -> tuple[Set[str], Set[str]]:
        """Parse R code to extract imports and function calls.

        Returns:
            Tuple of (imports, function_calls)
        """
        imports = set()
        function_calls = set()

        # Extract library/require calls
        import_patterns = [
            r'library\([\s"\']*(\w+)[\s"\']*\)',
            r'require\([\s"\']*(\w+)[\s"\']*\)',
            r'library\("([^"]+)"\)',
            r'source\("([^"]+)"\)',
        ]

        for pattern in import_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                imports.add(match.group(1))

        # Look for hds function calls (R syntax: hds_function() or hds::function())
        hds_func_patterns = [
            r'\b(hds[\w._]*)\s*\(',
            r'\bhds::([\w.]+)\s*\(',
        ]

        for pattern in hds_func_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                func_name = match.group(1)
                if "::" in pattern:
                    func_name = f"hds::{func_name}"
                function_calls.add(func_name)

        return imports, function_calls

    def index_documents(self, documents: List[Dict], force_rebuild: bool = False):
        """Index documents and build metadata.

        Args:
            documents: List of document dicts with 'content' and 'metadata'
            force_rebuild: Force rebuild of index
        """
        if not force_rebuild and os.path.exists(self.metadata_cache_file):
            logger.info("Metadata already cached, skipping indexing")
            return

        logger.info(f"Analyzing {len(documents)} documents...")

        for i, doc in enumerate(documents):
            if i % 50 == 0:
                logger.info(f"  Processed {i}/{len(documents)} documents...")

            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            file_path = metadata.get("source", metadata.get("path", "unknown"))
            repo = metadata.get("repo", "unknown")

            # Skip non-code files
            if metadata.get("type") == "repo_info":
                continue

            # Analyze file
            code_meta = self.analyze_file(content, file_path, repo)

            # Store in metadata dict
            self.metadata[repo][file_path] = code_meta

            # Update reverse indices
            for table in code_meta.tables_used:
                self.table_to_repos[table].add(repo)
                self.table_to_files[table].append({
                    "repo": repo,
                    "file": file_path,
                    "type": code_meta.file_type
                })

            for func in code_meta.function_calls:
                self.function_to_repos[func].add(repo)
                self.function_to_files[func].append({
                    "repo": repo,
                    "file": file_path,
                    "type": code_meta.file_type
                })

        logger.info(f"✓ Analyzed {len(documents)} documents")
        logger.info(f"  Found {len(self.table_to_repos)} unique tables referenced")
        logger.info(f"  Found {len(self.function_to_repos)} unique functions called")

        # Save cache
        self._save_cache()

    def get_table_usage(self, table_name: str) -> Dict[str, Any]:
        """Get usage information for a specific table.

        Args:
            table_name: Name of the table (e.g., "hds_curated_assets__demographics")

        Returns:
            Dictionary with usage statistics and file references
        """
        repos = self.table_to_repos.get(table_name, set())
        files = self.table_to_files.get(table_name, [])

        # Group by file type
        by_type = defaultdict(list)
        for file_info in files:
            by_type[file_info["type"]].append(file_info)

        return {
            "table": table_name,
            "total_repos": len(repos),
            "repos": sorted(list(repos)),
            "total_files": len(files),
            "files_by_type": dict(by_type),
            "all_files": files
        }

    def get_function_usage(self, function_pattern: str = "hds") -> Dict[str, Any]:
        """Get usage information for functions matching a pattern.

        Args:
            function_pattern: Pattern to match (e.g., "hds" for all hds functions)

        Returns:
            Dictionary with function usage statistics
        """
        matching_functions = {}

        for func, repos in self.function_to_repos.items():
            if function_pattern.lower() in func.lower():
                files = self.function_to_files.get(func, [])

                # Group by file type
                by_type = defaultdict(list)
                for file_info in files:
                    by_type[file_info["type"]].append(file_info)

                matching_functions[func] = {
                    "total_repos": len(repos),
                    "repos": sorted(list(repos)),
                    "total_files": len(files),
                    "files_by_type": dict(by_type),
                    "all_files": files
                }

        return {
            "pattern": function_pattern,
            "total_functions_found": len(matching_functions),
            "functions": matching_functions
        }

    def find_similar_projects(self, query: str,
                             hybrid_retriever=None,
                             k: int = 10) -> List[Dict[str, Any]]:
        """Find projects similar to a query using semantic search.

        Args:
            query: Search query (e.g., "smoking algorithm", "diabetes algorithm")
            hybrid_retriever: HybridRetriever instance for semantic search
            k: Number of results to return

        Returns:
            List of similar projects with metadata
        """
        if hybrid_retriever is None:
            logger.warning("No hybrid retriever provided, returning empty results")
            return []

        # Use hybrid search to find similar documents
        results = hybrid_retriever.similarity_search(query, k=k)

        # Group by repository and aggregate
        repo_matches = defaultdict(lambda: {
            "repo": "",
            "matched_files": [],
            "relevance_score": 0.0,
            "file_types": set(),
            "tables_used": set(),
            "functions_used": set()
        })

        for doc in results:
            repo = doc.metadata.get("repo", "unknown")
            file_path = doc.metadata.get("source", doc.metadata.get("path", "unknown"))

            # Get metadata for this file
            if repo in self.metadata and file_path in self.metadata[repo]:
                code_meta = self.metadata[repo][file_path]

                repo_matches[repo]["repo"] = repo
                repo_matches[repo]["matched_files"].append({
                    "file": file_path,
                    "type": code_meta.file_type,
                    "snippet": doc.page_content[:200] + "..."
                })
                repo_matches[repo]["relevance_score"] += 1.0  # Could use actual scores
                repo_matches[repo]["file_types"].add(code_meta.file_type)
                repo_matches[repo]["tables_used"].update(code_meta.tables_used)
                repo_matches[repo]["functions_used"].update(code_meta.function_calls)

        # Convert sets to lists for JSON serialization
        results_list = []
        for repo_data in repo_matches.values():
            repo_data["file_types"] = list(repo_data["file_types"])
            repo_data["tables_used"] = list(repo_data["tables_used"])
            repo_data["functions_used"] = list(repo_data["functions_used"])
            results_list.append(repo_data)

        # Sort by relevance
        results_list.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results_list

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the analyzed codebase."""
        total_repos = len(self.metadata)
        total_files = sum(len(files) for files in self.metadata.values())

        # Count file types
        file_types = defaultdict(int)
        for repo_files in self.metadata.values():
            for code_meta in repo_files.values():
                file_types[code_meta.file_type] += 1

        # Count tracked tables that were found
        tracked_tables_found = {
            table: len(self.table_to_repos[table])
            for table in self.TRACKED_TABLES
            if table in self.table_to_repos
        }

        return {
            "total_repos": total_repos,
            "total_files": total_files,
            "file_types": dict(file_types),
            "total_unique_tables": len(self.table_to_repos),
            "total_unique_functions": len(self.function_to_repos),
            "tracked_tables_found": tracked_tables_found,
            "tracked_tables_count": len(tracked_tables_found)
        }

    def _save_cache(self):
        """Save metadata to cache file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # Convert metadata to serializable format
            serializable_data = {
                "metadata": {
                    repo: {
                        file_path: {
                            "file_path": meta.file_path,
                            "repo": meta.repo,
                            "language": meta.language,
                            "tables_used": list(meta.tables_used),
                            "imports": list(meta.imports),
                            "function_calls": list(meta.function_calls),
                            "file_type": meta.file_type,
                        }
                        for file_path, meta in files.items()
                    }
                    for repo, files in self.metadata.items()
                },
                "table_to_repos": {
                    table: list(repos) for table, repos in self.table_to_repos.items()
                },
                "table_to_files": dict(self.table_to_files),
                "function_to_repos": {
                    func: list(repos) for func, repos in self.function_to_repos.items()
                },
                "function_to_files": dict(self.function_to_files),
            }

            with open(self.metadata_cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"✓ Saved metadata cache to {self.metadata_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")

    def _load_cache(self):
        """Load metadata from cache file."""
        if not os.path.exists(self.metadata_cache_file):
            return

        try:
            logger.info("Loading metadata from cache...")
            with open(self.metadata_cache_file, 'r') as f:
                data = json.load(f)

            # Restore metadata
            self.metadata = defaultdict(dict)
            for repo, files in data["metadata"].items():
                for file_path, meta_dict in files.items():
                    meta = CodeMetadata(
                        file_path=meta_dict["file_path"],
                        repo=meta_dict["repo"],
                        language=meta_dict["language"],
                        tables_used=set(meta_dict["tables_used"]),
                        imports=set(meta_dict["imports"]),
                        function_calls=set(meta_dict["function_calls"]),
                        file_type=meta_dict["file_type"],
                    )
                    self.metadata[repo][file_path] = meta

            # Restore reverse indices
            self.table_to_repos = defaultdict(set, {
                table: set(repos) for table, repos in data["table_to_repos"].items()
            })
            self.table_to_files = defaultdict(list, data["table_to_files"])
            self.function_to_repos = defaultdict(set, {
                func: set(repos) for func, repos in data["function_to_repos"].items()
            })
            self.function_to_files = defaultdict(list, data["function_to_files"])

            logger.info(f"✓ Loaded metadata for {len(self.metadata)} repos from cache")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")

    def clear_cache(self):
        """Clear the metadata cache."""
        if os.path.exists(self.metadata_cache_file):
            os.remove(self.metadata_cache_file)
            logger.info("✓ Metadata cache cleared")

        # Reset in-memory data
        self.metadata = defaultdict(dict)
        self.table_to_repos = defaultdict(set)
        self.table_to_files = defaultdict(list)
        self.function_to_repos = defaultdict(set)
        self.function_to_files = defaultdict(list)
