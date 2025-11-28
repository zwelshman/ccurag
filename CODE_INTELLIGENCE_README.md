# BHFDSC Code Intelligence Dashboard

A dedicated Streamlit application for analyzing the BHFDSC codebase through static code analysis.

## Overview

The Code Intelligence Dashboard provides organizational insights into the BHFDSC GitHub repositories by:
- Tracking HDS curated asset table usage across projects
- Identifying function dependencies and standardized code usage
- Analyzing module imports and library usage
- Finding repositories that use multiple data sources together

## Features

### 📊 Dashboard
- Overview statistics of repositories and files analyzed
- Visualization of most-used HDS curated assets
- Quick metrics on table, function, and module usage

### 📁 Table Usage Tracker
- Search for specific HDS curated asset tables
- Find which repositories use specific tables
- View files by type (Python, R, SQL) that reference tables
- Categorized views: Demographics, COVID, Deaths, HES Assets

### ⚙️ Function Usage Tracker
- Search for function patterns (e.g., "hds", "curate", "phenotype")
- Track standardized function usage across repositories
- Identify code reuse and common patterns

### 📦 Module Usage Tracker
- Track Python module imports (e.g., `hds_functions`, `pandas`)
- Find repositories using specific libraries
- Analyze import patterns across the organization

### 🔗 Cross-Analysis
- Find repositories using multiple data sources together
- Predefined analyses (e.g., COVID + Deaths)
- Custom multi-table analysis

## Getting Started

### Prerequisites

1. Python environment with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Environment variables configured (see `.env`):
   - `GITHUB_TOKEN`: Personal access token for GitHub API
   - Other configuration in `config.py`

### Running the App

```bash
streamlit run code_intelligence_app.py
```

The app will be available at `http://localhost:8501`

### First-Time Setup

1. **Build Metadata Index**: On first run, use the "Build Metadata Index" button in the configuration panel
   - This process analyzes all Python, R, and SQL files from BHFDSC repositories
   - Takes approximately 10-30 minutes depending on repository size
   - Creates `code_metadata.json` with extracted information

2. **Metadata Storage**:
   - Runtime cache: `.cache/code_metadata.json` (not committed)
   - Committed storage: `data_index/code_metadata.json` (committed to Git)
   - The app checks both locations automatically

## How It Works

### Static Code Analysis

The Code Intelligence system uses Abstract Syntax Tree (AST) parsing to extract structured metadata from code files:

1. **Python Analysis**: Uses Python's `ast` module to parse `.py` files
   - Extracts function calls, imports, table references
   - Identifies SQL queries within Python code

2. **R Analysis**: Custom parser for `.R` files
   - Extracts function calls and library imports
   - Identifies table references in dbplyr/dplyr chains

3. **SQL Analysis**: Regex-based parsing of `.sql` files
   - Extracts FROM/JOIN table references
   - Tracks database operations

### Tracked Assets

The system specifically tracks HDS curated assets:
- Demographics tables
- COVID data tables
- Deaths and mortality tables
- HES (Hospital Episode Statistics) tables
- Custom table patterns

See `code_analyzer.py` for the full list of tracked tables.

## Architecture

```
code_intelligence_app.py
    ↓
load_code_analyzer() [cached]
    ↓
CodeAnalyzer (code_analyzer.py)
    ↓
Load metadata from cache
    ↓
.cache/code_metadata.json OR data_index/code_metadata.json
```

### Key Components

- **code_intelligence_app.py**: Main Streamlit application
- **code_analyzer.py**: Core analysis engine with AST parsing
- **build_metadata_index.py**: Script to build/rebuild metadata index
- **config.py**: Configuration settings

## Differences from Main App

This dedicated app differs from `app.py` in several ways:

1. **Focused Scope**: Only code intelligence features (no Q&A/RAG system)
2. **Simplified UI**: Cleaner interface without vector store management
3. **Faster Loading**: No LLM or vector database initialization
4. **Independent**: Can run without Pinecone or Anthropic API keys

## Use Cases

### For Data Scientists
- "Which projects have analyzed COVID + diabetes together?"
- "Where is the demographics table used?"
- "Find all uses of the standardized HDS functions"

### For Team Leads
- Track adoption of standardized curated assets
- Identify code duplication opportunities
- Understand cross-project dependencies

### For New Team Members
- Discover existing code patterns
- Find similar projects to learn from
- Understand organizational data structure

## Performance

- **Metadata Building**: 10-30 minutes (one-time, can be committed to Git)
- **Query Time**: <100ms for most lookups (no API calls, pure dictionary lookups)
- **Memory Usage**: ~50-100MB for metadata cache
- **Disk Usage**: ~5-20MB for metadata JSON file

## Future Enhancements

Potential improvements to consider:
- Call graph analysis for function dependencies
- Data lineage tracking through pipelines
- Temporal analysis of code evolution
- Integration with CI/CD for automatic updates
- API endpoints for programmatic access

## Troubleshooting

### Metadata index not building
- Check GitHub token has correct permissions
- Verify network access to GitHub API
- Check logs for specific parsing errors

### Missing tables/functions
- Verify the pattern is in `TRACKED_TABLES` or being searched correctly
- Check if metadata index was built recently
- Try rebuilding metadata index with force_rebuild=True

### Slow performance
- Check if metadata is loaded from cache (should be instant)
- Verify cache files exist in `.cache/` or `data_index/`
- Consider reducing scope of analysis if dataset is very large

## Contributing

To add new tracked assets:
1. Update `TRACKED_TABLES` in `code_analyzer.py`
2. Rebuild metadata index
3. Test queries in the app

## License

See main repository LICENSE file.
