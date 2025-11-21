# BHFDSC Repository Q&A System

A hybrid RAG and code intelligence system for the [BHF Data Science Centre (BHFDSC)](https://github.com/BHFDSC) GitHub organization. Combines semantic search with static code analysis to provide both conversational Q&A and precise organizational intelligence about your codebase.

## Features

### Core RAG Features
- **Hybrid Search**: Combines BM25 keyword matching with vector semantic search for optimal retrieval
- **Persistent Cloud Storage**: Uses Pinecone vector database - data persists across sessions
- **Comprehensive Indexing**: Indexes README files and code from all organization repositories
- **Intelligent Q&A**: Uses Claude AI with RAG to answer questions based on repository content
- **Source Attribution**: Shows which repositories and files were used to generate answers
- **Interactive UI**: Clean Streamlit interface
- **Multiple File Types**: Indexes Markdown, Python, R, Jupyter notebooks, and more

### Organizational Intelligence (NEW)
- **Static Code Analysis**: Extracts structured metadata from Python, R, and SQL code
- **Table Dependency Tracking**: Find which repositories use specific HDS curated assets
- **Function Usage Analysis**: Track usage of standardized functions across projects
- **Semantic Project Clustering**: Discover similar algorithms and duplicate work
- **Cross-Dataset Analysis**: Identify projects using multiple data sources
- **File Classification**: Automatically categorize as curation, analysis, phenotyping, etc.

## Architecture

### RAG Query Path
```
User Question
    ↓
Streamlit UI
    ↓
Hybrid Search (BM25 + Vector)
    ↓
Retrieved Context + Question → Anthropic Claude API
    ↓
Answer + Sources
```

### Organizational Intelligence Path
```
Code Files
    ↓
Static Code Analyzer
    ↓
Metadata Extraction (tables, functions, patterns)
    ↓
Structured Queries (get_table_usage, get_function_usage, find_similar_projects)
    ↓
Precise Organizational Insights
```

**Key Components**:
- **GitHub Indexer**: Fetches repositories and file contents
- **Code Analyzer**: Parses code to extract structured metadata
- **Hybrid Retriever**: Combines BM25 keyword and vector semantic search
- **Pinecone**: Cloud vector database for embeddings storage
- **Claude**: Anthropic's AI for question answering
- **Streamlit**: Web interface

## Prerequisites

- Python 3.9 or higher
- [Anthropic API key](https://console.anthropic.com/)
- [Pinecone API key](https://www.pinecone.io/)
- (Optional) GitHub Personal Access Token for higher rate limits

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd ccurag
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional
GITHUB_TOKEN=your_github_token
GITHUB_ORG=BHFDSC

# Pinecone Settings
PINECONE_INDEX_NAME=ccuindex
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Quick Start

### Option 1: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy the `app.py` file
4. Add your API keys in Streamlit secrets settings
5. Use the "Setup" page to index repositories

### Option 2: Run Locally

#### Step 1: Index the Repositories

Start the app and use the UI to index:

```bash
streamlit run app.py
```

Then go to the "Setup" page and click "Index Repositories".

**Note**: Initial indexing takes 10-30 minutes depending on the number of repositories.

#### Step 2: Ask Questions

Type your questions in the chat interface. Example questions:

- "What COVID-19 and cardiovascular research projects are in this organization?"
- "Which repositories contain Python code for data analysis?"
- "What phenotyping algorithms are used in the CCU projects?"
- "Show me repositories that work with linked electronic health records"

## Organizational Intelligence Queries

Beyond conversational Q&A, you can run precise structured queries to understand code dependencies and patterns.

### Step 1: Build Metadata Index

First, analyze all repositories to extract structured metadata:

```bash
python build_metadata_index.py
```

This will:
- Parse all Python, R, and SQL code files
- Extract table references (HDS curated assets)
- Extract function imports and calls
- Classify files by type (curation, analysis, phenotyping, etc.)
- Cache metadata for fast querying

**Note**: This takes 10-30 minutes and only needs to be run once (or when repos are updated).

### Step 2: Run Organizational Queries

Use the example script to run intelligence queries:

```bash
python example_analyzer_usage.py
```

This demonstrates three types of queries:

#### 1. Table Dependency Tracking

Find which repositories use specific HDS curated assets:

```python
from code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer()

# Which repos use the demographics table?
usage = analyzer.get_table_usage("hds_curated_assets__demographics")
print(f"Used in {usage['total_repos']} repositories")
print(f"Repos: {usage['repos']}")
```

**Tracked HDS Curated Assets**:
- Demographics: `date_of_birth`, `ethnicity`, `sex`, `lsoa`, `demographics` (individual & multisource)
- COVID: `covid_positive`
- Deaths: `deaths_single`, `deaths_cause_of_death`
- HES: `hes_apc_*` (diagnoses, procedures, episodes, spells)

#### 2. Function Usage Analysis

Track which projects use standardized HDS functions:

```python
# Find all hds_functions usage
usage = analyzer.get_function_usage("hds")
print(f"Found {usage['total_functions_found']} unique HDS functions")

# See which repos call each function
for func_name, data in usage['functions'].items():
    print(f"{func_name}: {data['total_repos']} repos")
```

#### 3. Semantic Project Clustering

Discover similar algorithms and potential duplicate work:

```python
from hybrid_retriever import HybridRetriever

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever(vector_store)
hybrid_retriever.build_bm25_index(documents)

# Find similar projects
results = analyzer.find_similar_projects(
    query="smoking algorithm",
    hybrid_retriever=hybrid_retriever,
    k=10
)

for project in results:
    print(f"{project['repo']}: {project['relevance_score']} matching files")
    print(f"  Tables used: {project['tables_used']}")
    print(f"  File types: {project['file_types']}")
```

### Example Queries

**Cross-dataset analysis**:
```python
# Which projects use both COVID and deaths data?
covid_repos = set(analyzer.get_table_usage("hds_curated_assets__covid_positive")['repos'])
deaths_repos = set(analyzer.get_table_usage("hds_curated_assets__deaths_single")['repos'])
both = covid_repos & deaths_repos
```

**Find potential standardization opportunities**:
```python
# Find multiple teams building similar algorithms
smoking_projects = analyzer.find_similar_projects("smoking algorithm")
diabetes_projects = analyzer.find_similar_projects("diabetes algorithm")
```

**Audit data source usage**:
```python
# See all HES data usage across the organization
hes_diagnosis = analyzer.get_table_usage("hds_curated_assets__hes_apc_diagnosis")
hes_procedure = analyzer.get_table_usage("hds_curated_assets__hes_apc_procedure")
```

## Configuration

Key settings in `config.py`:

- `ANTHROPIC_MODEL`: Claude model to use (default: claude-opus-4-1)
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `CHUNK_SIZE`: Document chunk size (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `MAX_FILES_PER_REPO`: Maximum files to index per repository (default: 50)
- `INDEXED_FILE_EXTENSIONS`: File types to index

## Project Structure

```
ccurag/
├── app.py                       # Streamlit application
├── config.py                    # Configuration settings
├── github_indexer.py            # Repository fetching and indexing
├── vector_store.py              # Vector store interface
├── vector_store_pinecone.py     # Pinecone implementation
├── qa_chain.py                  # Question-answering system
├── hybrid_retriever.py          # Hybrid BM25 + vector search
├── code_analyzer.py             # Static code analysis & metadata extraction
├── build_hybrid_index.py        # Build hybrid search index
├── build_metadata_index.py      # Build code metadata index
├── example_analyzer_usage.py    # Example organizational queries
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## How It Works

### Indexing Phase
1. Fetches all repositories from the BHFDSC GitHub organization
2. Downloads README files and code files (Python, R, Markdown, SQL, etc.)
3. Splits documents into chunks
4. Creates embeddings using Sentence Transformers
5. Stores embeddings in Pinecone
6. **Builds BM25 index** for keyword matching (cached locally)

### Code Analysis Phase (NEW)
1. Parses Python/R code with AST and regex
2. Extracts SQL table references from queries
3. Identifies function imports and calls (especially hds_functions)
4. Classifies files by type (curation, analysis, phenotyping, etc.)
5. Builds reverse indices for fast lookup (table→repos, function→repos)
6. Caches metadata as JSON for instant querying

### RAG Query Phase
1. User asks a question via Streamlit UI
2. **Hybrid search**: Question processed by both BM25 and vector search
3. Scores combined with adaptive weighting (code queries favor BM25, conceptual queries favor vector)
4. Top-20 most relevant document chunks retrieved
5. Context + question sent to Claude API
6. Claude generates answer based on context
7. Answer and sources displayed to user

### Structured Query Phase (NEW)
1. Load cached metadata from code analysis
2. Run precise queries (table usage, function usage, similar projects)
3. Return exact counts, file locations, and cross-references
4. No LLM needed - instant, deterministic results

## Troubleshooting

### "ANTHROPIC_API_KEY is required"

Make sure you've created a `.env` file with your API keys.

### "PINECONE_API_KEY is required"

Add your Pinecone API key to the `.env` file.

### GitHub Rate Limiting

Add a GitHub Personal Access Token to your `.env` file:

```env
GITHUB_TOKEN=your_github_token
```

## Technologies Used

- **Anthropic Claude**: AI language model
- **Pinecone**: Cloud vector database
- **Sentence Transformers**: Text embeddings
- **Streamlit**: Web application framework
- **PyGithub**: GitHub API library

## About BHFDSC

The BHF Data Science Centre focuses on improving cardiovascular health through large-scale data analysis. Their repositories contain research code and analysis pipelines for studying COVID-19's impact on cardiovascular disease.

## License

This project is for educational and research purposes. Please respect the licenses of the BHFDSC repositories.

## Contributing

Contributions are welcome! Please submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.
