# BHFDSC Repository Q&A System

A Retrieval-Augmented Generation (RAG) application that enables users to ask questions about all repositories in the [BHF Data Science Centre (BHFDSC)](https://github.com/BHFDSC) GitHub organization. The system uses Anthropic's Claude AI, LangChain, and Streamlit to provide intelligent answers based on the organization's ~100 repositories focused on cardiovascular health research.

## Features

- **Comprehensive Repository Indexing**: Automatically fetches and indexes all repositories from the BHFDSC organization
- **Intelligent Q&A**: Uses Claude AI with RAG to answer questions based on actual repository content
- **Source Attribution**: Shows which repositories and files were used to generate each answer
- **Interactive UI**: Simple Streamlit interface for asking questions
- **Supports Multiple File Types**: Indexes Markdown, Python, R, Jupyter notebooks, and more

## Architecture

```
User Question
    ↓
Streamlit UI
    ↓
LangChain RAG Pipeline
    ↓
ChromaDB Vector Store ←→ Anthropic Claude API
    ↓
Answer + Sources
```

## Prerequisites

- Python 3.9 or higher
- Anthropic API key
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

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GITHUB_TOKEN=your_github_token_here  # Optional
GITHUB_ORG=BHFDSC
```

## Usage

### Step 1: Index the Repositories

Before running the application, you need to index all repositories from the BHFDSC organization:

```bash
python index_repos.py
```

This will:
- Fetch all repositories from the BHFDSC organization
- Download README files and other relevant files
- Create embeddings and store them in ChromaDB
- Save the vector store to the `chroma_db/` directory

**Note**: This process may take 10-30 minutes depending on the number of repositories and your internet connection.

### Step 2: Run the Streamlit App

Once indexing is complete, start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Step 3: Ask Questions

Type your questions in the chat interface. Example questions:

- "What COVID-19 and cardiovascular research projects are in this organization?"
- "Which repositories contain Python code for data analysis?"
- "What phenotyping algorithms are used in the CCU projects?"
- "Show me repositories that work with linked electronic health records"
- "What machine learning methods are used in these projects?"

## Configuration

You can customize the application by modifying `config.py`:

- `ANTHROPIC_MODEL`: Change the Claude model (default: claude-3-5-sonnet-20241022)
- `CHUNK_SIZE`: Adjust document chunk size for embeddings
- `MAX_FILES_PER_REPO`: Limit files indexed per repository
- `INDEXED_FILE_EXTENSIONS`: Add or remove file types to index

## Project Structure

```
ccurag/
├── app.py                 # Streamlit application
├── config.py             # Configuration management
├── github_indexer.py     # GitHub repository fetching
├── vector_store.py       # Vector store management
├── qa_chain.py           # QA system with LangChain
├── index_repos.py        # Script to index repositories
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
├── .gitignore           # Git ignore file
├── README.md            # This file
└── chroma_db/           # Vector store (generated)
```

## How It Works

1. **Indexing Phase**:
   - Fetches all repositories from the BHFDSC GitHub organization
   - Downloads README files and selected file types (Python, R, Markdown, etc.)
   - Splits documents into chunks
   - Creates embeddings using Sentence Transformers
   - Stores embeddings in ChromaDB

2. **Query Phase**:
   - User asks a question via Streamlit UI
   - Question is embedded and used to search the vector store
   - Top-k most relevant document chunks are retrieved
   - Retrieved context + question are sent to Claude API
   - Claude generates an answer based on the context
   - Answer and sources are displayed to the user

## Troubleshooting

### Error: "ANTHROPIC_API_KEY is required"

Make sure you've created a `.env` file with your Anthropic API key.

### Error: "Vector store not found"

Run `python index_repos.py` first to index the repositories.

### GitHub Rate Limiting

If you encounter rate limiting errors, add a GitHub Personal Access Token to your `.env` file:

```env
GITHUB_TOKEN=your_github_token_here
```

### Slow Indexing

The indexing process can take time. You can reduce `MAX_FILES_PER_REPO` in `config.py` to index fewer files per repository.

## Technologies Used

- **Anthropic Claude**: Advanced AI language model
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for embeddings
- **Streamlit**: Web application framework
- **PyGithub**: GitHub API library
- **Sentence Transformers**: Embedding models

## About BHFDSC

The BHF Data Science Centre (BHFDSC) focuses on improving cardiovascular health through large-scale data analysis. Their repositories contain research code and analysis pipelines for studying COVID-19's impact on cardiovascular disease using linked electronic health records.

## License

This project is for educational and research purposes. Please respect the licenses of the BHFDSC repositories when using information from them.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository.
