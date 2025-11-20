# BHFDSC Repository Q&A System

A Retrieval-Augmented Generation (RAG) application that enables users to ask questions about all repositories in the [BHF Data Science Centre (BHFDSC)](https://github.com/BHFDSC) GitHub organization. The system uses Anthropic's Claude AI and Streamlit to provide intelligent answers based on the organization's ~100 repositories focused on cardiovascular health research.

## Features

- **Persistent Cloud Storage**: Uses Pinecone vector database - data persists across app restarts
- **Comprehensive Repository Indexing**: Automatically fetches and indexes all repositories from the BHFDSC organization
- **Intelligent Q&A**: Uses Claude AI with RAG to answer questions based on actual repository content
- **Source Attribution**: Shows which repositories and files were used to generate each answer
- **Interactive UI**: Simple Streamlit interface for asking questions
- **Supports Multiple File Types**: Indexes Markdown, Python, R, Jupyter notebooks, and more
- **Easy Deployment**: Works both locally and on Streamlit Cloud
- **Built-in Setup Page**: Index repositories directly from the web interface
- **Flexible Backend**: Switch between Pinecone (cloud) and ChromaDB (local) via configuration

## Architecture

```
User Question
    ↓
Streamlit UI
    ↓
Similarity Search (Pinecone Cloud)
    ↓
Retrieved Context + Question → Anthropic Claude API
    ↓
Answer + Sources
```

**Vector Database**: Pinecone (default) provides persistent cloud storage. Can switch to ChromaDB for local development.

## Prerequisites

- Python 3.9 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- Pinecone API key ([Get one here](https://www.pinecone.io/))
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
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional
GITHUB_TOKEN=your_github_token_here
GITHUB_ORG=BHFDSC

# Vector Store (default: pinecone)
VECTOR_STORE_BACKEND=pinecone
PINECONE_INDEX_NAME=ccuindex
```

📖 **See [PINECONE_SETUP.md](PINECONE_SETUP.md) for detailed Pinecone configuration guide**

## Quick Start

### Option 1: Deploy to Streamlit Cloud (Recommended for Quick Testing)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy the `app.py` file
4. Add your `ANTHROPIC_API_KEY` in the Streamlit secrets settings
5. Use the built-in "Setup" page to index repositories

📖 See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Option 2: Run Locally

#### Step 1: Index the Repositories

You can index repositories either via the UI or command line:

**Via UI (Recommended):**
1. Start the app: `streamlit run app.py`
2. Go to the "Setup" page
3. Click "Index/Re-index Repositories"

**Via Command Line:**
```bash
python index_repos.py
```

This process will:
- Fetch all repositories from the BHFDSC organization (~100 repos)
- Download README files and other relevant files
- Create embeddings and store them in your configured vector store (Pinecone or ChromaDB)
- Track progress with checkpoint/resume capability

**Note**: Indexing takes 10-30 minutes depending on your internet connection and vector store backend.

#### Step 2: Run the Streamlit App

Once indexing is complete, start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

#### Step 3: Ask Questions

Type your questions in the chat interface. Example questions:

- "What COVID-19 and cardiovascular research projects are in this organization?"
- "Which repositories contain Python code for data analysis?"
- "What phenotyping algorithms are used in the CCU projects?"
- "Show me repositories that work with linked electronic health records"
- "What machine learning methods are used in these projects?"

## Deployment to Streamlit Cloud

The application is designed to work seamlessly on Streamlit Cloud with built-in UI for indexing repositories.

**Recommended:** Use Pinecone (default) for production deployments. Pinecone is a cloud vector database that provides:
- Persistent storage across app restarts
- Fast similarity search
- Free tier for testing (100k vectors)
- Simple setup and configuration

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions and [PINECONE_SETUP.md](PINECONE_SETUP.md) for Pinecone configuration.

**Alternative:** ChromaDB can be used for local development but requires re-indexing on each Streamlit Cloud restart.

## Configuration

You can customize the application by modifying `config.py`:

- `ANTHROPIC_MODEL`: Change the Claude model (default: claude-haiku-4-5-20251001)
- `CHUNK_SIZE`: Adjust document chunk size for embeddings
- `MAX_FILES_PER_REPO`: Limit files indexed per repository
- `INDEXED_FILE_EXTENSIONS`: Add or remove file types to index
- `VECTOR_STORE_BACKEND`: Switch between "pinecone" (default) or "chroma"

## Project Structure

```
ccurag/
├── app.py                      # Streamlit application
├── config.py                   # Configuration management
├── utils.py                    # Shared utilities (Document class, text splitting)
├── github_indexer.py           # GitHub repository fetching with checkpoint/resume
├── vector_store.py             # Vector store factory
├── vector_store_pinecone.py    # Pinecone vector store implementation
├── vector_store_chroma.py      # ChromaDB vector store implementation (legacy)
├── qa_chain.py                 # QA system with Anthropic Claude
├── index_repos.py              # CLI script to index repositories
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore file
├── README.md                   # This file
├── DEPLOYMENT.md               # Deployment guide
├── PINECONE_SETUP.md           # Pinecone setup guide
└── chroma_db/                  # Local vector store (if using ChromaDB)
```

## How It Works

1. **Indexing Phase**:
   - Fetches all repositories from the BHFDSC GitHub organization
   - Downloads README files and selected file types (Python, R, Markdown, etc.)
   - Splits documents into chunks (configurable size and overlap)
   - Creates embeddings using Sentence Transformers
   - Stores embeddings in Pinecone (cloud) or ChromaDB (local)
   - Supports checkpoint/resume for interrupted indexing

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

- **Anthropic Claude**: Advanced AI language model (via Anthropic Python SDK)
- **Pinecone**: Cloud vector database for embeddings (default)
- **ChromaDB**: Local vector database option
- **Sentence Transformers**: Embedding models (all-MiniLM-L6-v2)
- **Streamlit**: Web application framework
- **PyGithub**: GitHub API library

## About BHFDSC

The BHF Data Science Centre (BHFDSC) focuses on improving cardiovascular health through large-scale data analysis. Their repositories contain research code and analysis pipelines for studying COVID-19's impact on cardiovascular disease using linked electronic health records.

## License

This project is for educational and research purposes. Please respect the licenses of the BHFDSC repositories when using information from them.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository.
