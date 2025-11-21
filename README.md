# BHFDSC Repository Q&A System

A simple Retrieval-Augmented Generation (RAG) application that enables users to ask questions about repositories in the [BHF Data Science Centre (BHFDSC)](https://github.com/BHFDSC) GitHub organization. Uses Anthropic's Claude AI, Pinecone vector database, and Streamlit.

## Features

- **Simple Architecture**: Straightforward RAG implementation with minimal complexity
- **Persistent Cloud Storage**: Uses Pinecone vector database - data persists across sessions
- **Comprehensive Indexing**: Indexes README files and code from all organization repositories
- **Intelligent Q&A**: Uses Claude AI with RAG to answer questions based on repository content
- **Source Attribution**: Shows which repositories and files were used to generate answers
- **Interactive UI**: Clean Streamlit interface
- **Multiple File Types**: Indexes Markdown, Python, R, Jupyter notebooks, and more

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

**Key Components**:
- **GitHub Indexer**: Fetches repositories and file contents
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
├── app.py                    # Streamlit application
├── config.py                 # Configuration settings
├── github_indexer.py         # Repository fetching and indexing
├── vector_store.py           # Vector store interface
├── vector_store_pinecone.py  # Pinecone implementation
├── qa_chain.py               # Question-answering system
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## How It Works

**Indexing Phase**:
1. Fetches all repositories from the BHFDSC GitHub organization
2. Downloads README files and code files (Python, R, Markdown, etc.)
3. Splits documents into chunks
4. Creates embeddings using Sentence Transformers
5. Stores embeddings in Pinecone

**Query Phase**:
1. User asks a question via Streamlit UI
2. Question is embedded and used to search Pinecone
3. Top-5 most relevant document chunks are retrieved
4. Context + question sent to Claude API
5. Claude generates answer based on context
6. Answer and sources displayed to user

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
