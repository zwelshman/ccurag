# BHFDSC Repository Q&A System

A hybrid RAG (Retrieval-Augmented Generation) system for the [BHF Data Science Centre (BHFDSC)](https://github.com/BHFDSC) GitHub organization. Combines semantic search with BM25 keyword matching to provide conversational Q&A about your codebase.

## 🌟 Overview

This system provides:

**Conversational Q&A**: Ask natural language questions about repositories and get AI-generated answers based on actual code and documentation

**Key Innovation**: Hybrid search combining BM25 keyword matching with vector semantic search, optimized for code repositories with markdown documentation.

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Comprehensive technical documentation explaining how the system works
- **[HYBRID_SEARCH.md](HYBRID_SEARCH.md)**: Deep dive into the hybrid BM25 + vector search implementation
- **[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md)**: Potential improvements and extensions

## ✨ Features

- **Hybrid Search**: Combines BM25 keyword matching with vector semantic search for optimal retrieval
  - BM25 excels at exact term matching (function names, identifiers)
  - Vector search understands meaning and context
  - Adaptive weighting based on query type (code vs. conceptual)
- **Persistent Storage**: Uses Pinecone cloud vector database for embeddings - data persists across sessions. Local cache files are stored in `.cache/` directory and can be committed to `data_index/` folder in Git for easy sharing across deployments
- **Comprehensive Indexing**: Indexes README files and code from all organization repositories
- **Intelligent Q&A**: Uses Anthropic Claude 4.5 Sonnet with RAG to answer questions
- **Source Attribution**: Shows which repositories and files were used to generate answers with relevance scores
- **Interactive UI**: Clean Streamlit interface with separate tabs for Q&A, Documentation, and Setup
- **Multiple File Types**: Indexes `.md`, `.py`, `.r`, `.R`, `.sql`, `.ipynb`, `.csv`, `.txt`, `.rst`, `.html`, `.yaml`, `.yml`, `.json`

## 🏗️ Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (Streamlit Web App)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │   Q&A System   │
                  │  (RAG-based)   │
                  └────────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Hybrid Retriever│
                  │  BM25 + Vector  │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
    ┌──────────────┐  ┌─────────┐  ┌──────────────┐
    │   BM25 Index │  │ Pinecone│  │  Anthropic   │
    │  (keyword)   │  │ (vector)│  │Claude (LLM)  │
    └──────────────┘  └─────────┘  └──────────────┘
```

### Query Flow

```
User Question → Hybrid Search (BM25 + Vector) → Top-20 Documents →
Format Context + Question → Anthropic Claude API → Answer + Sources
```

**Flow Details**:
1. **User Query**: Natural language question entered in Streamlit UI
2. **Hybrid Search**:
   - BM25 keyword search for exact term matching
   - Vector semantic search for meaning and context
   - Adaptive weight combination (favor BM25 for code, vector for concepts)
3. **Context Building**: Top-20 most relevant document chunks
4. **Claude API**: Generates answer with source citations
5. **Display**: Answer shown with expandable source documents and relevance scores

### Key Components

- **GitHub Indexer** (`github_indexer.py`): Fetches repositories and file contents via GitHub API
- **Vector Store** (`vector_store_pinecone.py`): Manages Pinecone vector database and embeddings
- **Hybrid Retriever** (`hybrid_retriever.py`): Combines BM25 and vector search with adaptive weighting
- **QA System** (`qa_chain.py`): Formats prompts and calls Anthropic Claude API
- **Local Storage** (`cloud_storage.py`): Local filesystem cache storage - stores files in `.cache/` directory and optionally in `data_index/` folder for Git-based sharing
- **Streamlit App** (`app.py`): Web interface with Q&A, Documentation, and Setup tabs

## 🛠️ Technologies Used

### Core Technologies

#### Anthropic Claude API
- **Model**: `claude-sonnet-4-5` (Claude 4.5 Sonnet)
- **Purpose**: Generate natural language answers from retrieved context
- **Features**: 200K token context window (~500 pages of code), excellent technical accuracy
- **Why Claude?**: Superior performance on code understanding, instruction following for source citations, cost-effective

#### Pinecone Vector Database
- **Type**: Managed, serverless cloud vector database
- **Purpose**: Store and search document embeddings
- **Configuration**: 768 dimensions, cosine similarity
- **Why Pinecone?**: Fully managed (no infrastructure), fast (<50ms queries), persistent storage

#### Sentence Transformers (BAAI/llm-embedder)
- **Model**: `BAAI/llm-embedder`
- **Output**: 768-dimensional dense vectors
- **Purpose**: Convert text to embeddings for semantic search
- **Why This Model?**: Optimized for retrieval tasks, good balance of quality and speed (~100 docs/sec on CPU)

#### BM25 (rank-bm25)
- **Algorithm**: Okapi BM25
- **Purpose**: Keyword-based lexical search
- **Features**: Custom code-aware tokenization (CamelCase splitting, snake_case preservation)
- **Why BM25?**: Proven industry standard, fast, excellent for exact code identifier matching

#### Streamlit
- **Purpose**: Web UI framework
- **Features**: Reactive components, built-in caching, easy deployment to Streamlit Cloud
- **Why Streamlit?**: Rapid prototyping, pure Python, simple deployment

### How RAG Works

**RAG (Retrieval-Augmented Generation)** enhances LLMs by providing them with relevant external information:

1. **The Problem**: LLMs have knowledge cutoffs and don't know about your private/recent codebases
2. **The Solution**:
   - **Retrieval**: Search for relevant documents from your knowledge base
   - **Augmentation**: Add retrieved documents as context to the prompt
   - **Generation**: LLM generates answer based on provided context
3. **The Result**: Factual, source-cited answers grounded in your actual codebase

**Why RAG over Fine-tuning?**
- Updates in minutes (re-index) vs. hours/days (retrain)
- Always cites exact sources
- Lower cost ($10-50/month vs. $100-1000s)
- Perfect for frequently changing data

### How Hybrid Search Works

**The Challenge**: Neither BM25 nor vector search alone is perfect for code repositories.

**BM25 Strengths**: Exact term matching (function names, identifiers, acronyms)
**Vector Strengths**: Semantic understanding, paraphrased queries, conceptual relationships

**The Solution**: Combine both with adaptive weighting!

```python
# Adaptive weight calculation
if query_has_code_patterns(query):  # CamelCase, snake_case, ()
    weights = (0.7, 0.3)  # Favor BM25 for code queries
else:
    weights = (0.3, 0.7)  # Favor vector for conceptual queries

# Combine scores
final_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
```

**Result**: Best of both worlds! Code queries get exact matches, conceptual queries get semantic understanding.

**Research Support**: Studies show hybrid search achieves 75-85% accuracy vs. 60-70% for vector-only or BM25-only on code search tasks.

## Prerequisites

- Python 3.9 or higher
- [Anthropic API key](https://console.anthropic.com/) - for Claude AI
- [Pinecone API key](https://www.pinecone.io/) - for vector database
- (Optional) [GitHub Personal Access Token](https://github.com/settings/tokens) - increases rate limit from 60/hr to 5000/hr

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
├── build_hybrid_index.py        # Build hybrid search index
├── example_hybrid_usage.py      # Example hybrid search usage
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🚀 Using the Application

### Navigation

The application has 3 main tabs:

#### 1. Q&A Tab
- **Purpose**: Ask natural language questions about repositories
- **Features**:
  - Chat-style interface with conversation history
  - Expandable source documents with relevance scores
  - Example questions to get started
  - Hybrid search configuration panel
    - Build/clear BM25 index
    - View current search method (hybrid vs. vector-only)
    - Adjust number of source documents
- **Example Questions**:
  - "What COVID-19 and cardiovascular research projects are in this organization?"
  - "Which repositories contain Python code for data analysis?"
  - "What phenotyping algorithms are used in the CCU projects?"
  - "Show me repositories that work with linked electronic health records"

#### 2. Documentation Tab
- **Purpose**: Learn how the system works
- **Sections**:
  - **Architecture**: System design and component interactions
  - **How It Works**: Detailed explanations of RAG and hybrid search
  - **Technologies**: Deep dive into Anthropic Claude, Pinecone, BM25, embeddings
  - **Future Enhancements**: Potential improvements and extensions
- **Content**: Comprehensive technical documentation built into the app

#### 3. Setup Tab
- **Purpose**: Manage Pinecone vector database
- **Actions**:
  - Index repositories (one-time setup, 10-30 minutes)
  - View index statistics
  - Delete vector store (if needed)
  - View configuration (GitHub org, model, chunk sizes)
- **Note**: BM25 index is managed in the Q&A tab

### Workflow

**Initial Setup** (one-time, 10-30 minutes):
1. Go to **Setup** tab → Click "Index Repositories"
2. Go to **Q&A** tab → Open "Hybrid Search Configuration" → Click "Build BM25 Index"

**Using Q&A**:
1. Navigate to **Q&A** tab
2. Type your question in the chat input
3. Review answer and expandable source documents
4. Follow up with additional questions to dig deeper

## 🔧 How It Works

### Indexing Phase (One-time setup)

1. **Fetch Repositories**: GitHub Indexer fetches all repos from BHFDSC organization
2. **Download Files**: Collects README, code files (.py, .r, .R, .sql, .ipynb, .md, etc.)
3. **Chunk Documents**: Splits files into 1000-character chunks with 200-character overlap
4. **Create Embeddings**: Sentence Transformers converts text to 768-dim vectors
5. **Upload to Pinecone**: Batch upload embeddings with metadata (repo, file, URL)
6. **Build BM25 Index** (separate step):
   - Fetch documents from Pinecone
   - Tokenize with code-aware tokenizer
   - Build BM25Okapi index
   - Cache to `.cache/bm25_index.pkl` (copy to `data_index/` folder and commit to Git for sharing across deployments)

### RAG Query Phase (Runtime)

1. **User Question**: Natural language query entered in UI
2. **Hybrid Search**:
   - **BM25 Search**: Tokenize query, calculate BM25 scores, normalize to 0-1
   - **Vector Search**: Embed query, search Pinecone, get similarity scores
   - **Adaptive Weighting**: Analyze query for code patterns, adjust weights
   - **Combine Scores**: `final = (w_bm25 * bm25_score) + (w_vector * vector_score)`
3. **Rank & Select**: Sort by combined score, take top-20 documents
4. **Format Context**: Build structured context with metadata (repo, file, type)
5. **Construct Prompt**:
   - System: Define assistant role and guidelines
   - Context: Formatted retrieved documents
   - Question: User's query
   - Instructions: "Base answer on context, cite sources..."
6. **Call Claude API**: Generate answer with source citations
7. **Display Results**: Answer + expandable source documents with relevance scores

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

## 🔬 Technologies Used

- **Anthropic Claude** (`claude-sonnet-4-5`): AI language model for generating answers
- **Pinecone**: Managed cloud vector database for embeddings
- **Sentence Transformers** (`BAAI/llm-embedder`): Text embeddings for semantic search
- **BM25** (`rank-bm25`): Keyword-based lexical search
- **Streamlit**: Web application framework for UI
- **PyGithub**: GitHub API library for fetching repositories
- **Git**: Version control for cache persistence and sharing

**For detailed technical information, see [ARCHITECTURE.md](ARCHITECTURE.md).**

## 🚀 Future Enhancements

This is a proof-of-concept system with significant potential for extension. Key areas for improvement include:

### Short-term (1-3 months)
- **Enhanced Code Understanding**: Call graph analysis, data flow tracking, complexity metrics
- **Improved Search Quality**: Query expansion, re-ranking with cross-encoders, negative example learning
- **User Experience**: Conversation history, filters, syntax highlighting, direct GitHub links

### Medium-term (3-6 months)
- **Advanced Analytics**: Temporal analysis, contributor insights, pattern detection
- **Automated Code Quality**: Style consistency checking, security scanning, dependency audits
- **Multi-modal Capabilities**: Diagram understanding, deep notebook analysis, API spec parsing

### Long-term (6-12 months)
- **AI-Powered Code Generation**: Template generation, test generation, documentation generation
- **Collaborative Features**: Shared annotations, question history, expert routing
- **Integration Ecosystem**: VS Code extension, Slack/Teams bot, GitHub Actions, CI/CD
- **Advanced RAG**: Hierarchical retrieval, multi-hop reasoning, agentic workflows

**For detailed descriptions and implementation guidance, see [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md).**

## 📖 About BHFDSC

The [BHF Data Science Centre (BHFDSC)](https://bhfdatasciencecentre.org/) focuses on improving cardiovascular health through large-scale data analysis. Their GitHub organization contains research code and analysis pipelines for studying COVID-19's impact on cardiovascular disease, using data from millions of patients across England.

**Topics**: CVD-COVID-UK, cardiovascular disease, electronic health records, phenotyping, data curation

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the BHFDSC repositories.

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request.

Areas where contributions would be especially valuable:
- Improved tokenization for domain-specific languages
- Enhanced prompt engineering for better answer quality
- Performance optimizations for large codebases
- Better visualization of search results

## 💬 Support

For issues or questions:
- Open an issue on GitHub
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Review the [HYBRID_SEARCH.md](HYBRID_SEARCH.md) for search-related questions

## 🙏 Acknowledgments

- **Anthropic** for Claude API
- **Pinecone** for vector database
- **Hugging Face** for Sentence Transformers
- **BHFDSC** for their open research code
- The open-source community for the amazing tools that made this possible
