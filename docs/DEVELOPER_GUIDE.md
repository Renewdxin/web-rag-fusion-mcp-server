# Developer Guide

This guide provides a comprehensive overview of the RAG MCP Server for developers, covering architecture, key components, configuration, and setup.

## 1. Architecture Overview

The RAG MCP Server is designed with a modular and scalable architecture. It consists of a main server that orchestrates three core components: a Vector Store, a Web Searcher, and a Document Processor.

### High-Level Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│  RAGMCPServer   │◄──►│  External APIs  │
│ (IDE, CLI, etc.)│    │ (Orchestrator)  │    │ (OpenAI, Tavily)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
        ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐
        │VectorStore    │ │WebSearch     │ │DocumentProcessor  │
        │Manager        │ │Manager       │ │(File Processing)  │
        │(Local Search) │ │(Web Search)  │ │                   │
        └───────────────┘ └──────────────┘ └───────────────────┘
```

### Key Design Patterns
- **Singleton**: For global configuration management.
- **Factory**: To create document loaders for different file formats.
- **Strategy**: For pluggable search and chunking algorithms.
- **Adapter**: To standardize interfaces for external APIs like Tavily.

## 2. Core Components

### RAGMCPServer
- **File**: `src/mcp_server.py`
- **Description**: The main entry point and orchestrator. It handles MCP requests, validates inputs using JSON Schema, and routes them to the appropriate tool (`search_knowledge_base`, `web_search`, or `smart_search`).

### VectorStoreManager
- **File**: `src/vector_store.py`
- **Description**: Manages all interactions with the ChromaDB vector database. It handles document embedding (using OpenAI), indexing, and semantic similarity searches. It features connection retries, batching, and performance monitoring.

### WebSearchManager
- **File**: `src/web_search.py`
- **Description**: Manages web search operations using the Tavily API. It includes features like exponential backoff for retries, result caching (1-hour TTL), content quality scoring, and API quota tracking.

### DocumentProcessor
- **File**: `src/document_processor.py`
- **Description**: Handles the ingestion and processing of various document formats (PDF, DOCX, TXT, MD, HTML). It uses intelligent, sentence-aware chunking to split documents into manageable pieces for embedding.

### Smart Search Logic
- **File**: `src/mcp_server.py` (`_smart_search_internal` method)
- **Description**: This hybrid search first queries the local `VectorStoreManager`. If the results meet a specific similarity threshold (default: 0.75), it returns them directly. If not, it triggers a `WebSearchManager` search to supplement or replace the local results, providing a more comprehensive answer.

## 3. Setup and Deployment

### Local Development
1.  **Prerequisites**: Python 3.9+, Poetry (or pip).
2.  **Clone**: `git clone <repository-url> && cd rag-mcp-server`
3.  **Environment**: Create a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Dependencies**: Install required packages.
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configuration**: Create a `.env` file from the example.
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to add your `OPENAI_API_KEY` and `TAVILY_API_KEY`.
6.  **Run Server**:
    ```bash
    python src/mcp_server.py
    ```

### Docker Deployment
1.  **Prerequisites**: Docker and Docker Compose.
2.  **Configuration**: Create and configure your `.env` file as described above.
3.  **Build & Run**:
    ```bash
    docker-compose up --build
    ```
The server will be exposed on the port defined in `docker-compose.yml` (default: 8000).

## 4. Configuration (`.env`)

The server is configured via environment variables loaded from a `.env` file.

### Required
| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your API key for OpenAI, used for generating embeddings. |
| `TAVILY_API_KEY` | Your API key for Tavily, used for performing web searches. |

### Optional
| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_PATH` | `./vector_store` | Path to store the ChromaDB vector data. |
| `COLLECTION_NAME` | `knowledge_base` | Name of the collection within ChromaDB. |
| `SIMILARITY_THRESHOLD`| `0.75` | The score needed for `smart_search` to rely only on local results. |
| `LOG_LEVEL` | `INFO` | The logging level (e.g., `DEBUG`, `INFO`, `WARNING`). |

## 5. Key APIs

This is not an exhaustive list, but it highlights the main entry points for the core components.

### `RAGMCPServer`
- `async _search_knowledge_base(...)`: Performs a semantic search on the local vector store.
- `async _search_web(...)`: Executes a web search via the `WebSearchManager`.
- `async _smart_search_internal(...)`: Implements the hybrid local-then-web search logic.

### `VectorStoreManager`
- `async add_documents(...)`: Embeds and stores documents in ChromaDB.
- `async similarity_search_with_score(...)`: Executes a similarity search against the vector store.

### `WebSearchManager`
- `async search(...)`: Performs a web search, handling caching and retries.

### `DocumentProcessor`
- `async process_file(...)`: Processes a single file into document chunks.
- `async process_directory(...)`: Processes all supported files in a given directory.
