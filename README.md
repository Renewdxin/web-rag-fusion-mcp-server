# RAG MCP Server

RAG MCP Server is a powerful, production-ready server that enhances language models with advanced retrieval-augmented generation (RAG) capabilities. It intelligently combines semantic search across local documents with real-time web search to provide comprehensive and accurate context to your models.

## Core Features

- **Knowledge Base Search**: Find relevant information within your local documents.
- **Web Search**: Get up-to-date information from the internet.
- **Smart Search**: A hybrid approach that uses both knowledge base and web search for the most relevant results.

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (recommended for easy setup)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://tavily.com/#api)

### Installation & Running

#### Docker (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rag-mcp-server.git
    cd rag-mcp-server
    ```

2.  **Create your environment file:**
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your `OPENAI_API_KEY` and `TAVILY_API_KEY`.

3.  **Build and run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The server will be available at `http://localhost:8000`.

#### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rag-mcp-server.git
    cd rag-mcp-server
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Create your environment file:**
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your `OPENAI_API_KEY` and `TAVILY_API_KEY`.

4.  **Run the server:**
    ```bash
    python3 src/mcp_server.py
    ```
    The server will be available at `http://localhost:8000`.

## How to Use

You can interact with the server using any HTTP client, such as `curl` or Postman. The server exposes a single `/mcp` endpoint that accepts POST requests.

### Request Format

The request body should be a JSON object with the following structure:

```json
{
  "tool_name": "tool_to_use",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### Available Tools

#### 1. `search_knowledge_base`

Searches for documents within your local knowledge base.

**Example:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "search_knowledge_base",
           "parameters": {
             "query": "What are the latest advancements in AI?",
             "top_k": 5
           }
         }'
```

#### 2. `web_search`

Performs a web search using the Tavily API.

**Example:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "web_search",
           "parameters": {
             "query": "What is the weather in San Francisco?",
             "max_results": 3
           }
         }'
```

#### 3. `smart_search`

Combines both local knowledge base and web search for comprehensive results.

**Example:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "smart_search",
           "parameters": {
             "query": "Compare our internal sales data with public market trends."
           }
         }'
```

## Configuration

The server is configured through the `.env` file. Here are some of the key settings:

- `OPENAI_API_KEY`: **(Required)** Your OpenAI API key for document embeddings.
- `TAVILY_API_KEY`: **(Required)** Your Tavily API key for web search.
- `VECTOR_STORE_PATH`: The local directory where your knowledge base vectors are stored. Defaults to `./vector_store`.
- `LOG_LEVEL`: The logging level for the server. Defaults to `INFO`.

For more advanced configuration options, see the `config/settings.py` file.