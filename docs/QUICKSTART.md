# Quickstart Guide

This guide will help you get the RAG MCP Server up and running quickly.

## 1. Installation

We recommend using Docker for the easiest setup.

### With Docker (Recommended)

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

### Local Setup

1.  **Prerequisites**: Python 3.9+.
2.  **Clone and enter the directory**.
3.  **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Configure your `.env` file** as shown in the Docker steps.
5.  **Run the server**:
    ```bash
    python3 src/mcp_server.py
    ```

## 2. How to Use

Once the server is running (on `http://localhost:8000` by default), you can send requests to the `/mcp` endpoint.

### Available Tools

- `search_knowledge_base`: Searches your local files.
- `web_search`: Searches the internet.
- `smart_search`: Intelligently combines local and web search.

### Example `curl` Request

Here is an example of how to use the `smart_search` tool with `curl`:

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "smart_search",
           "parameters": {
             "query": "What are the latest trends in artificial intelligence?"
           }
         }'
```

## 3. Adding Documents

To add your own documents to the knowledge base, place your files (PDF, TXT, MD, DOCX) into a directory (e.g., a new directory named `my_documents`).

You will then need to run a script to process and embed these documents. Refer to the `DEVELOPER_GUIDE.md` for details on how to use the `DocumentProcessor` to ingest your files.

