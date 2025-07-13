# MCP-Powered Agentic RAG Server

A standards-compliant **Model Context Protocol (MCP) Server** that provides intelligent Retrieval-Augmented Generation (RAG) capabilities. This server can be used with any MCP-compatible client (like Claude Desktop) to provide local-first knowledge search with intelligent web search fallback.

## 🔌 What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that enables AI applications to securely connect to external data sources and tools. Unlike traditional API approaches, MCP provides:

- **Standardized Communication**: JSON-RPC based protocol for AI-tool interaction
- **Client Agnostic**: Works with any MCP-compatible client
- **Tool Discovery**: Automatic capability discovery and schema validation
- **Security**: Controlled access to sensitive data and operations

## 🚀 Features

- **Standards-Compliant**: Implements official MCP specification
- **Intelligent RAG Search**: Local knowledge base search with similarity scoring
- **Adaptive Web Search**: Automatic fallback to web search when local knowledge is insufficient
- **Configurable Thresholds**: Customizable similarity thresholds for search decisions
- **Multi-Source Integration**: Combines local and web information intelligently
- **Source Attribution**: Clear citations and confidence scores
- **Easy Integration**: Works with Claude Desktop and other MCP clients

## 🏗️ Architecture

```
MCP Client (Claude Desktop) ←→ MCP Protocol ←→ RAG Server
                                                    ↓
                                            Tool Registry
                                                    ↓
                                    ┌───────────────────────────┐
                                    │  search_knowledge_base    │
                                    │  web_search              │
                                    │  smart_search           │
                                    └───────────────────────────┘
                                                    ↓
                                    ┌───────────────────────────┐
                                    │  ChromaDB Vector Store    │
                                    │  Tavily Web Search       │
                                    └───────────────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- ChromaDB for vector storage
- Tavily API key (for web search)
- OpenAI API key (for embeddings, optional)

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/mcp-rag-server.git
cd mcp-rag-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install MCP SDK

```bash
pip install mcp
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# Required: Web Search API
TAVILY_API_KEY=tvly-your-tavily-api-key

# Optional: Embeddings (if using OpenAI embeddings)
OPENAI_API_KEY=sk-your-openai-key

# Vector Store Configuration
VECTOR_STORE_PATH=./vector_store
COLLECTION_NAME=knowledge_base

# MCP Server Settings
MCP_SERVER_NAME=rag-agent
SIMILARITY_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
```

### Key Configuration Parameters

- `SIMILARITY_THRESHOLD`: Score threshold for triggering web search (0-1, default: 0.75)
- `VECTOR_STORE_PATH`: Path to ChromaDB storage directory
- `TAVILY_API_KEY`: Required for web search functionality

## 🚀 Quick Start

### 1. Initialize Knowledge Base

```python
# scripts/init_knowledge_base.py
from vector_store import VectorStoreManager
from document_loader import load_documents

# Initialize vector store
vector_manager = VectorStoreManager("./vector_store")
await vector_manager.initialize_collection("knowledge_base")

# Load your documents
documents = load_documents("./data/")
await vector_manager.add_documents(documents)
```

Run the initialization:

```bash
python scripts/init_knowledge_base.py
```

### 2. Test the MCP Server

```bash
# Test server functionality
python mcp_server.py
```

### 3. Configure MCP Client

For **Claude Desktop**, add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/mcp-rag-server",
      "env": {
        "TAVILY_API_KEY": "your-tavily-key"
      }
    }
  }
}
```

**Claude Desktop Config Locations:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## 🔧 Available Tools

### 1. `search_knowledge_base`
Search the local vector database with similarity scoring.

**Input:**
```json
{
  "query": "string (required)",
  "top_k": "integer (optional, default: 5)"
}
```

**Example Usage in Claude:**
```
Please search our knowledge base for information about API documentation.
```

### 2. `web_search`
Search the web using Tavily API when local knowledge is insufficient.

**Input:**
```json
{
  "query": "string (required)",
  "max_results": "integer (optional, default: 5)"
}
```

### 3. `smart_search` (Recommended)
Intelligent search that tries local knowledge first, then web search if needed.

**Input:**
```json
{
  "query": "string (required)",
  "similarity_threshold": "number (optional, default: 0.75)",
  "local_top_k": "integer (optional, default: 5)",
  "web_max_results": "integer (optional, default: 5)"
}
```

**Example Usage in Claude:**
```
I need comprehensive information about our product pricing. Use smart search to check internal docs first, then web if needed.
```

## 📚 Usage Examples

### Basic Knowledge Search

```
Human: What are our company's core values?

Claude: I'll search your knowledge base for information about your company's core values.

[Uses search_knowledge_base tool]

Based on your internal documentation, your company's core values are:
1. Customer First - Prioritizing customer needs in all decisions
2. Innovation - Continuously improving and embracing new technologies
3. Integrity - Maintaining ethical standards in all business practices
...

Source: employee_handbook.pdf (Similarity: 0.94)
```

### Smart Search with Fallback

```
Human: What are the latest trends in AI development for 2024?

Claude: I'll use smart search to check your knowledge base first, then supplement with current web information if needed.

[Uses smart_search tool]

Local Knowledge Results (Max Score: 0.45)
- Found some general AI information but scores are below threshold (0.75)

Web Search Results:
1. "Top AI Trends for 2024" - TechCrunch
   - Generative AI adoption in enterprise
   - Multimodal AI systems
   - AI safety and regulation developments
...

Since local knowledge didn't contain current AI trends information, I've provided the latest information from web sources.
```

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Test MCP Protocol Compliance

```bash
python tests/test_mcp_protocol.py
```

### Integration Testing

```bash
# Test with actual MCP client
python tests/integration/test_claude_desktop.py
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t mcp-rag-server .

# Run with environment file
docker run --env-file .env -v $(pwd)/vector_store:/app/vector_store mcp-rag-server
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-rag-server:
    build: .
    volumes:
      - ./vector_store:/app/vector_store
      - ./data:/app/data
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 📊 Monitoring and Debugging

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python mcp_server.py
```

### Monitor Tool Usage

The server logs all tool calls and their results:

```
2024-01-15 10:30:15 - INFO - Tool called: smart_search
2024-01-15 10:30:15 - INFO - Query: "company revenue Q1"
2024-01-15 10:30:16 - INFO - Local search max score: 0.92
2024-01-15 10:30:16 - INFO - Decision: Local knowledge sufficient
```

### Performance Metrics

Check server performance:

```python
# In your client
# Monitor response times and success rates
```

## 🔍 Troubleshooting

### Common Issues

1. **"MCP server not found"**
   ```bash
   # Check Claude Desktop config path
   # Ensure absolute paths are used
   # Verify Python environment
   ```

2. **"ChromaDB not initialized"**
   ```bash
   python scripts/init_knowledge_base.py
   ```

3. **"Tavily API key invalid"**
   ```bash
   # Check .env file
   # Verify API key format: tvly-...
   ```

4. **"No similarity results"**
   ```bash
   # Check if documents are properly indexed
   # Verify embedding model compatibility
   # Lower similarity threshold temporarily
   ```

### Debug Mode

```bash
# Run in debug mode
export LOG_LEVEL=DEBUG
export MCP_DEBUG=true
python mcp_server.py
```

### Check MCP Client Connection

For Claude Desktop, check the logs:
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%LOCALAPPDATA%\Claude\logs\`

## 📈 Advanced Usage

### Custom Embedding Models

```python
# custom_embeddings.py
from chromadb.utils import embedding_functions

# Use custom embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

### Multi-Collection Support

```python
# Support multiple knowledge bases
collections = {
    "technical_docs": "Technical Documentation",
    "company_policies": "Company Policies",
    "product_specs": "Product Specifications"
}
```

### Custom Similarity Thresholds by Topic

```python
# Dynamic threshold adjustment
topic_thresholds = {
    "technical": 0.80,    # Higher threshold for technical queries
    "general": 0.70,      # Lower threshold for general queries
    "current_events": 0.60 # Even lower for time-sensitive queries
}
```

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .
```

### Adding New Tools

1. Define tool schema in `tool_schemas.py`
2. Implement tool handler in `mcp_server.py`
3. Add tests in `tests/`
4. Update documentation

### Submitting Changes

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run test suite: `pytest`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) for the open standard
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Tavily](https://tavily.com/) for web search capabilities
- [Anthropic](https://www.anthropic.com/) for Claude and MCP development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/mcp-rag-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mcp-rag-server/discussions)
- **MCP Documentation**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)

## 🗺️ Roadmap

- [ ] **Multi-client Support**: Support for more MCP clients beyond Claude Desktop
- [ ] **Advanced Vector Stores**: Pinecone, Weaviate, Qdrant integration
- [ ] **Document Processing Pipeline**: Automated document ingestion and processing
- [ ] **Semantic Caching**: Intelligent caching of search results
- [ ] **Multi-language Support**: Support for non-English knowledge bases
- [ ] **Graph RAG**: Integration with knowledge graphs
- [ ] **Real-time Updates**: Live document synchronization
- [ ] **Analytics Dashboard**: Usage analytics and performance monitoring

---

**Built with ❤️ using the Model Context Protocol**

For the latest updates and detailed documentation, visit our [GitHub repository](https://github.com/yourusername/mcp-rag-server).
