# RAG MCP Server

A powerful Model Context Protocol (MCP) server that provides Retrieval-Augmented Generation (RAG) capabilities for language models. It combines local knowledge base search with real-time web search to provide comprehensive and accurate context for your AI assistants.

## 🚀 Features

- **📚 Knowledge Base Search**: Quick search in local documents
- **🌐 Web Search**: Get latest information from the internet
- **🧠 Smart Search**: Hybrid approach combining both knowledge base and web search
- **🔌 MCP Compatible**: Perfect integration with Claude Code, Claude Desktop, and other MCP clients

## 📋 Prerequisites

- Python 3.9+ or Docker
- [OpenAI API Key](https://platform.openai.com/api-keys)
- Search Service API Key:
  - [Perplexity API Key](https://www.perplexity.ai/settings/api) (Recommended)
  - Or [Exa API Key](https://exa.ai/)

## ⚡ Quick Start

### Method 1: Docker Deployment (Recommended)

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd rag-mcp-server
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your API keys:
   ```bash
   # Required Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   SEARCH_API_KEY=your-perplexity-or-exa-api-key-here
   
   # Search Backend (perplexity or exa)
   SEARCH_BACKEND=perplexity
   
   # Environment
   ENVIRONMENT=prod
   ```

3. **Start Service**
   ```bash
   # Build and start production
   docker-compose up rag-mcp-server --build -d
   
   # Check status
   docker-compose ps
   
   # View logs
   docker-compose logs rag-mcp-server -f
   ```

### Method 2: Local Deployment

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

3. **Run Server**
   ```bash
   python src/mcp_server.py
   ```

## 🔧 Client Integration

### Claude Code Integration

1. **Find Configuration File**
   - macOS: `~/Library/Application Support/Claude Code/settings.json`
   - Linux: `~/.config/claude-code/settings.json`
   - Windows: `%APPDATA%/Claude Code/settings.json`

2. **Add MCP Server Configuration**
   
   Add to `settings.json`:
   ```json
   {
     "mcpServers": {
       "rag-server": {
         "command": "python",
         "args": ["/path/to/your/rag-mcp-server/src/mcp_server.py"],
         "env": {
           "OPENAI_API_KEY": "your-openai-api-key",
           "SEARCH_API_KEY": "your-search-api-key",
           "SEARCH_BACKEND": "perplexity",
           "ENVIRONMENT": "prod"
         }
       }
     }
   }
   ```

   Or use Docker version:
   ```json
   {
     "mcpServers": {
       "rag-server": {
         "command": "docker",
         "args": [
           "run", "--rm", "-i",
           "--env-file", "/path/to/your/rag-mcp-server/.env",
           "rag-mcp-server:production"
         ]
       }
     }
   }
   ```

3. **Restart Claude Code**

### Claude Desktop Integration

1. **Find Configuration File**
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add Configuration**
   ```json
   {
     "mcpServers": {
       "rag-server": {
         "command": "python",
         "args": ["/path/to/your/rag-mcp-server/src/mcp_server.py"],
         "env": {
           "OPENAI_API_KEY": "your-openai-api-key",
           "SEARCH_API_KEY": "your-search-api-key",
           "SEARCH_BACKEND": "perplexity"
         }
       }
     }
   }
   ```

## 📖 Usage

After successful integration, you can use the following tools in MCP-compatible clients:

### 🔍 Available Tools

#### 1. `search_knowledge_base` - Knowledge Base Search
Search for relevant information in local documents.

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Number of results to return, default 5

**Example:**
```
Search our knowledge base for "API security best practices"
```

#### 2. `web_search` - Web Search
Perform real-time web search using Perplexity or Exa.

**Parameters:**
- `query` (required): Search query
- `max_results` (optional): Maximum results, default 5

**Example:**
```
Search for latest AI development trends
```

#### 3. `smart_search` - Smart Search
Combines knowledge base and web search for comprehensive results.

**Parameters:**
- `query` (required): Search query

**Example:**
```
Compare our internal sales data with public market trends
```

## 📁 Document Management

### Adding Documents to Knowledge Base

1. **Create documents directory**
   ```bash
   mkdir -p documents
   ```

2. **Add document files**
   Supported formats: PDF, TXT, DOCX, MD
   ```bash
   cp your-documents.pdf documents/
   ```

3. **Rebuild index** (Server automatically detects and processes new documents)

## ⚙️ Advanced Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key              # OpenAI API key
SEARCH_API_KEY=your-search-key          # Search service API key

# Search Configuration
SEARCH_BACKEND=perplexity               # perplexity or exa
SIMILARITY_THRESHOLD=0.75               # Similarity threshold (0.0-1.0)
SIMILARITY_TOP_K=10                     # Number of relevant documents

# Document Processing
CHUNK_SIZE=1024                         # Document chunk size
CHUNK_OVERLAP=200                       # Chunk overlap size
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model

# System Configuration
ENVIRONMENT=prod                        # dev, test, prod
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
VECTOR_STORE_PATH=./data/vector_store.db # Vector database path
```

## 🛠️ Troubleshooting

### Common Issues

**1. Server won't start**
```bash
# Check API keys
cat .env | grep API_KEY

# Check dependencies
pip install -r requirements.txt

# View error logs
docker-compose logs rag-mcp-server
```

**2. Empty search results**
```bash
# Check documents
ls -la documents/

# Check vector database
ls -la data/

# Lower similarity threshold
# In .env: SIMILARITY_THRESHOLD=0.5
```

**3. MCP client connection fails**
- Ensure configuration file path is correct
- Check environment variables are set
- Restart MCP client application
- Check client error logs

### Development Mode

```bash
# Use development configuration
docker-compose --profile dev up rag-mcp-dev

# Or local development
ENVIRONMENT=dev python src/mcp_server.py
```

## 🔒 Security Recommendations

1. **Protect API Keys**
   - Don't commit `.env` files to version control
   - Use environment variables or key management services
   - Rotate API keys regularly

2. **Network Security**
   - Use HTTPS in production
   - Limit network access permissions
   - Enable rate limiting

## 📊 Monitoring

### Performance Monitoring

```bash
# Enable Prometheus monitoring
docker-compose --profile monitoring up

# Access monitoring dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Data Backup

```bash
# Backup vector database
docker run --rm -v mcp_rag_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/rag_data_backup.tar.gz -C /data .

# Restore data
docker run --rm -v mcp_rag_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/rag_data_backup.tar.gz -C /data
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.