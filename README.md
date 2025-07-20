# Multi-Provider RAG Server

[![Release](https://img.shields.io/github/v/release/Renewdxin/mcp)](https://github.com/Renewdxin/mcp/releases)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🌍 [中文文档](README_zh.md) | English

A powerful RAG (Retrieval-Augmented Generation) server with **dynamic embedding provider support** via the Model Context Protocol (MCP). Switch between OpenAI and DashScope/Qwen providers at runtime without code changes.

## ✨ Key Features

- 🔄 **Dynamic Provider Switching** - Runtime switching between OpenAI and DashScope
- 🏗️ **Multi-Index Support** - Different providers for different document collections
- 🛡️ **Robust Error Handling** - Automatic fallback and comprehensive error recovery
- 🌐 **Web Search Integration** - Enhanced search via Perplexity/Exa APIs
- ⚙️ **Environment-Based Config** - Zero-code configuration changes
- 🚀 **Production Ready** - Rate limiting, metrics, and monitoring

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Renewdxin/multi-provider-rag.git
cd multi-provider-rag
pip install -r requirements.txt
```

### 2. Configuration

Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Basic OpenAI Setup:**
```bash
EMBED_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
EMBEDDING_MODEL=text-embedding-3-small
```

**DashScope/Qwen Setup:**
```bash
EMBED_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_dashscope_key_here
EMBEDDING_MODEL=text-embedding-v1
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. Run the Server

```bash
python -m src.mcp_server
```

## 🔧 Usage Examples

### Dynamic Provider Selection

```python
from src.embedding_provider import get_embed_model

# Use OpenAI
openai_model = get_embed_model("openai")

# Use DashScope
dashscope_model = get_embed_model("dashscope", model="text-embedding-v1")

# Environment-based selection
embed_model = get_embed_model_from_env()  # Uses EMBED_PROVIDER
```

### Multiple Indexes with Different Providers

```python
from src.embedding_provider import create_index_with_provider

# Create specialized indexes
docs_index = create_index_with_provider("openai", documents)
code_index = create_index_with_provider("dashscope", code_docs)
```

## 🌐 Supported Providers

| Provider | Models | Endpoint | Features |
|----------|--------|----------|----------|
| **OpenAI** | `text-embedding-ada-002`<br>`text-embedding-3-small`<br>`text-embedding-3-large` | `https://api.openai.com/v1` | High quality, global availability |
| **DashScope** | `text-embedding-v1`<br>`text-embedding-v2` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | China-optimized, cost-effective |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EMBED_PROVIDER` | Embedding provider (`openai`/`dashscope`) | `openai` | No |
| `EMBEDDING_MODEL` | Model name (provider-specific) | `text-embedding-3-small` | No |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes (for OpenAI) |
| `DASHSCOPE_API_KEY` | DashScope API key | - | Yes (for DashScope) |
| `SEARCH_API_KEY` | Perplexity/Exa API key | - | Optional |
| `VECTOR_STORE_PATH` | Vector database path | `./data/vector_store.db` | No |

### Provider Switching

Switch providers instantly by updating environment variables:

```bash
# Switch to DashScope
export EMBED_PROVIDER=dashscope
export DASHSCOPE_API_KEY=your_key

# Switch to OpenAI  
export EMBED_PROVIDER=openai
export OPENAI_API_KEY=your_key
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# With custom configuration
docker-compose -f docker-compose.yml up -d
```

## 📖 API Reference

### MCP Tools

- **`search_knowledge_base`** - Search local vector database
- **`web_search`** - Search web via Perplexity/Exa
- **`smart_search`** - Hybrid local + web search
- **`add_document`** - Add documents to knowledge base

### Python API

```python
# Core embedding functions
from src.embedding_provider import (
    get_embed_model,
    get_embed_model_from_env,
    create_index_with_provider,
    validate_provider_config
)

# RAG engine
from src.llamaindex_processor import RAGEngine

# MCP server
from src.mcp_server import RAGMCPServer
```

## 🔍 Provider Validation

Check your provider configuration:

```python
from src.embedding_provider import validate_provider_config

# Validate OpenAI setup
openai_status = validate_provider_config("openai")
print(f"OpenAI ready: {openai_status['valid']}")

# Validate DashScope setup  
dashscope_status = validate_provider_config("dashscope")
print(f"DashScope ready: {dashscope_status['valid']}")
```

## 🚀 Production Features

- **Rate Limiting** - Configurable request throttling
- **Monitoring** - Prometheus metrics integration
- **Logging** - Structured logging with loguru
- **Error Recovery** - Automatic provider fallback
- **Health Checks** - Built-in validation endpoints

## 📊 Performance

- **Provider Switching** - Zero downtime switching
- **Caching** - Intelligent query engine caching
- **Batch Processing** - Optimized bulk operations
- **Memory Efficient** - Lazy loading and cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Full documentation](docs/embedding_providers.md)
- **Releases**: [GitHub Releases](https://github.com/Renewdxin/mcp/releases)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/Renewdxin/mcp/issues)

## ⭐ Star History

If this project helps you, please consider giving it a star! ⭐

---

**Built with** ❤️ **for the AI community**