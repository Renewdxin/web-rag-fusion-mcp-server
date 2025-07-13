# User Guide

This comprehensive guide helps you get started with the RAG MCP Server, covering installation, configuration, usage, and troubleshooting.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using the Server](#using-the-server)
- [Search Features](#search-features)
- [Document Management](#document-management)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Getting Started

The RAG MCP Server is a Model Context Protocol (MCP) server that provides intelligent search capabilities by combining local knowledge base search with web search. It offers three main tools:

1. **🔍 Knowledge Base Search** - Search your local document collection
2. **🌐 Web Search** - Search the internet via Tavily API
3. **🧠 Smart Search** - Intelligent hybrid search combining both sources

### Key Benefits

- **Comprehensive Search**: Access both local and web information
- **Intelligent Results**: AI-powered relevance scoring and filtering
- **Fast Performance**: Advanced caching and optimization
- **Rich Formatting**: Beautiful, easy-to-read search results
- **Developer-Friendly**: Full MCP protocol compatibility

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for embeddings)
- Tavily API key (for web search)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rag-mcp-server
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install additional format support
pip install python-docx beautifulsoup4 PyPDF2
```

### Step 3: Set Up Environment

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-proj-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# Optional Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
VECTOR_STORE_PATH=./data
SIMILARITY_THRESHOLD=0.75
```

### Step 4: Verify Installation

```bash
python -c "from config import config; config.validate()"
```

If successful, you'll see: `Configuration validation passed`

## Configuration

### Basic Configuration

The server uses environment variables for configuration. Here are the most important settings:

#### Required Settings

```bash
# OpenAI API key for document embeddings
OPENAI_API_KEY=sk-proj-your-key

# Tavily API key for web search
TAVILY_API_KEY=tvly-your-key
```

#### Optional Settings

```bash
# Server Environment
ENVIRONMENT=development          # development, staging, production
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR

# Storage Paths
VECTOR_STORE_PATH=./data        # Vector database storage
COLLECTION_NAME=rag_documents   # ChromaDB collection name

# Search Configuration
SIMILARITY_THRESHOLD=0.75       # Minimum similarity for results (0.0-1.0)
MAX_RESULTS_DEFAULT=10         # Default number of search results

# Performance Settings
MAX_RETRIES=3                  # API retry attempts
TIMEOUT_SECONDS=30             # Request timeout
WEB_SEARCH_TIMEOUT=45          # Web search specific timeout

# Quotas and Limits
TAVILY_QUOTA_LIMIT=1000        # Daily Tavily API quota
```

### Environment-Specific Configuration

#### Development
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
VECTOR_STORE_PATH=./dev_data
SIMILARITY_THRESHOLD=0.6  # Lower threshold for testing
```

#### Production
```bash
ENVIRONMENT=production
LOG_LEVEL=WARNING
VECTOR_STORE_PATH=/opt/rag-server/data
SIMILARITY_THRESHOLD=0.8  # Higher threshold for quality
TAVILY_QUOTA_LIMIT=10000  # Higher quota
```

## Using the Server

### Starting the Server

```bash
# Run the server
python src/mcp_server.py
```

The server will:
1. Validate configuration
2. Initialize vector store and web search
3. Start listening for MCP requests
4. Log status messages

### MCP Client Integration

The server implements the Model Context Protocol and can be used with any MCP-compatible client:

#### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp-server/src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "TAVILY_API_KEY": "your-key"
      }
    }
  }
}
```

#### VS Code Integration

Use with MCP-compatible VS Code extensions:

```json
{
  "mcp.servers": [
    {
      "name": "rag-server",
      "command": "python src/mcp_server.py",
      "args": [],
      "cwd": "/path/to/rag-mcp-server"
    }
  ]
}
```

## Search Features

### 1. Knowledge Base Search

Search your local document collection with intelligent semantic matching.

**Features:**
- 🔍 Semantic similarity search
- 📊 Relevance scoring (0.000-1.000)
- 🎯 Keyword highlighting
- 📁 Clickable source paths
- ⏱️ Search timing display
- 🏷️ Rich metadata display

**Parameters:**
```json
{
  "query": "machine learning algorithms",     // Required: search query
  "top_k": 10,                              // Optional: number of results (1-20)
  "filter_dict": {"file_type": "pdf"},      // Optional: metadata filters
  "include_metadata": true                   // Optional: include metadata
}
```

**Example Response:**
```
🔍 Found 3 results for 'machine learning algorithms'

**1. 🟢 Similarity: 0.892**
📂 Source: [research_paper.pdf](./docs/research_paper.pdf)

📖 Content:
**Machine learning** **algorithms** are computational methods that enable 
systems to learn patterns from data. Common **algorithms** include neural 
networks, decision trees, and support vector machines...

ℹ️ Metadata:
📄 Source: ./docs/research_paper.pdf
📄 Filename: research_paper.pdf
📄 File_Type: pdf
📋 Additional:
   • chunk_index: 5
   • total_chunks: 23
   • quality_score: 0.95

--------------------------------------------------

⏱️ Search completed in 0.234 seconds
🎯 Keywords used: machine, learning, algorithms
```

### 2. Web Search

Search the internet using the powerful Tavily API with intelligent filtering.

**Features:**
- 🌐 Real-time web search
- 🚫 Ad content filtering
- ✅ Quality scoring
- 📋 Result caching (1 hour TTL)
- 📊 API quota tracking
- 🎯 Query optimization

**Parameters:**
```json
{
  "query": "latest AI developments 2024",    // Required: search query
  "max_results": 8,                         // Optional: results (1-20)
  "search_depth": "advanced",               // Optional: basic/advanced
  "include_answer": true,                   // Optional: AI summary
  "include_raw_content": false,             // Optional: raw content
  "exclude_domains": ["spam-site.com"]      // Optional: blocked domains
}
```

**Example Response:**
```
🌐 Found 5 web results for 'latest AI developments 2024'

**1. 🟢 Score: 0.945**
📰 Title: Breakthrough AI Models Transform Industry
🌐 Source: [techcrunch.com](https://techcrunch.com/article)

📄 Content:
Major **AI** breakthroughs in **2024** include advanced language models, 
improved computer vision, and significant progress in robotics. These 
**developments** are reshaping industries from healthcare to finance...

✅ Quality Score: 0.89

--------------------------------------------------

⏱️ Search completed in 1.234 seconds
🔄 Fresh results from web
🎯 Keywords used: latest, developments, 2024
📊 API Usage: 15.2% of daily quota
```

### 3. Smart Search

Intelligent hybrid search that combines local knowledge with web search for comprehensive results.

**Features:**
- 🧠 Intelligent search strategy
- 🔄 Automatic fallback logic
- 📊 Source combination
- 🎯 Relevance optimization
- 📈 Performance tracking

**Parameters:**
```json
{
  "query": "quantum computing applications",  // Required: search query
  "local_max_results": 5,                   // Optional: max local results
  "web_max_results": 3,                     // Optional: max web results
  "local_threshold": 0.7,                   // Optional: local quality threshold
  "min_local_results": 2,                   // Optional: min local before web
  "combine_strategy": "relevance_score",    // Optional: combination method
  "include_sources": true                   // Optional: source information
}
```

**How Smart Search Works:**

1. **Local Search First**: Searches your knowledge base
2. **Quality Assessment**: Evaluates result quality and coverage
3. **Web Supplement**: Adds web results if local results are insufficient
4. **Intelligent Combining**: Merges results using the specified strategy
5. **Relevance Ranking**: Final ranking based on combined relevance scores

## Document Management

### Supported Formats

The server supports multiple document formats with intelligent processing:

| Format | Extension | Features |
|--------|-----------|----------|
| **PDF** | `.pdf` | Text extraction, metadata parsing |
| **Text** | `.txt`, `.md` | Encoding detection, structure preservation |
| **Word** | `.docx` | Content extraction, property parsing |
| **HTML** | `.html`, `.htm` | Clean text extraction, metadata extraction |

### Adding Documents to Knowledge Base

#### Method 1: Direct File Processing

```python
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

# Initialize components
processor = DocumentProcessor(chunk_size=1000, overlap=200)
vector_store = VectorStoreManager()

# Process documents
processed_docs = await processor.process_file(Path("document.pdf"))
documents = processor.convert_to_documents(processed_docs)

# Add to vector store
await vector_store.add_documents(documents)
```

#### Method 2: Batch Directory Processing

```python
# Process entire directory
processed_docs, stats = await processor.process_directory(
    Path("./documents"),
    recursive=True,
    progress_callback=lambda current, total, status: print(f"{current}/{total}: {status}")
)

print(f"Processed {stats.processed_files} files, {stats.total_chunks} chunks")
```

### Document Processing Features

#### Intelligent Chunking

- **Sentence Boundary Detection**: Preserves sentence structure
- **Paragraph Respect**: Maintains paragraph boundaries when possible
- **Configurable Overlap**: Ensures context continuity between chunks
- **Size Optimization**: Balances chunk size with content coherence

#### Metadata Extraction

Automatically extracts comprehensive metadata:

```json
{
  "source": "/path/to/document.pdf",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 1048576,
  "creation_time": "2024-01-15T10:30:00",
  "modification_time": "2024-01-20T14:45:00",
  "file_hash": "sha256:abc123...",
  "chunk_index": 5,
  "total_chunks": 23,
  "chunk_size": 987,
  "processing_time": 0.234
}
```

#### Content Validation

- **Quality Checks**: Ensures content meets minimum quality standards
- **Format Validation**: Verifies content structure and completeness
- **Deduplication**: Prevents duplicate content using content hashing
- **Error Recovery**: Graceful handling of corrupted or unreadable files

## Advanced Usage

### Custom Search Filters

Filter search results using metadata:

```json
// Search only PDF files from last month
{
  "query": "research findings",
  "filter_dict": {
    "file_type": "pdf",
    "creation_time": {"$gte": "2024-01-01"}
  }
}

// Search specific authors or sources
{
  "query": "machine learning",
  "filter_dict": {
    "author": "Dr. Smith",
    "source": {"$regex": "research.*pdf"}
  }
}
```

### Performance Optimization

#### Cache Management

Monitor and manage caches for optimal performance:

```python
# Get cache statistics
from src.web_search import WebSearchManager

web_search = WebSearchManager(api_key="your-key")
stats = await web_search.get_stats()

print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.1f}%")
print(f"Cache size: {stats['cache_stats']['cache_size_mb']:.2f} MB")

# Clear old cache entries
await web_search.cache.clear_cache(older_than_days=7)
```

#### Concurrency Control

Adjust processing concurrency based on your system:

```python
# High-performance server
processor = DocumentProcessor(
    chunk_size=800,      # Smaller chunks for better granularity
    overlap=100,         # Reduced overlap for speed
    max_concurrency=10   # Higher concurrency
)

# Resource-constrained environment
processor = DocumentProcessor(
    chunk_size=1500,     # Larger chunks
    overlap=300,         # More overlap for context
    max_concurrency=2    # Lower concurrency
)
```

### API Integration Examples

#### Programmatic Search

```python
import asyncio
from src.mcp_server import RAGMCPServer

async def search_example():
    server = RAGMCPServer()
    
    # Knowledge base search
    kb_results = await server._search_knowledge_base(
        request_id="req_001",
        query="neural networks",
        top_k=5
    )
    
    # Web search
    web_results = await server._search_web(
        request_id="req_002",
        query="latest neural network research",
        max_results=3
    )
    
    return kb_results, web_results

# Run search
kb_results, web_results = asyncio.run(search_example())
```

#### Batch Operations

```python
async def batch_search(queries):
    server = RAGMCPServer()
    tasks = []
    
    for i, query in enumerate(queries):
        task = server._search_knowledge_base(
            request_id=f"batch_{i}",
            query=query,
            top_k=3
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# Search multiple queries
queries = ["AI ethics", "machine learning", "data science"]
results = await batch_search(queries)
```

## Troubleshooting

### Common Issues

#### 1. Configuration Errors

**Problem**: `ConfigurationError: OPENAI_API_KEY not found`
```bash
# Solution: Set environment variable
export OPENAI_API_KEY=sk-proj-your-key
# Or add to .env file
echo "OPENAI_API_KEY=sk-proj-your-key" >> .env
```

**Problem**: `Invalid similarity threshold: 1.5`
```bash
# Solution: Use valid range (0.0-1.0)
SIMILARITY_THRESHOLD=0.75
```

#### 2. Vector Store Issues

**Problem**: `ChromaDB connection failed`
```bash
# Solution: Check permissions and disk space
ls -la ./data/
df -h ./data/

# Create directory if missing
mkdir -p ./data
chmod 755 ./data
```

**Problem**: `No embeddings found in collection`
```bash
# Solution: Add documents to collection
python -c "
from src.document_processor import process_directory_simple
from pathlib import Path
docs, stats = await process_directory_simple(Path('./documents'))
print(f'Processed {len(docs)} documents')
"
```

#### 3. Web Search Issues

**Problem**: `QuotaExceededError: Daily quota exceeded`
```bash
# Solutions:
# 1. Check usage: Set TAVILY_QUOTA_LIMIT in .env
# 2. Wait for reset (daily quotas reset at midnight UTC)
# 3. Upgrade Tavily plan for higher limits
```

**Problem**: `RateLimitError: Too many requests`
```bash
# Solution: Automatic retry with backoff (built-in)
# Manually wait a few seconds between requests
```

#### 4. Document Processing Issues

**Problem**: `UnsupportedFormatError: Cannot process .xyz files`
```bash
# Solution: Check supported formats
python -c "
from src.document_processor import DocumentProcessor
processor = DocumentProcessor()
print('Supported formats:', processor.get_supported_formats())
"
```

**Problem**: `ContentValidationError: Empty content`
```bash
# Solution: Check file integrity
file ./documents/problem_file.pdf
# Ensure file is not corrupted or password-protected
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug log level
export LOG_LEVEL=DEBUG

# Run with debug output
python src/mcp_server.py 2>&1 | tee debug.log
```

### Performance Issues

#### Slow Search Performance

1. **Check Index Size**: Large collections may need optimization
2. **Adjust Similarity Threshold**: Higher thresholds = faster search
3. **Monitor Memory Usage**: Ensure sufficient RAM for embeddings
4. **Review Query Complexity**: Simplify overly complex queries

#### High Memory Usage

1. **Reduce Batch Size**: Lower `batch_size` in document processing
2. **Clear Caches**: Periodically clear old cache entries
3. **Optimize Chunk Size**: Balance between accuracy and memory
4. **Monitor Embeddings**: Large embedding caches use significant memory

### Health Checks

Monitor server health with built-in checks:

```python
# Check vector store health
vector_store = VectorStoreManager()
stats = await vector_store.get_collection_stats()
print(f"Collection health: {stats['document_count']} documents")

# Check web search health
web_search = WebSearchManager(api_key="your-key")
stats = await web_search.get_stats()
print(f"API health: {stats['search_stats']['success_rate']:.1f}% success rate")
```

## Best Practices

### Configuration Management

1. **Use Environment Variables**: Never hardcode API keys
2. **Separate Environments**: Different configs for dev/staging/production
3. **Regular Key Rotation**: Update API keys periodically
4. **Monitor Usage**: Track API quotas and usage patterns

### Document Organization

1. **Logical Structure**: Organize documents in meaningful directories
2. **Consistent Naming**: Use clear, descriptive filenames
3. **Regular Updates**: Keep document collection current
4. **Quality Control**: Review and validate important documents

### Search Optimization

1. **Tune Similarity Thresholds**: Adjust based on your content and needs
2. **Use Specific Queries**: More specific queries yield better results
3. **Leverage Filters**: Use metadata filters to narrow search scope
4. **Monitor Performance**: Track search times and adjust parameters

### Performance Optimization

1. **Cache Management**: Monitor cache hit rates and sizes
2. **Batch Operations**: Process multiple documents together
3. **Concurrency Tuning**: Adjust based on system resources
4. **Regular Maintenance**: Clean up old cache entries and logs

### Security

1. **API Key Security**: Protect keys, use rotation, monitor access
2. **Input Validation**: Server validates all inputs automatically
3. **Access Control**: Implement appropriate access controls for your use case
4. **Regular Updates**: Keep dependencies and libraries updated

### Monitoring and Maintenance

1. **Log Analysis**: Review logs regularly for errors and performance issues
2. **Usage Tracking**: Monitor API usage and quotas
3. **Performance Monitoring**: Track search times and cache hit rates
4. **Regular Backups**: Backup vector store collections and configurations

## Getting Help

### Documentation Resources

- **API Reference**: Detailed method documentation
- **Architecture Guide**: System design and component relationships
- **Configuration Guide**: Comprehensive configuration options

### Community Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Wiki**: Community-contributed guides and examples

### Professional Support

For enterprise deployments and custom integrations, consider:

- **Consulting Services**: Architecture review and optimization
- **Custom Development**: Feature extensions and integrations
- **Training**: Team training and best practices workshops

---

*Happy searching! 🚀*