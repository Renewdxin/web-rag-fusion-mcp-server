# API Reference

This document provides comprehensive API reference for all classes, methods, and interfaces in the RAG MCP Server system.

## Table of Contents

- [RAGMCPServer](#ragmcpserver)
- [VectorStoreManager](#vectorstoremanager)
- [WebSearchManager](#websearchmanager)
- [DocumentProcessor](#documentprocessor)
- [Configuration](#configuration)
- [Data Types](#data-types)
- [Error Handling](#error-handling)

## RAGMCPServer

The main MCP server class that orchestrates all RAG operations.

### Class Definition

```python
class RAGMCPServer:
    """
    RAG (Retrieval-Augmented Generation) MCP Server.
    
    Provides intelligent search capabilities through local vector database
    and web search integration via Tavily API.
    """
```

### Constructor

```python
def __init__(self) -> None:
    """Initialize the RAG MCP Server."""
```

### Public Methods

#### `async run() -> None`

Run the RAG MCP Server using stdio transport.

**Description:** Performs startup validation, initializes the server, and runs until shutdown is requested.

**Raises:**
- `ConfigurationError`: If configuration validation fails
- `ImportError`: If required dependencies are missing

**Example:**
```python
server = RAGMCPServer()
await server.run()
```

### Tool Methods

#### `async _search_knowledge_base(request_id: str, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None, include_metadata: bool = True) -> List[TextContent]`

Search the local vector knowledge base.

**Parameters:**
- `request_id` (str): Unique request identifier for logging
- `query` (str): Search query string (1-1000 characters)
- `top_k` (int): Number of results to return (1-20, default: 5)
- `filter_dict` (Optional[Dict]): Metadata filters for search results
- `include_metadata` (bool): Whether to include document metadata (default: True)

**Returns:** `List[TextContent]` - Formatted search results

**Features:**
- Query preprocessing (lowercase, special char removal, keyword extraction)
- Result highlighting with markdown bold formatting
- Similarity score display with color coding
- Clickable source file paths
- Comprehensive metadata formatting
- Search execution time tracking

**Example:**
```python
results = await server._search_knowledge_base(
    request_id="req_123",
    query="machine learning algorithms",
    top_k=10,
    filter_dict={"file_type": "pdf"},
    include_metadata=True
)
```

#### `async _search_web(request_id: str, query: str, max_results: int = 5, search_depth: str = "basic", include_answer: bool = True, include_raw_content: bool = False, exclude_domains: Optional[List[str]] = None) -> List[TextContent]`

Search the web using Tavily API.

**Parameters:**
- `request_id` (str): Unique request identifier for logging
- `query` (str): Search query string (1-400 characters)
- `max_results` (int): Maximum number of results (1-20, default: 5)
- `search_depth` (str): Search depth ("basic" or "advanced", default: "basic")
- `include_answer` (bool): Whether to include AI answer (default: True)
- `include_raw_content` (bool): Whether to include raw content (default: False)
- `exclude_domains` (List[str]): Domains to exclude from results

**Returns:** `List[TextContent]` - Formatted web search results

**Features:**
- Exponential backoff retry mechanism
- Content filtering for ads and irrelevant content
- Query optimization with stop word removal
- Result caching with 1-hour TTL
- API usage tracking and quota management
- Quality scoring for content

**Example:**
```python
results = await server._search_web(
    request_id="req_456",
    query="latest AI developments",
    max_results=8,
    search_depth="advanced",
    exclude_domains=["example.com", "spam-site.com"]
)
```

#### `async _smart_search_internal(request_id: str, query: str, local_max_results: int = 5, web_max_results: int = 3, local_threshold: float = 0.7, min_local_results: int = 2, combine_strategy: str = "relevance_score", include_sources: bool = True) -> List[TextContent]`

Perform intelligent hybrid search combining local and web results.

**Parameters:**
- `request_id` (str): Unique request identifier
- `query` (str): Search query string
- `local_max_results` (int): Max local results (1-20, default: 5)
- `web_max_results` (int): Max web results (0-10, default: 3)
- `local_threshold` (float): Local similarity threshold (0.0-1.0, default: 0.7)
- `min_local_results` (int): Min local results before web search (0-10, default: 2)
- `combine_strategy` (str): Result combination strategy ("interleave", "local_first", "relevance_score")
- `include_sources` (bool): Include source information (default: True)

**Returns:** `List[TextContent]` - Combined search results

## VectorStoreManager

Manages ChromaDB vector store operations with OpenAI embeddings.

### Class Definition

```python
class VectorStoreManager:
    """
    Production-ready vector store manager with clean architecture.
    
    Features:
    - Pluggable embedding functions
    - Async-first design with proper resource management
    - Comprehensive error handling and validation
    - Performance optimizations and monitoring
    - Type safety and clean interfaces
    """
```

### Constructor

```python
def __init__(
    self,
    collection_name: str = "rag_documents",
    persist_directory: Optional[Union[str, Path]] = None,
    embedding_function: Optional[EmbeddingFunction] = None,
    enable_deduplication: bool = True,
    connection_retries: int = 3,
    connection_retry_delay: float = 1.0
) -> None:
```

**Parameters:**
- `collection_name` (str): Name of the ChromaDB collection
- `persist_directory` (Optional[Union[str, Path]]): Directory to persist the database
- `embedding_function` (Optional[EmbeddingFunction]): Embedding function to use
- `enable_deduplication` (bool): Whether to enable content deduplication
- `connection_retries` (int): Number of connection retry attempts
- `connection_retry_delay` (float): Delay between connection retries

### Public Methods

#### `async initialize_collection(metadata: Optional[Dict[str, Any]] = None, embedding_function: Optional[EmbeddingFunction] = None) -> Collection`

Initialize collection with idempotent creation.

**Parameters:**
- `metadata` (Optional[Dict]): Collection metadata
- `embedding_function` (Optional[EmbeddingFunction]): Custom embedding function

**Returns:** `Collection` - ChromaDB collection instance

#### `async add_documents(documents: List[Document], batch_size: int = 100, progress_callback: Optional[ProgressCallback] = None, overwrite: bool = False) -> OperationStats`

Add documents with comprehensive processing and monitoring.

**Parameters:**
- `documents` (List[Document]): List of documents to add
- `batch_size` (int): Batch size for processing (1-1000, default: 100)
- `progress_callback` (Optional[ProgressCallback]): Progress callback function
- `overwrite` (bool): Whether to overwrite existing documents

**Returns:** `OperationStats` - Operation statistics

**Features:**
- Document validation and deduplication
- Batch processing with progress tracking
- Comprehensive error handling
- Performance monitoring

#### `async similarity_search_with_score(query: str, k: int = 10, filter_dict: Optional[Dict[str, Any]] = None, include_metadata: bool = True) -> List[SearchResult]`

Perform similarity search with scores.

**Parameters:**
- `query` (str): Search query
- `k` (int): Number of results to return (1-100)
- `filter_dict` (Optional[Dict]): Metadata filters
- `include_metadata` (bool): Whether to include metadata

**Returns:** `List[SearchResult]` - Search results with scores

#### `async backup_collection(backup_path: Union[str, Path], include_embeddings: bool = True, compress: bool = True) -> Dict[str, Any]`

Create collection backup with async file operations.

**Parameters:**
- `backup_path` (Union[str, Path]): Path for backup
- `include_embeddings` (bool): Whether to include embeddings
- `compress` (bool): Whether to compress backup

**Returns:** `Dict[str, Any]` - Backup statistics

#### `async get_collection_stats() -> Dict[str, Any]`

Get comprehensive collection statistics.

**Returns:** `Dict[str, Any]` - Collection statistics including document count, metadata, schema info

#### `async validate_embeddings(sample_size: int = 10) -> Dict[str, Any]`

Validate embedding dimensions and consistency.

**Parameters:**
- `sample_size` (int): Sample size for validation (1-100)

**Returns:** `Dict[str, Any]` - Validation results

## WebSearchManager

Manages web search operations via Tavily API with advanced features.

### Class Definition

```python
class WebSearchManager:
    """
    Comprehensive web search manager with Tavily API integration.
    
    Features:
    - Tavily API integration with timeout and retry mechanisms
    - Exponential backoff retry strategy
    - Comprehensive error handling
    - Result caching with TTL
    - Content filtering and query optimization
    - API usage tracking
    """
```

### Constructor

```python
def __init__(
    self,
    api_key: str,
    timeout: int = 30,
    max_retries: int = 3,
    cache_ttl_hours: int = 1,
    cache_max_size: int = 1000,
    quota_limit: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> None:
```

**Parameters:**
- `api_key` (str): Tavily API key (required)
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum retry attempts
- `cache_ttl_hours` (int): Cache TTL in hours
- `cache_max_size` (int): Maximum cache entries
- `quota_limit` (Optional[int]): Daily API quota limit
- `cache_dir` (Optional[Path]): Cache directory path

### Public Methods

#### `async search(query: str, max_results: int = 5, search_depth: str = "basic", include_answer: bool = True, include_raw_content: bool = False, exclude_domains: Optional[List[str]] = None) -> Tuple[List[WebSearchResult], Dict[str, Any]]`

Perform web search with comprehensive error handling and optimization.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of results (1-20)
- `search_depth` (str): Search depth ("basic" or "advanced")
- `include_answer` (bool): Whether to include AI answer
- `include_raw_content` (bool): Whether to include raw content
- `exclude_domains` (Optional[List[str]]): Domains to exclude

**Returns:** `Tuple[List[WebSearchResult], Dict[str, Any]]` - Search results and metadata

**Features:**
- Query optimization (stop word removal, key phrase extraction)
- Content filtering (ad removal, quality scoring)
- Result caching with intelligent cache keys
- API usage tracking and quota management
- Exponential backoff retry with smart error handling

#### `async get_stats() -> Dict[str, Any]`

Get comprehensive search statistics.

**Returns:** `Dict[str, Any]` - Search, cache, and usage statistics

#### `async close() -> None`

Clean up resources.

## DocumentProcessor

Processes various document formats with intelligent chunking.

### Class Definition

```python
class DocumentProcessor:
    """
    Main document processor with multi-format support and intelligent chunking.
    
    Features:
    - Multi-format file support (PDF, TXT, MD, DOCX, HTML)
    - Intelligent chunking with sentence boundaries
    - Comprehensive metadata extraction
    - Async batch processing with concurrency control
    - Progress tracking and caching
    - Content validation and deduplication
    """
```

### Constructor

```python
def __init__(
    self,
    chunk_size: int = 1000,
    overlap: int = 200,
    max_concurrency: int = 5,
    cache_dir: Optional[Path] = None,
    custom_tags: Optional[Dict[str, str]] = None
) -> None:
```

**Parameters:**
- `chunk_size` (int): Target size for text chunks
- `overlap` (int): Overlap between chunks
- `max_concurrency` (int): Maximum concurrent file processing
- `cache_dir` (Optional[Path]): Directory for processing cache
- `custom_tags` (Optional[Dict[str, str]]): Custom tags to add to all documents

### Public Methods

#### `async process_file(file_path: Path, preserve_paragraphs: bool = True, custom_metadata: Optional[Dict[str, Any]] = None) -> List[ProcessedDocument]`

Process single file into chunks with metadata.

**Parameters:**
- `file_path` (Path): Path to file to process
- `preserve_paragraphs` (bool): Whether to preserve paragraph boundaries
- `custom_metadata` (Optional[Dict]): Additional metadata to include

**Returns:** `List[ProcessedDocument]` - List of processed document chunks

#### `async process_files(file_paths: List[Path], preserve_paragraphs: bool = True, progress_callback: Optional[ProgressCallback] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[ProcessedDocument], ProcessingStats]`

Process multiple files with batch processing and progress tracking.

**Parameters:**
- `file_paths` (List[Path]): List of file paths to process
- `preserve_paragraphs` (bool): Whether to preserve paragraph boundaries
- `progress_callback` (Optional[ProgressCallback]): Progress callback function
- `custom_metadata` (Optional[Dict]): Additional metadata for all files

**Returns:** `Tuple[List[ProcessedDocument], ProcessingStats]` - Processed documents and statistics

#### `async process_directory(directory: Path, recursive: bool = True, file_pattern: str = "*", progress_callback: Optional[ProgressCallback] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[ProcessedDocument], ProcessingStats]`

Process all supported files in a directory.

**Parameters:**
- `directory` (Path): Directory to process
- `recursive` (bool): Whether to process subdirectories
- `file_pattern` (str): File pattern to match
- `progress_callback` (Optional[ProgressCallback]): Progress callback function
- `custom_metadata` (Optional[Dict]): Additional metadata for all files

**Returns:** `Tuple[List[ProcessedDocument], ProcessingStats]` - Processed documents and statistics

#### `get_supported_formats() -> List[str]`

Get list of all supported file formats.

**Returns:** `List[str]` - List of supported file extensions

#### `can_process(file_path: Path) -> bool`

Check if file can be processed.

**Parameters:**
- `file_path` (Path): Path to file

**Returns:** `bool` - Whether file can be processed

## Configuration

### Config Class

```python
class Config:
    """Configuration management with validation and environment support."""
```

#### Key Properties

- `OPENAI_API_KEY` (str): OpenAI API key
- `TAVILY_API_KEY` (str): Tavily API key  
- `ENVIRONMENT` (str): Runtime environment
- `LOG_LEVEL` (str): Logging level
- `VECTOR_STORE_PATH` (str): Vector store directory
- `SIMILARITY_THRESHOLD` (float): Similarity threshold
- `MAX_RETRIES` (int): Maximum retry attempts
- `TIMEOUT_SECONDS` (int): Request timeout

#### Methods

##### `validate() -> None`

Validate all configuration settings.

**Raises:** `ConfigurationError` - If validation fails

## Data Types

### Document

```python
@dataclass
class Document:
    """Document with content hash for deduplication."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    content_hash: Optional[str] = None
```

### SearchResult

```python
@dataclass
class SearchResult:
    """Search result with document and score."""
    document: Document
    score: float
```

### WebSearchResult

```python
@dataclass
class WebSearchResult:
    """Structured web search result."""
    title: str
    url: str
    content: str
    snippet: str = ""
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_domain: str = ""
    content_type: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ProcessedDocument

```python
@dataclass
class ProcessedDocument:
    """Container for processed document with metadata."""
    content: str
    metadata: Dict[str, Any]
    source_file: Path
    chunk_index: Optional[int] = None
    processing_time: float = 0.0
```

### OperationStats

```python
@dataclass
class OperationStats:
    """Statistics for batch operations."""
    total_items: int
    processed_items: int
    failed_items: int
    execution_time: float
    success_rate: float
    additional_stats: Dict[str, Any] = field(default_factory=dict)
```

### ProcessingStats

```python
@dataclass
class ProcessingStats:
    """Statistics for document processing operations."""
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    execution_time: float
    bytes_processed: int
    cache_hits: int = 0
    cache_misses: int = 0
    files_by_type: Dict[str, int] = field(default_factory=dict)
```

### SearchStats

```python
@dataclass
class SearchStats:
    """Search operation statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    quota_used: int = 0
    quota_remaining: Optional[int] = None
```

## Error Handling

### Exception Hierarchy

```python
# Base exceptions
class VectorStoreError(Exception): ...
class WebSearchError(Exception): ...
class DocumentProcessingError(Exception): ...
class ConfigurationError(Exception): ...

# Specific exceptions
class EmbeddingError(VectorStoreError): ...
class ValidationError(VectorStoreError): ...
class RateLimitError(WebSearchError): ...
class QuotaExceededError(WebSearchError): ...
class UnsupportedFormatError(DocumentProcessingError): ...
class ContentValidationError(DocumentProcessingError): ...
```

### Error Handling Patterns

#### Graceful Degradation

```python
try:
    results = await web_search.search(query)
except QuotaExceededError:
    # Fallback to local search only
    results = await local_search.search(query)
```

#### Retry with Backoff

```python
for attempt in range(max_retries):
    try:
        return await api_call()
    except RateLimitError as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
            continue
        raise
```

## Usage Examples

### Basic Server Setup

```python
from src.mcp_server import RAGMCPServer

# Initialize and run server
server = RAGMCPServer()
await server.run()
```

### Vector Store Operations

```python
from src.vector_store import VectorStoreManager, Document

# Initialize vector store
vector_store = VectorStoreManager()
await vector_store.initialize_collection()

# Add documents
documents = [
    Document(page_content="Example content", metadata={"source": "test.txt"})
]
stats = await vector_store.add_documents(documents)

# Search documents
results = await vector_store.similarity_search_with_score("example query", k=5)
```

### Web Search Operations

```python
from src.web_search import WebSearchManager

# Initialize web search
web_search = WebSearchManager(api_key="your-tavily-key")

# Perform search
results, metadata = await web_search.search(
    query="AI developments",
    max_results=10,
    search_depth="advanced"
)
```

### Document Processing

```python
from src.document_processor import DocumentProcessor
from pathlib import Path

# Initialize processor
processor = DocumentProcessor(chunk_size=800, overlap=100)

# Process single file
results = await processor.process_file(Path("document.pdf"))

# Process directory
results, stats = await processor.process_directory(
    Path("./documents"),
    recursive=True
)
```