"""
Refactored Vector Store Manager for ChromaDB integration.

This module provides a clean, production-ready architecture for vector store operations:
- Abstract interfaces for pluggable embedding functions
- Async-first design with proper resource management
- Comprehensive error handling and validation
- Performance optimizations and caching
- Type safety with proper annotations

Architecture:
- EmbeddingFunction: Abstract base for embedding implementations
- CacheManager: Async cache operations with connection pooling
- DocumentProcessor: Handles deduplication and validation
- MetadataManager: Collection metadata operations
- VectorStoreManager: Main orchestrator class
"""

import asyncio
import hashlib
import json
import logging
import pickle
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

# Async file operations
try:
    import aiofiles
    import aiosqlite

    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False
    aiofiles = None
    aiosqlite = None

# OpenAI imports
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

# ChromaDB imports
try:
    import chromadb
    from chromadb.api.types import Collection
    from chromadb.config import Settings
    from chromadb.errors import ConnectionError as ChromaConnectionError
    from chromadb.errors import InvalidCollectionException

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Collection = Any

# Local imports
try:
    from config.settings import ConfigurationError, config
except ImportError:
    config = None
    ConfigurationError = Exception


# Type definitions
class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int) -> None: ...


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""

    embeddings: List[List[float]]
    dimensions: int
    cache_hits: int = 0
    cache_misses: int = 0
    execution_time: float = 0.0


@dataclass
class Document:
    """Document with content hash for deduplication."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    content_hash: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.content_hash is None:
            self.content_hash = hashlib.sha256(
                self.page_content.encode("utf-8")
            ).hexdigest()


@dataclass
class SearchResult:
    """Search result with document and score."""

    document: Document
    score: float


@dataclass
class OperationStats:
    """Statistics for batch operations."""

    total_items: int
    processed_items: int
    failed_items: int
    execution_time: float
    success_rate: float
    additional_stats: Dict[str, Any] = field(default_factory=dict)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class EmbeddingError(VectorStoreError):
    """Exception for embedding-related errors."""

    pass


class ValidationError(VectorStoreError):
    """Exception for validation errors."""

    pass


# Abstract interfaces
class EmbeddingFunction(ABC):
    """Abstract base class for embedding functions."""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple documents."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Optional[int]:
        """Expected embedding dimensions."""
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass


class CacheManager:
    """Async cache manager with connection pooling."""

    def __init__(self, cache_path: Path, pool_size: int = 5):
        self.cache_path = cache_path
        self.pool_size = pool_size
        self._connection_pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.CacheManager")

    async def initialize(self) -> None:
        """Initialize cache database and connection pool."""
        if self._initialized:
            return

        if not ASYNC_IO_AVAILABLE:
            raise ImportError("aiosqlite required for async cache operations")

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.cache_path)
            await self._connection_pool.put(conn)

        # Create tables
        async with self._get_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings
                (
                    content_hash
                    TEXT
                    PRIMARY
                    KEY,
                    model
                    TEXT
                    NOT
                    NULL,
                    embedding
                    BLOB
                    NOT
                    NULL,
                    dimensions
                    INTEGER
                    NOT
                    NULL,
                    created_at
                    TIMESTAMP
                    DEFAULT
                    CURRENT_TIMESTAMP,
                    access_count
                    INTEGER
                    DEFAULT
                    1,
                    last_accessed
                    TIMESTAMP
                    DEFAULT
                    CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_hash
                    ON embeddings(model, content_hash)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at
                    ON embeddings(created_at)
                """
            )
            await conn.commit()

        self._initialized = True
        self.logger.info(f"Cache initialized at {self.cache_path}")

    @asynccontextmanager
    async def _get_connection(self):
        """Get connection from pool."""
        if not self._initialized:
            await self.initialize()

        conn = await self._connection_pool.get()
        try:
            yield conn
        finally:
            await self._connection_pool.put(conn)

    async def get_embedding(
            self, content_hash: str, model: str
    ) -> Optional[List[float]]:
        """Retrieve embedding from cache."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT embedding, dimensions FROM embeddings WHERE content_hash = ? AND model = ?",
                    (content_hash, model),
                )
                row = await cursor.fetchone()

                if row:
                    # Update access statistics
                    await conn.execute(
                        "UPDATE embeddings SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE content_hash = ?",
                        (content_hash,),
                    )
                    await conn.commit()

                    embedding = pickle.loads(row[0])
                    return embedding

        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def store_embedding(
            self, content_hash: str, model: str, embedding: List[float]
    ) -> None:
        """Store embedding in cache."""
        try:
            async with self._get_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO embeddings (content_hash, model, embedding, dimensions) VALUES (?, ?, ?, ?)",
                    (content_hash, model, pickle.dumps(embedding), len(embedding)),
                )
                await conn.commit()
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")

    async def get_stats(self, model: str) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*), AVG(access_count), SUM(access_count) FROM embeddings WHERE model = ?",
                    (model,),
                )
                count, avg_access, total_access = await cursor.fetchone()

                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ? AND created_at > datetime('now', '-1 day')",
                    (model,),
                )
                recent_count = (await cursor.fetchone())[0]

                return {
                    "total_cached_embeddings": count or 0,
                    "average_access_count": round(avg_access or 0, 2),
                    "total_cache_hits": total_access or 0,
                    "recent_additions": recent_count or 0,
                    "cache_size_mb": (
                        self.cache_path.stat().st_size / (1024 * 1024)
                        if self.cache_path.exists()
                        else 0
                    ),
                }
        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {}

    async def clear_cache(
            self, model: str, older_than_days: Optional[int] = None
    ) -> int:
        """Clear cache entries."""
        try:
            async with self._get_connection() as conn:
                if older_than_days:
                    cursor = await conn.execute(
                        "DELETE FROM embeddings WHERE model = ? AND created_at < datetime('now', ? || ' days')",
                        (model, f"-{older_than_days}"),
                    )
                else:
                    cursor = await conn.execute(
                        "DELETE FROM embeddings WHERE model = ?", (model,)
                    )

                deleted_count = cursor.rowcount
                await conn.commit()
                return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return 0

    async def close(self) -> None:
        """Close all connections in pool."""
        while not self._connection_pool.empty():
            try:
                conn = await asyncio.wait_for(self._connection_pool.get(), timeout=1.0)
                await conn.close()
            except asyncio.TimeoutError:
                break

        self._initialized = False
        self.logger.info("Cache connections closed")


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * self.max_requests / self.time_window,
            )
            self.last_update = now

            if self.tokens < 1:
                # Wait for next token
                wait_time = (1 - self.tokens) * self.time_window / self.max_requests
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """OpenAI embedding function with async operations and caching."""

    def __init__(
            self,
            api_key: str,
            model: str = "text-embedding-3-small",
            cache_dir: Optional[Path] = None,
            max_requests_per_minute: int = 3000,
            max_retries: int = 3,
            retry_delay: float = 1.0,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required: pip install openai")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._dimensions: Optional[int] = None

        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests_per_minute, 60.0)

        # Caching
        cache_path = (cache_dir or Path(tempfile.gettempdir())) / "openai_embeddings.db"
        self.cache = CacheManager(cache_path)

        self.logger = logging.getLogger(f"{__name__}.OpenAIEmbeddingFunction")

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    def _get_content_hash(self, text: str) -> str:
        """Generate content hash."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _create_embeddings_with_retry(
            self, texts: List[str]
    ) -> List[List[float]]:
        """Create embeddings with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                await self.rate_limiter.acquire()

                response = await self.client.embeddings.create(
                    model=self.model, input=texts
                )

                embeddings = [data.embedding for data in response.data]

                # Set dimensions on first successful call
                if embeddings and self._dimensions is None:
                    self._dimensions = len(embeddings[0])
                    self.logger.info(f"Set embedding dimensions to {self._dimensions}")

                return embeddings

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Embedding creation failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise EmbeddingError(
                        f"Failed to create embeddings after {self.max_retries + 1} attempts: {e}"
                    )

    async def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple documents with caching."""
        if not texts:
            return EmbeddingResult(embeddings=[], dimensions=0)

        start_time = time.time()
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        cache_hits = 0
        cache_misses = 0

        # Check cache for each text
        for i, text in enumerate(texts):
            content_hash = self._get_content_hash(text)
            cached_embedding = await self.cache.get_embedding(content_hash, self.model)

            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                cache_misses += 1

        # Create embeddings for uncached texts
        if texts_to_embed:
            new_embeddings = await self._create_embeddings_with_retry(texts_to_embed)

            # Cache and fill results
            for j, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                content_hash = self._get_content_hash(text)
                await self.cache.store_embedding(content_hash, self.model, embedding)

                original_index = indices_to_embed[j]
                embeddings[original_index] = embedding

        execution_time = time.time() - start_time

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=self._dimensions
                       or (len(embeddings[0]) if embeddings and embeddings[0] else 0),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            execution_time=execution_time,
        )

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = await self.embed_documents([text])
        return result.embeddings[0] if result.embeddings else []

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_stats(self.model)

    async def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear embedding cache."""
        return await self.cache.clear_cache(self.model, older_than_days)

    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()
        await self.cache.close()


class DocumentProcessor:
    """Handles document validation and deduplication."""

    def __init__(self, enable_deduplication: bool = True):
        self.enable_deduplication = enable_deduplication
        self._content_hashes: set = set()
        self.logger = logging.getLogger(f"{__name__}.DocumentProcessor")

    def validate_document(self, document: Document) -> None:
        """Validate document content and metadata."""
        if not document.page_content or not document.page_content.strip():
            raise ValidationError(f"Document {document.id} has empty content")

        if len(document.page_content) > 1_000_000:  # 1MB limit
            raise ValidationError(f"Document {document.id} content too large")

        # Validate metadata is JSON serializable
        try:
            json.dumps(document.metadata)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Document {document.id} has non-serializable metadata: {e}"
            )

    def add_content_hash(self, content_hash: str) -> None:
        """Add content hash to deduplication set."""
        self._content_hashes.add(content_hash)

    def load_content_hashes(self, hashes: List[str]) -> None:
        """Load existing content hashes."""
        self._content_hashes.update(hashes)
        self.logger.info(f"Loaded {len(hashes)} content hashes for deduplication")

    def process_documents(
            self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """Process documents with validation and deduplication."""
        unique_docs = []
        duplicate_docs = []

        for doc in documents:
            try:
                self.validate_document(doc)

                if (
                        self.enable_deduplication
                        and doc.content_hash in self._content_hashes
                ):
                    duplicate_docs.append(doc)
                else:
                    unique_docs.append(doc)
                    if self.enable_deduplication:
                        self._content_hashes.add(doc.content_hash)

            except ValidationError as e:
                self.logger.warning(f"Document validation failed: {e}")
                # Could add to failed_docs list if needed

        return unique_docs, duplicate_docs


class MetadataManager:
    """Manages collection metadata operations."""

    def __init__(self, collection: Collection):
        self.collection = collection
        self.logger = logging.getLogger(f"{__name__}.MetadataManager")

    def get_metadata(self) -> Dict[str, Any]:
        """Get current collection metadata."""
        return self.collection.metadata or {}

    def update_metadata(
            self, updates: Dict[str, Any], merge: bool = True
    ) -> Dict[str, Any]:
        """Update collection metadata with versioning."""
        try:
            current_metadata = self.get_metadata() if merge else {}

            # Add version tracking
            current_metadata.setdefault("version", 0)
            current_metadata["version"] += 1
            current_metadata["last_updated"] = datetime.utcnow().isoformat()
            current_metadata["updated_by"] = "VectorStoreManager"

            # Merge updates
            current_metadata.update(updates)

            # Update collection
            self.collection.modify(metadata=current_metadata)

            self.logger.info(
                f"Updated metadata to version {current_metadata['version']}"
            )
            return current_metadata

        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")
            raise VectorStoreError(f"Metadata update failed: {e}")

    def get_schema_info(self) -> Dict[str, Any]:
        """Analyze collection schema."""
        try:
            sample = self.collection.get(limit=10, include=["documents", "metadatas"])

            schema_info = {
                "document_count": self.collection.count(),
                "sample_size": len(sample["documents"]) if sample["documents"] else 0,
                "metadata_fields": set(),
                "metadata_types": {},
                "average_doc_length": 0,
            }

            if sample["documents"]:
                doc_lengths = [len(doc) for doc in sample["documents"]]
                schema_info.update(
                    {
                        "average_doc_length": sum(doc_lengths) / len(doc_lengths),
                        "min_doc_length": min(doc_lengths),
                        "max_doc_length": max(doc_lengths),
                    }
                )

                if sample["metadatas"]:
                    for metadata in sample["metadatas"]:
                        if metadata:
                            for key, value in metadata.items():
                                schema_info["metadata_fields"].add(key)
                                schema_info["metadata_types"][key] = type(
                                    value
                                ).__name__

            schema_info["metadata_fields"] = list(schema_info["metadata_fields"])
            return schema_info

        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {"error": str(e)}


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

    def __init__(
            self,
            collection_name: str = "rag_documents",
            persist_directory: Optional[Union[str, Path]] = None,
            embedding_function: Optional[EmbeddingFunction] = None,
            enable_deduplication: bool = True,
            connection_retries: int = 3,
            connection_retry_delay: float = 1.0,
    ):
        """
        Initialize VectorStoreManager.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: Embedding function to use
            enable_deduplication: Whether to enable content deduplication
            connection_retries: Number of connection retry attempts
            connection_retry_delay: Delay between connection retries
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB required: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = Path(
            persist_directory or (config.VECTOR_STORE_PATH if config else "./data")
        ).parent
        self.connection_retries = connection_retries
        self.connection_retry_delay = connection_retry_delay

        # Initialize components
        self.embedding_function = (
                embedding_function or self._create_default_embedding_function()
        )
        self.document_processor = DocumentProcessor(enable_deduplication)

        # State
        self._client: Optional[Any] = None
        self._collection: Optional[Collection] = None
        self._metadata_manager: Optional[MetadataManager] = None
        self._is_initialized = False

        # Health tracking
        self._last_health_check = 0.0
        self._health_check_interval = 30.0

        # Logging
        self.logger = logging.getLogger(f"{__name__}.VectorStoreManager")

    def _create_default_embedding_function(self) -> Optional[EmbeddingFunction]:
        """Create default OpenAI embedding function if available."""
        if OPENAI_AVAILABLE and config and config.OPENAI_API_KEY:
            try:
                return OpenAIEmbeddingFunction(
                    api_key=config.OPENAI_API_KEY,
                    cache_dir=self.persist_directory / "embeddings_cache",
                )
            except Exception as e:
                self.logger.warning(f"Failed to create default embedding function: {e}")
        return None

    async def _initialize_client(self) -> Optional[Any]:
        """Initialize ChromaDB client with retry logic."""
        if not CHROMADB_AVAILABLE:
            raise VectorStoreError(
                "ChromaDB is not available - please install chromadb"
            )

        for attempt in range(self.connection_retries + 1):
            try:
                self.logger.info(
                    f"Initializing ChromaDB client (attempt {attempt + 1})"
                )

                # Ensure directory exists
                self.persist_directory.mkdir(parents=True, exist_ok=True)

                # Configure ChromaDB
                settings = Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False,
                    allow_reset=True,
                )

                client = chromadb.PersistentClient(
                    path=str(self.persist_directory), settings=settings
                )

                # Test connection
                client.heartbeat()
                collections = client.list_collections()

                self.logger.info(
                    f"ChromaDB client initialized. Found {len(collections)} collections"
                )
                self._last_health_check = time.time()

                return client

            except Exception as e:
                if attempt < self.connection_retries:
                    wait_time = self.connection_retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Connection failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise VectorStoreError(
                        f"Failed to initialize ChromaDB after {self.connection_retries + 1} attempts: {e}"
                    )

    async def _ensure_healthy_connection(self) -> Optional[Any]:
        """Ensure ChromaDB connection is healthy."""
        current_time = time.time()

        if (
                self._client is None
                or current_time - self._last_health_check > self._health_check_interval
        ):

            if self._client is not None:
                try:
                    self._client.heartbeat()
                    self._last_health_check = current_time
                    return self._client
                except Exception as e:
                    self.logger.warning(f"Health check failed: {e}. Reconnecting...")
                    self._client = None
                    self._collection = None
                    self._metadata_manager = None

            self._client = await self._initialize_client()

        return self._client

    async def initialize_collection(
            self,
            metadata: Optional[Dict[str, Any]] = None,
            embedding_function: Optional[EmbeddingFunction] = None,
    ) -> Collection:
        """Initialize collection with idempotent creation."""
        try:
            client = await self._ensure_healthy_connection()
            ef = embedding_function or self.embedding_function

            collection_metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "VectorStoreManager",
                "version": 1,
                **(metadata or {}),
            }

            try:
                # Try to get existing collection
                collection = client.get_collection(
                    name=self.collection_name, embedding_function=ef
                )
                self.logger.info(f"Found existing collection '{self.collection_name}'")

            except InvalidCollectionException:
                # Create new collection
                collection = client.create_collection(
                    name=self.collection_name,
                    metadata=collection_metadata,
                    embedding_function=ef,
                )
                self.logger.info(f"Created new collection '{self.collection_name}'")

            # Initialize components
            self._collection = collection
            self._metadata_manager = MetadataManager(collection)
            self._is_initialized = True

            # Load existing content hashes for deduplication
            if self.document_processor.enable_deduplication:
                await self._load_content_hashes()

            count = collection.count()
            self.logger.info(
                f"Collection '{self.collection_name}' ready with {count} documents"
            )

            return collection

        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise VectorStoreError(f"Collection initialization failed: {e}")

    async def _load_content_hashes(self) -> None:
        """Load existing content hashes efficiently."""
        if not self._collection:
            return

        try:
            # Get content hashes in batches to avoid memory issues
            batch_size = 1000
            offset = 0
            hashes = []

            while True:
                batch = self._collection.get(
                    limit=batch_size, offset=offset, include=["metadatas"]
                )

                if not batch["metadatas"]:
                    break

                for metadata in batch["metadatas"]:
                    if metadata and "content_hash" in metadata:
                        hashes.append(metadata["content_hash"])

                if len(batch["metadatas"]) < batch_size:
                    break

                offset += batch_size

            self.document_processor.load_content_hashes(hashes)

        except Exception as e:
            self.logger.warning(f"Failed to load content hashes: {e}")

    async def add_documents(
            self,
            documents: List[Document],
            batch_size: int = 100,
            progress_callback: Optional[ProgressCallback] = None,
            overwrite: bool = False,
    ) -> OperationStats:
        """
        Add documents with comprehensive processing and monitoring.

        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
            progress_callback: Optional progress callback
            overwrite: Whether to overwrite existing documents

        Returns:
            Operation statistics
        """
        if not documents:
            raise ValidationError("Documents list cannot be empty")

        if batch_size <= 0 or batch_size > 1000:
            raise ValidationError("Batch size must be between 1 and 1000")

        # Ensure collection is initialized
        if not self._is_initialized:
            await self.initialize_collection()

        start_time = time.time()
        total_docs = len(documents)

        # Process documents (validation and deduplication)
        unique_docs, duplicate_docs = self.document_processor.process_documents(
            documents
        )

        self.logger.info(
            f"Processing {total_docs} documents: {len(unique_docs)} unique, "
            f"{len(duplicate_docs)} duplicates"
        )

        if not unique_docs:
            return OperationStats(
                total_items=total_docs,
                processed_items=0,
                failed_items=0,
                execution_time=time.time() - start_time,
                success_rate=100.0,
                additional_stats={
                    "unique_documents": 0,
                    "duplicate_documents": len(duplicate_docs),
                    "collection_size": self._collection.count(),
                },
            )

        # Process in batches
        processed_docs = 0
        failed_docs = 0

        try:
            for i in range(0, len(unique_docs), batch_size):
                batch = unique_docs[i: i + batch_size]
                batch_start = time.time()

                try:
                    await self._process_document_batch(batch, overwrite)
                    processed_docs += len(batch)

                    batch_time = time.time() - batch_start
                    self.logger.debug(
                        f"Batch {i // batch_size + 1} processed in {batch_time:.2f}s"
                    )

                    # Call progress callback
                    if progress_callback:
                        try:
                            progress_callback(processed_docs, len(unique_docs))
                        except Exception as e:
                            self.logger.warning(f"Progress callback failed: {e}")

                except Exception as e:
                    batch_failed = len(batch)
                    failed_docs += batch_failed
                    self.logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Critical error during document addition: {e}")
            raise VectorStoreError(f"Document addition failed: {e}")

        # Calculate statistics
        execution_time = time.time() - start_time
        success_rate = (
            ((processed_docs) / len(unique_docs)) * 100 if unique_docs else 100.0
        )

        stats = OperationStats(
            total_items=total_docs,
            processed_items=processed_docs,
            failed_items=failed_docs,
            execution_time=execution_time,
            success_rate=success_rate,
            additional_stats={
                "unique_documents": len(unique_docs),
                "duplicate_documents": len(duplicate_docs),
                "documents_per_second": (
                    processed_docs / execution_time if execution_time > 0 else 0
                ),
                "collection_size": self._collection.count(),
            },
        )

        self.logger.info(
            f"Document addition completed: {processed_docs}/{len(unique_docs)} processed "
            f"({len(duplicate_docs)} duplicates) in {execution_time:.2f}s"
        )

        return stats

    async def _process_document_batch(
            self, documents: List[Document], overwrite: bool
    ) -> None:
        """Process a batch of documents."""
        if not self._collection:
            raise VectorStoreError("Collection not initialized")

        # Prepare data
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            ids.append(doc.id)
            texts.append(doc.page_content)

            # Prepare metadata with content hash and timestamp
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "content_hash": doc.content_hash,
                    "added_timestamp": datetime.utcnow().isoformat(),
                    "document_id": doc.id,
                }
            )
            metadatas.append(metadata)

        # Handle existing documents
        if not overwrite:
            existing = self._collection.get(ids=ids, include=[])
            existing_ids = set(existing["ids"])

            if existing_ids:
                # Filter out existing documents
                filtered_data = [
                    (doc_id, text, metadata)
                    for doc_id, text, metadata in zip(ids, texts, metadatas)
                    if doc_id not in existing_ids
                ]

                if not filtered_data:
                    self.logger.debug("All documents in batch already exist")
                    return

                ids, texts, metadatas = zip(*filtered_data)
                ids, texts, metadatas = list(ids), list(texts), list(metadatas)

        # Add to collection
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

    async def similarity_search_with_score(
            self,
            query: str,
            k: int = 10,
            filter_dict: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
    ) -> List[SearchResult]:
        """
        Perform similarity search with scores.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata

        Returns:
            List of search results with scores
        """
        if not query.strip():
            raise ValidationError("Query cannot be empty")

        if k <= 0 or k > 100:
            raise ValidationError("k must be between 1 and 100")

        if not self._is_initialized:
            await self.initialize_collection()

        try:
            # Prepare query parameters
            query_params = {
                "query_texts": [query],
                "n_results": k,
                "include": ["documents", "distances"],
            }

            if include_metadata:
                query_params["include"].append("metadatas")

            if filter_dict:
                query_params["where"] = filter_dict

            # Perform search
            start_time = time.time()
            results = self._collection.query(**query_params)
            search_time = time.time() - start_time

            # Process results
            search_results = []

            if results["documents"] and results["documents"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                distances = results["distances"][0]
                metadatas = (
                    results.get("metadatas", [[{}] * len(documents)])[0]
                    if include_metadata
                    else [{}] * len(documents)
                )

                for doc_id, text, distance, metadata in zip(
                        ids, documents, distances, metadatas
                ):
                    # Convert distance to similarity score
                    similarity_score = max(0.0, 1.0 - distance)

                    document = Document(
                        page_content=text, metadata=metadata or {}, id=doc_id
                    )

                    search_results.append(
                        SearchResult(document=document, score=similarity_score)
                    )

            self.logger.debug(
                f"Search completed in {search_time:.3f}s, found {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search operation failed: {e}")

    async def backup_collection(
            self,
            backup_path: Union[str, Path],
            include_embeddings: bool = True,
            compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Create collection backup with async file operations.

        Args:
            backup_path: Path for backup
            include_embeddings: Whether to include embeddings
            compress: Whether to compress backup

        Returns:
            Backup statistics
        """
        if not self._is_initialized:
            await self.initialize_collection()

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        try:
            start_time = time.time()

            # Get collection data
            include_fields = ["documents", "metadatas"]
            if include_embeddings:
                include_fields.append("embeddings")

            all_data = self._collection.get(include=include_fields)

            # Prepare backup data
            backup_data = {
                "collection_name": self.collection_name,
                "collection_metadata": self._collection.metadata,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "document_count": len(all_data["ids"]) if all_data["ids"] else 0,
                "include_embeddings": include_embeddings,
                "data": all_data,
            }

            # Save backup with async file operations
            timestamp = int(time.time())
            backup_file = (
                    backup_path / f"{self.collection_name}_backup_{timestamp}.json"
            )

            if ASYNC_IO_AVAILABLE:
                async with aiofiles.open(backup_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(backup_data, indent=2, ensure_ascii=False))
            else:
                # Fallback to sync operations
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)

            execution_time = time.time() - start_time
            backup_size = backup_file.stat().st_size / (1024 * 1024)

            stats = {
                "backup_file": str(backup_file),
                "document_count": backup_data["document_count"],
                "backup_size_mb": round(backup_size, 2),
                "execution_time": round(execution_time, 2),
                "include_embeddings": include_embeddings,
            }

            self.logger.info(
                f"Backup completed: {backup_data['document_count']} documents, {backup_size:.2f} MB"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise VectorStoreError(f"Backup operation failed: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        if not self._is_initialized:
            await self.initialize_collection()

        try:
            stats = {
                "collection_name": self.collection_name,
                "document_count": self._collection.count(),
                "metadata": self._collection.metadata,
                "is_initialized": self._is_initialized,
                "last_health_check": self._last_health_check,
                "deduplication_enabled": self.document_processor.enable_deduplication,
                "content_hashes_loaded": len(self.document_processor._content_hashes),
            }

            # Add embedding function stats
            if self.embedding_function:
                if hasattr(self.embedding_function, "get_cache_stats"):
                    stats["embedding_cache"] = (
                        await self.embedding_function.get_cache_stats()
                    )
                stats["embedding_dimensions"] = self.embedding_function.dimensions

            # Add schema info
            if self._metadata_manager:
                stats["schema_info"] = self._metadata_manager.get_schema_info()

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            raise VectorStoreError(f"Could not retrieve statistics: {e}")

    async def validate_embeddings(self, sample_size: int = 10) -> Dict[str, Any]:
        """Validate embedding dimensions and consistency."""
        if not self._is_initialized:
            await self.initialize_collection()

        try:
            sample_data = self._collection.get(
                limit=min(sample_size, 100),  # Cap sample size
                include=["embeddings", "documents"],
            )

            if not sample_data["embeddings"]:
                return {
                    "status": "no_embeddings",
                    "message": "No embeddings found in collection",
                }

            embeddings = [emb for emb in sample_data["embeddings"] if emb]

            if not embeddings:
                return {
                    "status": "no_valid_embeddings",
                    "message": "No valid embeddings found",
                }

            dimensions = [len(emb) for emb in embeddings]
            unique_dimensions = set(dimensions)

            validation_result = {
                "status": (
                    "consistent" if len(unique_dimensions) == 1 else "inconsistent"
                ),
                "sample_size": len(embeddings),
                "dimensions": {
                    "min": min(dimensions),
                    "max": max(dimensions),
                    "average": round(sum(dimensions) / len(dimensions), 2),
                    "unique_values": sorted(unique_dimensions),
                },
                "issues": [],
            }

            # Check against expected dimensions
            if self.embedding_function and self.embedding_function.dimensions:
                expected = self.embedding_function.dimensions
                if expected not in unique_dimensions:
                    validation_result["issues"].append(
                        f"Expected dimensions {expected} not found in sample"
                    )

            if len(unique_dimensions) > 1:
                validation_result["issues"].append(
                    f"Inconsistent embedding dimensions: {sorted(unique_dimensions)}"
                )

            return validation_result

        except Exception as e:
            self.logger.error(f"Embedding validation failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_metadata_manager(self) -> MetadataManager:
        """Get metadata manager."""
        if not self._metadata_manager:
            raise VectorStoreError("Collection not initialized")
        return self._metadata_manager

    async def close(self) -> None:
        """Clean up all resources."""
        self.logger.info("Closing VectorStoreManager")

        # Close embedding function
        if self.embedding_function:
            await self.embedding_function.close()

        # Reset state
        self._collection = None
        self._client = None
        self._metadata_manager = None
        self._is_initialized = False

        # Clear document processor state
        self.document_processor._content_hashes.clear()

        self.logger.info("VectorStoreManager closed successfully")


# Factory functions for convenience
async def create_openai_vector_store(
        collection_name: str = "rag_documents",
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        persist_directory: Optional[Path] = None,
        **kwargs,
) -> VectorStoreManager:
    """Create VectorStoreManager with OpenAI embeddings."""
    if not api_key and config:
        api_key = config.OPENAI_API_KEY

    if not api_key:
        raise ValueError("OpenAI API key required")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model=model,
        cache_dir=persist_directory / "embeddings_cache" if persist_directory else None,
    )

    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        **kwargs,
    )

    await manager.initialize_collection()
    return manager


# Export main classes
__all__ = [
    "VectorStoreManager",
    "Document",
    "SearchResult",
    "OperationStats",
    "EmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "DocumentProcessor",
    "MetadataManager",
    "CacheManager",
    "VectorStoreError",
    "EmbeddingError",
    "ValidationError",
    "create_openai_vector_store",
]
