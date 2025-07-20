"""
Simplified Vector Store compatibility layer.

This module provides minimal compatibility for existing code that expects
vector_store.py components, while delegating all actual functionality to LlamaIndex.

Since LlamaIndex now handles all vector store operations through ChromaVectorStore,
embedding management, and document processing, this module serves as a thin
compatibility layer for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Document:
    """Compatibility Document class for legacy code."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    content_hash: Optional[str] = None


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a collection is not found."""
    pass


class EmbeddingError(VectorStoreError):
    """Raised when embedding operations fail."""
    pass


@dataclass
class SearchResult:
    """Container for search results."""
    documents: List[Document]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_results: int = 0
    execution_time: float = 0.0


# Simplified compatibility functions that delegate to LlamaIndex
def create_simple_document(content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
    """Create a simple document for compatibility."""
    return Document(
        page_content=content,
        metadata=metadata or {},
        id=str(hash(content))[:12],
    )


def extract_text_from_documents(documents: List[Document]) -> List[str]:
    """Extract text content from documents."""
    return [doc.page_content for doc in documents]


def add_timestamps(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Add timestamp metadata."""
    enhanced_metadata = metadata.copy()
    enhanced_metadata.update({
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    })
    return enhanced_metadata


# Export only essential classes for compatibility
__all__ = [
    "Document",
    "VectorStoreError", 
    "CollectionNotFoundError",
    "EmbeddingError",
    "SearchResult",
    "create_simple_document",
    "extract_text_from_documents",
    "add_timestamps",
]