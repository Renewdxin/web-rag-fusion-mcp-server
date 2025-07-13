"""
RAG MCP Server - Source Package

This package contains the main modules for the RAG (Retrieval-Augmented Generation) 
Model Context Protocol server.

Modules:
- mcp_server: Main MCP server implementation
- vector_store: Vector database management with ChromaDB
- web_search: Web search functionality via Tavily API  
- document_processor: Document processing and chunking
"""

__version__ = "0.1.0"
__author__ = "RAG MCP Team"

# Export main classes for convenience
from .vector_store import VectorStoreManager, Document, SearchResult
from .web_search import WebSearchManager, WebSearchResult
from .document_processor import DocumentProcessor
from .mcp_server import RAGMCPServer

__all__ = [
    "VectorStoreManager",
    "Document", 
    "SearchResult",
    "WebSearchManager",
    "WebSearchResult", 
    "DocumentProcessor",
    "RAGMCPServer",
]