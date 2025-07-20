"""
RAG MCP Server - Source Package

This package contains the main modules for the RAG (Retrieval-Augmented Generation) 
Model Context Protocol server.

Modules:
- mcp_server: Main MCP server implementation
- llamaindex_processor: Advanced RAG engine with LlamaIndex
- web_search: Web search functionality via Tavily API
"""

__version__ = "0.1.0"
__author__ = "Sean Renn"

# Export core classes only
from .llamaindex_processor import RAGEngine
from .mcp_server import RAGMCPServer

__all__ = [
    "RAGEngine",
    "RAGMCPServer",
]