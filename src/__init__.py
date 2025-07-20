"""
RAG MCP Server - Source Package

This package contains the main modules for the RAG (Retrieval-Augmented Generation) 
Model Context Protocol server with dynamic embedding provider support.

Modules:
- mcp_server: Main MCP server implementation
- llamaindex_processor: Advanced RAG engine with LlamaIndex
- web_search: Web search functionality via Tavily API
- embedding_provider: Dynamic embedding provider system (OpenAI/DashScope)
"""

__version__ = "0.1.0"
__author__ = "Sean Renn"

# Export core classes only
from .llamaindex_processor import RAGEngine
from .mcp_server import RAGMCPServer
from .embedding_provider import get_embed_model, get_embed_model_from_env, create_index_with_provider

__all__ = [
    "RAGEngine",
    "RAGMCPServer",
    "get_embed_model",
    "get_embed_model_from_env", 
    "create_index_with_provider",
]