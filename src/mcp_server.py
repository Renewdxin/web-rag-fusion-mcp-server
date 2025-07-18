"""
RAG MCP Server implementation.

This module implements a Model Context Protocol (MCP) server that provides
Retrieval-Augmented Generation (RAG) capabilities through three main tools:
- search_knowledge_base: Search local vector knowledge base
- web_search: Search web via Tavily API
- smart_search: Intelligent search combining local and web results

The server provides comprehensive logging, graceful shutdown handling,
and startup validation for all dependencies and configurations.
"""

import asyncio
import logging
import re
import signal
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

# JSON Schema validation
try:
    import jsonschema
    from jsonschema import ValidationError, validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

# Local imports
from config.settings import ConfigurationError, config
from .vector_store import SearchResult, VectorStoreError, VectorStoreManager, Document
from .web_search import (
    WebSearchError,
    WebSearchManager,
    WebSearchResult,
)
from .document_processor import DocumentProcessor, ProcessingStats
from .search_optimizer import SearchOptimizer, RankingStrategy

# LlamaIndex imports for enhanced RAG
try:
    from .llamaindex_processor import LlamaIndexProcessor
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LlamaIndexProcessor = None


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()

        # Remove old requests outside time window
        while self.requests and self.requests[0] <= now - self.time_window:
            self.requests.popleft()

        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def time_until_allowed(self) -> float:
        """Get seconds until next request is allowed."""
        if not self.requests:
            return 0.0

        oldest_request = self.requests[0]
        return max(0.0, self.time_window - (time.time() - oldest_request))


class RAGMCPServer:
    """
    RAG (Retrieval-Augmented Generation) MCP Server.

    Provides intelligent search capabilities through local vector database
    and web search integration via Tavily API.
    """

    def __init__(self):
        """Initialize the RAG MCP Server."""
        self.server = Server("rag-agent")
        self.logger = self._setup_logging()
        self._setup_signal_handlers()
        self._register_handlers()

        # Rate limiting (100 requests per minute by default)
        self.rate_limiter = RateLimiter(
            max_requests=config.MAX_RETRIES * 20, time_window=60  # Scale with config
        )

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "tool_usage": defaultdict(int),
        }

        # Cache tool schemas for validation
        self._tool_schemas = {}

        # Initialize vector store manager (lazy loading)
        self._vector_store: Optional[VectorStoreManager] = None

        # Initialize LlamaIndex processor (lazy loading)
        self._llamaindex_processor: Optional[LlamaIndexProcessor] = None
        self._use_llamaindex = LLAMAINDEX_AVAILABLE and getattr(config, 'USE_LLAMAINDEX', True)

        # Initialize web search manager (lazy loading)
        self._web_search: Optional[WebSearchManager] = None

        # Initialize document processor (lazy loading)
        self._document_processor: Optional[DocumentProcessor] = None

        # Initialize search optimizer (lazy loading)
        self._search_optimizer: Optional[SearchOptimizer] = None

        # Document management storage
        self._documents_metadata: Dict[str, Dict[str, Any]] = {}
        self._document_tags: Dict[str, List[str]] = {}
        self._document_usage: Dict[str, Dict[str, Any]] = {}
        self._document_versions: Dict[str, List[Dict[str, Any]]] = {}

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging."""
        logger = logging.getLogger("rag-mcp-server")
        logger.setLevel(getattr(logging, config.LOG_LEVEL))

        # Console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers for SIGINT and SIGTERM."""

        def signal_handler(signum: int, frame) -> None:
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _register_handlers(self) -> None:
        """Register MCP server handlers."""
        self.server.list_tools()(self._list_tools)
        self.server.call_tool()(self._call_tool)

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracing."""
        return f"req_{uuid.uuid4().hex[:8]}"

    async def _validate_tool_arguments(
            self, tool_name: str, arguments: Dict[str, Any]
    ) -> None:
        """
        Validate tool arguments against JSON Schema.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments provided for the tool

        Raises:
            ValueError: If arguments are invalid
        """
        if not JSONSCHEMA_AVAILABLE:
            self.logger.warning("jsonschema not available, skipping validation")
            return

        # Get tool schema from cached tools
        if tool_name not in self._tool_schemas:
            # Cache schemas on first use
            tools = await self._list_tools()
            for tool in tools:
                self._tool_schemas[tool.name] = tool.inputSchema

        if tool_name not in self._tool_schemas:
            raise ValueError(f"Unknown tool: {tool_name}")

        schema = self._tool_schemas[tool_name]

        try:
            validate(instance=arguments, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for tool '{tool_name}': {e.message}")

    def _update_metrics(
            self, tool_name: str, execution_time: float, success: bool
    ) -> None:
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["tool_usage"][tool_name] += 1

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        # Update rolling average response time
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
                                                        current_avg * (total_requests - 1) + execution_time
                                                ) / total_requests

    async def _list_tools(self) -> List[Tool]:
        """
        List available RAG tools.

        Returns:
            List of tool definitions with complete JSON Schema specifications.
        """
        self.logger.debug("Listing available tools")

        return [
            Tool(
                name="search_knowledge_base",
                description=(
                    "Search the local vector knowledge base for relevant information. "
                    "This tool performs semantic similarity search across stored documents "
                    "and returns the most relevant results based on the query."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "filter_dict": {
                            "type": "object",
                            "description": "Optional metadata filters for search results",
                            "additionalProperties": True,
                            "default": None,
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Whether to include document metadata in results",
                            "default": True,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="web_search",
                description=(
                    "Search the web using Tavily API for current and comprehensive information. "
                    "This tool is ideal for finding recent information, news, or data not "
                    "available in the local knowledge base."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for web search",
                            "minLength": 1,
                            "maxLength": 400,
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of web results to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "search_depth": {
                            "type": "string",
                            "description": "Depth of search results",
                            "enum": ["basic", "advanced"],
                            "default": "basic",
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": "Whether to include AI-generated answer summary",
                            "default": True,
                        },
                        "include_raw_content": {
                            "type": "boolean",
                            "description": "Whether to include raw content from sources",
                            "default": False,
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                            },
                            "description": "List of domains to exclude from search results",
                            "maxItems": 10,
                            "default": [],
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="smart_search",
                description=(
                    "Sophisticated intelligent hybrid search with advanced decision logic. "
                    "Searches local knowledge base first, evaluates result quality against threshold, "
                    "and intelligently decides whether to supplement with web search. Provides "
                    "cross-source deduplication, relevance explanations, confidence scoring, "
                    "and source credibility assessment for comprehensive search results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for intelligent hybrid search",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Threshold value for triggering web search (0.0-1.0). If max local score < threshold, web search is triggered",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.75,
                        },
                        "local_top_k": {
                            "type": "integer",
                            "description": "Maximum results to retrieve from local knowledge base",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "web_max_results": {
                            "type": "integer",
                            "description": "Maximum results to retrieve from web search if triggered",
                            "minimum": 0,
                            "maximum": 20,
                            "default": 5,
                        },
                        "include_sources": {
                            "type": "boolean",
                            "description": "Whether to include detailed source information for all results",
                            "default": True,
                        },
                        "combine_strategy": {
                            "type": "string",
                            "description": "Strategy for combining and ranking results from multiple sources",
                            "enum": ["interleave", "local_first", "relevance_score"],
                            "default": "relevance_score",
                        },
                        "min_local_results": {
                            "type": "integer",
                            "description": "Minimum local results needed before considering web search sufficient",
                            "minimum": 0,
                            "maximum": 10,
                            "default": 2,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="add_document",
                description=(
                    "Add a document to the knowledge base. Supports both file paths and raw content. "
                    "Automatically detects file type and processes the document with intelligent chunking. "
                    "Returns document ID and processing statistics. Supports batch additions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Raw text content (alternative to file_path)",
                            "minLength": 1,
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to file to process (alternative to content)",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata for the document",
                            "additionalProperties": True,
                            "default": {},
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to associate with the document",
                            "default": [],
                        },
                        "batch_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple file paths for batch processing",
                            "default": [],
                        },
                        "auto_detect_type": {
                            "type": "boolean",
                            "description": "Whether to auto-detect file type",
                            "default": True,
                        },
                    },
                    "oneOf": [
                        {"required": ["content"]},
                        {"required": ["file_path"]},
                        {"required": ["batch_files"]},
                    ],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="update_document",
                description=(
                    "Update an existing document in the knowledge base. Supports partial updates "
                    "with version history maintenance. Re-indexes affected document chunks and "
                    "implements conflict detection and resolution."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "ID of the document to update",
                            "minLength": 1,
                        },
                        "content": {
                            "type": "string",
                            "description": "New content for the document",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Metadata fields to update",
                            "additionalProperties": True,
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Updated tags for the document",
                        },
                        "partial_update": {
                            "type": "boolean",
                            "description": "Whether to perform a partial update (merge with existing)",
                            "default": True,
                        },
                        "create_version": {
                            "type": "boolean",
                            "description": "Whether to create a version backup",
                            "default": True,
                        },
                        "force_update": {
                            "type": "boolean",
                            "description": "Force update even if conflicts detected",
                            "default": False,
                        },
                    },
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="delete_document",
                description=(
                    "Delete a document from the knowledge base with soft delete capability. "
                    "Provides recovery option, cleans up orphaned chunks, and updates related indices."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "ID of the document to delete",
                            "minLength": 1,
                        },
                        "soft_delete": {
                            "type": "boolean",
                            "description": "Whether to perform soft delete (recoverable)",
                            "default": True,
                        },
                        "cleanup_chunks": {
                            "type": "boolean",
                            "description": "Whether to clean up associated chunks",
                            "default": True,
                        },
                        "update_indices": {
                            "type": "boolean",
                            "description": "Whether to update related indices",
                            "default": True,
                        },
                        "backup_before_delete": {
                            "type": "boolean",
                            "description": "Create backup before deletion",
                            "default": True,
                        },
                    },
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="list_documents",
                description=(
                    "List documents in the knowledge base with comprehensive filtering and pagination. "
                    "Supports metadata filtering, tag-based search, date range queries, and various sort options. "
                    "Returns document statistics and detailed information."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page": {
                            "type": "integer",
                            "description": "Page number (1-based)",
                            "minimum": 1,
                            "default": 1,
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Number of items per page",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                        },
                        "filter_metadata": {
                            "type": "object",
                            "description": "Filter by metadata fields",
                            "additionalProperties": True,
                        },
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags (OR logic)",
                        },
                        "filter_type": {
                            "type": "string",
                            "description": "Filter by document type",
                        },
                        "date_from": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Filter documents created after this date",
                        },
                        "date_to": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Filter documents created before this date",
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["date", "size", "type", "name", "usage"],
                            "description": "Sort criteria",
                            "default": "date",
                        },
                        "sort_order": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "description": "Sort order",
                            "default": "desc",
                        },
                        "include_stats": {
                            "type": "boolean",
                            "description": "Include document statistics",
                            "default": True,
                        },
                        "include_deleted": {
                            "type": "boolean",
                            "description": "Include soft-deleted documents",
                            "default": False,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="manage_tags",
                description=(
                    "Manage document tags - add, remove, or list tags. Supports tag-based search "
                    "filtering and bulk tag operations across multiple documents."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "list", "search"],
                            "description": "Tag management action",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Document ID (for add/remove actions)",
                        },
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple document IDs for bulk operations",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to add or remove",
                        },
                        "search_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to search for (for search action)",
                        },
                        "tag_filter": {
                            "type": "string",
                            "description": "Filter tags by pattern (for list action)",
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="bulk_operations",
                description=(
                    "Perform bulk operations on multiple documents - add, update, delete, or tag operations. "
                    "Provides batch processing with progress tracking and error handling."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "update", "delete", "tag"],
                            "description": "Bulk operation type",
                        },
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                            "description": "List of documents/operations to process",
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Number of documents to process in each batch",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                        },
                        "continue_on_error": {
                            "type": "boolean",
                            "description": "Continue processing if individual operations fail",
                            "default": True,
                        },
                        "progress_callback": {
                            "type": "boolean",
                            "description": "Provide progress updates",
                            "default": True,
                        },
                    },
                    "required": ["operation", "documents"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="document_analytics",
                description=(
                    "Get document usage analytics including access frequency, search hits, "
                    "popular documents, and usage patterns over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Specific document ID for detailed analytics",
                        },
                        "analytics_type": {
                            "type": "string",
                            "enum": ["overview", "usage", "search_hits", "popular", "trends"],
                            "description": "Type of analytics to retrieve",
                            "default": "overview",
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "week", "month", "year", "all"],
                            "description": "Time range for analytics",
                            "default": "month",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                        },
                        "include_charts": {
                            "type": "boolean",
                            "description": "Include ASCII charts in results",
                            "default": False,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="optimize_search",
                description=(
                    "Optimize search results with advanced techniques including query expansion, "
                    "hybrid ranking, summarization, personalization, and spell correction. "
                    "Provides enhanced search quality and user experience."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to optimize",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier for personalization",
                            "default": "anonymous",
                        },
                        "ranking_strategy": {
                            "type": "string",
                            "enum": ["vector_only", "bm25_only", "hybrid_balanced", "hybrid_vector_weighted", "hybrid_bm25_weighted", "personalized"],
                            "description": "Ranking strategy to use",
                            "default": "hybrid_balanced",
                        },
                        "enable_personalization": {
                            "type": "boolean",
                            "description": "Enable personalization features",
                            "default": True,
                        },
                        "enable_summarization": {
                            "type": "boolean",
                            "description": "Enable result summarization",
                            "default": True,
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="get_search_analytics",
                description=(
                    "Get comprehensive search analytics including query patterns, user behavior, "
                    "performance metrics, and actionable insights for search optimization."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analytics_type": {
                            "type": "string",
                            "enum": ["overview", "query_analytics", "user_behavior", "performance", "content_insights", "trends"],
                            "description": "Type of analytics to retrieve",
                            "default": "overview",
                        },
                        "time_period": {
                            "type": "string",
                            "enum": ["day", "week", "month", "all"],
                            "description": "Time period for analytics",
                            "default": "week",
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "description": "Include optimization recommendations",
                            "default": True,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="track_user_feedback",
                description=(
                    "Track user feedback and interactions to improve search personalization "
                    "and relevance. Includes click tracking, dwell time, and explicit feedback."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "minLength": 1,
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query that generated the results",
                            "minLength": 1,
                        },
                        "clicked_results": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs that were clicked",
                            "default": [],
                        },
                        "dwell_times": {
                            "type": "object",
                            "description": "Dwell time (seconds) for each document",
                            "additionalProperties": {"type": "number"},
                            "default": {},
                        },
                        "feedback_scores": {
                            "type": "object",
                            "description": "Explicit feedback scores (-1 to 1) for documents",
                            "additionalProperties": {"type": "number"},
                            "default": {},
                        },
                    },
                    "required": ["user_id", "query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="create_ab_test",
                description=(
                    "Create A/B test experiment to compare different search strategies "
                    "and optimize search performance based on user behavior metrics."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "Unique identifier for the experiment",
                            "minLength": 1,
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name for the experiment",
                            "minLength": 1,
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what the experiment tests",
                            "minLength": 1,
                        },
                        "variant_a": {
                            "type": "object",
                            "description": "Configuration for variant A (control)",
                            "additionalProperties": True,
                        },
                        "variant_b": {
                            "type": "object",
                            "description": "Configuration for variant B (test)",
                            "additionalProperties": True,
                        },
                        "traffic_split": {
                            "type": "number",
                            "description": "Traffic split for variant A (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                        "duration_days": {
                            "type": "integer",
                            "description": "Duration of experiment in days",
                            "minimum": 1,
                            "maximum": 90,
                            "default": 7,
                        },
                        "success_metric": {
                            "type": "string",
                            "description": "Primary success metric to track",
                            "default": "click_through_rate",
                        },
                    },
                    "required": ["experiment_id", "name", "description", "variant_a", "variant_b"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="get_ab_test_results",
                description=(
                    "Get results and performance metrics from A/B test experiments "
                    "to understand which search strategies perform better."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "Experiment ID to get results for",
                        },
                        "include_details": {
                            "type": "boolean",
                            "description": "Include detailed metrics and analysis",
                            "default": True,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Comprehensive tool execution handler with validation, rate limiting,
        timeout control, and performance monitoring.

        Args:
            name: The name of the tool to execute
            arguments: Arguments provided for the tool

        Returns:
            CallToolResult with TextContent response
        """
        # Generate unique request ID for tracing
        request_id = self._generate_request_id()
        start_time = time.time()

        # Initial logging
        self.logger.info(
            f"[{request_id}] Executing tool: {name} with arguments: {arguments}"
        )

        try:
            # Rate limiting check
            if not self.rate_limiter.is_allowed():
                wait_time = self.rate_limiter.time_until_allowed()
                error_msg = f"Rate limit exceeded. Try again in {wait_time:.1f} seconds"
                self.logger.warning(f"[{request_id}] {error_msg}")
                self._update_metrics(name, time.time() - start_time, False)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {error_msg}")]
                )

            # Validate input arguments against JSON Schema
            try:
                await self._validate_tool_arguments(name, arguments)
                self.logger.debug(f"[{request_id}] Arguments validation passed")
            except ValueError as e:
                error_msg = f"Validation error: {str(e)}"
                self.logger.error(f"[{request_id}] {error_msg}")
                self._update_metrics(name, time.time() - start_time, False)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {error_msg}")]
                )

            # Execute tool with timeout control
            timeout_seconds = config.TIMEOUT_SECONDS
            self.logger.debug(
                f"[{request_id}] Starting tool execution with {timeout_seconds}s timeout"
            )

            try:
                async with asyncio.timeout(timeout_seconds):
                    # Pattern match on tool name to dispatch to appropriate handler
                    if name == "search_knowledge_base":
                        result = await self._search_knowledge_base(request_id, **arguments)
                    elif name == "web_search":
                        result = await self._search_web(request_id, **arguments)
                    elif name == "smart_search":
                        # Map parameters to match function signature
                        smart_search_args = {
                            "query": arguments["query"],
                            "similarity_threshold": arguments.get("similarity_threshold", 0.75),
                            "local_top_k": arguments.get("local_top_k", 5),
                            "web_max_results": arguments.get("web_max_results", 5),
                            "include_sources": arguments.get("include_sources", True),
                            "combine_strategy": arguments.get("combine_strategy", "relevance_score"),
                            "min_local_results": arguments.get("min_local_results", 2),
                        }
                        result = await self._smart_search_internal(request_id, **smart_search_args)
                    elif name == "add_document":
                        result = await self._add_document(request_id, **arguments)
                    elif name == "update_document":
                        result = await self._update_document(request_id, **arguments)
                    elif name == "delete_document":
                        result = await self._delete_document(request_id, **arguments)
                    elif name == "list_documents":
                        result = await self._list_documents(request_id, **arguments)
                    elif name == "manage_tags":
                        result = await self._manage_tags(request_id, **arguments)
                    elif name == "bulk_operations":
                        result = await self._bulk_operations(request_id, **arguments)
                    elif name == "document_analytics":
                        result = await self._document_analytics(request_id, **arguments)
                    elif name == "optimize_search":
                        result = await self._optimize_search(request_id, **arguments)
                    elif name == "get_search_analytics":
                        result = await self._get_search_analytics(request_id, **arguments)
                    elif name == "track_user_feedback":
                        result = await self._track_user_feedback(request_id, **arguments)
                    elif name == "create_ab_test":
                        result = await self._create_ab_test(request_id, **arguments)
                    elif name == "get_ab_test_results":
                        result = await self._get_ab_test_results(request_id, **arguments)
                    else:
                        error_msg = f"Unknown tool: {name}"
                        self.logger.error(f"[{request_id}] {error_msg}")
                        self._update_metrics(name, time.time() - start_time, False)
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"Error: {error_msg}")]
                        )

                # Success - log performance metrics
                execution_time = time.time() - start_time
                self.logger.info(f"[{request_id}] Tool '{name}' completed successfully in {execution_time:.3f}s")
                self._update_metrics(name, execution_time, True)

                # Convert result to CallToolResult format
                if isinstance(result, list):
                    # Assume it's already List[TextContent]
                    return CallToolResult(content=result)
                elif isinstance(result, str):
                    return CallToolResult(
                        content=[TextContent(type="text", text=result)]
                    )
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(result))]
                    )

            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                error_msg = f"Tool execution timed out after {timeout_seconds}s"
                self.logger.error(f"[{request_id}] {error_msg}")
                self._update_metrics(name, execution_time, False)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {error_msg}")]
                )

        except Exception as e:
            # Catch-all error handling with detailed logging
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error executing tool {name}: {str(e)}"
            self.logger.error(f"[{request_id}] {error_msg}", exc_info=True)
            self._update_metrics(name, execution_time, False)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {error_msg}")]
            )

    async def _get_vector_store(self) -> VectorStoreManager:
        """Get or initialize vector store manager."""
        if self._vector_store is None:
            try:
                self._vector_store = VectorStoreManager(
                    collection_name=(
                        config.COLLECTION_NAME
                        if hasattr(config, "COLLECTION_NAME")
                        else "rag_documents"
                    ),
                    persist_directory=(
                        config.VECTOR_STORE_PATH
                        if hasattr(config, "VECTOR_STORE_PATH")
                        else "./data"
                    ),
                )
                await self._vector_store.initialize_collection()
                self.logger.info("Vector store initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
                raise VectorStoreError(f"Vector store initialization failed: {e}")
        return self._vector_store

    async def _get_llamaindex_processor(self) -> Optional[LlamaIndexProcessor]:
        """Get or initialize LlamaIndex processor."""
        if not self._use_llamaindex or not LLAMAINDEX_AVAILABLE:
            return None
            
        if self._llamaindex_processor is None:
            try:
                self._llamaindex_processor = LlamaIndexProcessor(
                    collection_name=(
                        config.COLLECTION_NAME
                        if hasattr(config, "COLLECTION_NAME")
                        else "rag_documents_llamaindex"
                    ),
                    chunk_size=getattr(config, 'CHUNK_SIZE', 1024),
                    chunk_overlap=getattr(config, 'CHUNK_OVERLAP', 200),
                    embedding_model=getattr(config, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    similarity_top_k=getattr(config, 'SIMILARITY_TOP_K', 10),
                    similarity_cutoff=getattr(config, 'SIMILARITY_CUTOFF', 0.7),
                )
                self.logger.info("LlamaIndex processor initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize LlamaIndex processor: {e}")
                # Fall back to traditional vector store
                self._use_llamaindex = False
                return None
        return self._llamaindex_processor

    async def _get_web_search(self) -> WebSearchManager:
        """Get or initialize web search manager."""
        if self._web_search is None:
            try:
                if not config.TAVILY_API_KEY:
                    raise WebSearchError("Tavily API key not configured")

                self._web_search = WebSearchManager(
                    api_key=config.TAVILY_API_KEY,
                    timeout=getattr(config, "WEB_SEARCH_TIMEOUT", 30),
                    max_retries=getattr(config, "MAX_RETRIES", 3),
                    quota_limit=getattr(config, "TAVILY_QUOTA_LIMIT", None),
                )
                self.logger.info("Web search manager initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize web search manager: {e}")
                raise WebSearchError(f"Web search initialization failed: {e}")
        return self._web_search

    async def _get_document_processor(self) -> DocumentProcessor:
        """Get or initialize document processor."""
        if self._document_processor is None:
            try:
                self._document_processor = DocumentProcessor(
                    chunk_size=getattr(config, "CHUNK_SIZE", 1000),
                    overlap=getattr(config, "CHUNK_OVERLAP", 200),
                    max_concurrency=getattr(config, "MAX_CONCURRENCY", 5),
                )
                self.logger.info("Document processor initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize document processor: {e}")
                raise DocumentProcessingError(f"Document processor initialization failed: {e}")
        return self._document_processor

    async def _get_search_optimizer(self) -> SearchOptimizer:
        """Get or initialize search optimizer."""
        if self._search_optimizer is None:
            try:
                self._search_optimizer = SearchOptimizer(
                    enable_query_expansion=True,
                    enable_hybrid_ranking=True,
                    enable_summarization=True,
                    enable_personalization=True,
                    enable_spell_correction=True,
                    enable_semantic_analysis=True,
                    enable_analytics=True,
                    enable_ab_testing=True,
                )
                self.logger.info("Search optimizer initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize search optimizer: {e}")
                # Continue without search optimizer
                self._search_optimizer = None
        return self._search_optimizer

    def _preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess search query for better results.

        Args:
            query: Raw search query

        Returns:
            Dictionary with processed query components
        """
        # Convert to lowercase
        processed_query = query.lower().strip()

        # Remove special characters but keep spaces and hyphens
        cleaned_query = re.sub(r"[^a-zA-Z0-9\s\-]", " ", processed_query)

        # Extract keywords (words longer than 2 characters)
        keywords = [
            word.strip() for word in cleaned_query.split() if len(word.strip()) > 2
        ]

        # Create search variations
        variations = [
            query,  # Original query
            processed_query,  # Lowercase
            cleaned_query,  # Cleaned
            " ".join(keywords),  # Keywords only
        ]

        return {
            "original": query,
            "processed": processed_query,
            "cleaned": cleaned_query,
            "keywords": keywords,
            "variations": list(set(variations)),  # Remove duplicates
        }

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata for display in a readable format.

        Args:
            metadata: Document metadata dictionary

        Returns:
            Formatted metadata string
        """
        if not metadata:
            return "No metadata available"

        formatted_lines = []

        # Key metadata fields to display prominently
        priority_fields = ["source", "filename", "file_type", "created_at", "modified"]

        # Display priority fields first
        for field in priority_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                formatted_lines.append(f"📄 {field.title()}: {value}")

        # Display other metadata
        other_fields = {k: v for k, v in metadata.items() if k not in priority_fields}
        if other_fields:
            formatted_lines.append("📋 Additional:")
            for key, value in list(other_fields.items())[
                              :3
                              ]:  # Limit to 3 additional fields
                if isinstance(value, (str, int, float, bool)):
                    value_str = str(value)
                    if len(value_str) > 30:
                        value_str = value_str[:27] + "..."
                    formatted_lines.append(f"   • {key}: {value_str}")

        return "\n".join(formatted_lines)

    def _highlight_content(
            self, content: str, keywords: List[str], max_length: int = 500
    ) -> str:
        """
        Highlight search keywords in content and truncate if necessary.

        Args:
            content: Document content
            keywords: Keywords to highlight
            max_length: Maximum content length

        Returns:
            Highlighted and potentially truncated content
        """
        if not keywords:
            # No keywords to highlight, just truncate if needed
            if len(content) <= max_length:
                return content
            return content[: max_length - 3] + "..."

        # Create case-insensitive pattern for all keywords
        keyword_pattern = "|".join(re.escape(kw) for kw in keywords if len(kw) > 2)

        if not keyword_pattern:
            # No valid keywords, just truncate
            if len(content) <= max_length:
                return content
            return content[: max_length - 3] + "..."

        try:
            # Find keyword matches
            matches = list(re.finditer(keyword_pattern, content, re.IGNORECASE))

            if not matches:
                # No matches found, just truncate
                if len(content) <= max_length:
                    return content
                return content[: max_length - 3] + "..."

            # If content is short enough, highlight in place
            if len(content) <= max_length:
                highlighted = re.sub(
                    keyword_pattern,
                    lambda m: f"**{m.group()}**",
                    content,
                    flags=re.IGNORECASE,
                )
                return highlighted

            # For long content, show context around first match
            first_match = matches[0]
            start_pos = max(0, first_match.start() - 100)
            end_pos = min(len(content), first_match.end() + 300)

            excerpt = content[start_pos:end_pos]
            if start_pos > 0:
                excerpt = "..." + excerpt
            if end_pos < len(content):
                excerpt = excerpt + "..."

            # Highlight keywords in excerpt
            highlighted = re.sub(
                keyword_pattern,
                lambda m: f"**{m.group()}**",
                excerpt,
                flags=re.IGNORECASE,
            )

            return highlighted

        except Exception as e:
            self.logger.warning(f"Error highlighting content: {e}")
            # Fallback to simple truncation
            if len(content) <= max_length:
                return content
            return content[: max_length - 3] + "..."

    def _sort_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Sort search results by similarity score and timestamp.

        Args:
            results: List of search results

        Returns:
            Sorted list of search results
        """

        def sort_key(result: SearchResult):
            # Primary sort: similarity score (descending)
            similarity = result.score

            # Secondary sort: timestamp if available (descending - newer first)
            timestamp = 0
            if result.document.metadata:
                # Try different timestamp field names
                for field in ["timestamp", "created_at", "modified", "processing_time"]:
                    if field in result.document.metadata:
                        try:
                            if isinstance(result.document.metadata[field], str):
                                # Try to parse datetime string
                                timestamp = datetime.fromisoformat(
                                    result.document.metadata[field].replace(
                                        "Z", "+00:00"
                                    )
                                ).timestamp()
                            elif isinstance(
                                    result.document.metadata[field], (int, float)
                            ):
                                timestamp = float(result.document.metadata[field])
                            break
                        except (ValueError, TypeError):
                            continue

            return (-similarity, -timestamp)  # Both descending

        return sorted(results, key=sort_key)

    async def _search_knowledge_base(
            self,
            request_id: str,
            query: str,
            top_k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
    ) -> List[TextContent]:
        """
        Search the local vector knowledge base.

        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            top_k: Number of results to return (max 20)
            filter_dict: Optional metadata filters
            include_metadata: Whether to include document metadata

        Returns:
            List of TextContent with formatted search results
        """
        search_start_time = time.time()

        try:
            # Validate parameters
            if not query or not query.strip():
                return [
                    TextContent(
                        type="text",
                        text="❌ Error: Query cannot be empty. Please provide a search query.",
                    )
                ]

            if top_k < 1 or top_k > 20:
                return [
                    TextContent(
                        type="text", text="❌ Error: top_k must be between 1 and 20."
                    )
                ]

            self.logger.info(
                f"[{request_id}] Searching knowledge base for: '{query}' (top_k={top_k})"
            )

            # Preprocess query
            query_info = self._preprocess_query(query)
            self.logger.debug(
                f"[{request_id}] Query preprocessing: {query_info['keywords']}"
            )

            # Try LlamaIndex processor first if available
            llamaindex_processor = await self._get_llamaindex_processor()
            
            if llamaindex_processor:
                # Use LlamaIndex for more efficient search
                self.logger.debug(f"[{request_id}] Using LlamaIndex processor")
                
                # Perform search with LlamaIndex
                nodes = await llamaindex_processor.search(
                    query=query_info["processed"],
                    top_k=top_k,
                    similarity_cutoff=0.6,  # Reasonable cutoff for results
                )
                
                # Convert nodes to SearchResult format for compatibility
                search_results = []
                for node in nodes:
                    # Create a Document-like object
                    doc_content = node.node.text
                    doc_metadata = node.node.metadata or {}
                    
                    # Create SearchResult
                    from .vector_store import SearchResult
                    result = SearchResult(
                        document=Document(
                            page_content=doc_content,
                            metadata=doc_metadata,
                            id=node.node.id_ if hasattr(node.node, 'id_') else None,
                        ),
                        score=node.score,
                        metadata=doc_metadata,
                    )
                    search_results.append(result)
                    
            else:
                # Fallback to traditional vector store
                self.logger.debug(f"[{request_id}] Using traditional vector store")
                vector_store = await self._get_vector_store()
                
                # Perform similarity search
                search_results = await vector_store.similarity_search_with_score(
                    query=query_info["processed"],
                    k=top_k,
                    filter_dict=filter_dict,
                    include_metadata=include_metadata,
                )

            # Sort results
            sorted_results = self._sort_results(search_results)

            search_time = time.time() - search_start_time

            # Handle empty results
            if not sorted_results:
                friendly_message = (
                    f"🔍 **No results found for '{query}'**\n\n"
                    f"💡 **Suggestions:**\n"
                    f"• Try different keywords or phrases\n"
                    f"• Use broader terms\n"
                    f"• Check spelling\n"
                    f"• Try searching for partial matches\n\n"
                    f"⏱️ Search completed in {search_time:.3f} seconds"
                )
                return [TextContent(type="text", text=friendly_message)]

            # Format results
            response_lines = [
                f"🔍 **Found {len(sorted_results)} result{'s' if len(sorted_results) != 1 else ''} for '{query}'**\n"
            ]

            for i, result in enumerate(sorted_results, 1):
                # Format similarity score
                score_str = f"{result.score:.3f}"
                score_emoji = (
                    "🟢"
                    if result.score >= 0.8
                    else "🟡" if result.score >= 0.6 else "🔴"
                )

                # Extract source file path
                source_path = "Unknown source"
                if result.document.metadata and "source" in result.document.metadata:
                    source_path = str(result.document.metadata["source"])
                    # Make path clickable if it's a valid file path
                    if source_path.startswith("/") or source_path.startswith("."):
                        try:
                            path_obj = Path(source_path)
                            if path_obj.exists():
                                source_path = f"[{path_obj.name}]({source_path})"
                            else:
                                source_path = f"📄 {path_obj.name}"
                        except Exception:
                            source_path = f"📄 {source_path}"

                # Highlight and truncate content
                highlighted_content = self._highlight_content(
                    result.document.page_content, query_info["keywords"], max_length=500
                )

                # Build result entry
                result_lines = [
                    f"**{i}. {score_emoji} Similarity: {score_str}**",
                    f"📂 **Source:** {source_path}",
                    f"",  # Empty line
                    f"📖 **Content:**",
                    highlighted_content,
                ]

                # Add metadata if requested
                if include_metadata and result.document.metadata:
                    metadata_str = self._format_metadata(result.document.metadata)
                    result_lines.extend(
                        [f"", f"ℹ️ **Metadata:**", metadata_str]  # Empty line
                    )

                result_lines.append("\n" + "-" * 50 + "\n")  # Separator
                response_lines.extend(result_lines)

            # Add search statistics
            response_lines.extend(
                [
                    f"⏱️ **Search completed in {search_time:.3f} seconds**",
                    f"🎯 **Keywords used:** {', '.join(query_info['keywords'][:5])}",  # Show first 5 keywords
                ]
            )

            response_text = "\n".join(response_lines)

            self.logger.info(
                f"[{request_id}] Knowledge base search completed: {len(sorted_results)} results in {search_time:.3f}s"
            )

            return [TextContent(type="text", text=response_text)]

        except VectorStoreError as e:
            error_msg = f"❌ **Vector store error:** {str(e)}"
            self.logger.error(f"[{request_id}] Vector store error: {e}")
            return [TextContent(type="text", text=error_msg)]

        except Exception as e:
            error_msg = f"❌ **Search failed:** {str(e)}"
            self.logger.error(
                f"[{request_id}] Unexpected error during search: {e}", exc_info=True
            )
            return [TextContent(type="text", text=error_msg)]

    async def _search_web(
            self,
            request_id: str,
            query: str,
            max_results: int = 5,
            search_depth: str = "basic",
            include_answer: bool = True,
            include_raw_content: bool = False,
            exclude_domains: Optional[List[str]] = None,
    ) -> List[TextContent]:
        """
        Search the web using Tavily API.

        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            max_results: Maximum number of results
            search_depth: Search depth (basic/advanced)
            include_answer: Whether to include AI answer
            include_raw_content: Whether to include raw content
            exclude_domains: Domains to exclude

        Returns:
            List of TextContent with formatted web search results
        """
        search_start_time = time.time()
        exclude_domains = exclude_domains or []

        try:
            # Validate parameters
            if not query or not query.strip():
                return [
                    TextContent(
                        type="text",
                        text="❌ Error: Query cannot be empty. Please provide a search query.",
                    )
                ]

            if max_results < 1 or max_results > 20:
                return [
                    TextContent(
                        type="text",
                        text="❌ Error: max_results must be between 1 and 20.",
                    )
                ]

            self.logger.info(
                f"[{request_id}] Searching web for: '{query}' (max_results={max_results})"
            )

            # Get web search manager
            web_search = await self._get_web_search()

            # Perform web search
            search_results, metadata = await web_search.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                exclude_domains=exclude_domains,
            )

            search_time = time.time() - search_start_time

            # Handle errors in metadata
            if "error" in metadata:
                error_type = metadata.get("error_type", "WebSearchError")
                error_msg = metadata["error"]

                if error_type == "QuotaExceededError":
                    friendly_message = (
                        f"🚫 **Daily API quota exceeded**\n\n"
                        f"The web search service has reached its daily limit. "
                        f"Please try again tomorrow or contact support for higher limits.\n\n"
                        f"⏱️ Search attempted in {search_time:.3f} seconds"
                    )
                elif error_type == "RateLimitError":
                    friendly_message = (
                        f"⏳ **Rate limit exceeded**\n\n"
                        f"Too many requests in a short time. Please wait a moment and try again.\n\n"
                        f"⏱️ Search attempted in {search_time:.3f} seconds"
                    )
                else:
                    friendly_message = (
                        f"🔍 **Web search temporarily unavailable**\n\n"
                        f"**Error:** {error_msg}\n\n"
                        f"💡 **Suggestions:**\n"
                        f"• Try again in a few moments\n"
                        f"• Check your internet connection\n"
                        f"• Try a different search query\n\n"
                        f"⏱️ Search attempted in {search_time:.3f} seconds"
                    )

                return [TextContent(type="text", text=friendly_message)]

            # Handle empty results
            if not search_results:
                friendly_message = (
                    f"🔍 **No web results found for '{query}'**\n\n"
                    f"💡 **Suggestions:**\n"
                    f"• Try different keywords\n"
                    f"• Use more general terms\n"
                    f"• Check spelling\n"
                    f"• Try searching for related topics\n\n"
                    f"⏱️ Search completed in {search_time:.3f} seconds"
                )

                # Add cache info if available
                if metadata.get("cache_hit"):
                    friendly_message += "\n📋 Result retrieved from cache"

                return [TextContent(type="text", text=friendly_message)]

            # Format results
            response_lines = [
                f"🌐 **Found {len(search_results)} web result{'s' if len(search_results) != 1 else ''} for '{query}'**\n"
            ]

            for i, result in enumerate(search_results, 1):
                # Format score
                score_str = f"{result.score:.3f}"
                score_emoji = (
                    "🟢"
                    if result.score >= 0.8
                    else "🟡" if result.score >= 0.6 else "🔴"
                )

                # Format domain
                domain_emoji = "🌐"
                domain_name = result.source_domain or "Unknown"

                # Truncate content for display
                display_content = result.content
                if len(display_content) > 400:
                    display_content = display_content[:397] + "..."

                # Build result entry
                result_lines = [
                    f"**{i}. {score_emoji} Score: {score_str}**",
                    f"📰 **Title:** {result.title}",
                    f"{domain_emoji} **Source:** [{domain_name}]({result.url})",
                    f"",  # Empty line
                    f"📄 **Content:**",
                    display_content,
                ]

                # Add metadata if available
                if result.metadata:
                    quality_score = result.metadata.get("quality_score")
                    if quality_score is not None:
                        quality_emoji = (
                            "✅"
                            if quality_score >= 0.7
                            else "⚠️" if quality_score >= 0.5 else "❌"
                        )
                        result_lines.append(
                            f"\n{quality_emoji} **Quality Score:** {quality_score:.2f}"
                        )

                result_lines.append("\n" + "-" * 50 + "\n")  # Separator
                response_lines.extend(result_lines)

            # Add search metadata
            response_lines.extend(
                [f"⏱️ **Search completed in {search_time:.3f} seconds**"]
            )

            # Add cache info
            if metadata.get("cache_hit"):
                response_lines.append("📋 **Result retrieved from cache**")
            else:
                response_lines.append("🔄 **Fresh results from web**")

            # Add query optimization info if available
            query_opt = metadata.get("query_optimization")
            if query_opt and query_opt.get("keywords"):
                keywords = query_opt["keywords"][:5]  # Show first 5 keywords
                response_lines.append(f"🎯 **Keywords used:** {', '.join(keywords)}")

            # Add quota info if available
            quota_info = metadata.get("quota_info")
            if quota_info and quota_info.get("quota_limit"):
                usage_pct = quota_info.get("usage_percentage", 0)
                response_lines.append(
                    f"📊 **API Usage:** {usage_pct:.1f}% of daily quota"
                )

            response_text = "\n".join(response_lines)

            self.logger.info(
                f"[{request_id}] Web search completed: {len(search_results)} results in {search_time:.3f}s"
            )

            return [TextContent(type="text", text=response_text)]

        except WebSearchError as e:
            error_msg = f"🔍 **Web search error:** {str(e)}"
            self.logger.error(f"[{request_id}] Web search error: {e}")
            return [TextContent(type="text", text=error_msg)]

        except Exception as e:
            error_msg = f"❌ **Search failed:** {str(e)}"
            self.logger.error(
                f"[{request_id}] Unexpected error during web search: {e}", exc_info=True
            )
            return [TextContent(type="text", text=error_msg)]

    async def _smart_search_internal(
            self,
            request_id: str,
            query: str,
            similarity_threshold: float = 0.75,
            local_top_k: int = 5,
            web_max_results: int = 5,
            include_sources: bool = True,
            combine_strategy: str = "relevance_score",
            min_local_results: int = 2,
    ) -> List[TextContent]:
        """
        Perform sophisticated intelligent hybrid search with advanced decision logic.

        This method implements a comprehensive search strategy that:
        1. Searches local knowledge base first
        2. Evaluates result quality against similarity threshold
        3. Makes intelligent decisions about web search necessity
        4. Combines and deduplicates results across sources
        5. Provides relevance explanations and confidence scoring
        6. Assesses source credibility

        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            similarity_threshold: Threshold for triggering web search (0.0-1.0)
            local_top_k: Maximum local results to retrieve
            web_max_results: Maximum web results to retrieve
            include_sources: Whether to include detailed source information
            combine_strategy: Strategy for combining results
            min_local_results: Minimum local results before considering web search

        Returns:
            List of TextContent with sophisticated search results and analysis
        """
        start_time = time.time()
        self.logger.debug(
            f"[{request_id}] Starting smart search with threshold {similarity_threshold}"
        )

        try:
            # Step 1: Search local knowledge base
            self.logger.debug(f"[{request_id}] Step 1: Searching local knowledge base")
            local_search_start = time.time()

            # Try LlamaIndex first if available
            llamaindex_processor = await self._get_llamaindex_processor()
            
            if llamaindex_processor:
                self.logger.debug(f"[{request_id}] Using LlamaIndex for smart search")
                
                # Use LlamaIndex for local search
                nodes = await llamaindex_processor.search(
                    query=query,
                    top_k=local_top_k,
                    similarity_cutoff=0.6,
                )
                
                # Convert to SearchResult format
                local_raw_results = []
                for node in nodes:
                    doc_content = node.node.text
                    doc_metadata = node.node.metadata or {}
                    
                    from .vector_store import SearchResult
                    result = SearchResult(
                        document=Document(
                            page_content=doc_content,
                            metadata=doc_metadata,
                            id=node.node.id_ if hasattr(node.node, 'id_') else None,
                        ),
                        score=node.score,
                        metadata=doc_metadata,
                    )
                    local_raw_results.append(result)
                    
            else:
                # Fallback to traditional vector store
                vector_store = await self._get_vector_store()
                local_raw_results = await vector_store.search(
                    query=query, top_k=local_top_k, filter_dict=None
                )

            local_search_time = time.time() - local_search_start

            # Calculate maximum score and analyze local results quality
            max_local_score = 0.0
            high_quality_local_count = 0

            if local_raw_results:
                max_local_score = max(result.score for result in local_raw_results)
                high_quality_local_count = sum(
                    1 for result in local_raw_results if result.score >= 0.8
                )

            self.logger.info(
                f"[{request_id}] Local search: {len(local_raw_results)} results, max_score={max_local_score:.3f}"
            )

            # Step 2: Decision logic implementation
            web_search_triggered = False
            decision_reason = ""
            confidence_level = "HIGH"

            if not local_raw_results:
                # No local results - perform web search directly
                web_search_triggered = True
                decision_reason = (
                    "No local knowledge found - searching web for comprehensive results"
                )
                confidence_level = "MEDIUM"
            elif max_local_score < similarity_threshold:
                # Low quality local results - supplement with web search
                web_search_triggered = True
                decision_reason = f"Local results quality below threshold ({max_local_score:.3f} < {similarity_threshold}) - enhancing with web search"
                confidence_level = "MEDIUM"
            elif len(local_raw_results) < min_local_results:
                # Insufficient local results - get more from web
                web_search_triggered = True
                decision_reason = f"Insufficient local results ({len(local_raw_results)} < {min_local_results}) - supplementing with web search"
                confidence_level = "MEDIUM"
            else:
                # High quality local results sufficient
                decision_reason = f"Local knowledge sufficient (max_score={max_local_score:.3f} >= {similarity_threshold})"
                confidence_level = "HIGH"

            # Step 3: Web search if triggered
            web_raw_results = []
            web_search_time = 0.0

            if web_search_triggered and web_max_results > 0:
                self.logger.debug(f"[{request_id}] Step 3: Performing web search")
                web_search_start = time.time()

                try:
                    web_search_mgr = await self._get_web_search()
                    web_raw_results, web_metadata = await web_search_mgr.search(
                        query=query, max_results=web_max_results, include_answer=True
                    )
                    web_search_time = time.time() - web_search_start
                    self.logger.info(
                        f"[{request_id}] Web search: {len(web_raw_results)} results"
                    )

                except Exception as e:
                    self.logger.warning(f"[{request_id}] Web search failed: {e}")
                    web_raw_results = []
                    confidence_level = "LOW" if not local_raw_results else "MEDIUM"

            # Step 4: Cross-source result deduplication
            deduplicated_results = self._deduplicate_results(
                local_raw_results, web_raw_results
            )

            # Step 5: Source credibility assessment and relevance scoring
            enhanced_results = self._assess_result_credibility(
                deduplicated_results, query
            )

            # Step 6: Format comprehensive response
            total_time = time.time() - start_time
            response_sections = []

            # Header with decision summary
            response_sections.append(
                f"🧠 **Smart Search Results for '{query}'**\n"
                f"**Decision:** {decision_reason}\n"
                f"**Confidence Level:** {confidence_level}\n"
                f"**Total Processing Time:** {total_time:.3f}s\n"
            )

            # Local Knowledge Base Results Section
            response_sections.append("## 📚 Local Knowledge Base Results\n")

            if local_raw_results:
                response_sections.append(
                    f"**Found {len(local_raw_results)} local result{'s' if len(local_raw_results) != 1 else ''}**\n"
                    f"**Maximum Similarity Score:** {max_local_score:.3f}\n"
                    f"**High Quality Results:** {high_quality_local_count}/{len(local_raw_results)}\n"
                    f"**Search Time:** {local_search_time:.3f}s\n"
                )

                for i, result in enumerate(
                        local_raw_results[:3], 1
                ):  # Show top 3 local results
                    relevance_explanation = self._generate_relevance_explanation(
                        result, query, "local"
                    )
                    source_path = (
                        result.metadata.get("source", "Unknown source")
                        if result.metadata
                        else "Unknown source"
                    )

                    # Score visualization
                    score_emoji = (
                        "🟢"
                        if result.score >= 0.8
                        else "🟡" if result.score >= 0.6 else "🔴"
                    )

                    response_sections.append(
                        f"**{i}. {score_emoji} Similarity: {result.score:.3f}**\n"
                        f"📂 **Source:** [{source_path}]({source_path})\n"
                        f"📖 **Content Preview:** {result.content[:200]}{'...' if len(result.content) > 200 else ''}\n"
                        f"🎯 **Relevance:** {relevance_explanation}\n"
                        f"{'─' * 40}\n"
                    )
            else:
                response_sections.append(
                    "❌ **No local knowledge found for this query**\n"
                )

            # Web Search Results Section (if performed)
            if web_search_triggered:
                response_sections.append("\n## 🌐 Web Search Results\n")

                if web_raw_results:
                    response_sections.append(
                        f"**Found {len(web_raw_results)} web result{'s' if len(web_raw_results) != 1 else ''}**\n"
                        f"**Search Time:** {web_search_time:.3f}s\n"
                        f"**Status:** {'✅ Fresh results' if web_search_time > 0 else '📋 Cached results'}\n"
                    )

                    for i, result in enumerate(
                            web_raw_results[:3], 1
                    ):  # Show top 3 web results
                        relevance_explanation = self._generate_relevance_explanation(
                            result, query, "web"
                        )
                        credibility_score = self._calculate_credibility_score(result)

                        # Score and credibility visualization
                        score_emoji = (
                            "🟢"
                            if result.score >= 0.8
                            else "🟡" if result.score >= 0.6 else "🔴"
                        )
                        credibility_emoji = (
                            "🏆"
                            if credibility_score >= 0.8
                            else "✅" if credibility_score >= 0.6 else "⚠️"
                        )

                        response_sections.append(
                            f"**{i}. {score_emoji} Quality Score: {result.score:.3f}**\n"
                            f"📰 **Title:** {result.title}\n"
                            f"🌐 **Source:** [{result.source_domain or 'Unknown'}]({result.url})\n"
                            f"{credibility_emoji} **Credibility:** {credibility_score:.2f}/1.0\n"
                            f"📄 **Content Preview:** {result.content[:200]}{'...' if len(result.content) > 200 else ''}\n"
                            f"🎯 **Relevance:** {relevance_explanation}\n"
                            f"{'─' * 40}\n"
                        )
                else:
                    response_sections.append(
                        "❌ **No web results found or web search failed**\n"
                    )

            # Smart Recommendations Section
            response_sections.append("\n## 🎯 Smart Recommendations\n")

            recommendations = self._generate_smart_recommendations(
                local_raw_results,
                web_raw_results,
                max_local_score,
                similarity_threshold,
                confidence_level,
                query,
            )
            response_sections.append(recommendations)

            # Search Strategy Summary
            response_sections.append("\n## 📊 Search Strategy Analysis\n")
            response_sections.append(
                f"**Strategy Used:** {combine_strategy}\n"
                f"**Threshold Applied:** {similarity_threshold}\n"
                f"**Web Search Triggered:** {'✅ Yes' if web_search_triggered else '❌ No'}\n"
                f"**Total Sources Consulted:** {len(local_raw_results) + len(web_raw_results)}\n"
                f"**Deduplication Applied:** ✅ Cross-source duplicate removal\n"
                f"**Overall Confidence:** {confidence_level}\n"
            )

            # Performance metrics
            response_sections.append(
                f"\n⏱️ **Performance Metrics:**\n"
                f"• Local search: {local_search_time:.3f}s\n"
                f"• Web search: {web_search_time:.3f}s\n"
                f"• Total processing: {total_time:.3f}s\n"
            )

            final_response = "\n".join(response_sections)

            self.logger.info(
                f"[{request_id}] Smart search completed: "
                f"local={len(local_raw_results)}, web={len(web_raw_results)}, "
                f"confidence={confidence_level}, time={total_time:.3f}s"
            )

            return [TextContent(type="text", text=final_response)]

        except Exception as e:
            error_msg = f"❌ **Smart search failed:** {str(e)}\n\nPlease try again or contact support if the issue persists."
            self.logger.error(f"[{request_id}] Smart search error: {e}", exc_info=True)
            return [TextContent(type="text", text=error_msg)]

    def _deduplicate_results(
            self, local_results: List[SearchResult], web_results: List[WebSearchResult]
    ) -> Dict[str, List]:
        """
        Perform cross-source result deduplication using content similarity and URL matching.

        Args:
            local_results: Results from local knowledge base
            web_results: Results from web search

        Returns:
            Dictionary with deduplicated and categorized results
        """
        from difflib import SequenceMatcher

        def content_similarity(text1: str, text2: str) -> float:
            """Calculate content similarity between two text snippets."""
            return SequenceMatcher(
                None, text1.lower()[:500], text2.lower()[:500]
            ).ratio()

        # Track duplicates and keep best version
        deduplicated_local = []
        deduplicated_web = []
        duplicate_pairs = []

        # Check for duplicates between local and web results
        for local_result in local_results:
            is_duplicate = False
            local_content = local_result.content.lower()

            for web_result in web_results:
                web_content = web_result.content.lower()

                # Check content similarity (threshold: 0.8 for high similarity)
                similarity = content_similarity(local_content, web_content)

                if similarity >= 0.8:
                    duplicate_pairs.append(
                        {
                            "local": local_result,
                            "web": web_result,
                            "similarity": similarity,
                            "kept": (
                                "local"
                                if local_result.score >= web_result.score
                                else "web"
                            ),
                        }
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated_local.append(local_result)

        # Add web results that aren't duplicates
        for web_result in web_results:
            is_duplicate = any(pair["web"] == web_result for pair in duplicate_pairs)
            if not is_duplicate:
                deduplicated_web.append(web_result)

        return {
            "local": deduplicated_local,
            "web": deduplicated_web,
            "duplicates": duplicate_pairs,
        }

    def _assess_result_credibility(
            self, results_dict: Dict[str, List], query: str
    ) -> Dict[str, List]:
        """
        Assess source credibility and enhance results with credibility scores.

        Args:
            results_dict: Deduplicated results dictionary
            query: Original search query for context

        Returns:
            Enhanced results with credibility assessment
        """
        # Enhance local results (they have inherent high credibility)
        enhanced_local = []
        for result in results_dict.get("local", []):
            # Local results get high credibility based on curation
            result.credibility_score = 0.9  # High credibility for curated local content
            result.credibility_factors = [
                "Curated local content",
                "Direct source access",
            ]
            enhanced_local.append(result)

        # Enhance web results with credibility assessment
        enhanced_web = []
        for result in results_dict.get("web", []):
            credibility_score = self._calculate_credibility_score(result)
            result.credibility_score = credibility_score
            enhanced_web.append(result)

        return {
            "local": enhanced_local,
            "web": enhanced_web,
            "duplicates": results_dict.get("duplicates", []),
        }

    def _calculate_credibility_score(self, result) -> float:
        """
        Calculate credibility score for a web search result.

        Args:
            result: WebSearchResult to assess

        Returns:
            Credibility score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        factors = []

        # Domain reputation (simplified assessment)
        if hasattr(result, "source_domain") and result.source_domain:
            domain = result.source_domain.lower()

            # High credibility domains
            if any(
                    trusted in domain
                    for trusted in [
                        "edu",
                        "gov",
                        "org",
                        "wikipedia",
                        "reuters",
                        "bbc",
                        "nature",
                        "science",
                        "ieee",
                        "acm",
                        "arxiv",
                        "pubmed",
                    ]
            ):
                score += 0.3
                factors.append("Trusted domain")

            # Medium credibility domains
            elif any(
                    medium in domain
                    for medium in ["com", "net", "co.", "news", "tech", "research"]
            ):
                score += 0.1
                factors.append("Established domain")

        # Content quality indicators
        if hasattr(result, "content") and result.content:
            content_length = len(result.content)

            # Appropriate length (not too short, not too long)
            if 100 <= content_length <= 2000:
                score += 0.1
                factors.append("Appropriate content length")

            # Check for academic/professional language patterns
            academic_terms = [
                "research",
                "study",
                "analysis",
                "methodology",
                "findings",
                "conclusion",
            ]
            if any(term in result.content.lower() for term in academic_terms):
                score += 0.1
                factors.append("Academic/professional content")

        # Title quality
        if hasattr(result, "title") and result.title:
            if len(result.title) >= 10 and not result.title.isupper():
                score += 0.05
                factors.append("Well-formatted title")

        # Result score quality
        if hasattr(result, "score") and result.score >= 0.8:
            score += 0.1
            factors.append("High relevance score")

        # Store factors for explanation
        result.credibility_factors = factors

        return min(1.0, score)  # Cap at 1.0

    def _generate_relevance_explanation(
            self, result, query: str, source_type: str
    ) -> str:
        """
        Generate explanation for why a result is relevant to the query.

        Args:
            result: Search result (local or web)
            query: Original search query
            source_type: "local" or "web"

        Returns:
            Human-readable relevance explanation
        """
        query_terms = set(query.lower().split())
        content_lower = result.content.lower() if hasattr(result, "content") else ""

        # Find matching terms
        matching_terms = []
        for term in query_terms:
            if term in content_lower and len(term) > 2:  # Skip short words
                matching_terms.append(term)

        # Calculate match percentage
        match_percentage = (
            (len(matching_terms) / len(query_terms)) * 100 if query_terms else 0
        )

        # Generate explanation based on source type and matches
        if source_type == "local":
            base_explanation = (
                f"Matches {len(matching_terms)}/{len(query_terms)} query terms"
            )
            if hasattr(result, "score"):
                if result.score >= 0.9:
                    return f"{base_explanation} with excellent semantic similarity"
                elif result.score >= 0.8:
                    return f"{base_explanation} with high semantic similarity"
                elif result.score >= 0.7:
                    return f"{base_explanation} with good semantic similarity"
                else:
                    return f"{base_explanation} with moderate semantic similarity"
        else:  # web results
            base_explanation = f"Contains {len(matching_terms)} key terms from query"
            if hasattr(result, "score"):
                if result.score >= 0.8:
                    return f"{base_explanation}, highly relevant content"
                elif result.score >= 0.6:
                    return f"{base_explanation}, moderately relevant content"
                else:
                    return f"{base_explanation}, potentially relevant content"

        return "Relevance detected through content analysis"

    def _generate_smart_recommendations(
            self,
            local_results: List,
            web_results: List,
            max_local_score: float,
            similarity_threshold: float,
            confidence_level: str,
            query: str,
    ) -> str:
        """
        Generate intelligent recommendations based on search results analysis.

        Args:
            local_results: Local search results
            web_results: Web search results
            max_local_score: Highest local similarity score
            similarity_threshold: Applied threshold
            confidence_level: Overall confidence level
            query: Original search query

        Returns:
            Formatted recommendations string
        """
        recommendations = []

        # Analyze result quality and provide recommendations
        if confidence_level == "HIGH":
            if local_results and max_local_score >= 0.9:
                recommendations.append(
                    "✅ **Excellent local knowledge found** - Your knowledge base contains highly relevant information for this query."
                )
            else:
                recommendations.append(
                    "✅ **Good local knowledge available** - Sufficient information found in your knowledge base."
                )

        elif confidence_level == "MEDIUM":
            if local_results and web_results:
                recommendations.append(
                    "🔄 **Hybrid approach applied** - Combined local and web sources for comprehensive coverage."
                )
            elif web_results and not local_results:
                recommendations.append(
                    "🌐 **Web sources utilized** - No local knowledge found, relying on current web information."
                )
            elif local_results and not web_results:
                recommendations.append(
                    "📚 **Local sources only** - Web search unavailable, using available local knowledge."
                )

            # Suggest improvements
            if max_local_score < similarity_threshold:
                recommendations.append(
                    f"💡 **Suggestion:** Consider adding more documents about '{query}' to improve future local search results."
                )

        else:  # LOW confidence
            recommendations.append(
                "⚠️ **Limited results found** - Consider refining your search query or adding relevant documents to your knowledge base."
            )

            # Provide specific suggestions
            query_words = query.split()
            if len(query_words) > 5:
                recommendations.append(
                    "💡 **Try shorter queries** - Use 2-4 key terms for better results."
                )
            elif len(query_words) == 1:
                recommendations.append(
                    "💡 **Try more specific queries** - Add context or related terms."
                )

        # Source diversity analysis
        total_sources = len(local_results) + len(web_results)
        if total_sources >= 5:
            recommendations.append(
                f"📊 **Good source diversity** - Found {total_sources} results across multiple sources."
            )
        elif total_sources >= 2:
            recommendations.append(
                f"📊 **Moderate source coverage** - Found {total_sources} relevant sources."
            )

        # Credibility assessment
        if web_results:
            high_credibility_count = sum(
                1
                for result in web_results
                if hasattr(result, "credibility_score")
                and result.credibility_score >= 0.8
            )
            if high_credibility_count > 0:
                recommendations.append(
                    f"🏆 **High credibility sources** - {high_credibility_count} web result(s) from trusted sources."
                )

        return (
            "\n".join(recommendations)
            if recommendations
            else "No specific recommendations available."
        )

    # Document Management Tool Handlers
    
    async def _add_document(
        self,
        request_id: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        batch_files: Optional[List[str]] = None,
        auto_detect_type: bool = True,
    ) -> List[TextContent]:
        """Add document(s) to the knowledge base."""
        try:
            doc_processor = await self._get_document_processor()
            vector_store = await self._get_vector_store()
            
            metadata = metadata or {}
            tags = tags or []
            batch_files = batch_files or []
            
            # Track processing results
            results = []
            total_chunks = 0
            processing_time = time.time()
            
            if batch_files:
                # Batch processing
                for file in batch_files:
                    try:
                        processed_docs = await doc_processor.process_file(Path(file))
                        for doc in processed_docs:
                            doc_id = str(uuid.uuid4())
                            doc.metadata.update(metadata)
                            doc.metadata["document_id"] = doc_id
                            doc.metadata["tags"] = tags
                            doc.metadata["created_at"] = datetime.now().isoformat()
                            
                            # Store in vector database
                            await vector_store.add_documents([Document(
                                page_content=doc.content,
                                metadata=doc.metadata,
                                id=doc_id
                            )])
                            
                            # Track document metadata
                            self._documents_metadata[doc_id] = doc.metadata
                            self._document_tags[doc_id] = tags
                            self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                            
                            total_chunks += 1
                            results.append(f"✅ {Path(file).name}: {doc_id}")
                    except Exception as e:
                        results.append(f"❌ {Path(file).name}: {str(e)}")
                        
            elif file_path:
                # Single file processing
                processed_docs = await doc_processor.process_file(Path(file_path))
                for doc in processed_docs:
                    doc_id = str(uuid.uuid4())
                    doc.metadata.update(metadata)
                    doc.metadata["document_id"] = doc_id
                    doc.metadata["tags"] = tags
                    doc.metadata["created_at"] = datetime.now().isoformat()
                    
                    await vector_store.add_documents([Document(
                        page_content=doc.content,
                        metadata=doc.metadata,
                        id=doc_id
                    )])
                    
                    self._documents_metadata[doc_id] = doc.metadata
                    self._document_tags[doc_id] = tags
                    self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                    
                    total_chunks += 1
                    results.append(f"✅ Document added: {doc_id}")
                    
            elif content:
                # Raw content processing
                doc_id = str(uuid.uuid4())
                enhanced_metadata = {
                    **metadata,
                    "document_id": doc_id,
                    "tags": tags,
                    "created_at": datetime.now().isoformat(),
                    "source": "raw_content",
                    "content_length": len(content)
                }
                
                await vector_store.add_documents([Document(
                    page_content=content,
                    metadata=enhanced_metadata,
                    id=doc_id
                )])
                
                self._documents_metadata[doc_id] = enhanced_metadata
                self._document_tags[doc_id] = tags
                self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                
                total_chunks += 1
                results.append(f"✅ Content added: {doc_id}")
                
            processing_time = time.time() - processing_time
            
            # Format response
            response = f"""📄 **Document Addition Complete**

**Processing Summary:**
• Documents processed: {len(results)}
• Total chunks created: {total_chunks}
• Processing time: {processing_time:.2f}s

**Results:**
{chr(10).join(results)}

**Statistics:**
• Success rate: {len([r for r in results if r.startswith('✅')])/len(results)*100:.1f}%
• Failed: {len([r for r in results if r.startswith('❌')])}
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Document addition failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error adding document:** {str(e)}")]

    async def _update_document(
        self,
        request_id: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        partial_update: bool = True,
        create_version: bool = True,
        force_update: bool = False,
    ) -> List[TextContent]:
        """Update an existing document."""
        try:
            if document_id not in self._documents_metadata:
                return [TextContent(type="text", text=f"❌ **Document not found:** {document_id}")]
            
            # Create version backup if requested
            if create_version:
                version_data = {
                    "content": content,
                    "metadata": self._documents_metadata[document_id].copy(),
                    "tags": self._document_tags[document_id].copy(),
                    "timestamp": datetime.now().isoformat(),
                    "version": len(self._document_versions.get(document_id, [])) + 1
                }
                
                if document_id not in self._document_versions:
                    self._document_versions[document_id] = []
                self._document_versions[document_id].append(version_data)
            
            # Update content if provided
            if content:
                vector_store = await self._get_vector_store()
                # Remove old document
                await vector_store.delete_documents([document_id])
                # Add updated document
                updated_metadata = self._documents_metadata[document_id].copy()
                updated_metadata["modified_at"] = datetime.now().isoformat()
                
                await vector_store.add_documents([Document(
                    page_content=content,
                    metadata=updated_metadata,
                    id=document_id
                )])
            
            # Update metadata
            if metadata:
                if partial_update:
                    self._documents_metadata[document_id].update(metadata)
                else:
                    self._documents_metadata[document_id] = metadata
                self._documents_metadata[document_id]["modified_at"] = datetime.now().isoformat()
            
            # Update tags
            if tags is not None:
                self._document_tags[document_id] = tags
            
            return [TextContent(type="text", text=f"✅ **Document updated successfully:** {document_id}")]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Document update failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error updating document:** {str(e)}")]

    async def _delete_document(
        self,
        request_id: str,
        document_id: str,
        soft_delete: bool = True,
        cleanup_chunks: bool = True,
        update_indices: bool = True,
        backup_before_delete: bool = True,
    ) -> List[TextContent]:
        """Delete a document from the knowledge base."""
        try:
            if document_id not in self._documents_metadata:
                return [TextContent(type="text", text=f"❌ **Document not found:** {document_id}")]
            
            # Create backup if requested
            if backup_before_delete:
                backup_data = {
                    "metadata": self._documents_metadata[document_id].copy(),
                    "tags": self._document_tags[document_id].copy(),
                    "usage": self._document_usage[document_id].copy(),
                    "deleted_at": datetime.now().isoformat()
                }
                
                if document_id not in self._document_versions:
                    self._document_versions[document_id] = []
                self._document_versions[document_id].append(backup_data)
            
            if soft_delete:
                # Mark as deleted but keep metadata
                self._documents_metadata[document_id]["deleted"] = True
                self._documents_metadata[document_id]["deleted_at"] = datetime.now().isoformat()
                action = "soft deleted"
            else:
                # Remove from vector store and all tracking
                if cleanup_chunks:
                    vector_store = await self._get_vector_store()
                    await vector_store.delete_documents([document_id])
                
                # Remove from all tracking dictionaries
                del self._documents_metadata[document_id]
                del self._document_tags[document_id]
                del self._document_usage[document_id]
                action = "permanently deleted"
            
            return [TextContent(type="text", text=f"✅ **Document {action} successfully:** {document_id}")]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Document deletion failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error deleting document:** {str(e)}")]

    async def _list_documents(
        self,
        request_id: str,
        page: int = 1,
        page_size: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
        filter_tags: Optional[List[str]] = None,
        filter_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_by: str = "date",
        sort_order: str = "desc",
        include_stats: bool = True,
        include_deleted: bool = False,
    ) -> List[TextContent]:
        """List documents with filtering and pagination."""
        try:
            # Apply filters
            filtered_docs = []
            for doc_id, metadata in self._documents_metadata.items():
                # Skip deleted documents unless requested
                if metadata.get("deleted", False) and not include_deleted:
                    continue
                
                # Apply metadata filters
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                # Apply tag filters
                if filter_tags:
                    doc_tags = self._document_tags.get(doc_id, [])
                    if not any(tag in doc_tags for tag in filter_tags):
                        continue
                
                # Apply type filter
                if filter_type and metadata.get("file_type") != filter_type:
                    continue
                
                # Apply date filters
                created_at = metadata.get("created_at")
                if date_from and created_at and created_at < date_from:
                    continue
                if date_to and created_at and created_at > date_to:
                    continue
                
                filtered_docs.append((doc_id, metadata))
            
            # Sort documents
            def sort_key(item):
                doc_id, metadata = item
                if sort_by == "date":
                    return metadata.get("created_at", "")
                elif sort_by == "size":
                    return metadata.get("content_length", 0)
                elif sort_by == "type":
                    return metadata.get("file_type", "")
                elif sort_by == "name":
                    return metadata.get("source", "")
                elif sort_by == "usage":
                    return self._document_usage.get(doc_id, {}).get("access_count", 0)
                return ""
            
            filtered_docs.sort(key=sort_key, reverse=(sort_order == "desc"))
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_docs = filtered_docs[start_idx:end_idx]
            
            # Format response
            response_lines = [f"📋 **Document List (Page {page}/{(len(filtered_docs) + page_size - 1) // page_size})**\n"]
            
            if include_stats:
                total_docs = len(self._documents_metadata)
                deleted_docs = len([m for m in self._documents_metadata.values() if m.get("deleted", False)])
                response_lines.append(f"**Statistics:**")
                response_lines.append(f"• Total documents: {total_docs}")
                response_lines.append(f"• Active documents: {total_docs - deleted_docs}")
                response_lines.append(f"• Deleted documents: {deleted_docs}")
                response_lines.append(f"• Filtered results: {len(filtered_docs)}\n")
            
            for i, (doc_id, metadata) in enumerate(page_docs, 1):
                tags = self._document_tags.get(doc_id, [])
                usage = self._document_usage.get(doc_id, {})
                
                status = "🗑️ Deleted" if metadata.get("deleted", False) else "✅ Active"
                source = metadata.get("source", "Unknown")
                created = metadata.get("created_at", "Unknown")[:19]  # Trim to date/time
                
                response_lines.append(f"**{start_idx + i}. {status} - {doc_id[:8]}...**")
                response_lines.append(f"📂 Source: {source}")
                response_lines.append(f"📅 Created: {created}")
                response_lines.append(f"🏷️ Tags: {', '.join(tags) if tags else 'None'}")
                response_lines.append(f"👁️ Views: {usage.get('access_count', 0)}")
                response_lines.append("")
            
            return [TextContent(type="text", text="\n".join(response_lines))]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Document listing failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error listing documents:** {str(e)}")]

    async def _manage_tags(
        self,
        request_id: str,
        action: str,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        search_tags: Optional[List[str]] = None,
        tag_filter: Optional[str] = None,
    ) -> List[TextContent]:
        """Manage document tags."""
        try:
            if action == "list":
                all_tags = set()
                for doc_tags in self._document_tags.values():
                    all_tags.update(doc_tags)
                
                if tag_filter:
                    all_tags = {tag for tag in all_tags if tag_filter.lower() in tag.lower()}
                
                response = f"🏷️ **Available Tags ({len(all_tags)} total):**\n\n"
                response += "\n".join(f"• {tag}" for tag in sorted(all_tags))
                return [TextContent(type="text", text=response)]
            
            elif action == "search":
                if not search_tags:
                    return [TextContent(type="text", text="❌ **Error:** search_tags required for search action")]
                
                matching_docs = []
                for doc_id, doc_tags in self._document_tags.items():
                    if any(tag in doc_tags for tag in search_tags):
                        matching_docs.append(doc_id)
                
                response = f"🔍 **Documents with tags {search_tags}:**\n\n"
                response += "\n".join(f"• {doc_id}" for doc_id in matching_docs)
                return [TextContent(type="text", text=response)]
            
            elif action in ["add", "remove"]:
                if not tags:
                    return [TextContent(type="text", text=f"❌ **Error:** tags required for {action} action")]
                
                target_docs = []
                if document_id:
                    target_docs = [document_id]
                elif document_ids:
                    target_docs = document_ids
                else:
                    return [TextContent(type="text", text="❌ **Error:** document_id or document_ids required")]
                
                results = []
                for doc_id in target_docs:
                    if doc_id not in self._document_tags:
                        results.append(f"❌ {doc_id}: Not found")
                        continue
                    
                    if action == "add":
                        for tag in tags:
                            if tag not in self._document_tags[doc_id]:
                                self._document_tags[doc_id].append(tag)
                        results.append(f"✅ {doc_id}: Added tags {tags}")
                    else:  # remove
                        for tag in tags:
                            if tag in self._document_tags[doc_id]:
                                self._document_tags[doc_id].remove(tag)
                        results.append(f"✅ {doc_id}: Removed tags {tags}")
                
                return [TextContent(type="text", text=f"🏷️ **Tag {action} results:**\n\n" + "\n".join(results))]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Tag management failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error managing tags:** {str(e)}")]

    async def _bulk_operations(
        self,
        request_id: str,
        operation: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 10,
        continue_on_error: bool = True,
        progress_callback: bool = True,
    ) -> List[TextContent]:
        """Perform bulk operations on multiple documents."""
        try:
            results = []
            total = len(documents)
            processed = 0
            
            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]
                
                for doc_data in batch:
                    try:
                        if operation == "add":
                            result = await self._add_document(request_id, **doc_data)
                        elif operation == "update":
                            result = await self._update_document(request_id, **doc_data)
                        elif operation == "delete":
                            result = await self._delete_document(request_id, **doc_data)
                        elif operation == "tag":
                            result = await self._manage_tags(request_id, **doc_data)
                        
                        results.append(f"✅ Operation {processed + 1}: Success")
                        processed += 1
                        
                    except Exception as e:
                        error_msg = f"❌ Operation {processed + 1}: {str(e)}"
                        results.append(error_msg)
                        if not continue_on_error:
                            break
                        processed += 1
                
                if progress_callback:
                    self.logger.info(f"[{request_id}] Bulk {operation}: {processed}/{total} completed")
            
            success_count = len([r for r in results if r.startswith("✅")])
            failure_count = len([r for r in results if r.startswith("❌")])
            
            response = f"""⚡ **Bulk {operation.title()} Operation Complete**

**Summary:**
• Total operations: {total}
• Successful: {success_count}
• Failed: {failure_count}
• Success rate: {success_count/total*100:.1f}%

**Detailed Results:**
{chr(10).join(results[:20])}  # Show first 20 results
{f'... and {len(results) - 20} more' if len(results) > 20 else ''}
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Bulk operation failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error in bulk operation:** {str(e)}")]

    async def _document_analytics(
        self,
        request_id: str,
        document_id: Optional[str] = None,
        analytics_type: str = "overview",
        time_range: str = "month",
        limit: int = 20,
        include_charts: bool = False,
    ) -> List[TextContent]:
        """Get document usage analytics."""
        try:
            if analytics_type == "overview":
                total_docs = len(self._documents_metadata)
                active_docs = len([m for m in self._documents_metadata.values() if not m.get("deleted", False)])
                total_views = sum(usage.get("access_count", 0) for usage in self._document_usage.values())
                
                # Get top tags
                tag_counts = {}
                for tags in self._document_tags.values():
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                response = f"""📊 **Document Analytics Overview**

**Document Statistics:**
• Total documents: {total_docs}
• Active documents: {active_docs}
• Deleted documents: {total_docs - active_docs}
• Total views: {total_views}
• Average views per document: {total_views/max(total_docs, 1):.1f}

**Top Tags:**
{chr(10).join(f'• {tag}: {count} documents' for tag, count in top_tags)}

**Recent Activity:**
• Documents added today: {len([m for m in self._documents_metadata.values() if m.get('created_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])}
• Documents modified today: {len([m for m in self._documents_metadata.values() if m.get('modified_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])}
"""
                
            elif analytics_type == "popular":
                # Get most accessed documents
                popular_docs = sorted(
                    [(doc_id, usage.get("access_count", 0)) for doc_id, usage in self._document_usage.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:limit]
                
                response = f"📈 **Most Popular Documents (Top {limit}):**\n\n"
                for i, (doc_id, views) in enumerate(popular_docs, 1):
                    metadata = self._documents_metadata.get(doc_id, {})
                    source = metadata.get("source", "Unknown")[:50]
                    response += f"{i}. {doc_id[:8]}... - {views} views\n   📂 {source}\n\n"
                
            elif analytics_type == "usage" and document_id:
                if document_id not in self._document_usage:
                    return [TextContent(type="text", text=f"❌ **Document not found:** {document_id}")]
                
                usage = self._document_usage[document_id]
                metadata = self._documents_metadata[document_id]
                tags = self._document_tags.get(document_id, [])
                
                response = f"""📊 **Document Usage Analytics: {document_id}**

**Metadata:**
• Source: {metadata.get('source', 'Unknown')}
• Created: {metadata.get('created_at', 'Unknown')}
• Type: {metadata.get('file_type', 'Unknown')}
• Size: {metadata.get('content_length', 'Unknown')} characters

**Usage Statistics:**
• Total views: {usage.get('access_count', 0)}
• Last accessed: {usage.get('last_accessed', 'Never')}
• Tags: {', '.join(tags) if tags else 'None'}

**Versions:**
• Version history: {len(self._document_versions.get(document_id, []))} versions
"""
            
            else:
                response = f"📊 **Analytics type '{analytics_type}' not yet implemented**"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Analytics generation failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error generating analytics:** {str(e)}")]

    # Search Optimization Tool Handlers
    
    async def _optimize_search(
        self,
        request_id: str,
        query: str,
        user_id: str = "anonymous",
        ranking_strategy: str = "hybrid_balanced",
        enable_personalization: bool = True,
        enable_summarization: bool = True,
        max_results: int = 10,
    ) -> List[TextContent]:
        """Optimize search results using SearchOptimizer."""
        try:
            # Get search optimizer
            optimizer = await self._get_search_optimizer()
            if not optimizer:
                return [TextContent(type="text", text="❌ **Search optimizer not available**")]
            
            # First, get base search results
            vector_store = await self._get_vector_store()
            base_results = await vector_store.similarity_search_with_score(
                query=query,
                k=max_results * 2,  # Get more results for better optimization
                include_metadata=True
            )
            
            # Convert ranking strategy string to enum
            strategy_map = {
                "vector_only": RankingStrategy.VECTOR_ONLY,
                "bm25_only": RankingStrategy.BM25_ONLY,
                "hybrid_balanced": RankingStrategy.HYBRID_BALANCED,
                "hybrid_vector_weighted": RankingStrategy.HYBRID_VECTOR_WEIGHTED,
                "hybrid_bm25_weighted": RankingStrategy.HYBRID_BM25_WEIGHTED,
                "personalized": RankingStrategy.PERSONALIZED,
            }
            strategy = strategy_map.get(ranking_strategy, RankingStrategy.HYBRID_BALANCED)
            
            # Optimize search results
            optimized_results = await optimizer.optimize_search(
                query=query,
                user_id=user_id,
                search_results=base_results,
                ranking_strategy=strategy,
                enable_personalization=enable_personalization,
                enable_summarization=enable_summarization,
                max_results=max_results
            )
            
            # Format response
            response_lines = [
                f"🔍 **Optimized Search Results for '{query}'**\n",
                f"**User:** {user_id}",
                f"**Strategy:** {ranking_strategy}",
                f"**Optimizations Applied:** {', '.join(optimized_results['metadata']['optimizations_applied'])}",
                f"**Processing Time:** {optimized_results['metadata']['processing_time']:.3f}s\n",
            ]
            
            # Add query analysis if available
            if 'query_analysis' in optimized_results['metadata']:
                analysis = optimized_results['metadata']['query_analysis']
                response_lines.extend([
                    f"**Query Analysis:**",
                    f"• Type: {analysis['type']}",
                    f"• Intent: {analysis['intent']}",
                    f"• Complexity: {analysis['complexity']:.2f}",
                    f"• Confidence: {analysis['confidence']:.2f}\n"
                ])
            
            # Add expanded query information
            if 'expanded_query' in optimized_results['metadata']:
                expanded = optimized_results['metadata']['expanded_query']
                if expanded['synonyms']:
                    response_lines.append(f"**Query Expansion:** {len(expanded['synonyms'])} terms expanded")
                if expanded['boost_terms']:
                    response_lines.append(f"**Boost Terms:** {', '.join(expanded['boost_terms'])}")
                response_lines.append("")
            
            # Add spelling suggestions if available
            if 'spelling_suggestions' in optimized_results['metadata']:
                suggestions = optimized_results['metadata']['spelling_suggestions']
                if suggestions:
                    response_lines.append(f"**Spelling Suggestions:** {len(suggestions)} corrections found\n")
            
            # Display optimized results
            response_lines.append(f"**Results ({len(optimized_results['optimized_results'])}):**\n")
            
            for i, result_data in enumerate(optimized_results['optimized_results'], 1):
                doc = result_data['document']
                ranking_score = result_data['ranking_score']
                summary = result_data['summary']
                
                # Format document info
                source = doc.metadata.get('source', 'Unknown')[:50]
                response_lines.extend([
                    f"**{i}. Score: {ranking_score.final_score:.3f}**",
                    f"📂 **Source:** {source}",
                    f"📊 **Ranking:** Vector: {ranking_score.vector_score:.3f}, BM25: {ranking_score.bm25_score:.3f}, Personal: {ranking_score.personalization_score:.3f}",
                ])
                
                # Add summary if available
                if summary and enable_summarization:
                    response_lines.extend([
                        f"📄 **Summary:** {summary.summary_text[:200]}{'...' if len(summary.summary_text) > 200 else ''}",
                        f"🎯 **Relevance:** {summary.relevance_score:.3f}",
                    ])
                else:
                    # Fallback to content preview
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    response_lines.append(f"📄 **Content:** {preview}")
                
                response_lines.append("─" * 50)
            
            return [TextContent(type="text", text="\n".join(response_lines))]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Search optimization failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error optimizing search:** {str(e)}")]

    async def _get_search_analytics(
        self,
        request_id: str,
        analytics_type: str = "overview",
        time_period: str = "week",
        include_recommendations: bool = True,
    ) -> List[TextContent]:
        """Get search analytics dashboard data."""
        try:
            optimizer = await self._get_search_optimizer()
            if not optimizer:
                return [TextContent(type="text", text="❌ **Search optimizer not available**")]
            
            dashboard_data = optimizer.get_analytics_dashboard()
            
            if "error" in dashboard_data:
                return [TextContent(type="text", text=f"❌ **Analytics error:** {dashboard_data['error']}")]
            
            # Format response based on analytics type
            if analytics_type == "overview":
                overview = dashboard_data.get('overview', {})
                if 'no_data' in overview:
                    return [TextContent(type="text", text="📊 **No analytics data available yet**")]
                
                response = f"""📊 **Search Analytics Overview**

**Query Statistics:**
• Total queries: {overview.get('total_queries', 0)}
• Daily queries: {overview.get('daily_queries', 0)}
• Weekly queries: {overview.get('weekly_queries', 0)}
• Unique users: {overview.get('unique_users', 0)}

**Performance Metrics:**
• Average response time: {overview.get('avg_response_time', 0):.3f}s
• Click-through rate: {overview.get('click_through_rate', 0):.1%}
• Average dwell time: {overview.get('avg_dwell_time', 0):.2f}s
• Queries per user: {overview.get('queries_per_user', 0):.2f}
"""
                
            elif analytics_type == "query_analytics":
                query_data = dashboard_data.get('query_analytics', {})
                if 'no_data' in query_data:
                    return [TextContent(type="text", text="📊 **No query analytics data available yet**")]
                
                response = f"""📊 **Query Analytics**

**Query Types:**
{chr(10).join(f'• {qtype}: {count}' for qtype, count in query_data.get('query_types', {}).items())}

**Search Intents:**
{chr(10).join(f'• {intent}: {count}' for intent, count in query_data.get('intents', {}).items())}

**Performance:**
• Average complexity: {query_data.get('avg_complexity', 0):.3f}
• Failure rate: {query_data.get('failure_rate', 0):.1%}
• Average query length: {query_data.get('avg_query_length', 0):.2f} words

**Popular Queries:**
{chr(10).join(f'• {query}: {count}' for query, count in query_data.get('popular_queries', [])[:5])}
"""
                
            elif analytics_type == "user_behavior":
                user_data = dashboard_data.get('user_behavior', {})
                if 'no_data' in user_data:
                    return [TextContent(type="text", text="📊 **No user behavior data available yet**")]
                
                response = f"""📊 **User Behavior Analytics**

**Session Statistics:**
• Average session length: {user_data.get('avg_session_length', 0):.2f} queries
• Average session duration: {user_data.get('avg_session_duration', 0):.2f}s
• Total sessions: {user_data.get('total_sessions', 0)}
• Returning users: {user_data.get('returning_users', 0)}

**Engagement Metrics:**
• Engagement rate: {user_data.get('engagement_rate', 0):.1%}
• Bounce rate: {user_data.get('bounce_rate', 0):.1%}
• Average click position: {user_data.get('avg_click_position', 0):.2f}
"""
                
            elif analytics_type == "performance":
                perf_data = dashboard_data.get('performance', {})
                if 'no_data' in perf_data:
                    return [TextContent(type="text", text="📊 **No performance data available yet**")]
                
                response = f"""📊 **Performance Analytics**

**Overall Performance:**
• Success rate: {perf_data.get('success_rate', 0):.1%}
• Average response time: {perf_data.get('avg_response_time', 0):.3f}s
• Total operations: {perf_data.get('total_operations', 0)}

**Response Time Distribution:**
• Min: {perf_data.get('min_response_time', 0):.3f}s
• Max: {perf_data.get('max_response_time', 0):.3f}s
"""
                
            else:
                response = f"📊 **Analytics for {analytics_type}**\n\nDetailed analytics data available. Use specific analytics type for focused insights."
            
            # Add recommendations if requested
            if include_recommendations and 'recommendations' in dashboard_data:
                recommendations = dashboard_data['recommendations']
                if recommendations:
                    response += f"\n\n**💡 Recommendations:**\n"
                    for rec in recommendations[:3]:  # Show top 3
                        priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                        response += f"\n{priority_emoji} **{rec['title']}**\n{rec['description']}\n*Action:* {rec['action']}\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Analytics retrieval failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error retrieving analytics:** {str(e)}")]

    async def _track_user_feedback(
        self,
        request_id: str,
        user_id: str,
        query: str,
        clicked_results: List[str] = None,
        dwell_times: Dict[str, float] = None,
        feedback_scores: Dict[str, int] = None,
    ) -> List[TextContent]:
        """Track user feedback for personalization."""
        try:
            optimizer = await self._get_search_optimizer()
            if not optimizer:
                return [TextContent(type="text", text="❌ **Search optimizer not available**")]
            
            clicked_results = clicked_results or []
            dwell_times = dwell_times or {}
            feedback_scores = feedback_scores or {}
            
            # Track feedback
            optimizer.track_user_feedback(
                user_id=user_id,
                query=query,
                clicked_results=clicked_results,
                dwell_times=dwell_times,
                feedback_scores=feedback_scores
            )
            
            # Update document usage analytics
            for doc_id in clicked_results:
                if doc_id in self._document_usage:
                    self._document_usage[doc_id]["access_count"] += 1
                    self._document_usage[doc_id]["last_accessed"] = datetime.now().isoformat()
            
            response = f"""✅ **User Feedback Tracked**

**User:** {user_id}
**Query:** {query}
**Clicked Results:** {len(clicked_results)}
**Dwell Time Data:** {len(dwell_times)} documents
**Feedback Scores:** {len(feedback_scores)} documents

**Summary:**
• Average dwell time: {sum(dwell_times.values()) / len(dwell_times):.2f}s
• Average feedback score: {sum(feedback_scores.values()) / len(feedback_scores):.2f}
• Personalization data updated for future searches
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Feedback tracking failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error tracking feedback:** {str(e)}")]

    async def _create_ab_test(
        self,
        request_id: str,
        experiment_id: str,
        name: str,
        description: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_days: int = 7,
        success_metric: str = "click_through_rate",
    ) -> List[TextContent]:
        """Create A/B test experiment."""
        try:
            optimizer = await self._get_search_optimizer()
            if not optimizer:
                return [TextContent(type="text", text="❌ **Search optimizer not available**")]
            
            # Create A/B test
            optimizer.create_ab_test(
                experiment_id=experiment_id,
                name=name,
                description=description,
                variant_a=variant_a,
                variant_b=variant_b,
                traffic_split=traffic_split,
                duration_days=duration_days,
                success_metric=success_metric
            )
            
            response = f"""🧪 **A/B Test Created Successfully**

**Experiment Details:**
• ID: {experiment_id}
• Name: {name}
• Description: {description}
• Duration: {duration_days} days
• Traffic Split: {traffic_split * 100:.1f}% / {(1 - traffic_split) * 100:.1f}%
• Success Metric: {success_metric}

**Variant A (Control):**
{chr(10).join(f'• {key}: {value}' for key, value in variant_a.items())}

**Variant B (Test):**
{chr(10).join(f'• {key}: {value}' for key, value in variant_b.items())}

**Status:** Active - experiment is now running
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] A/B test creation failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error creating A/B test:** {str(e)}")]

    async def _get_ab_test_results(
        self,
        request_id: str,
        experiment_id: str = None,
        include_details: bool = True,
    ) -> List[TextContent]:
        """Get A/B test results."""
        try:
            optimizer = await self._get_search_optimizer()
            if not optimizer:
                return [TextContent(type="text", text="❌ **Search optimizer not available**")]
            
            results = optimizer.get_ab_test_results()
            
            if "error" in results:
                return [TextContent(type="text", text=f"❌ **A/B test error:** {results['error']}")]
            
            if experiment_id:
                # Get specific experiment results
                if experiment_id in results['experiments']:
                    exp_data = results['experiments'][experiment_id]
                    
                    response = f"""🧪 **A/B Test Results: {experiment_id}**

**Experiment:** {exp_data['name']}
**Status:** {exp_data['status']}
**Duration:** {exp_data['start_date']} to {exp_data['end_date']}
**Has Results:** {'Yes' if exp_data['has_results'] else 'No'}
**Winner:** {exp_data['winner'] or 'No significant difference'}
"""
                else:
                    response = f"❌ **Experiment not found:** {experiment_id}"
            else:
                # Get summary of all experiments
                response = f"""🧪 **A/B Test Summary**

**Overview:**
• Total experiments: {results['total_experiments']}
• Active experiments: {results['active_experiments']}
• Completed experiments: {results['completed_experiments']}

**Experiments:**
"""
                
                for exp_id, exp_data in results['experiments'].items():
                    status_emoji = "🟢" if exp_data['status'] == 'active' else "🔵" if exp_data['status'] == 'completed' else "⚪"
                    winner_text = f" (Winner: {exp_data['winner']})" if exp_data['winner'] else ""
                    response += f"\n{status_emoji} **{exp_id}:** {exp_data['name']}{winner_text}"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] A/B test results retrieval failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error retrieving A/B test results:** {str(e)}")]

    def _validate_startup_dependencies(self) -> None:
        """
        Validate startup configuration and check dependencies.

        Raises:
            ConfigurationError: If configuration validation fails
            ImportError: If required dependencies are missing
        """
        self.logger.info("Validating startup dependencies...")

        # Validate configuration
        try:
            config.validate()
            self.logger.info("Configuration validation passed")
        except ConfigurationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

        # Check required dependencies
        required_modules = [
            ("mcp.server", "Model Context Protocol server"),
            ("mcp.server.stdio", "MCP stdio server"),
            ("mcp.types", "MCP types"),
        ]

        missing_modules = []
        for module_name, description in required_modules:
            try:
                __import__(module_name)
                self.logger.debug(f"✓ {description} available")
            except ImportError:
                missing_modules.append((module_name, description))
                self.logger.error(f"✗ {description} missing")

        if missing_modules:
            error_msg = "Missing required dependencies:\n" + "\n".join(
                f"  - {desc} ({module})" for module, desc in missing_modules
            )
            raise ImportError(error_msg)

        # Validate API keys are present (if web search will be used)
        if not config.TAVILY_API_KEY:
            self.logger.warning(
                "TAVILY_API_KEY not configured - web search will be unavailable"
            )
        else:
            self.logger.info("Tavily API key configured - web search available")

        self.logger.info("All startup dependencies validated successfully")

    async def run(self) -> None:
        """
        Run the RAG MCP Server using stdio transport.

        This method performs startup validation, initializes the server,
        and runs until shutdown is requested.
        """
        try:
            self.logger.info("Starting RAG MCP Server...")

            # Validate startup dependencies
            self._validate_startup_dependencies()

            self.logger.info("RAG MCP Server initialized successfully")
            self.logger.info(f"Environment: {config.ENVIRONMENT}")
            self.logger.info(f"Log level: {config.LOG_LEVEL}")

            # Run the server with stdio transport
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("Server running on stdio transport...")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error in server: {e}", exc_info=True)
            raise
        finally:
            # Clean up resources
            if self._web_search:
                await self._web_search.close()
            self.logger.info("RAG MCP Server shutdown complete")


# Server instance for module-level access
rag_server = RAGMCPServer()


async def main() -> None:
    await rag_server.run()


if __name__ == "__main__":
    asyncio.run(main())
