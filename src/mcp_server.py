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
from mcp.types import CallToolResult, TextContent, Tool

# JSON Schema validation
try:
    import jsonschema
    from jsonschema import ValidationError, validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

# LlamaIndex availability check
try:
    import llama_index
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Local imports
from config.settings import ConfigurationError, config
from config.tool_loader import ToolConfigLoader
from .web_search import (
    WebSearchError,
    WebSearchManager,
    WebSearchResult,
)
from .llamaindex_processor import RAGEngine


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

        # Load tool configurations
        self.tool_loader = ToolConfigLoader()

        # Tool schema cache for validation
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}

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

        # Initialize web search manager (lazy loading)
        self._web_search: Optional[WebSearchManager] = None

        # Initialize RAG engine (lazy loading)
        self._rag_engine: Optional[RAGEngine] = None

        # Document management storage
        self._documents_metadata: Dict[str, Dict[str, Any]] = {}
        self._document_tags: Dict[str, List[str]] = {}
        self._document_usage: Dict[str, Dict[str, Any]] = {}
        self._document_versions: Dict[str, List[Dict[str, Any]]] = {}

        # Tool dispatch table
        self._tool_handlers = {
            "search_knowledge_base": self._search_knowledge_base,
            "web_search": self._search_web,
            "smart_search": self._smart_search_dispatch,
            "add_document": self._add_document,
        }

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
        self.server.list_tools = self._list_tools
        self.server.call_tool = self._call_tool

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
        return self.tool_loader.get_tool_definitions()
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
                    # Use dictionary dispatch for clean tool handling
                    if name in self._tool_handlers:
                        handler = self._tool_handlers[name]
                        result = await handler(request_id, **arguments)
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


    async def _get_rag_engine(self) -> RAGEngine:
        """Get or initialize RAG engine."""
        if self._rag_engine is None:
            try:
                if not LLAMAINDEX_AVAILABLE:
                    raise ImportError("LlamaIndex is not available")
                
                self._rag_engine = RAGEngine(
                    collection_name="rag_documents",
                    chunk_size=1024,
                    chunk_overlap=200,
                    embedding_model="text-embedding-3-small",
                    llm_model="gpt-3.5-turbo",
                    similarity_top_k=10,
                )
                self.logger.info("RAG engine initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG engine: {e}")
                # Continue without RAG engine
                self._rag_engine = None
        return self._rag_engine

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



    async def _search_knowledge_base(
            self,
            request_id: str,
            query: str,
            top_k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
    ) -> List[TextContent]:
        """
        Search the local vector knowledge base using RAGEngine.

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

            # Get RAG engine
            rag_engine = await self._get_rag_engine()
            
            if not rag_engine:
                return [
                    TextContent(
                        type="text",
                        text="❌ Error: RAG engine not available. Please check LlamaIndex installation.",
                    )
                ]

            # Prepare user preferences for filtering
            user_preferences = None
            if filter_dict:
                user_preferences = {}
                if "tags" in filter_dict:
                    user_preferences["tags"] = filter_dict["tags"]
                # Add other filters as needed
                for key, value in filter_dict.items():
                    if key != "tags":
                        user_preferences[key] = value

            # Execute query using RAGEngine
            response = await rag_engine.query(
                query_text=query,
                user_preferences=user_preferences
            )

            search_time = time.time() - search_start_time

            # Handle empty or error response
            if not response or not response.response:
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

            # Format response with source citations
            response_lines = [
                f"🤖 **RAG Response for '{query}'**\n",
                f"📝 **Answer:**",
                response.response,
                f"",  # Empty line
            ]

            # Add source citations
            if response.source_nodes:
                response_lines.extend([
                    f"📚 **Sources ({len(response.source_nodes)} documents):**",
                    f""
                ])
                
                for i, node in enumerate(response.source_nodes[:top_k], 1):
                    # Format similarity score
                    score = getattr(node, 'score', 0.0)
                    score_str = f"{score:.3f}"
                    score_emoji = (
                        "🟢" if score >= 0.8
                        else "🟡" if score >= 0.6 
                        else "🔴"
                    )

                    # Extract source information
                    source_path = "Unknown source"
                    metadata = getattr(node.node, 'metadata', {})
                    if metadata and "source" in metadata:
                        source_path = str(metadata["source"])
                        try:
                            path_obj = Path(source_path)
                            source_path = f"📄 {path_obj.name}"
                        except Exception:
                            source_path = f"📄 {source_path}"

                    # Get content preview
                    content = getattr(node.node, 'text', '')
                    content_preview = content[:200] + "..." if len(content) > 200 else content

                    # Build source entry
                    source_lines = [
                        f"**{i}. {score_emoji} Relevance: {score_str}**",
                        f"📂 **Source:** {source_path}",
                        f"📖 **Content:** {content_preview}",
                    ]

                    # Add metadata if requested and available
                    if include_metadata and metadata:
                        filtered_metadata = {k: v for k, v in metadata.items() 
                                           if k not in ['source', 'text']}
                        if filtered_metadata:
                            metadata_str = ", ".join([f"{k}: {v}" for k, v in filtered_metadata.items()])
                            source_lines.append(f"ℹ️ **Metadata:** {metadata_str}")

                    source_lines.append("")  # Empty line
                    response_lines.extend(source_lines)

            # Add search statistics
            response_lines.extend([
                f"⏱️ **Search completed in {search_time:.3f} seconds**",
                f"🔍 **Query processed with hybrid retrieval (Vector + BM25)**",
            ])

            response_text = "\n".join(response_lines)

            self.logger.info(
                f"[{request_id}] RAG search completed: {len(response.source_nodes) if response.source_nodes else 0} sources in {search_time:.3f}s"
            )

            return [TextContent(type="text", text=response_text)]

        except Exception as e:
            error_msg = f"❌ **RAG search failed:** {str(e)}"
            self.logger.error(
                f"[{request_id}] Unexpected error during RAG search: {e}", exc_info=True
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

    async def _smart_search_dispatch(self, request_id: str, **arguments) -> List[TextContent]:
        """Dispatch method for smart_search tool - maps arguments and calls internal method."""
        smart_search_args = {
            "query": arguments["query"],
            "similarity_threshold": arguments.get("similarity_threshold", 0.75),
            "local_top_k": arguments.get("local_top_k", 5),
            "web_max_results": arguments.get("web_max_results", 5),
            "include_sources": arguments.get("include_sources", True),
            "combine_strategy": arguments.get("combine_strategy", "relevance_score"),
            "min_local_results": arguments.get("min_local_results", 2),
        }
        return await self._smart_search_internal(request_id, **smart_search_args)

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
        Simplified smart search using RAGEngine with web search fallback.

        This method implements an intelligent search strategy:
        1. Uses RAGEngine for local knowledge base search with hybrid retrieval
        2. Evaluates result quality against similarity threshold
        3. Falls back to web search if local results are insufficient
        4. Combines results with clear source attribution

        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            similarity_threshold: Threshold for triggering web search (0.0-1.0)
            local_top_k: Maximum local results to retrieve
            web_max_results: Maximum web results to retrieve
            include_sources: Whether to include detailed source information
            combine_strategy: Strategy for combining results (ignored for now)
            min_local_results: Minimum local results before considering web search

        Returns:
            List of TextContent with smart search results
        """
        start_time = time.time()
        self.logger.info(f"[{request_id}] Starting smart search with RAGEngine")

        try:
            # Step 1: Search using RAGEngine
            rag_engine = await self._get_rag_engine()
            
            local_response = None
            local_confidence = 0.0
            
            if rag_engine:
                self.logger.debug(f"[{request_id}] Using RAGEngine for local search")
                
                # Execute query with RAGEngine
                rag_response = await rag_engine.query(
                    query_text=query,
                    user_preferences=None
                )
                
                if rag_response and rag_response.response and rag_response.source_nodes:
                    local_response = rag_response
                    # Calculate confidence based on source node scores
                    if rag_response.source_nodes:
                        scores = [getattr(node, 'score', 0.0) for node in rag_response.source_nodes]
                        local_confidence = max(scores) if scores else 0.0
                    
                    self.logger.info(
                        f"[{request_id}] RAG search: {len(rag_response.source_nodes)} sources, "
                        f"confidence={local_confidence:.3f}"
                    )

            web_search_triggered = False
            decision_reason = ""
            confidence_level = "HIGH"
            web_response = None
            
            # Decide if web search is needed based on local results
            if not local_response or not local_response.source_nodes:
                # No local results - perform web search
                web_search_triggered = True
                decision_reason = "No local knowledge found - searching web for comprehensive results"
                confidence_level = "MEDIUM"
            elif local_confidence < similarity_threshold:
                # Low confidence local results - supplement with web search
                web_search_triggered = True
                decision_reason = (
                    f"Local results confidence below threshold ({local_confidence:.3f} < {similarity_threshold}) "
                    f"- enhancing with web search"
                )
                confidence_level = "MEDIUM"
            elif len(local_response.source_nodes) < min_local_results:
                # Insufficient local source diversity
                web_search_triggered = True
                decision_reason = (
                    f"Insufficient local sources ({len(local_response.source_nodes)} < {min_local_results}) "
                    f"- supplementing with web search"
                )
                confidence_level = "MEDIUM"
            else:
                # High quality local results sufficient
                decision_reason = (
                    f"Local knowledge sufficient (confidence={local_confidence:.3f} >= {similarity_threshold})"
                )
                confidence_level = "HIGH"

            # Step 3: Web search if triggered
            web_search_time = 0.0
            
            if web_search_triggered and web_max_results > 0:
                self.logger.debug(f"[{request_id}] Performing web search as fallback")
                web_search_start = time.time()

                try:
                    # Use the existing web search functionality
                    web_results = await self._search_web(
                        request_id=request_id,
                        query=query,
                        max_results=web_max_results,
                        include_answer=True,
                    )
                    web_search_time = time.time() - web_search_start
                    
                    # Extract the web search response text
                    if web_results and web_results[0].text:
                        web_response = web_results[0].text
                    
                    self.logger.info(f"[{request_id}] Web search completed in {web_search_time:.3f}s")

                except Exception as e:
                    self.logger.warning(f"[{request_id}] Web search failed: {e}")
                    web_response = None
                    confidence_level = "LOW" if not local_response else "MEDIUM"

            # Step 4: Format combined response
            total_time = time.time() - start_time
            
            response_lines = [
                f"🧠 **Smart Search Results for '{query}'**\n",
                f"🎯 **Search Strategy:** RAGEngine + Web Fallback",
                f"📊 **Confidence Level:** {confidence_level}",
                f"🔍 **Decision:** {decision_reason}",
                f"⏱️ **Total Time:** {total_time:.3f}s\n",
            ]

            # Add local RAG results
            if local_response and local_response.response:
                response_lines.extend([
                    "## 🏠 Local Knowledge (RAG)",
                    f"**Answer:** {local_response.response}",
                    f"**Confidence:** {local_confidence:.3f}",
                    f"**Sources:** {len(local_response.source_nodes)} documents\n",
                ])
                
                # Add source details if requested
                if include_sources and local_response.source_nodes:
                    response_lines.append("**Source Details:**")
                    for i, node in enumerate(local_response.source_nodes[:3], 1):  # Show top 3
                        score = getattr(node, 'score', 0.0)
                        score_emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                        
                        metadata = getattr(node.node, 'metadata', {})
                        source_path = metadata.get('source', 'Unknown')
                        if source_path != 'Unknown':
                            try:
                                source_path = f"📄 {Path(source_path).name}"
                            except Exception:
                                source_path = f"📄 {source_path}"
                        
                        content_preview = getattr(node.node, 'text', '')[:150]
                        response_lines.append(
                            f"{i}. {score_emoji} **{source_path}** (score: {score:.3f})\n"
                            f"   {content_preview}{'...' if len(content_preview) >= 150 else ''}"
                        )
                    response_lines.append("")

            # Add web search results if performed
            if web_search_triggered:
                if web_response:
                    response_lines.extend([
                        "## 🌐 Web Search Supplement",
                        f"**Search Time:** {web_search_time:.3f}s",
                        web_response,
                    ])
                else:
                    response_lines.extend([
                        "## 🌐 Web Search",
                        "❌ Web search was attempted but failed to return results",
                    ])

            # Add footer
            response_lines.extend([
                "\n---",
                f"🔧 **Search powered by:** RAGEngine (Hybrid Vector + BM25) + Web Fallback",
                f"📈 **Result quality:** {confidence_level}",
            ])

            final_response = "\n".join(response_lines)

            self.logger.info(
                f"[{request_id}] Smart search completed: "
                f"local_sources={len(local_response.source_nodes) if local_response and local_response.source_nodes else 0}, "
                f"web_triggered={web_search_triggered}, confidence={confidence_level}, time={total_time:.3f}s"
            )

            return [TextContent(type="text", text=final_response)]

        except Exception as e:
            error_msg = f"❌ **Smart search failed:** {str(e)}\n\nPlease try again or contact support if the issue persists."
            self.logger.error(f"[{request_id}] Smart search error: {e}", exc_info=True)
            return [TextContent(type="text", text=error_msg)]












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
        """Add document(s) to the knowledge base using RAGEngine."""
        try:
            # Get RAG engine
            rag_engine = await self._get_rag_engine()
            if not rag_engine:
                return [TextContent(
                    type="text",
                    text="❌ Error: RAG engine not available. Please check LlamaIndex installation."
                )]
            
            metadata = metadata or {}
            tags = tags or []
            batch_files = batch_files or []
            
            # Add processing metadata
            metadata.update({
                "created_at": datetime.now().isoformat(),
                "tags": tags,
                "processor": "rag_engine_direct"
            })
            
            # Track processing results
            results = []
            total_docs = 0
            processing_time = time.time()
            
            if batch_files:
                # Batch file processing using SimpleDirectoryReader
                from llama_index.core import SimpleDirectoryReader
                
                for file_path in batch_files:
                    try:
                        # Use LlamaIndex SimpleDirectoryReader for each file
                        documents = SimpleDirectoryReader(
                            input_files=[file_path],
                            exclude_hidden=True,
                        ).load_data()
                        
                        # Add metadata to each document
                        for doc in documents:
                            doc_id = str(uuid.uuid4())
                            doc.metadata.update(metadata)
                            doc.metadata["document_id"] = doc_id
                            doc.metadata["source"] = file_path
                        
                        # Add documents to RAGEngine
                        success = await rag_engine.add_documents(documents)
                        
                        if success:
                            # Track document metadata
                            for doc in documents:
                                doc_id = doc.metadata["document_id"]
                                self._documents_metadata[doc_id] = doc.metadata
                                self._document_tags[doc_id] = tags
                                self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                            
                            total_docs += len(documents)
                            results.append(f"✅ {Path(file_path).name}: {len(documents)} documents")
                        else:
                            results.append(f"❌ {Path(file_path).name}: Failed to add to RAG engine")
                            
                    except Exception as e:
                        results.append(f"❌ {Path(file_path).name}: {str(e)}")
                        
            elif file_path:
                # Single file processing using SimpleDirectoryReader
                from llama_index.core import SimpleDirectoryReader
                
                try:
                    documents = SimpleDirectoryReader(
                        input_files=[file_path],
                        exclude_hidden=True,
                    ).load_data()
                    
                    # Add metadata to each document
                    for doc in documents:
                        doc_id = str(uuid.uuid4())
                        doc.metadata.update(metadata)
                        doc.metadata["document_id"] = doc_id
                        doc.metadata["source"] = file_path
                    
                    # Add documents to RAGEngine
                    success = await rag_engine.add_documents(documents)
                    
                    if success:
                        # Track document metadata
                        for doc in documents:
                            doc_id = doc.metadata["document_id"]
                            self._documents_metadata[doc_id] = doc.metadata
                            self._document_tags[doc_id] = tags
                            self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                        
                        total_docs = len(documents)
                        results.append(f"✅ {Path(file_path).name}: {len(documents)} documents")
                    else:
                        results.append(f"❌ {Path(file_path).name}: Failed to add to RAG engine")
                        
                except Exception as e:
                    results.append(f"❌ {Path(file_path).name}: {str(e)}")
                    
            elif content:
                # Raw content processing using LlamaIndex Document
                from llama_index.core import Document as LlamaDocument
                
                try:
                    doc_id = str(uuid.uuid4())
                    enhanced_metadata = {
                        **metadata,
                        "document_id": doc_id,
                        "source": "raw_content",
                        "content_length": len(content)
                    }
                    
                    # Create LlamaIndex document
                    llama_doc = LlamaDocument(
                        text=content,
                        metadata=enhanced_metadata,
                        doc_id=doc_id
                    )
                    
                    # Add document to RAGEngine
                    success = await rag_engine.add_documents([llama_doc])
                    
                    if success:
                        # Track document metadata
                        self._documents_metadata[doc_id] = enhanced_metadata
                        self._document_tags[doc_id] = tags
                        self._document_usage[doc_id] = {"access_count": 0, "last_accessed": None}
                        
                        total_docs = 1
                        results.append(f"✅ Content added: {doc_id}")
                    else:
                        results.append(f"❌ Content: Failed to add to RAG engine")
                        
                except Exception as e:
                    results.append(f"❌ Content: {str(e)}")
                
            processing_time = time.time() - processing_time
            
            # Format response
            response = f"""📄 **Document Addition Complete**

**Processing Summary:**
• Documents processed: {len(results)}
• Total documents created: {total_docs}
• Processing time: {processing_time:.2f}s

**Results:**
{chr(10).join(results)}

**Statistics:**
• Success rate: {len([r for r in results if r.startswith('✅')])/len(results)*100:.1f}% if results else 100.0%
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
                rag_engine = await self._get_rag_engine()
                # Note: RAGEngine handles document updates internally
                # For now, we'll track the content change in metadata
                updated_metadata = self._documents_metadata[document_id].copy()
                updated_metadata["modified_at"] = datetime.now().isoformat()
                updated_metadata["content_updated"] = True
            
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
                # Remove from RAG engine and all tracking
                if cleanup_chunks:
                    rag_engine = await self._get_rag_engine()
                    # Note: RAGEngine handles document deletion internally
                    # For now, we'll just mark as deleted in metadata
                
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
        """[DEPRECATED] Redirects to search_knowledge_base with RAGEngine."""
        
        # Add deprecation warning
        deprecation_warning = [
            TextContent(
                type="text",
                text=(
                    "⚠️ **DEPRECATED:** The optimize_search tool is deprecated. "
                    "The new search_knowledge_base tool with RAGEngine automatically provides "
                    "hybrid ranking, query optimization, and personalization.\n\n"
                    "Redirecting your query to search_knowledge_base...\n"
                )
            )
        ]
        
        # Prepare filter_dict for personalization
        filter_dict = None
        if enable_personalization and user_id != "anonymous":
            # You could add user-specific tags here if available
            filter_dict = {"user_id": user_id}
        
        # Call the new search_knowledge_base method
        search_results = await self._search_knowledge_base(
            request_id=request_id,
            query=query,
            top_k=max_results,
            filter_dict=filter_dict,
            include_metadata=True,
        )
        
        # Combine deprecation warning with search results
        return deprecation_warning + search_results

    async def _get_search_analytics(
        self,
        request_id: str,
        analytics_type: str = "overview",
        time_period: str = "week",
        include_recommendations: bool = True,
    ) -> List[TextContent]:
        """[DEPRECATED] Search analytics functionality."""
        return [TextContent(
            type="text", 
            text=(
                "⚠️ **DEPRECATED:** Search analytics functionality has been deprecated. "
                "The SearchOptimizer has been replaced with RAGEngine, which provides "
                "built-in search optimization without requiring separate analytics tracking. "
                "Use the enhanced search_knowledge_base tool instead."
            )
        )]

    async def _track_user_feedback(
        self,
        request_id: str,
        user_id: str,
        query: str,
        clicked_results: Optional[List[str]] = None,
        dwell_times: Optional[Dict[str, float]] = None,
        feedback_scores: Optional[Dict[str, float]] = None,
    ) -> List[TextContent]:
        """[DEPRECATED] User feedback tracking functionality."""
        return [TextContent(
            type="text", 
            text=(
                "⚠️ **DEPRECATED:** User feedback tracking has been deprecated. "
                "The SearchOptimizer has been replaced with RAGEngine, which provides "
                "built-in search optimization without requiring manual feedback tracking. "
                "Use the enhanced search_knowledge_base tool instead."
            )
        )]

    async def _create_ab_test(
        self,
        request_id: str,
        experiment_id: str,
        name: str,
        control_strategy: str = "hybrid_balanced",
        test_strategy: str = "personalized",
        traffic_split: float = 0.5,
        duration_days: int = 7,
        success_metrics: Optional[List[str]] = None,
    ) -> List[TextContent]:
        """[DEPRECATED] A/B testing functionality."""
        return [TextContent(
            type="text", 
            text=(
                "⚠️ **DEPRECATED:** A/B testing functionality has been deprecated. "
                "The SearchOptimizer has been replaced with RAGEngine, which provides "
                "optimized search results without requiring A/B testing configuration. "
                "Use the enhanced search_knowledge_base tool instead."
            )
        )]

    async def _get_ab_test_results(
        self,
        request_id: str,
        experiment_id: str = None,
        include_details: bool = True,
    ) -> List[TextContent]:
        """[DEPRECATED] A/B test results functionality."""
        return [TextContent(
            type="text", 
            text=(
                "⚠️ **DEPRECATED:** A/B test results functionality has been deprecated. "
                "The SearchOptimizer has been replaced with RAGEngine, which provides "
                "optimized search results without requiring A/B testing. "
                "Use the enhanced search_knowledge_base tool instead."
            )
        )]

    def _validate_startup_dependencies(self) -> None:
        """
        Validate startup configuration and check dependencies.

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self.logger.info("Validating startup dependencies...")

        # Check vector store path
        if not config.VECTOR_STORE_PATH:
            raise ConfigurationError("VECTOR_STORE_PATH is required")

        # Create vector store directory if it doesn't exist
        vector_store_path = Path(config.VECTOR_STORE_PATH)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Vector store path: {vector_store_path}")

        # Check OpenAI API key (optional for basic functionality)
        if not config.OPENAI_API_KEY:
            self.logger.warning(
                "OpenAI API key not configured - some features may be limited"
            )
        else:
            self.logger.info("OpenAI API key configured - enhanced features available")

        # Check Tavily API key (optional for web search)
        if not config.TAVILY_API_KEY:
            self.logger.warning("Tavily API key not configured - web search disabled")
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

            # Create stdio transport and server
            try:
                from mcp.server.stdio import stdio_server
            except ImportError:
                # Fallback for different MCP versions
                from mcp.server import serve_stdio as stdio_server

            # Initialize server
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("Server started successfully, listening for MCP requests...")
                
                # Run the server until shutdown
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal, stopping server...")
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
