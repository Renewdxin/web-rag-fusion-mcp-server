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
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Production-ready libraries (must be available)
from aiolimiter import AsyncLimiter
# JSON Schema validation
from jsonschema import ValidationError, validate
from loguru import logger
# MCP imports
from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from tenacity import retry, stop_after_attempt, wait_exponential

# LlamaIndex availability check
try:
    import llama_index
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Local imports
from config.settings import ConfigurationError, config
from config.tool_loader import ToolConfigLoader
from src.web_search import (
    WebSearchError,
    WebSearchManager,
)
from src.llamaindex_processor import RAGEngine


class RAGMCPServer:
    """
    RAG (Retrieval-Augmented Generation) MCP Server.

    Provides intelligent search capabilities through local vector database
    and web search integration via Tavily API.
    """

    def __init__(self) -> None:
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
        if config.ENABLE_RATE_LIMITING:
            self.rate_limiter = AsyncLimiter(
                max_rate=config.RATE_LIMIT_REQUESTS, 
                time_period=config.RATE_LIMIT_WINDOW
            )
        else:
            # No rate limiting
            self.rate_limiter = None

        # Performance metrics with Prometheus (if enabled)
        if config.ENABLE_PROMETHEUS_METRICS:
            self.registry = CollectorRegistry()
            self.request_counter = Counter(
                'mcp_requests_total', 
                'Total number of MCP requests',
                ['tool_name', 'status'],
                registry=self.registry
            )
            self.request_duration = Histogram(
                'mcp_request_duration_seconds',
                'Duration of MCP requests',
                ['tool_name'],
                registry=self.registry
            )
            self.active_requests = Gauge(
                'mcp_active_requests',
                'Number of active MCP requests',
                registry=self.registry
            )
        else:
            # Fallback to simple metrics
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "tool_usage": {},
            }

        # Initialize web search manager (lazy loading)
        self._web_search: Optional[WebSearchManager] = None

        # Initialize RAG engine (lazy loading)
        self._rag_engine: Optional[RAGEngine] = None

        # Basic document tracking (simplified)
        self._document_count = 0

        # Tool dispatch table
        self._tool_handlers = {
            "search_knowledge_base": self._search_knowledge_base,
            "web_search": self._search_web,
            "smart_search": self._smart_search_dispatch,
            "add_document": self._add_document,
        }

    def _setup_logging(self):
        """Configure logging using loguru or fallback to standard logging."""
        if config.USE_LOGURU:
            # Configure loguru
            logger.remove()  # Remove default handler
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
                level=config.LOG_LEVEL,
                colorize=True
            )
            return logger
        else:
            # Fallback to standard logging
            import logging
            std_logger = logging.getLogger("rag-mcp-server")
            std_logger.setLevel(getattr(logging, config.LOG_LEVEL))

            # Console handler
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            std_logger.addHandler(handler)

            return std_logger

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
        # Register proper MCP protocol handlers
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return await self._list_tools()
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]):
            result = await self._call_tool(name, arguments)
            # Return the content directly, not the CallToolResult wrapper
            if isinstance(result, CallToolResult):
                self.logger.debug(f"Returning CallToolResult.content: {len(result.content)} items")
                return result.content
            else:
                # Fallback - should not happen with current implementation
                self.logger.warning(f"Unexpected result type: {type(result)}")
                return [TextContent(type="text", text=str(result))]
        
        # Register resource and prompt handlers (empty for now)
        @self.server.list_resources()
        async def handle_list_resources():
            return []
            
        @self.server.list_prompts()
        async def handle_list_prompts():
            return []

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracing."""
        import uuid
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
        """Update performance metrics using Prometheus or fallback."""
        if config.ENABLE_PROMETHEUS_METRICS:
            # Update Prometheus metrics
            status = "success" if success else "error"
            self.request_counter.labels(tool_name=tool_name, status=status).inc()
            self.request_duration.labels(tool_name=tool_name).observe(execution_time)
        else:
            # Fallback to simple metrics
            self.metrics["total_requests"] += 1
            if tool_name not in self.metrics["tool_usage"]:
                self.metrics["tool_usage"][tool_name] = 0
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
            if self.rate_limiter:
                async with self.rate_limiter:
                    pass  # This will handle rate limiting automatically
            
            # If we got here, we're within rate limits or no rate limiting

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
                self.logger.debug(f"[{request_id}] Result type: {type(result)}")
                
                if isinstance(result, list):
                    # Check if it's already List[TextContent]
                    if result and hasattr(result[0], 'type') and result[0].type == 'text':
                        self.logger.debug(f"[{request_id}] Result is List[TextContent]")
                        call_result = CallToolResult(content=result)
                        self.logger.debug(f"[{request_id}] Created CallToolResult successfully")
                        return call_result
                    else:
                        # Convert to TextContent format
                        self.logger.debug(f"[{request_id}] Converting list to TextContent format")
                        content = []
                        for item in result:
                            if hasattr(item, 'text'):
                                content.append(TextContent(type="text", text=str(item.text)))
                            else:
                                content.append(TextContent(type="text", text=str(item)))
                        call_result = CallToolResult(content=content)
                        return call_result
                elif isinstance(result, str):
                    call_result = CallToolResult(
                        content=[TextContent(type="text", text=result)]
                    )
                    return call_result
                elif hasattr(result, 'text'):
                    # Handle single TextContent object
                    call_result = CallToolResult(
                        content=[TextContent(type="text", text=str(result.text))]
                    )
                    return call_result
                else:
                    call_result = CallToolResult(
                        content=[TextContent(type="text", text=str(result))]
                    )
                    return call_result

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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    ) if config.ENABLE_RETRY_LOGIC else lambda f: f
    async def _get_web_search(self) -> WebSearchManager:
        """Get or initialize web search manager with retry logic."""
        if self._web_search is None:
            try:
                if not config.SEARCH_API_KEY:
                    raise WebSearchError("Search API key not configured")

                self._web_search = WebSearchManager(
                    api_key=config.SEARCH_API_KEY,
                    timeout=getattr(config, "WEB_SEARCH_TIMEOUT", 30),
                    max_retries=getattr(config, "MAX_RETRIES", 3),
                    quota_limit=getattr(config, "TAVILY_QUOTA_LIMIT", None),
                )
                if config.USE_LOGURU:
                    logger.info("Web search manager initialized successfully")
                else:
                    self.logger.info("Web search manager initialized successfully")
            except Exception as e:
                if config.USE_LOGURU:
                    logger.error(f"Failed to initialize web search manager: {e}")
                else:
                    self.logger.error(f"Failed to initialize web search manager: {e}")
                raise WebSearchError(f"Web search initialization failed: {e}")
        return self._web_search


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    ) if config.ENABLE_RETRY_LOGIC else lambda f: f
    async def _get_rag_engine(self) -> RAGEngine:
        """Get or initialize RAG engine with retry logic."""
        if self._rag_engine is None:
            try:
                if not LLAMAINDEX_AVAILABLE:
                    raise ImportError("LlamaIndex is not available")
                
                self._rag_engine = RAGEngine(
                    collection_name=config.COLLECTION_NAME,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                    embedding_model=config.EMBEDDING_MODEL,
                    llm_model=config.LLM_MODEL,
                    similarity_top_k=config.SIMILARITY_TOP_K,
                )
                if config.USE_LOGURU:
                    logger.info(f"RAG engine initialized successfully with LLM: {config.LLM_MODEL}, Embedding: {config.EMBEDDING_MODEL}")
                else:
                    self.logger.info(f"RAG engine initialized successfully with LLM: {config.LLM_MODEL}, Embedding: {config.EMBEDDING_MODEL}")
            except Exception as e:
                if config.USE_LOGURU:
                    logger.error(f"Failed to initialize RAG engine: {e}")
                else:
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
                        except Exception as e:
                            self.logger.debug(f"Error parsing source path: {e}")
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

            source_count = len(response.source_nodes) if response.source_nodes else 0
            self.logger.info(
                f"[{request_id}] RAG search completed: {source_count} sources in {search_time:.3f}s"
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
                            except Exception as e:
                                self.logger.debug(f"Error parsing source path: {e}")
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

            local_source_count = (
                len(local_response.source_nodes) 
                if local_response and local_response.source_nodes 
                else 0
            )
            self.logger.info(
                f"[{request_id}] Smart search completed: "
                f"local_sources={local_source_count}, "
                f"web_triggered={web_search_triggered}, confidence={confidence_level}, time={total_time:.3f}s"
            )

            return [TextContent(type="text", text=final_response)]

        except Exception as e:
            error_msg = (
                f"❌ **Smart search failed:** {str(e)}\n\n"
                f"Please try again or contact support if the issue persists."
            )
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
                "tags": ",".join(tags) if tags else "",  # Convert list to comma-separated string
                "processor": "rag_engine_direct"
            })
            
            # Track processing results
            results = []
            total_docs = 0
            processing_time = time.time()
            
            if batch_files:
                # Batch file processing
                from llama_index.core import SimpleDirectoryReader
                
                for file_path in batch_files:
                    try:
                        documents = SimpleDirectoryReader(
                            input_files=[file_path],
                            exclude_hidden=True,
                        ).load_data()
                        
                        # Add metadata to each document
                        for doc in documents:
                            doc.metadata.update(metadata)
                            doc.metadata["source"] = file_path
                        
                        # Add documents to RAGEngine
                        success = await rag_engine.add_documents(documents)
                        
                        if success:
                            total_docs += len(documents)
                            self._document_count += len(documents)
                            results.append(f"✅ {Path(file_path).name}: {len(documents)} documents")
                        else:
                            results.append(f"❌ {Path(file_path).name}: Failed to add to RAG engine")
                            
                    except Exception as e:
                        results.append(f"❌ {Path(file_path).name}: {str(e)}")
                        
            elif file_path:
                # Single file processing
                from llama_index.core import SimpleDirectoryReader
                
                try:
                    documents = SimpleDirectoryReader(
                        input_files=[file_path],
                        exclude_hidden=True,
                    ).load_data()
                    
                    # Add metadata to each document
                    for doc in documents:
                        doc.metadata.update(metadata)
                        doc.metadata["source"] = file_path
                    
                    # Add documents to RAGEngine
                    success = await rag_engine.add_documents(documents)
                    
                    if success:
                        total_docs = len(documents)
                        self._document_count += len(documents)
                        results.append(f"✅ {Path(file_path).name}: {len(documents)} documents")
                    else:
                        results.append(f"❌ {Path(file_path).name}: Failed to add to RAG engine")
                        
                except Exception as e:
                    results.append(f"❌ {Path(file_path).name}: {str(e)}")
                    
            elif content:
                # Raw content processing
                from llama_index.core import Document as LlamaDocument
                
                try:
                    enhanced_metadata = {
                        **metadata,
                        "source": "raw_content",
                        "content_length": len(content)
                    }
                    
                    # Create LlamaIndex document
                    llama_doc = LlamaDocument(
                        text=content,
                        metadata=enhanced_metadata,
                    )
                    
                    # Add document to RAGEngine
                    success = await rag_engine.add_documents([llama_doc])
                    
                    if success:
                        total_docs = 1
                        self._document_count += 1
                        results.append(f"✅ Content added successfully")
                    else:
                        results.append(f"❌ Content: Failed to add to RAG engine")
                        
                except Exception as e:
                    results.append(f"❌ Content: {str(e)}")
                
            processing_time = time.time() - processing_time
            
            # Format response
            success_count = len([r for r in results if r.startswith('✅')])
            success_rate = (success_count / len(results) * 100) if results else 100.0
            
            response = f"""📄 **Document Addition Complete**

**Processing Summary:**
• Documents processed: {len(results)}
• Total documents created: {total_docs}
• Processing time: {processing_time:.2f}s
• Total documents in knowledge base: {self._document_count}

**Results:**
{chr(10).join(results)}

**Statistics:**
• Success rate: {success_rate:.1f}%
• Failed: {len([r for r in results if r.startswith('❌')])}
"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Document addition failed: {e}")
            return [TextContent(type="text", text=f"❌ **Error adding document:** {str(e)}")]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics in a standardized format."""
        if config.ENABLE_PROMETHEUS_METRICS:
            # Return Prometheus metrics in text format
            return {
                "prometheus_metrics": generate_latest(self.registry).decode('utf-8'),
                "format": "prometheus"
            }
        else:
            # Return simple metrics
            return {
                "metrics": self.metrics,
                "format": "simple"
            }

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
        if not config.SEARCH_API_KEY:
            self.logger.warning("Search API key not configured - web search disabled")
        else:
            self.logger.info("Search API key configured - web search available")

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
            from mcp.server.stdio import stdio_server

            # Initialize server
            async with stdio_server() as (read_stream, write_stream):
                if config.USE_LOGURU:
                    logger.info("Server started successfully, listening for MCP requests...")
                else:
                    self.logger.info("Server started successfully, listening for MCP requests...")
                
                # Run the server until shutdown
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        except KeyboardInterrupt:
            if config.USE_LOGURU:
                logger.info("Received shutdown signal, stopping server...")
            else:
                self.logger.info("Received shutdown signal, stopping server...")
        except Exception as e:
            if config.USE_LOGURU:
                logger.error(f"Fatal error in server: {e}")
            else:
                self.logger.error(f"Fatal error in server: {e}", exc_info=True)
            raise
        finally:
            # Clean up resources
            if self._web_search:
                await self._web_search.close()
            if config.USE_LOGURU:
                logger.info("RAG MCP Server shutdown complete")
            else:
                self.logger.info("RAG MCP Server shutdown complete")


# Server instance for module-level access
rag_server = RAGMCPServer()


async def main() -> None:
    await rag_server.run()


if __name__ == "__main__":
    asyncio.run(main())
