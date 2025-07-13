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
import signal
import sys
import logging
import time
import uuid
import json
import re
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

# JSON Schema validation
try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

# Local imports
from config import config, ConfigurationError
from vector_store import VectorStoreManager, Document, SearchResult, VectorStoreError
from web_search import WebSearchManager, WebSearchResult, WebSearchError, RateLimitError, QuotaExceededError


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
            max_requests=config.MAX_RETRIES * 20,  # Scale with config
            time_window=60
        )
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'tool_usage': defaultdict(int)
        }
        
        # Cache tool schemas for validation
        self._tool_schemas = {}
        
        # Initialize vector store manager (lazy loading)
        self._vector_store: Optional[VectorStoreManager] = None
        
        # Initialize web search manager (lazy loading)
        self._web_search: Optional[WebSearchManager] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging."""
        logger = logging.getLogger("rag-mcp-server")
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    async def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> None:
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
    
    def _update_metrics(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.metrics['total_requests'] += 1
        self.metrics['tool_usage'][tool_name] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update rolling average response time
        total_requests = self.metrics['total_requests']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + execution_time) / total_requests
        )
    
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
                            "maxLength": 1000
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        },
                        "filter_dict": {
                            "type": "object",
                            "description": "Optional metadata filters for search results",
                            "additionalProperties": True,
                            "default": None
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Whether to include document metadata in results",
                            "default": True
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
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
                            "maxLength": 400
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of web results to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        },
                        "search_depth": {
                            "type": "string",
                            "description": "Depth of search results",
                            "enum": ["basic", "advanced"],
                            "default": "basic"
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": "Whether to include AI-generated answer summary",
                            "default": True
                        },
                        "include_raw_content": {
                            "type": "boolean",
                            "description": "Whether to include raw content from sources",
                            "default": False
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                            },
                            "description": "List of domains to exclude from search results",
                            "maxItems": 10,
                            "default": []
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="smart_search",
                description=(
                    "Intelligent hybrid search that combines local knowledge base and web search. "
                    "First searches the local vector database, then supplements with web search "
                    "if needed. Provides the most comprehensive and relevant results by leveraging "
                    "both local expertise and current web information."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for intelligent hybrid search",
                            "minLength": 1,
                            "maxLength": 1000
                        },
                        "local_max_results": {
                            "type": "integer",
                            "description": "Maximum results to retrieve from local knowledge base",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        },
                        "web_max_results": {
                            "type": "integer",
                            "description": "Maximum results to retrieve from web if local results insufficient",
                            "minimum": 0,
                            "maximum": 10,
                            "default": 3
                        },
                        "local_threshold": {
                            "type": "number",
                            "description": "Minimum local similarity score to consider results sufficient",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7
                        },
                        "min_local_results": {
                            "type": "integer",
                            "description": "Minimum local results needed before triggering web search",
                            "minimum": 0,
                            "maximum": 10,
                            "default": 2
                        },
                        "combine_strategy": {
                            "type": "string",
                            "description": "How to combine local and web results",
                            "enum": ["interleave", "local_first", "relevance_score"],
                            "default": "relevance_score"
                        },
                        "include_sources": {
                            "type": "boolean",
                            "description": "Whether to include source information for all results",
                            "default": True
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            )
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
        self.logger.info(f"[{request_id}] Executing tool: {name} with arguments: {arguments}")
        
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
            self.logger.debug(f"[{request_id}] Starting tool execution with {timeout_seconds}s timeout")
            
            try:
                async with asyncio.timeout(timeout_seconds):
                    # Pattern match on tool name to dispatch to appropriate handler
                    if name == "search_knowledge_base":
                        result = await self._search_knowledge_base(request_id, **arguments)
                    elif name == "web_search":
                        result = await self._search_web(request_id, **arguments)
                    elif name == "smart_search":
                        result = await self._smart_search_internal(request_id, **arguments)
                    else:
                        error_msg = f"Unknown tool: {name}"
                        self.logger.error(f"[{request_id}] {error_msg}")
                        self._update_metrics(name, time.time() - start_time, False)
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"Error: {error_msg}")]
                        )
                
                # Success - log performance metrics
                execution_time = time.time() - start_time
                self.logger.info(
                    f"[{request_id}] Tool '{name}' completed successfully in {execution_time:.3f}s"
                )
                self._update_metrics(name, execution_time, True)
                
                # Convert result to CallToolResult format
                if isinstance(result, list):
                    # Assume it's already List[TextContent]
                    return CallToolResult(content=result)
                elif isinstance(result, str):
                    return CallToolResult(content=[TextContent(type="text", text=result)])
                else:
                    return CallToolResult(content=[TextContent(type="text", text=str(result))])
                    
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
                    collection_name=config.COLLECTION_NAME if hasattr(config, 'COLLECTION_NAME') else "rag_documents",
                    persist_directory=config.VECTOR_STORE_PATH if hasattr(config, 'VECTOR_STORE_PATH') else "./data"
                )
                await self._vector_store.initialize_collection()
                self.logger.info("Vector store initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
                raise VectorStoreError(f"Vector store initialization failed: {e}")
        return self._vector_store
    
    async def _get_web_search(self) -> WebSearchManager:
        """Get or initialize web search manager."""
        if self._web_search is None:
            try:
                if not config.TAVILY_API_KEY:
                    raise WebSearchError("Tavily API key not configured")
                
                self._web_search = WebSearchManager(
                    api_key=config.TAVILY_API_KEY,
                    timeout=getattr(config, 'WEB_SEARCH_TIMEOUT', 30),
                    max_retries=getattr(config, 'MAX_RETRIES', 3),
                    quota_limit=getattr(config, 'TAVILY_QUOTA_LIMIT', None)
                )
                self.logger.info("Web search manager initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize web search manager: {e}")
                raise WebSearchError(f"Web search initialization failed: {e}")
        return self._web_search
    
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
        cleaned_query = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', processed_query)
        
        # Extract keywords (words longer than 2 characters)
        keywords = [word.strip() for word in cleaned_query.split() if len(word.strip()) > 2]
        
        # Create search variations
        variations = [
            query,  # Original query
            processed_query,  # Lowercase
            cleaned_query,  # Cleaned
            ' '.join(keywords)  # Keywords only
        ]
        
        return {
            'original': query,
            'processed': processed_query,
            'cleaned': cleaned_query,
            'keywords': keywords,
            'variations': list(set(variations))  # Remove duplicates
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
        priority_fields = ['source', 'filename', 'file_type', 'created_at', 'modified']
        
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
            for key, value in list(other_fields.items())[:3]:  # Limit to 3 additional fields
                if isinstance(value, (str, int, float, bool)):
                    value_str = str(value)
                    if len(value_str) > 30:
                        value_str = value_str[:27] + "..."
                    formatted_lines.append(f"   • {key}: {value_str}")
        
        return "\n".join(formatted_lines)
    
    def _highlight_content(self, content: str, keywords: List[str], max_length: int = 500) -> str:
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
            return content[:max_length-3] + "..."
        
        # Create case-insensitive pattern for all keywords
        keyword_pattern = '|'.join(re.escape(kw) for kw in keywords if len(kw) > 2)
        
        if not keyword_pattern:
            # No valid keywords, just truncate
            if len(content) <= max_length:
                return content
            return content[:max_length-3] + "..."
        
        try:
            # Find keyword matches
            matches = list(re.finditer(keyword_pattern, content, re.IGNORECASE))
            
            if not matches:
                # No matches found, just truncate
                if len(content) <= max_length:
                    return content
                return content[:max_length-3] + "..."
            
            # If content is short enough, highlight in place
            if len(content) <= max_length:
                highlighted = re.sub(
                    keyword_pattern, 
                    lambda m: f"**{m.group()}**", 
                    content, 
                    flags=re.IGNORECASE
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
                flags=re.IGNORECASE
            )
            
            return highlighted
            
        except Exception as e:
            self.logger.warning(f"Error highlighting content: {e}")
            # Fallback to simple truncation
            if len(content) <= max_length:
                return content
            return content[:max_length-3] + "..."
    
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
                for field in ['timestamp', 'created_at', 'modified', 'processing_time']:
                    if field in result.document.metadata:
                        try:
                            if isinstance(result.document.metadata[field], str):
                                # Try to parse datetime string
                                timestamp = datetime.fromisoformat(result.document.metadata[field].replace('Z', '+00:00')).timestamp()
                            elif isinstance(result.document.metadata[field], (int, float)):
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
        include_metadata: bool = True
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
                return [TextContent(
                    type="text", 
                    text="❌ Error: Query cannot be empty. Please provide a search query."
                )]
            
            if top_k < 1 or top_k > 20:
                return [TextContent(
                    type="text", 
                    text="❌ Error: top_k must be between 1 and 20."
                )]
            
            self.logger.info(f"[{request_id}] Searching knowledge base for: '{query}' (top_k={top_k})")
            
            # Preprocess query
            query_info = self._preprocess_query(query)
            self.logger.debug(f"[{request_id}] Query preprocessing: {query_info['keywords']}")
            
            # Get vector store
            vector_store = await self._get_vector_store()
            
            # Perform similarity search
            search_results = await vector_store.similarity_search_with_score(
                query=query_info['processed'],
                k=top_k,
                filter_dict=filter_dict,
                include_metadata=include_metadata
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
                score_emoji = "🟢" if result.score >= 0.8 else "🟡" if result.score >= 0.6 else "🔴"
                
                # Extract source file path
                source_path = "Unknown source"
                if result.document.metadata and 'source' in result.document.metadata:
                    source_path = str(result.document.metadata['source'])
                    # Make path clickable if it's a valid file path
                    if source_path.startswith('/') or source_path.startswith('.'):
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
                    result.document.page_content, 
                    query_info['keywords'],
                    max_length=500
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
                    result_lines.extend([
                        f"",  # Empty line
                        f"ℹ️ **Metadata:**",
                        metadata_str
                    ])
                
                result_lines.append("\n" + "-" * 50 + "\n")  # Separator
                response_lines.extend(result_lines)
            
            # Add search statistics
            response_lines.extend([
                f"⏱️ **Search completed in {search_time:.3f} seconds**",
                f"🎯 **Keywords used:** {', '.join(query_info['keywords'][:5])}"  # Show first 5 keywords
            ])
            
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
            self.logger.error(f"[{request_id}] Unexpected error during search: {e}", exc_info=True)
            return [TextContent(type="text", text=error_msg)]
    
    async def _search_web(
        self,
        request_id: str,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        exclude_domains: Optional[List[str]] = None
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
                return [TextContent(
                    type="text", 
                    text="❌ Error: Query cannot be empty. Please provide a search query."
                )]
            
            if max_results < 1 or max_results > 20:
                return [TextContent(
                    type="text", 
                    text="❌ Error: max_results must be between 1 and 20."
                )]
            
            self.logger.info(f"[{request_id}] Searching web for: '{query}' (max_results={max_results})")
            
            # Get web search manager
            web_search = await self._get_web_search()
            
            # Perform web search
            search_results, metadata = await web_search.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                exclude_domains=exclude_domains
            )
            
            search_time = time.time() - search_start_time
            
            # Handle errors in metadata
            if 'error' in metadata:
                error_type = metadata.get('error_type', 'WebSearchError')
                error_msg = metadata['error']
                
                if error_type == 'QuotaExceededError':
                    friendly_message = (
                        f"🚫 **Daily API quota exceeded**\n\n"
                        f"The web search service has reached its daily limit. "
                        f"Please try again tomorrow or contact support for higher limits.\n\n"
                        f"⏱️ Search attempted in {search_time:.3f} seconds"
                    )
                elif error_type == 'RateLimitError':
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
                if metadata.get('cache_hit'):
                    friendly_message += "\n📋 Result retrieved from cache"
                
                return [TextContent(type="text", text=friendly_message)]
            
            # Format results
            response_lines = [
                f"🌐 **Found {len(search_results)} web result{'s' if len(search_results) != 1 else ''} for '{query}'**\n"
            ]
            
            for i, result in enumerate(search_results, 1):
                # Format score
                score_str = f"{result.score:.3f}"
                score_emoji = "🟢" if result.score >= 0.8 else "🟡" if result.score >= 0.6 else "🔴"
                
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
                    quality_score = result.metadata.get('quality_score')
                    if quality_score is not None:
                        quality_emoji = "✅" if quality_score >= 0.7 else "⚠️" if quality_score >= 0.5 else "❌"
                        result_lines.append(f"\n{quality_emoji} **Quality Score:** {quality_score:.2f}")
                
                result_lines.append("\n" + "-" * 50 + "\n")  # Separator
                response_lines.extend(result_lines)
            
            # Add search metadata
            response_lines.extend([
                f"⏱️ **Search completed in {search_time:.3f} seconds**"
            ])
            
            # Add cache info
            if metadata.get('cache_hit'):
                response_lines.append("📋 **Result retrieved from cache**")
            else:
                response_lines.append("🔄 **Fresh results from web**")
            
            # Add query optimization info if available
            query_opt = metadata.get('query_optimization')
            if query_opt and query_opt.get('keywords'):
                keywords = query_opt['keywords'][:5]  # Show first 5 keywords
                response_lines.append(f"🎯 **Keywords used:** {', '.join(keywords)}")
            
            # Add quota info if available
            quota_info = metadata.get('quota_info')
            if quota_info and quota_info.get('quota_limit'):
                usage_pct = quota_info.get('usage_percentage', 0)
                response_lines.append(f"📊 **API Usage:** {usage_pct:.1f}% of daily quota")
            
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
            self.logger.error(f"[{request_id}] Unexpected error during web search: {e}", exc_info=True)
            return [TextContent(type="text", text=error_msg)]
    
    async def _smart_search_internal(
        self,
        request_id: str,
        query: str,
        local_max_results: int = 5,
        web_max_results: int = 3,
        local_threshold: float = 0.7,
        min_local_results: int = 2,
        combine_strategy: str = "relevance_score",
        include_sources: bool = True
    ) -> List[TextContent]:
        """
        Perform intelligent hybrid search combining local and web results.
        
        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            local_max_results: Max local results
            web_max_results: Max web results  
            local_threshold: Local similarity threshold
            min_local_results: Min local results before web search
            combine_strategy: How to combine results
            include_sources: Whether to include source info
            
        Returns:
            List of TextContent with combined search results
        """
        self.logger.debug(f"[{request_id}] Performing smart search for: {query}")
        
        # First, search local knowledge base
        local_results = await self._search_knowledge_base(
            request_id=request_id,
            query=query,
            top_k=local_max_results,
            include_metadata=include_sources
        )
        
        # Determine if web search is needed
        # TODO: Implement actual logic to evaluate local results quality
        need_web_search = True  # Placeholder logic
        
        response_text = "Smart Search Results:\n\n"
        response_text += "=== LOCAL KNOWLEDGE BASE ===\n"
        response_text += local_results[0].text + "\n"
        
        if need_web_search and web_max_results > 0:
            web_results = await self._search_web(
                request_id=request_id,
                query=query,
                max_results=web_max_results,
                include_answer=True
            )
            
            response_text += "=== WEB SEARCH SUPPLEMENT ===\n"
            response_text += web_results[0].text + "\n"
        
        response_text += f"=== SEARCH STRATEGY ===\n"
        response_text += f"Combine strategy: {combine_strategy}\n"
        response_text += f"Local threshold: {local_threshold}\n"
        response_text += f"Web search triggered: {need_web_search}\n"
        
        return [TextContent(type="text", text=response_text)]
    
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
            ('mcp.server', 'Model Context Protocol server'),
            ('mcp.server.stdio', 'MCP stdio server'),
            ('mcp.types', 'MCP types'),
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
            self.logger.warning("TAVILY_API_KEY not configured - web search will be unavailable")
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
                    self.server.create_initialization_options()
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
    """Main entry point for the RAG MCP Server."""
    await rag_server.run()


if __name__ == "__main__":
    asyncio.run(main())