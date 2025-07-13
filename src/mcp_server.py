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
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from datetime import datetime, timedelta

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
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score for results (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": None  # Will use config.SIMILARITY_THRESHOLD
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
                        result = await self._search_knowledge_base_internal(request_id, **arguments)
                    elif name == "web_search":
                        result = await self._search_web_internal(request_id, **arguments)
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
    
    async def _search_knowledge_base_internal(
        self,
        request_id: str,
        query: str,
        max_results: int = 10,
        similarity_threshold: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[TextContent]:
        """
        Search the local vector knowledge base.
        
        Args:
            request_id: Unique request identifier for logging
            query: Search query string
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            include_metadata: Whether to include document metadata
            
        Returns:
            List of TextContent with search results
        """
        self.logger.debug(f"[{request_id}] Searching knowledge base for: {query}")
        
        # Use config threshold if not provided
        threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        # TODO: Implement actual vector database search
        # This is a placeholder implementation
        results = [
            {
                "content": f"Knowledge base result for '{query}' (placeholder)",
                "similarity": 0.85,
                "metadata": {"source": "local_db", "doc_id": "doc_1"} if include_metadata else None
            }
        ]
        
        response_text = f"Found {len(results)} results in knowledge base:\n\n"
        for i, result in enumerate(results[:max_results], 1):
            response_text += f"{i}. Content: {result['content']}\n"
            response_text += f"   Similarity: {result['similarity']:.3f}\n"
            if result['metadata']:
                response_text += f"   Metadata: {result['metadata']}\n"
            response_text += "\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _search_web_internal(
        self,
        request_id: str,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        exclude_domains: List[str] = None
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
            List of TextContent with web search results
        """
        self.logger.debug(f"[{request_id}] Searching web for: {query}")
        exclude_domains = exclude_domains or []
        
        # TODO: Implement actual Tavily API integration
        # This is a placeholder implementation
        results = [
            {
                "title": f"Web result for '{query}' (placeholder)",
                "url": "https://example.com",
                "content": f"Web content about {query}",
                "score": 0.9
            }
        ]
        
        response_text = f"Found {len(results)} web results:\n\n"
        for i, result in enumerate(results[:max_results], 1):
            response_text += f"{i}. {result['title']}\n"
            response_text += f"   URL: {result['url']}\n"
            response_text += f"   Score: {result['score']:.3f}\n"
            response_text += f"   Content: {result['content']}\n\n"
        
        return [TextContent(type="text", text=response_text)]
    
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
        local_results = await self._search_knowledge_base_internal(
            request_id=request_id,
            query=query,
            max_results=local_max_results,
            similarity_threshold=local_threshold,
            include_metadata=include_sources
        )
        
        # Determine if web search is needed
        # TODO: Implement actual logic to evaluate local results quality
        need_web_search = True  # Placeholder logic
        
        response_text = "Smart Search Results:\n\n"
        response_text += "=== LOCAL KNOWLEDGE BASE ===\n"
        response_text += local_results[0].text + "\n"
        
        if need_web_search and web_max_results > 0:
            web_results = await self._search_web_internal(
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
            self.logger.info("RAG MCP Server shutdown complete")


# Server instance for module-level access
rag_server = RAGMCPServer()


async def main() -> None:
    """Main entry point for the RAG MCP Server."""
    await rag_server.run()


if __name__ == "__main__":
    asyncio.run(main())