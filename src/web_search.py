"""
Simple AI Search Integration using Official Libraries

This module provides clean integration with AI search services using their official Python libraries:
- Perplexity AI: Using perplexipy or OpenAI-compatible client
- Exa.ai: Using exa-py official library

Key advantages:
- Uses official libraries (more reliable than custom HTTP clients)
- Simple, focused implementation
- Easy to extend with new services

Installation requirements:
- pip install perplexipy (for Perplexity)
- pip install exa-py (for Exa.ai)
- pip install openai (alternative for Perplexity)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Perplexity support
try:
    from perplexipy import PerplexityClient
    PERPLEXITY_AVAILABLE = True
except ImportError:
    try:
        # Alternative: OpenAI-compatible client for Perplexity
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        PERPLEXITY_AVAILABLE = False
    except ImportError:
        OPENAI_AVAILABLE = False
        PERPLEXITY_AVAILABLE = False

# Exa.ai support  
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False


@dataclass
class SearchResult:
    """Standardized search result format."""
    title: str
    url: str
    content: str
    snippet: str = ""
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


# Legacy compatibility
WebSearchResult = SearchResult


class SearchBackend(ABC):
    """Abstract base class for search backends."""
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> tuple[List[SearchResult], Dict[str, Any]]:
        """Perform search and return results with metadata."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class PerplexityBackend(SearchBackend):
    """
    Perplexity AI Backend using official library.
    
    Uses perplexipy for simple integration with Perplexity's search capabilities.
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Perplexity API key is required")
        if not PERPLEXITY_AVAILABLE and not OPENAI_AVAILABLE:
            raise ImportError("Neither perplexipy nor openai is available. Install with: pip install perplexipy")
            
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.PerplexityBackend")
        
        if PERPLEXITY_AVAILABLE:
            self.client = PerplexityClient(api_key=api_key)
            self.use_perplexipy = True
        elif OPENAI_AVAILABLE:
            # Use OpenAI client with Perplexity endpoint
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
            self.use_perplexipy = False
        else:
            raise ImportError("No suitable Perplexity client available")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        model: str = "llama-3.1-sonar-small-128k-online",
        **kwargs
    ) -> tuple[List[SearchResult], Dict[str, Any]]:
        """Search using Perplexity AI."""
        start_time = time.time()
        
        try:
            if self.use_perplexipy:
                # Use perplexipy library
                result = await asyncio.to_thread(
                    self.client.query, 
                    query
                )
                
                # Parse perplexipy result
                if isinstance(result, str):
                    # Simple text result
                    search_results = [SearchResult(
                        title="Perplexity Answer",
                        url="",
                        content=result,
                        snippet=result[:200] + "..." if len(result) > 200 else result,
                        score=1.0,
                        metadata={"source": "perplexity", "type": "ai_answer"}
                    )]
                else:
                    # Handle other result formats
                    search_results = []
                    for item in (result if isinstance(result, list) else [result]):
                        search_results.append(SearchResult(
                            title=getattr(item, 'title', 'Perplexity Result'),
                            url=getattr(item, 'url', ''),
                            content=str(item),
                            snippet=str(item)[:200] + "..." if len(str(item)) > 200 else str(item),
                            score=1.0,
                            metadata={"source": "perplexity"}
                        ))
                        
            else:
                # Use OpenAI-compatible client
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful search assistant. Provide comprehensive answers with sources when possible."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.1)
                )
                
                answer = response.choices[0].message.content
                search_results = [SearchResult(
                    title="Perplexity AI Answer",
                    url="",
                    content=answer,
                    snippet=answer[:200] + "..." if len(answer) > 200 else answer,
                    score=1.0,
                    metadata={
                        "source": "perplexity", 
                        "model": model,
                        "type": "ai_answer"
                    }
                )]
            
            metadata = {
                "search_time": time.time() - start_time,
                "total_results": len(search_results),
                "query": query,
                "source": "perplexity_ai"
            }
            
            return search_results, metadata
            
        except Exception as e:
            self.logger.error(f"Perplexity search error: {e}")
            return [], {
                "error": str(e),
                "error_type": "PerplexityError",
                "search_time": time.time() - start_time
            }
    
    async def close(self) -> None:
        """Close client connections."""
        if hasattr(self.client, 'close'):
            await asyncio.to_thread(self.client.close)


class ExaBackend(SearchBackend):
    """
    Exa.ai Backend using official library.
    
    Uses exa-py for semantic search and content discovery.
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Exa API key is required")
        if not EXA_AVAILABLE:
            raise ImportError("exa-py not available. Install with: pip install exa-py")
            
        self.api_key = api_key
        self.client = Exa(api_key=api_key)
        self.logger = logging.getLogger(f"{__name__}.ExaBackend")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        search_type: str = "neural",
        include_text: bool = True,
        **kwargs
    ) -> tuple[List[SearchResult], Dict[str, Any]]:
        """Search using Exa.ai."""
        start_time = time.time()
        
        try:
            if include_text:
                # Search and get content in one call
                result = await asyncio.to_thread(
                    self.client.search_and_contents,
                    query=query,
                    num_results=max_results,
                    type=search_type,
                    text=True,
                    **kwargs
                )
            else:
                # Just search for URLs
                result = await asyncio.to_thread(
                    self.client.search,
                    query=query,
                    num_results=max_results,
                    type=search_type,
                    **kwargs
                )
            
            # Parse Exa results
            search_results = []
            for item in result.results:
                # Extract content
                content = ""
                if hasattr(item, 'text') and item.text:
                    content = item.text
                elif hasattr(item, 'highlights') and item.highlights:
                    content = " ".join(item.highlights)
                elif hasattr(item, 'summary') and item.summary:
                    content = item.summary
                
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                search_results.append(SearchResult(
                    title=item.title or "Exa Result",
                    url=item.url or "",
                    content=content,
                    snippet=snippet,
                    score=getattr(item, 'score', 0.0),
                    metadata={
                        "source": "exa",
                        "published_date": getattr(item, 'published_date', None),
                        "author": getattr(item, 'author', None),
                        "id": getattr(item, 'id', None)
                    }
                ))
            
            metadata = {
                "search_time": time.time() - start_time,
                "total_results": len(search_results),
                "query": query,
                "source": "exa_ai",
                "autoprompt_string": getattr(result, 'autoprompt_string', None)
            }
            
            return search_results, metadata
            
        except Exception as e:
            self.logger.error(f"Exa search error: {e}")
            return [], {
                "error": str(e),
                "error_type": "ExaError",
                "search_time": time.time() - start_time
            }
    
    async def close(self) -> None:
        """Close client connections."""
        # Exa client doesn't need explicit closing
        pass


class WebSearchManager:
    """
    Simplified Web Search Manager using official libraries.
    
    Clean interface for AI search services with automatic library detection.
    """
    
    def __init__(
        self,
        api_key: str,
        backend: str = "perplexity",
        timeout: int = 30,
        max_retries: int = 3,
        quota_limit: Optional[int] = None,
        **backend_config
    ):
        """
        Initialize with specified backend.
        
        Args:
            api_key: API key for the backend service
            backend: Backend type ("perplexity", "exa")
            timeout: Request timeout (for compatibility)
            max_retries: Maximum retry attempts (for compatibility)
            quota_limit: API quota limit (for compatibility)
            backend_config: Additional backend configuration
        """
        self.logger = logging.getLogger(f"{__name__}.WebSearchManager")
        
        # Create backend
        if backend == "perplexity":
            self.backend = PerplexityBackend(api_key=api_key)
        elif backend == "exa":
            self.backend = ExaBackend(api_key=api_key)
        else:
            raise ValueError(f"Unknown backend type: {backend}. Supported: perplexity, exa")
        
        self.logger.info(f"Initialized with {backend} backend using official library")
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform search using the configured backend.
        
        Returns:
            Tuple of (search_results, metadata)
        """
        if not query or not query.strip():
            return [], {"error": "Empty query provided"}
        
        self.logger.debug(f"Searching: '{query}' (max_results={max_results})")
        
        try:
            results, metadata = await self.backend.search(
                query=query.strip(),
                max_results=max_results,
                **kwargs
            )
            
            self.logger.info(f"Search completed: {len(results)} results")
            return results, metadata
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return [], {
                "error": str(e),
                "error_type": "SearchManagerError"
            }
    
    async def close(self) -> None:
        """Close and cleanup resources."""
        if self.backend:
            await self.backend.close()
            self.logger.info("Search manager closed")


# Legacy compatibility
class WebSearchError(Exception):
    """Web search error for backward compatibility."""
    pass


# Factory function for easy initialization
def create_search_manager(
    backend: str = "perplexity",
    api_key: Optional[str] = None,
    **config
) -> WebSearchManager:
    """
    Factory function to create WebSearchManager with specified backend.
    
    Examples:
        # Perplexity AI (Default - Best for research & comprehensive answers)
        manager = create_search_manager("perplexity", api_key="your_key")
        
        # Exa.ai (Best for semantic search & AI-optimized content)
        manager = create_search_manager("exa", api_key="your_key")
    """
    if not api_key:
        raise ValueError("API key is required")
    return WebSearchManager(api_key=api_key, backend=backend, **config) 