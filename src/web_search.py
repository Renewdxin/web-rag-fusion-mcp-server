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

from perplexipy import PerplexityClient
from openai import OpenAI
from exa_py import Exa

from config.settings import config


@dataclass
class SearchResult:
    """Standardized search result format."""
    title: str
    url: str
    content: str
    snippet: str = ""
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def source_domain(self) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            return parsed.netloc or "Unknown"
        except Exception:
            return "Unknown"


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
            
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.PerplexityBackend")
        
        if config.SEARCH_BACKEND == 'perplexity':
            try:
                # Try with api_key parameter first
                self.client = PerplexityClient(api_key=api_key)
                self.use_perplexipy = True
            except TypeError:
                # If api_key parameter not supported, try without it
                self.client = PerplexityClient()
                self.use_perplexipy = True
        else:
            # Use OpenAI client with Perplexity endpoint
            base_url = config.SEARCH_BASE_URL or "https://api.perplexity.ai"
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.use_perplexipy = False
    
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
            
        self.api_key = api_key
        # Note: Exa client may not support custom base_url, check their documentation
        # If they add support in the future, we can use config.SEARCH_BASE_URL here
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
            timeout: int = 30,
            max_retries: int = 3,
            quota_limit: Optional[int] = None,
        **backend_config
    ):
        """
        Initialize with specified backend.

        Args:
            api_key: API key for the backend service
            timeout: Request timeout (for compatibility)
            max_retries: Maximum retry attempts (for compatibility)
            quota_limit: API quota limit (for compatibility)
            backend_config: Additional backend configuration
        """
        self.logger = logging.getLogger(f"{__name__}.WebSearchManager")

        # Create backend
        if config.SEARCH_BACKEND == "perplexity":
            self.backend = PerplexityBackend(api_key=api_key)
        elif config.SEARCH_BACKEND == "exa":
            self.backend = ExaBackend(api_key=api_key)
        else:
            raise ValueError(f"Unknown backend type: {config.SEARCH_BACKEND}. Supported: perplexity, exa")
        
        self.logger.info(f"Initialized with {config.SEARCH_BACKEND} backend using official library")

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
    api_key: Optional[str] = None,
    **config
) -> WebSearchManager:
    """
    Factory function to create WebSearchManager with specified backend.
    
    Examples:
        # Perplexity AI (Default - Best for research & comprehensive answers)
        manager = create_search_manager(api_key="your_key")
        
        # Exa.ai (Best for semantic search & AI-optimized content)
        manager = create_search_manager(api_key="your_key")
    """
    if not api_key:
        raise ValueError("API key is required")
    return WebSearchManager(api_key=api_key, **config)