"""
Web Search Manager for Tavily API Integration.

This module provides comprehensive web search capabilities with:
- Tavily API integration with timeout and retry mechanisms
- Exponential backoff retry strategy
- Comprehensive error handling for various API scenarios
- Result caching with TTL and size management
- Content filtering and query optimization
- API usage tracking and quota management
- Rich result formatting and processing

Features:
- WebSearchResult: Structured search result data
- WebSearchManager: Main search orchestrator with caching
- QueryOptimizer: Query preprocessing and optimization
- ContentFilter: Ad removal and content cleaning
- UsageTracker: API quota monitoring
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict, defaultdict
import tempfile
import pickle

# HTTP client imports
try:
    import aiohttp
    import aiofiles
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    aiofiles = None

# Async database for caching
try:
    import aiosqlite
    ASYNC_DB_AVAILABLE = True
except ImportError:
    ASYNC_DB_AVAILABLE = False
    aiosqlite = None

# Local imports
try:
    from config import config
except ImportError:
    config = None


# Type definitions and data classes
@dataclass
class WebSearchResult:
    """Structured web search result."""
    title: str
    url: str
    content: str
    snippet: str = ""
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_domain: str = ""
    content_type: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.snippet and self.content:
            # Generate snippet from content if not provided
            self.snippet = self._create_snippet(self.content)
        
        if not self.source_domain and self.url:
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.url)
                self.source_domain = parsed.netloc.lower()
            except Exception:
                self.source_domain = "unknown"
    
    def _create_snippet(self, content: str, max_length: int = 200) -> str:
        """Create a snippet from content."""
        if not content:
            return ""
        
        # Clean content
        cleaned = re.sub(r'\s+', ' ', content.strip())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # Find a good breaking point near max_length
        break_point = cleaned.rfind(' ', 0, max_length - 3)
        if break_point == -1 or break_point < max_length * 0.7:
            break_point = max_length - 3
        
        return cleaned[:break_point] + "..."


@dataclass
class SearchStats:
    """Search operation statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    quota_used: int = 0
    quota_remaining: Optional[int] = None


class WebSearchError(Exception):
    """Base exception for web search errors."""
    pass


class RateLimitError(WebSearchError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(WebSearchError):
    """Raised when API quota is exceeded."""
    pass


class QueryOptimizer:
    """Query optimization and preprocessing."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QueryOptimizer")
        
        # Common stop words to remove
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'very', 'when', 'come', 'his', 'your', 'now', 'just', 'than',
            'only', 'may', 'can', 'could', 'should', 'would'
        }
        
        # Patterns for cleaning
        self.cleanup_patterns = [
            (r'[^\w\s\-\']', ' '),  # Remove special chars except hyphens and apostrophes
            (r'\s+', ' '),          # Multiple spaces to single space
            (r'\b\w{1,2}\b', ''),   # Remove 1-2 character words
        ]
    
    def optimize_query(self, query: str, max_length: int = 400) -> Dict[str, Any]:
        """
        Optimize search query for better results.
        
        Args:
            query: Original search query
            max_length: Maximum query length
            
        Returns:
            Dictionary with optimization results
        """
        if not query or not query.strip():
            return {
                'original': query,
                'optimized': '',
                'keywords': [],
                'key_phrases': [],
                'removed_words': []
            }
        
        original_query = query.strip()
        
        # Convert to lowercase for processing
        processed = original_query.lower()
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            processed = re.sub(pattern, replacement, processed)
        
        processed = processed.strip()
        
        # Split into words
        words = [word.strip() for word in processed.split() if word.strip()]
        
        # Remove stop words and track what was removed
        removed_words = []
        filtered_words = []
        
        for word in words:
            if word in self.stop_words:
                removed_words.append(word)
            else:
                filtered_words.append(word)
        
        # Extract key phrases (2-3 word combinations)
        key_phrases = []
        if len(filtered_words) >= 2:
            # 2-word phrases
            for i in range(len(filtered_words) - 1):
                phrase = f"{filtered_words[i]} {filtered_words[i+1]}"
                key_phrases.append(phrase)
            
            # 3-word phrases
            if len(filtered_words) >= 3:
                for i in range(len(filtered_words) - 2):
                    phrase = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                    key_phrases.append(phrase)
        
        # Build optimized query
        optimized_query = ' '.join(filtered_words)
        
        # Truncate if too long
        if len(optimized_query) > max_length:
            # Try to keep complete words
            truncated = optimized_query[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # If we can keep most of it
                optimized_query = truncated[:last_space]
            else:
                optimized_query = truncated
        
        result = {
            'original': original_query,
            'optimized': optimized_query,
            'keywords': filtered_words,
            'key_phrases': key_phrases[:5],  # Top 5 phrases
            'removed_words': removed_words
        }
        
        self.logger.debug(f"Query optimization: {len(words)} -> {len(filtered_words)} words")
        return result


class ContentFilter:
    """Content filtering and cleaning."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ContentFilter")
        
        # Patterns to identify ad content
        self.ad_patterns = [
            r'\b(advertisement|sponsored|promoted|ad\s*by)\b',
            r'\bsponsored\s*content\b',
            r'\bpaid\s*promotion\b',
            r'\b(buy\s*now|shop\s*now|click\s*here)\b',
            r'\bspecial\s*offer\b',
            r'\bdiscount\s*code\b',
            r'\bfree\s*shipping\b',
            r'\b\d+%\s*off\b',
        ]
        
        # Patterns for irrelevant content
        self.irrelevant_patterns = [
            r'\bcookie\s*(policy|notice|consent)\b',
            r'\bterms\s*of\s*(service|use)\b',
            r'\bprivacy\s*policy\b',
            r'\bsubscribe\s*to\s*newsletter\b',
            r'\bfollow\s*us\s*on\b',
            r'\bsocial\s*media\b',
            r'\bshare\s*this\s*article\b',
            r'\brelated\s*articles?\b',
            r'\bcomments?\s*section\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_ad_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ad_patterns]
        self.compiled_irrelevant_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.irrelevant_patterns]
    
    def filter_content(self, content: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """
        Filter and clean content.
        
        Args:
            content: Raw content to filter
            title: Article title for context
            url: Source URL for context
            
        Returns:
            Dictionary with filtering results
        """
        if not content:
            return {
                'original_content': content,
                'filtered_content': '',
                'removed_sections': [],
                'is_ad_content': False,
                'quality_score': 0.0
            }
        
        original_length = len(content)
        filtered_content = content
        removed_sections = []
        
        # Check for ad content
        is_ad_content = self._is_ad_content(content, title, url)
        
        # Remove irrelevant sections
        for pattern in self.compiled_irrelevant_patterns:
            matches = pattern.findall(filtered_content)
            if matches:
                removed_sections.extend(matches)
                filtered_content = pattern.sub('', filtered_content)
        
        # Clean up whitespace
        filtered_content = re.sub(r'\s+', ' ', filtered_content).strip()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(filtered_content, original_length, is_ad_content)
        
        result = {
            'original_content': content,
            'filtered_content': filtered_content,
            'removed_sections': removed_sections,
            'is_ad_content': is_ad_content,
            'quality_score': quality_score,
            'size_reduction': original_length - len(filtered_content)
        }
        
        return result
    
    def _is_ad_content(self, content: str, title: str = "", url: str = "") -> bool:
        """Check if content appears to be advertisement."""
        # Check content for ad patterns
        ad_matches = sum(1 for pattern in self.compiled_ad_patterns if pattern.search(content))
        
        # Check title for ad indicators
        title_ad_matches = sum(1 for pattern in self.compiled_ad_patterns if pattern.search(title))
        
        # Combine scores
        total_indicators = ad_matches + title_ad_matches * 2  # Weight title more heavily
        
        return total_indicators >= 2
    
    def _calculate_quality_score(self, content: str, original_length: int, is_ad: bool) -> float:
        """Calculate content quality score (0.0 to 1.0)."""
        if not content or is_ad:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor
        if len(content) >= 100:
            score += 0.2
        elif len(content) >= 50:
            score += 0.1
        
        # Sentence structure (periods indicate complete sentences)
        sentence_count = content.count('.')
        if sentence_count >= 3:
            score += 0.2
        elif sentence_count >= 1:
            score += 0.1
        
        # Word variety (unique words / total words)
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.2
        
        # Penalize excessive reduction
        if original_length > 0:
            reduction_ratio = len(content) / original_length
            if reduction_ratio < 0.3:  # More than 70% reduction
                score *= 0.7
        
        return min(1.0, score)


class UsageTracker:
    """API usage tracking and quota management."""
    
    def __init__(self, quota_limit: Optional[int] = None):
        self.logger = logging.getLogger(f"{__name__}.UsageTracker")
        self.quota_limit = quota_limit
        self.usage_data = {
            'daily_usage': defaultdict(int),
            'hourly_usage': defaultdict(int),
            'total_usage': 0,
            'last_reset': datetime.utcnow().date(),
            'rate_limit_resets': []
        }
    
    def track_request(self, cost: int = 1) -> None:
        """Track an API request."""
        now = datetime.utcnow()
        today = now.date()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset daily usage if needed
        if self.usage_data['last_reset'] != today:
            self.usage_data['daily_usage'].clear()
            self.usage_data['last_reset'] = today
        
        # Update usage counters
        self.usage_data['daily_usage'][today] += cost
        self.usage_data['hourly_usage'][current_hour] += cost
        self.usage_data['total_usage'] += cost
        
        self.logger.debug(f"API usage tracked: +{cost}, daily total: {self.usage_data['daily_usage'][today]}")
    
    def check_quota(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if we're within quota limits.
        
        Returns:
            Tuple of (can_proceed, quota_info)
        """
        today = datetime.utcnow().date()
        daily_usage = self.usage_data['daily_usage'][today]
        
        quota_info = {
            'daily_usage': daily_usage,
            'quota_limit': self.quota_limit,
            'quota_remaining': None,
            'usage_percentage': 0.0
        }
        
        if self.quota_limit:
            quota_info['quota_remaining'] = max(0, self.quota_limit - daily_usage)
            quota_info['usage_percentage'] = (daily_usage / self.quota_limit) * 100
            
            can_proceed = daily_usage < self.quota_limit
        else:
            can_proceed = True
        
        return can_proceed, quota_info
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        today = datetime.utcnow().date()
        can_proceed, quota_info = self.check_quota()
        
        return {
            'total_usage': self.usage_data['total_usage'],
            'daily_usage': self.usage_data['daily_usage'][today],
            'quota_info': quota_info,
            'can_make_requests': can_proceed,
            'rate_limit_events': len(self.usage_data['rate_limit_resets']),
            'last_reset_date': str(self.usage_data['last_reset'])
        }


class SearchCache:
    """Result caching with TTL and size management."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 1, max_size: int = 1000):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "web_search_cache"
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        self.cache_db_path = self.cache_dir / "search_cache.db"
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.SearchCache")
    
    async def initialize(self) -> None:
        """Initialize cache database."""
        if self._initialized:
            return
        
        if not ASYNC_DB_AVAILABLE:
            self.logger.warning("aiosqlite not available, caching disabled")
            return
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiosqlite.connect(self.cache_db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_hash TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        results BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at ON search_cache(created_at)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed ON search_cache(last_accessed)
                """)
                await db.commit()
            
            self._initialized = True
            self.logger.info(f"Search cache initialized at {self.cache_db_path}")
            
            # Clean expired entries
            await self._cleanup_expired()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key for query and parameters."""
        # Include relevant parameters in cache key
        cache_data = {
            'query': query.lower().strip(),
            'params': {k: v for k, v in kwargs.items() if k in ['max_results', 'search_depth']}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
    
    async def get(self, query: str, **kwargs) -> Optional[List[WebSearchResult]]:
        """Retrieve cached results."""
        if not self._initialized:
            await self.initialize()
        
        if not self._initialized:
            return None
        
        cache_key = self._get_cache_key(query, **kwargs)
        
        try:
            async with aiosqlite.connect(self.cache_db_path) as db:
                # Check if entry exists and is not expired
                cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)
                
                cursor = await db.execute("""
                    SELECT results FROM search_cache 
                    WHERE query_hash = ? AND created_at > ?
                """, (cache_key, cutoff_time.isoformat()))
                
                row = await cursor.fetchone()
                
                if row:
                    # Update access statistics
                    await db.execute("""
                        UPDATE search_cache 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP 
                        WHERE query_hash = ?
                    """, (cache_key,))
                    await db.commit()
                    
                    # Deserialize results
                    results = pickle.loads(row[0])
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return results
            
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def store(self, query: str, results: List[WebSearchResult], **kwargs) -> None:
        """Store results in cache."""
        if not self._initialized:
            await self.initialize()
        
        if not self._initialized:
            return
        
        cache_key = self._get_cache_key(query, **kwargs)
        
        try:
            async with aiosqlite.connect(self.cache_db_path) as db:
                # Serialize results
                serialized_results = pickle.dumps(results)
                
                # Store in cache
                await db.execute("""
                    INSERT OR REPLACE INTO search_cache (query_hash, query_text, results) 
                    VALUES (?, ?, ?)
                """, (cache_key, query[:200], serialized_results))
                
                await db.commit()
                
                # Check cache size and cleanup if needed
                await self._enforce_size_limit(db)
                
                self.logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")
                
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        if not self._initialized:
            return
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)
            
            async with aiosqlite.connect(self.cache_db_path) as db:
                cursor = await db.execute("""
                    DELETE FROM search_cache WHERE created_at < ?
                """, (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                await db.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
                    
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    async def _enforce_size_limit(self, db) -> None:
        """Enforce maximum cache size by removing oldest entries."""
        try:
            # Count current entries
            cursor = await db.execute("SELECT COUNT(*) FROM search_cache")
            count = (await cursor.fetchone())[0]
            
            if count > self.max_size:
                # Remove oldest entries
                entries_to_remove = count - self.max_size
                await db.execute("""
                    DELETE FROM search_cache 
                    WHERE query_hash IN (
                        SELECT query_hash FROM search_cache 
                        ORDER BY last_accessed ASC 
                        LIMIT ?
                    )
                """, (entries_to_remove,))
                
                await db.commit()
                self.logger.info(f"Removed {entries_to_remove} old cache entries to enforce size limit")
                
        except Exception as e:
            self.logger.warning(f"Cache size enforcement failed: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized:
            return {}
        
        try:
            async with aiosqlite.connect(self.cache_db_path) as db:
                # Total entries
                cursor = await db.execute("SELECT COUNT(*) FROM search_cache")
                total_entries = (await cursor.fetchone())[0]
                
                # Recent entries (last 24 hours)
                cutoff = datetime.utcnow() - timedelta(hours=24)
                cursor = await db.execute("""
                    SELECT COUNT(*) FROM search_cache WHERE created_at > ?
                """, (cutoff.isoformat(),))
                recent_entries = (await cursor.fetchone())[0]
                
                # Cache hit statistics
                cursor = await db.execute("""
                    SELECT AVG(access_count), SUM(access_count) FROM search_cache
                """)
                avg_access, total_access = await cursor.fetchone()
                
                return {
                    'total_entries': total_entries,
                    'recent_entries': recent_entries,
                    'max_size': self.max_size,
                    'ttl_hours': self.ttl_hours,
                    'average_access_count': round(avg_access or 0, 2),
                    'total_cache_hits': total_access or 0,
                    'cache_size_mb': self.cache_db_path.stat().st_size / (1024 * 1024) if self.cache_db_path.exists() else 0
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {}


class WebSearchManager:
    """
    Comprehensive web search manager with Tavily API integration.
    
    Features:
    - Tavily API integration with timeout and retry mechanisms
    - Exponential backoff retry strategy
    - Comprehensive error handling
    - Result caching with TTL
    - Content filtering and query optimization
    - API usage tracking
    """
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl_hours: int = 1,
        cache_max_size: int = 1000,
        quota_limit: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize WebSearchManager.
        
        Args:
            api_key: Tavily API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            cache_ttl_hours: Cache TTL in hours
            cache_max_size: Maximum cache entries
            quota_limit: Daily API quota limit
            cache_dir: Cache directory path
        """
        if not api_key:
            raise ValueError("Tavily API key is required")
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for web search: pip install aiohttp")
        
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.tavily.com"
        
        # Initialize components
        self.query_optimizer = QueryOptimizer()
        self.content_filter = ContentFilter()
        self.usage_tracker = UsageTracker(quota_limit)
        self.cache = SearchCache(cache_dir, cache_ttl_hours, cache_max_size)
        
        # Statistics
        self.stats = SearchStats()
        
        # HTTP session (will be created when needed)
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.logger = logging.getLogger(f"{__name__}.WebSearchManager")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'RAG-MCP-Server/1.0'
                }
            )
        return self._session
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        exclude_domains: Optional[List[str]] = None
    ) -> Tuple[List[WebSearchResult], Dict[str, Any]]:
        """
        Perform web search with comprehensive error handling and optimization.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: Search depth (basic/advanced)
            include_answer: Whether to include AI answer
            include_raw_content: Whether to include raw content
            exclude_domains: Domains to exclude
            
        Returns:
            Tuple of (search results, search metadata)
        """
        search_start_time = time.time()
        exclude_domains = exclude_domains or []
        
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if max_results < 1 or max_results > 20:
                raise ValueError("max_results must be between 1 and 20")
            
            # Check quota
            can_proceed, quota_info = self.usage_tracker.check_quota()
            if not can_proceed:
                raise QuotaExceededError(f"Daily quota exceeded: {quota_info['daily_usage']}/{quota_info['quota_limit']}")
            
            # Optimize query
            query_info = self.query_optimizer.optimize_query(query)
            search_query = query_info['optimized'] or query_info['original']
            
            self.logger.info(f"Searching web for: '{search_query}' (max_results={max_results})")
            
            # Check cache first
            cache_params = {
                'max_results': max_results,
                'search_depth': search_depth
            }
            cached_results = await self.cache.get(search_query, **cache_params)
            
            if cached_results:
                self.stats.cache_hits += 1
                search_time = time.time() - search_start_time
                
                metadata = {
                    'query_optimization': query_info,
                    'cache_hit': True,
                    'search_time': search_time,
                    'quota_info': quota_info
                }
                
                self.logger.info(f"Cache hit for query: '{search_query}' ({len(cached_results)} results)")
                return cached_results, metadata
            
            # Cache miss - perform API search
            self.stats.cache_misses += 1
            
            # Perform search with retry mechanism
            api_results = await self._search_with_retry(
                search_query, max_results, search_depth, 
                include_answer, include_raw_content, exclude_domains
            )
            
            # Process and filter results
            processed_results = await self._process_results(api_results, query_info)
            
            # Cache results
            await self.cache.store(search_query, processed_results, **cache_params)
            
            # Update statistics
            search_time = time.time() - search_start_time
            self.stats.total_execution_time += search_time
            self.stats.successful_requests += 1
            self.stats.total_requests += 1
            
            if self.stats.total_requests > 0:
                self.stats.average_response_time = self.stats.total_execution_time / self.stats.total_requests
            
            metadata = {
                'query_optimization': query_info,
                'cache_hit': False,
                'search_time': search_time,
                'quota_info': quota_info,
                'results_processed': len(processed_results),
                'api_response_time': search_time  # Will be updated with actual API time
            }
            
            self.logger.info(f"Web search completed: {len(processed_results)} results in {search_time:.3f}s")
            return processed_results, metadata
            
        except Exception as e:
            # Update error statistics
            search_time = time.time() - search_start_time
            self.stats.failed_requests += 1
            self.stats.total_requests += 1
            
            self.logger.error(f"Web search failed for query '{query}': {e}")
            
            # Return empty results with error metadata
            metadata = {
                'error': str(e),
                'error_type': type(e).__name__,
                'search_time': search_time,
                'quota_info': self.usage_tracker.check_quota()[1]
            }
            
            return [], metadata
    
    async def _search_with_retry(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        include_answer: bool,
        include_raw_content: bool,
        exclude_domains: List[str]
    ) -> Dict[str, Any]:
        """Perform API search with exponential backoff retry."""
        session = await self._get_session()
        
        # Prepare API request
        request_data = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results
        }
        
        if exclude_domains:
            request_data["exclude_domains"] = exclude_domains
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Track API usage
                self.usage_tracker.track_request()
                self.stats.api_calls += 1
                
                # Calculate delay for this attempt
                if attempt > 0:
                    delay = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
                    self.logger.info(f"Retrying search in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                # Make API request
                api_start_time = time.time()
                async with session.post(f"{self.base_url}/search", json=request_data) as response:
                    api_time = time.time() - api_start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.debug(f"API request successful in {api_time:.3f}s")
                        return result
                    
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 60))
                        error_msg = f"Rate limit exceeded, retry after {retry_after}s"
                        
                        if attempt < self.max_retries:
                            self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise RateLimitError(error_msg, retry_after)
                    
                    elif response.status == 402:  # Payment required / quota exceeded
                        error_text = await response.text()
                        raise QuotaExceededError(f"API quota exceeded: {error_text}")
                    
                    elif response.status >= 500:  # Server error - can retry
                        error_text = await response.text()
                        last_exception = WebSearchError(f"Server error ({response.status}): {error_text}")
                        if attempt < self.max_retries:
                            self.logger.warning(f"Server error, retrying (attempt {attempt + 1}): {last_exception}")
                            continue
                    
                    else:  # Client error - don't retry
                        error_text = await response.text()
                        raise WebSearchError(f"API error ({response.status}): {error_text}")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = WebSearchError(f"Network error: {str(e)}")
                if attempt < self.max_retries:
                    self.logger.warning(f"Network error, retrying (attempt {attempt + 1}): {last_exception}")
                    continue
            
            except (RateLimitError, QuotaExceededError):
                # Don't retry these errors
                raise
            
            except Exception as e:
                last_exception = WebSearchError(f"Unexpected error: {str(e)}")
                if attempt < self.max_retries:
                    self.logger.warning(f"Unexpected error, retrying (attempt {attempt + 1}): {last_exception}")
                    continue
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise WebSearchError("All retry attempts failed")
    
    async def _process_results(self, api_results: Dict[str, Any], query_info: Dict[str, Any]) -> List[WebSearchResult]:
        """Process and filter API results."""
        if not api_results or 'results' not in api_results:
            return []
        
        processed_results = []
        
        for item in api_results.get('results', []):
            try:
                # Extract basic information
                title = item.get('title', 'No title')
                url = item.get('url', '')
                content = item.get('content', '')
                raw_content = item.get('raw_content', '')
                score = item.get('score', 0.0)
                
                # Use raw_content if available and longer
                if raw_content and len(raw_content) > len(content):
                    content = raw_content
                
                # Filter content
                filter_result = self.content_filter.filter_content(content, title, url)
                
                # Skip if identified as ad content
                if filter_result['is_ad_content']:
                    self.logger.debug(f"Skipping ad content: {title[:50]}...")
                    continue
                
                # Skip if quality score is too low
                if filter_result['quality_score'] < 0.3:
                    self.logger.debug(f"Skipping low quality content: {title[:50]}...")
                    continue
                
                # Create search result
                result = WebSearchResult(
                    title=title,
                    url=url,
                    content=filter_result['filtered_content'],
                    score=score,
                    metadata={
                        'original_content_length': len(content),
                        'filtered_content_length': len(filter_result['filtered_content']),
                        'quality_score': filter_result['quality_score'],
                        'removed_sections': len(filter_result['removed_sections']),
                        'query_keywords': query_info.get('keywords', [])
                    }
                )
                
                processed_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to process search result: {e}")
                continue
        
        # Sort by score (descending)
        processed_results.sort(key=lambda x: x.score, reverse=True)
        
        return processed_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        cache_stats = await self.cache.get_cache_stats()
        usage_stats = self.usage_tracker.get_usage_stats()
        
        return {
            'search_stats': {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (self.stats.successful_requests / max(1, self.stats.total_requests)) * 100,
                'average_response_time': round(self.stats.average_response_time, 3),
                'api_calls': self.stats.api_calls
            },
            'cache_stats': {
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate': (self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses)) * 100,
                **cache_stats
            },
            'usage_stats': usage_stats
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self.logger.info("WebSearchManager closed")


# Export main classes
__all__ = [
    'WebSearchManager',
    'WebSearchResult',
    'SearchStats',
    'WebSearchError',
    'RateLimitError',
    'QuotaExceededError',
    'QueryOptimizer',
    'ContentFilter',
    'UsageTracker',
    'SearchCache'
]