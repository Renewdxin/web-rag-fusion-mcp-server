"""
Configuration management module with environment variable support, validation, and singleton pattern.

This module provides a centralized configuration system that:
- Loads settings from environment variables using python-dotenv
- Validates required configurations with descriptive error messages
- Supports different environments (dev, test, prod)
- Implements singleton pattern for consistent configuration access
- Provides type-annotated properties with sensible defaults

Config Schema:
    VECTOR_STORE_PATH (str): Path to the vector store database
    TAVILY_API_KEY (str): API key for Tavily service
    OPENAI_API_KEY (str): API key for OpenAI service
    SIMILARITY_THRESHOLD (float): Threshold for similarity matching (0.0-1.0)
    ENVIRONMENT (str): Application environment (dev, test, prod)
    LOG_LEVEL (str): Logging level (DEBUG, INFO, WARNING, ERROR)
    MAX_RETRIES (int): Maximum number of retry attempts
    TIMEOUT_SECONDS (int): Request timeout in seconds
    USE_LLAMAINDEX (bool): Whether to use LlamaIndex for enhanced RAG processing
    CHUNK_SIZE (int): Size of text chunks for document processing
    CHUNK_OVERLAP (int): Overlap between text chunks
    EMBEDDING_MODEL (str): OpenAI embedding model for LlamaIndex
    SIMILARITY_TOP_K (int): Number of similar documents to retrieve
    SIMILARITY_CUTOFF (float): Minimum similarity score for search results
    COLLECTION_NAME (str): Name of the document collection
"""

import logging
import os
from pathlib import Path
from typing import Optional, Literal

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """
    Singleton configuration class that manages application settings.
    
    Loads configuration from environment variables and provides validation
    with descriptive error messages for missing or invalid values.
    """

    _instance: Optional['Config'] = None
    _initialized: bool = False

    def __new__(cls) -> 'Config':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration if not already done."""
        if not self._initialized:
            self._load_environment()
            self._initialized = True

    @property
    def SEARCH_BACKEND(self) -> Literal['perplexity', 'exa']:
        """The search backend to use."""
        backend = os.getenv('SEARCH_BACKEND', 'perplexity').lower()
        if backend not in ('perplexity', 'exa'):
            raise ConfigurationError(f"SEARCH_BACKEND must be one of 'perplexity', 'exa', got '{backend}'")
        return backend # type: ignore

    def _load_environment(self) -> None:
        """Load environment variables from .env file if available."""
        if load_dotenv is not None:
            # Try to load from .env file in project root
            env_path = Path(__file__).parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        else:
            logging.warning(
                "python-dotenv not installed. Environment variables will only be loaded from system environment.")

    @property
    def VECTOR_STORE_PATH(self) -> str:
        """Path to the vector store database."""
        return os.getenv('VECTOR_STORE_PATH', './data/vector_store.db')

    @property
    def SEARCH_API_KEY(self) -> str:
        """API key for search service (Perplexity or Exa)."""
        # Check for new key first, fallback to old key for compatibility
        value = os.getenv('SEARCH_API_KEY') or os.getenv('TAVILY_API_KEY', '')
        return value

    @property
    def OPENAI_API_KEY(self) -> str:
        """API key for OpenAI service."""
        value = os.getenv('OPENAI_API_KEY', '')
        return value

    @property
    def OPENAI_BASE_URL(self) -> Optional[str]:
        """Base URL for OpenAI API (for proxy or alternative endpoints)."""
        return os.getenv('OPENAI_BASE_URL', None)

    @property
    def SEARCH_BASE_URL(self) -> Optional[str]:
        """Base URL for search API (for proxy or alternative endpoints)."""
        return os.getenv('SEARCH_BASE_URL', None)

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        """Threshold for similarity matching (0.0-1.0)."""
        try:
            value = float(os.getenv('SIMILARITY_THRESHOLD', '0.75'))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {value}")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid SIMILARITY_THRESHOLD: {e}")

    @property
    def ENVIRONMENT(self) -> Literal['dev', 'test', 'prod']:
        """Application environment."""
        env = os.getenv('ENVIRONMENT', 'dev').lower()
        if env not in ('dev', 'test', 'prod'):
            raise ConfigurationError(f"ENVIRONMENT must be one of 'dev', 'test', 'prod', got '{env}'")
        return env  # type: ignore

    @property
    def LOG_LEVEL(self) -> str:
        """Logging level."""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        if level not in valid_levels:
            raise ConfigurationError(f"LOG_LEVEL must be one of {valid_levels}, got '{level}'")
        return level

    @property
    def MAX_RETRIES(self) -> int:
        """Maximum number of retry attempts."""
        try:
            value = int(os.getenv('MAX_RETRIES', '3'))
            if value < 0:
                raise ValueError("MAX_RETRIES must be non-negative")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid MAX_RETRIES: {e}")

    @property
    def TIMEOUT_SECONDS(self) -> int:
        """Request timeout in seconds."""
        try:
            value = int(os.getenv('TIMEOUT_SECONDS', '30'))
            if value <= 0:
                raise ValueError("TIMEOUT_SECONDS must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid TIMEOUT_SECONDS: {e}")

    # LlamaIndex Configuration
    @property
    def USE_LLAMAINDEX(self) -> bool:
        """Whether to use LlamaIndex for enhanced RAG processing."""
        return os.getenv('USE_LLAMAINDEX', 'true').lower() in ('true', '1', 'yes', 'on')

    @property
    def CHUNK_SIZE(self) -> int:
        """Size of text chunks for document processing."""
        try:
            value = int(os.getenv('CHUNK_SIZE', '1024'))
            if value <= 0:
                raise ValueError("CHUNK_SIZE must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid CHUNK_SIZE: {e}")

    @property
    def CHUNK_OVERLAP(self) -> int:
        """Overlap between text chunks."""
        try:
            value = int(os.getenv('CHUNK_OVERLAP', '200'))
            if value < 0:
                raise ValueError("CHUNK_OVERLAP must be non-negative")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid CHUNK_OVERLAP: {e}")

    @property
    def EMBEDDING_MODEL(self) -> str:
        """OpenAI embedding model for LlamaIndex."""
        return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    @property
    def SIMILARITY_TOP_K(self) -> int:
        """Number of similar documents to retrieve."""
        try:
            value = int(os.getenv('SIMILARITY_TOP_K', '10'))
            if value <= 0:
                raise ValueError("SIMILARITY_TOP_K must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid SIMILARITY_TOP_K: {e}")

    @property
    def SIMILARITY_CUTOFF(self) -> float:
        """Minimum similarity score for search results."""
        try:
            value = float(os.getenv('SIMILARITY_CUTOFF', '0.7'))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"SIMILARITY_CUTOFF must be between 0.0 and 1.0, got {value}")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid SIMILARITY_CUTOFF: {e}")

    @property
    def COLLECTION_NAME(self) -> str:
        """Name of the document collection."""
        return os.getenv('COLLECTION_NAME', 'rag_documents')

    # System Components Configuration
    @property
    def ENABLE_PROMETHEUS_METRICS(self) -> bool:
        """Whether to enable Prometheus metrics collection."""
        return os.getenv('ENABLE_PROMETHEUS_METRICS', 'true').lower() in ('true', '1', 'yes', 'on')

    @property
    def ENABLE_RATE_LIMITING(self) -> bool:
        """Whether to enable request rate limiting."""
        return os.getenv('ENABLE_RATE_LIMITING', 'true').lower() in ('true', '1', 'yes', 'on')

    @property
    def RATE_LIMIT_REQUESTS(self) -> int:
        """Maximum requests per time window for rate limiting."""
        try:
            value = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
            if value <= 0:
                raise ValueError("RATE_LIMIT_REQUESTS must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid RATE_LIMIT_REQUESTS: {e}")

    @property
    def RATE_LIMIT_WINDOW(self) -> int:
        """Time window in seconds for rate limiting."""
        try:
            value = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
            if value <= 0:
                raise ValueError("RATE_LIMIT_WINDOW must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(f"Invalid RATE_LIMIT_WINDOW: {e}")

    @property
    def USE_LOGURU(self) -> bool:
        """Whether to use loguru for structured logging instead of standard logging."""
        return os.getenv('USE_LOGURU', 'true').lower() in ('true', '1', 'yes', 'on')

    @property
    def ENABLE_RETRY_LOGIC(self) -> bool:
        """Whether to enable tenacity-based retry logic for resilient operations."""
        return os.getenv('ENABLE_RETRY_LOGIC', 'true').lower() in ('true', '1', 'yes', 'on')

    def validate(self) -> None:
        """
        Validate all required configuration values.
        
        Raises:
            ConfigurationError: If any required configuration is missing or invalid.
        """
        errors = []

        # Check required string fields
        required_fields = {
            'SEARCH_API_KEY': self.SEARCH_API_KEY,
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
        }

        for field_name, field_value in required_fields.items():
            if not field_value or field_value.strip() == '':
                errors.append(f"{field_name} is required but not set or empty")

        # Validate vector store path
        vector_path = Path(self.VECTOR_STORE_PATH)
        vector_dir = vector_path.parent
        if not vector_dir.exists():
            try:
                vector_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                errors.append(f"Cannot create directory for VECTOR_STORE_PATH '{vector_path}': {e}")

        # Validate numeric fields (will raise ConfigurationError if invalid)
        try:
            self.SIMILARITY_THRESHOLD
        except ConfigurationError as e:
            errors.append(str(e))

        try:
            self.MAX_RETRIES
        except ConfigurationError as e:
            errors.append(str(e))

        try:
            self.TIMEOUT_SECONDS
        except ConfigurationError as e:
            errors.append(str(e))

        # Validate environment and log level
        try:
            self.ENVIRONMENT
        except ConfigurationError as e:
            errors.append(str(e))

        try:
            self.LOG_LEVEL
        except ConfigurationError as e:
            errors.append(str(e))

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigurationError(error_msg)

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == 'dev'

    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.ENVIRONMENT == 'test'

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == 'prod'

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for debugging.
        
        Note: Sensitive values (API keys) are masked.
        """
        return {
            'VECTOR_STORE_PATH': self.VECTOR_STORE_PATH,
            'SEARCH_API_KEY': '***' if self.SEARCH_API_KEY else '',
            'OPENAI_API_KEY': '***' if self.OPENAI_API_KEY else '',
            'OPENAI_BASE_URL': self.OPENAI_BASE_URL,
            'SEARCH_BASE_URL': self.SEARCH_BASE_URL,
            'SIMILARITY_THRESHOLD': self.SIMILARITY_THRESHOLD,
            'ENVIRONMENT': self.ENVIRONMENT,
            'LOG_LEVEL': self.LOG_LEVEL,
            'MAX_RETRIES': self.MAX_RETRIES,
            'TIMEOUT_SECONDS': self.TIMEOUT_SECONDS,
            'USE_LLAMAINDEX': self.USE_LLAMAINDEX,
            'ENABLE_PROMETHEUS_METRICS': self.ENABLE_PROMETHEUS_METRICS,
            'ENABLE_RATE_LIMITING': self.ENABLE_RATE_LIMITING,
            'RATE_LIMIT_REQUESTS': self.RATE_LIMIT_REQUESTS,
            'RATE_LIMIT_WINDOW': self.RATE_LIMIT_WINDOW,
            'USE_LOGURU': self.USE_LOGURU,
            'ENABLE_RETRY_LOGIC': self.ENABLE_RETRY_LOGIC,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"Config({', '.join(f'{k}={v}' for k, v in config_dict.items())})"


# Global configuration instance
config = Config()
