"""
Dynamic Embedding Provider Manager for RAG System

This module provides dynamic switching between embedding providers including:
- OpenAI (via official API)
- DashScope/Qwen (OpenAI-compatible via endpoint)

Features:
- Runtime provider switching via environment variables
- Per-index embedding model support
- Error handling and fallback mechanisms
- Clean provider abstraction
- Easy extensibility for additional providers
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding

logger = logging.getLogger(__name__)


class EmbeddingProviderError(Exception):
    """Raised when embedding provider initialization or operation fails."""
    pass


def get_embed_model(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Union[OpenAIEmbedding, DashScopeEmbedding]:
    """
    Create an embedding model instance based on the specified provider.
    
    Args:
        provider: The embedding provider ("openai" or "dashscope")
        model: The model name (provider-specific defaults used if None)
        api_key: API key (auto-detected from env if None)
        **kwargs: Additional arguments passed to the embedding model
        
    Returns:
        Configured embedding model instance
        
    Raises:
        ValueError: If provider is invalid
        EmbeddingProviderError: If API key is missing or initialization fails
        
    Examples:
        # Use OpenAI with defaults
        embed_model = get_embed_model("openai")
        
        # Use DashScope with custom model
        embed_model = get_embed_model("dashscope", model="text-embedding-v2")
        
        # Use OpenAI with custom API key
        embed_model = get_embed_model("openai", api_key="sk-...")
    """
    provider = provider.lower().strip()
    
    if provider == "openai":
        return _create_openai_embedding(model, api_key, **kwargs)
    elif provider == "dashscope":
        return _create_dashscope_embedding(model, api_key, **kwargs)
    else:
        raise ValueError(
            f"Invalid embedding provider: '{provider}'. "
            f"Supported providers: 'openai', 'dashscope'"
        )


def _create_openai_embedding(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> OpenAIEmbedding:
    """Create OpenAI embedding model with proper configuration."""
    # Use provided API key or auto-detect from environment
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise EmbeddingProviderError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Use provided model or default
    if model is None:
        model = "text-embedding-ada-002"
    
    # Prepare OpenAI client arguments
    openai_kwargs = {
        "api_key": api_key,
        "model": model,
        "embed_batch_size": kwargs.get("embed_batch_size", 100),
    }
    
    # Add base URL if specified (for proxy support)
    base_url = kwargs.get("api_base") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        # Ensure base URL is properly formatted for OpenAI
        if not base_url.endswith("/"):
            base_url += "/"
        if not base_url.endswith("v1/"):
            base_url += "v1/"
        openai_kwargs["base_url"] = base_url
    
    try:
        embedding_model = OpenAIEmbedding(**openai_kwargs)
        logger.info(f"Initialized OpenAI embedding model: {model}")
        return embedding_model
    except Exception as e:
        raise EmbeddingProviderError(f"Failed to initialize OpenAI embedding model: {e}")


def _create_dashscope_embedding(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> DashScopeEmbedding:
    """Create DashScope embedding model with proper configuration."""
    # Use provided API key or auto-detect from environment
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise EmbeddingProviderError(
            "DashScope API key not found. Please set DASHSCOPE_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Use provided model or default
    if model is None:
        model = "text-embedding-v1"
    
    # Prepare DashScope arguments
    dashscope_kwargs = {
        "api_key": api_key,
        "model_name": model,
    }
    
    # Add any additional DashScope-specific parameters
    if "embed_batch_size" in kwargs:
        dashscope_kwargs["embed_batch_size"] = kwargs["embed_batch_size"]
    
    try:
        embedding_model = DashScopeEmbedding(**dashscope_kwargs)
        logger.info(f"Initialized DashScope embedding model: {model}")
        return embedding_model
    except Exception as e:
        raise EmbeddingProviderError(f"Failed to initialize DashScope embedding model: {e}")


def get_embed_model_from_env(
    provider_env_var: str = "EMBED_PROVIDER",
    fallback_provider: str = "openai"
) -> Union[OpenAIEmbedding, DashScopeEmbedding]:
    """
    Create embedding model based on environment variable configuration.
    
    Args:
        provider_env_var: Environment variable name for provider selection
        fallback_provider: Provider to use if env var is not set
        
    Returns:
        Configured embedding model instance
        
    Raises:
        EmbeddingProviderError: If provider initialization fails
        
    Environment Variables:
        EMBED_PROVIDER: Provider name ("openai" or "dashscope")
        OPENAI_API_KEY: API key for OpenAI
        DASHSCOPE_API_KEY: API key for DashScope
        EMBEDDING_MODEL: Model name (provider-specific)
    """
    # Get provider from environment or use fallback
    provider = os.getenv(provider_env_var, fallback_provider).lower().strip()
    
    # Get model from environment (optional)
    model = os.getenv("EMBEDDING_MODEL")
    
    # Log the provider being used (with fallback warning if applicable)
    env_provider = os.getenv(provider_env_var)
    if env_provider is None:
        logger.warning(
            f"Environment variable {provider_env_var} not set. "
            f"Using fallback provider: {fallback_provider}"
        )
    
    try:
        embedding_model = get_embed_model(provider, model=model)
        logger.info(f"Embedding provider initialized: {provider}")
        return embedding_model
    except ValueError as e:
        # Invalid provider - try fallback
        if provider != fallback_provider:
            logger.warning(
                f"Invalid provider '{provider}' from {provider_env_var}. "
                f"Falling back to: {fallback_provider}"
            )
            try:
                return get_embed_model(fallback_provider, model=model)
            except Exception as fallback_error:
                raise EmbeddingProviderError(
                    f"Both primary provider '{provider}' and fallback '{fallback_provider}' failed. "
                    f"Primary error: {e}. Fallback error: {fallback_error}"
                )
        else:
            raise EmbeddingProviderError(f"Provider initialization failed: {e}")
    except EmbeddingProviderError as e:
        # API key missing or other initialization error
        if provider != fallback_provider:
            logger.warning(
                f"Provider '{provider}' failed: {e}. Trying fallback: {fallback_provider}"
            )
            try:
                return get_embed_model(fallback_provider, model=model)
            except Exception as fallback_error:
                raise EmbeddingProviderError(
                    f"Both primary provider '{provider}' and fallback '{fallback_provider}' failed. "
                    f"Primary error: {e}. Fallback error: {fallback_error}"
                )
        else:
            raise


def create_index_with_provider(
    provider: str,
    documents: Optional[list] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **index_kwargs
) -> 'VectorStoreIndex':
    """
    Create a VectorStoreIndex with a specific embedding provider.
    
    This allows creating multiple indexes with different providers in the same app.
    
    Args:
        provider: Embedding provider ("openai" or "dashscope")
        documents: Documents to index (optional)
        model: Model name (provider-specific defaults used if None)
        api_key: API key (auto-detected from env if None)
        **index_kwargs: Additional arguments passed to VectorStoreIndex
        
    Returns:
        VectorStoreIndex configured with the specified provider
        
    Example:
        # Create index with OpenAI embeddings
        openai_index = create_index_with_provider("openai", documents)
        
        # Create index with DashScope embeddings
        dashscope_index = create_index_with_provider("dashscope", documents)
    """
    from llama_index.core import VectorStoreIndex
    
    # Get embedding model for the specified provider
    embed_model = get_embed_model(provider, model=model, api_key=api_key)
    
    # Create index with the embedding model
    if documents is not None:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            **index_kwargs
        )
    else:
        index = VectorStoreIndex(
            [],
            embed_model=embed_model,
            **index_kwargs
        )
    
    logger.info(
        f"Created VectorStoreIndex with {provider} embeddings "
        f"({len(documents) if documents else 0} documents)"
    )
    
    return index


# Provider metadata for reference
PROVIDER_INFO = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "default_model": "text-embedding-ada-002",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI embeddings via official API"
    },
    "dashscope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "text-embedding-v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "description": "DashScope/Qwen embeddings via OpenAI-compatible API"
    }
}


def list_providers() -> Dict[str, Dict[str, str]]:
    """
    Get information about available embedding providers.
    
    Returns:
        Dictionary with provider information
    """
    return PROVIDER_INFO.copy()


def validate_provider_config(provider: str) -> Dict[str, Any]:
    """
    Validate configuration for a specific provider.
    
    Args:
        provider: Provider name to validate
        
    Returns:
        Dictionary with validation results and configuration info
    """
    provider = provider.lower().strip()
    
    if provider not in PROVIDER_INFO:
        return {
            "valid": False,
            "error": f"Unknown provider: {provider}",
            "available_providers": list(PROVIDER_INFO.keys())
        }
    
    provider_info = PROVIDER_INFO[provider]
    api_key_env = provider_info["api_key_env"]
    api_key = os.getenv(api_key_env)
    
    result = {
        "valid": bool(api_key),
        "provider": provider,
        "api_key_configured": bool(api_key),
        "api_key_env_var": api_key_env,
        "default_model": provider_info["default_model"],
        "description": provider_info["description"]
    }
    
    if not api_key:
        result["error"] = f"API key not configured. Please set {api_key_env} environment variable."
    
    return result


__all__ = [
    "get_embed_model",
    "get_embed_model_from_env", 
    "create_index_with_provider",
    "list_providers",
    "validate_provider_config",
    "EmbeddingProviderError",
    "PROVIDER_INFO"
]