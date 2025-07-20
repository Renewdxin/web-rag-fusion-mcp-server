# Dynamic Embedding Provider System

## Overview

The dynamic embedding provider system allows seamless switching between different embedding providers (OpenAI and DashScope/Qwen) at runtime without locking into a single provider. This system provides a clean abstraction layer and maintains compatibility with the existing LlamaIndex RAG setup.

## Features

- **Runtime Provider Switching**: Switch between providers using environment variables
- **Per-Index Provider Support**: Create multiple indexes with different providers in the same application
- **Error Handling & Fallbacks**: Robust error handling with automatic fallback to default providers
- **Clean Provider Abstraction**: Easy to extend for additional providers
- **Environment-Based Configuration**: Configure providers via environment variables
- **API Key Auto-Detection**: Automatically detect API keys from environment variables

## Supported Providers

### OpenAI
- **Provider Name**: `openai`
- **API Endpoint**: `https://api.openai.com/v1`
- **Default Model**: `text-embedding-ada-002`
- **API Key Environment Variable**: `OPENAI_API_KEY`
- **Models**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`

### DashScope/Qwen
- **Provider Name**: `dashscope`
- **API Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Default Model**: `text-embedding-v1`
- **API Key Environment Variable**: `DASHSCOPE_API_KEY`
- **Models**: `text-embedding-v1`, `text-embedding-v2`

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EMBED_PROVIDER` | Primary embedding provider (`openai` or `dashscope`) | `openai` | No |
| `EMBEDDING_MODEL` | Model name (provider-specific) | `text-embedding-3-small` | No |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes (for OpenAI) |
| `DASHSCOPE_API_KEY` | DashScope API key | - | Yes (for DashScope) |
| `OPENAI_BASE_URL` | OpenAI API base URL (for proxies) | - | No |

## Usage Examples

### Basic Usage

```python
from src.embedding_provider import get_embed_model

# Explicit provider selection
openai_model = get_embed_model("openai")
dashscope_model = get_embed_model("dashscope")

# Custom model
openai_large = get_embed_model("openai", model="text-embedding-3-large")
```

### Environment-Based Selection

```python
import os
from src.embedding_provider import get_embed_model_from_env

# Set provider via environment
os.environ["EMBED_PROVIDER"] = "dashscope"

# Get model based on environment
embed_model = get_embed_model_from_env()
```

### Multiple Indexes with Different Providers

```python
from src.embedding_provider import create_index_with_provider
from llama_index.core import Document

documents = [Document(text="Sample document")]

# Create indexes with different providers
openai_index = create_index_with_provider("openai", documents)
dashscope_index = create_index_with_provider("dashscope", documents)
```

### Configuration Validation

```python
from src.embedding_provider import validate_provider_config, list_providers

# Check provider availability
validation = validate_provider_config("openai")
if validation["valid"]:
    print("OpenAI is ready to use")

# List all providers
providers = list_providers()
```

## Integration with Existing Code

The system integrates seamlessly with the existing RAGEngine in `src/llamaindex_processor.py`. The embedding model is now automatically selected based on the `EMBED_PROVIDER` environment variable.

### Configuration in settings.py

New configuration properties have been added:

```python
@property
def EMBED_PROVIDER(self) -> str:
    """Embedding provider (openai or dashscope)."""
    return os.getenv('EMBED_PROVIDER', 'openai').lower()

@property
def DASHSCOPE_API_KEY(self) -> str:
    """API key for DashScope service."""
    return os.getenv('DASHSCOPE_API_KEY', '')
```

### RAGEngine Integration

The RAGEngine now uses the dynamic provider system in its `_setup_settings` method:

```python
def _setup_settings(self, embedding_model: str, llm_model: str):
    # Configure embedding model using the new provider system
    Settings.embed_model = get_embed_model_from_env(
        provider_env_var="EMBED_PROVIDER",
        fallback_provider="openai"
    )
```

## Error Handling

The system provides comprehensive error handling:

### EmbeddingProviderError
Raised when:
- API key is missing or invalid
- Provider initialization fails
- Network issues occur

### ValueError
Raised when:
- Invalid provider name is specified
- Invalid configuration parameters

### Fallback Mechanism
- If primary provider fails, automatically falls back to default (OpenAI)
- Logs warnings when fallbacks are used
- Provides detailed error messages for troubleshooting

## Testing

### Test Script
Run the comprehensive test suite:

```bash
python test_embedding_providers.py
```

### Example Usage
See practical examples:

```bash
python example_embedding_usage.py
```

### Test Coverage
- Provider initialization
- Embedding generation
- Index creation
- Provider switching
- Error handling
- Configuration validation

## Migration Guide

### From Original Code
If you're migrating from the original hardcoded approach:

1. **Update imports**:
   ```python
   # Old
   from llama_index.embeddings.openai import OpenAIEmbedding
   
   # New
   from src.embedding_provider import get_embed_model_from_env
   ```

2. **Update initialization**:
   ```python
   # Old
   Settings.embed_model = OpenAIEmbedding(
       api_key=config.OPENAI_API_KEY,
       model="text-embedding-ada-002"
   )
   
   # New
   Settings.embed_model = get_embed_model_from_env()
   ```

3. **Set environment variables**:
   ```bash
   export EMBED_PROVIDER=openai
   export OPENAI_API_KEY=your_key_here
   # or
   export EMBED_PROVIDER=dashscope
   export DASHSCOPE_API_KEY=your_key_here
   ```

## Best Practices

1. **Provider Selection**:
   - Use `openai` for maximum compatibility and model variety
   - Use `dashscope` for Alibaba Cloud integration or specific regional requirements

2. **API Key Management**:
   - Store API keys in environment variables, never in code
   - Use different keys for different environments (dev/staging/prod)
   - Regularly rotate API keys for security

3. **Error Handling**:
   - Always validate provider configuration before use
   - Implement proper fallback strategies
   - Log provider selection and errors for troubleshooting

4. **Performance**:
   - Cache embedding models when possible
   - Use appropriate batch sizes for bulk operations
   - Monitor API usage and costs

5. **Model Selection**:
   - Choose models based on your specific use case requirements
   - Consider embedding dimensions vs. performance trade-offs
   - Test different models with your data

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```
   Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable
   ```
   **Solution**: Set the appropriate API key environment variable

2. **Invalid Provider**:
   ```
   Error: Invalid embedding provider: 'unknown'. Supported providers: 'openai', 'dashscope'
   ```
   **Solution**: Use a supported provider name

3. **Network Issues**:
   ```
   Error: Failed to initialize OpenAI embedding model: Connection error
   ```
   **Solution**: Check network connectivity and API endpoint accessibility

4. **Model Not Found**:
   ```
   Error: Model 'invalid-model' not found
   ```
   **Solution**: Use a valid model name for the selected provider

### Debug Mode
Enable debug logging to see detailed provider selection:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Provider Status Check
Use the validation function to check provider status:

```python
from src.embedding_provider import validate_provider_config

for provider in ["openai", "dashscope"]:
    status = validate_provider_config(provider)
    print(f"{provider}: {status}")
```

## Future Extensions

The system is designed to be easily extensible for additional providers:

1. **Add Provider Info**: Update `PROVIDER_INFO` in `embedding_provider.py`
2. **Create Provider Function**: Add `_create_<provider>_embedding()` function
3. **Update Main Function**: Add provider case to `get_embed_model()`
4. **Add Tests**: Include provider in test suite
5. **Update Documentation**: Add provider to this guide

Example providers that could be added:
- Hugging Face Embeddings
- Cohere Embeddings
- Azure OpenAI
- Google Vertex AI Embeddings