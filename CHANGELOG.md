# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-20

### Added
- **Dynamic Embedding Provider System**
  - Support for OpenAI and DashScope/Qwen embedding providers
  - Runtime provider switching via `EMBED_PROVIDER` environment variable
  - Per-index embedding model support for multiple providers in same application
  - Comprehensive error handling and automatic fallback mechanisms
  - Provider validation and configuration checking utilities

- **New Environment Variables**
  - `EMBED_PROVIDER`: Choose between 'openai' and 'dashscope' providers
  - `DASHSCOPE_API_KEY`: API key for DashScope/Qwen services
  - Enhanced `EMBEDDING_MODEL` support for provider-specific models

- **Enhanced Configuration**
  - Updated `.env` and `.env.example` with new provider options
  - Provider-specific model recommendations and examples
  - Comprehensive configuration documentation

- **Testing & Examples**
  - `test_embedding_providers.py`: Comprehensive test suite for both providers
  - `example_embedding_usage.py`: Practical usage examples and scenarios
  - Provider switching demonstrations and error handling tests

- **Documentation**
  - `docs/embedding_providers.md`: Complete provider system documentation
  - Migration guide from hardcoded to dynamic provider system
  - Troubleshooting guide and best practices
  - Environment variable reference and configuration examples

### Changed
- **RAGEngine Integration**
  - Refactored `src/llamaindex_processor.py` to use dynamic provider system
  - Replaced hardcoded embedding initialization with flexible provider selection
  - Maintained backward compatibility with existing configurations

- **Configuration System**
  - Enhanced `config/settings.py` with new provider-related properties
  - Added validation for provider-specific configuration
  - Improved error messages and configuration guidance

### Technical Details
- **Core Module**: `src/embedding_provider.py`
  - `get_embed_model(provider, model, api_key)`: Main provider factory function
  - `get_embed_model_from_env()`: Environment-based provider selection
  - `create_index_with_provider()`: Per-index provider support
  - `validate_provider_config()`: Configuration validation utilities

- **Provider Support**
  - **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
  - **DashScope**: `text-embedding-v1`, `text-embedding-v2`
  - OpenAI-compatible endpoint support for DashScope integration

- **Error Handling**
  - Custom `EmbeddingProviderError` for provider-specific issues
  - Automatic fallback to OpenAI when primary provider fails
  - Comprehensive logging and debugging information

### Dependencies
- All required dependencies already present in `requirements.txt`
- `llama-index-embeddings-openai>=0.1.0`
- `llama-index-embeddings-dashscope>=0.3.0`

### Migration Notes
- Existing configurations continue to work without changes
- To use DashScope: Set `EMBED_PROVIDER=dashscope` and `DASHSCOPE_API_KEY`
- Provider switching requires only environment variable changes
- No code changes needed for basic provider switching

### Breaking Changes
- None. This release maintains full backward compatibility.

### Known Issues
- None identified in current release.

### Contributors
- Implementation powered by Claude Code
- Co-Authored-By: Claude <noreply@anthropic.com>