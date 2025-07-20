# Release Notes - v0.1.0

## 🚀 Dynamic Embedding Provider System

This release introduces a comprehensive dynamic embedding provider system that allows seamless switching between OpenAI and DashScope/Qwen embedding providers without code changes.

## ✨ Key Features

### 🔄 **Runtime Provider Switching**
- Switch between OpenAI and DashScope providers using environment variables
- No code changes required - just update your `.env` file
- Automatic provider detection and initialization

### 🏗️ **Per-Index Provider Support**
- Create multiple vector indexes with different providers in the same application
- Mix and match providers based on your specific use case requirements
- Independent configuration for different document collections

### 🛡️ **Robust Error Handling**
- Comprehensive error handling with detailed error messages
- Automatic fallback to OpenAI when primary provider fails
- Configuration validation with helpful guidance

### ⚙️ **Easy Configuration**
```bash
# Use OpenAI (default)
EMBED_PROVIDER=openai
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small

# Use DashScope/Qwen
EMBED_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-v1
```

## 📋 **What's New**

### New Files Added
- `src/embedding_provider.py` - Core provider system
- `test_embedding_providers.py` - Comprehensive test suite
- `example_embedding_usage.py` - Usage examples
- `docs/embedding_providers.md` - Complete documentation
- `VERSION` - Version tracking
- `CHANGELOG.md` - Detailed change log

### Enhanced Files
- `config/settings.py` - Added provider configuration
- `src/llamaindex_processor.py` - Integrated dynamic providers
- `.env` and `.env.example` - Updated with provider options

## 🔧 **Usage Examples**

### Basic Provider Selection
```python
from src.embedding_provider import get_embed_model

# Explicit provider selection
openai_model = get_embed_model("openai")
dashscope_model = get_embed_model("dashscope")
```

### Environment-Based Configuration
```python
from src.embedding_provider import get_embed_model_from_env

# Uses EMBED_PROVIDER environment variable
embed_model = get_embed_model_from_env()
```

### Multiple Indexes
```python
from src.embedding_provider import create_index_with_provider

# Different providers for different use cases
docs_index = create_index_with_provider("openai", documents)
code_index = create_index_with_provider("dashscope", code_documents)
```

## 🧪 **Testing**

Test the new system with the included test suite:

```bash
# Run comprehensive tests
python test_embedding_providers.py

# See practical examples
python example_embedding_usage.py
```

## 📖 **Documentation**

Comprehensive documentation is available in `docs/embedding_providers.md`, including:
- Complete setup instructions
- Provider comparison and recommendations
- Troubleshooting guide
- Migration instructions
- Best practices

## 🔄 **Migration**

### For Existing Users
- **No breaking changes** - existing configurations continue to work
- **Optional upgrade** - add new environment variables to use new features
- **Gradual adoption** - can switch providers per component

### Quick Migration Steps
1. Add `EMBED_PROVIDER=openai` to your `.env` file (maintains current behavior)
2. Optionally add `DASHSCOPE_API_KEY` if you want to use DashScope
3. Test with `python test_embedding_providers.py`
4. Switch providers by changing `EMBED_PROVIDER` value

## 🎯 **Benefits**

- **Flexibility**: Switch providers without code changes
- **Cost Optimization**: Choose the most cost-effective provider for each use case
- **Regional Compliance**: Use DashScope for China operations, OpenAI for global
- **Performance**: Select the best-performing provider for your specific data
- **Redundancy**: Automatic fallback ensures system reliability

## 🔮 **Future Roadmap**

The provider system is designed for easy extension. Future versions may include:
- Hugging Face Embeddings
- Cohere Embeddings
- Azure OpenAI
- Google Vertex AI
- Custom embedding providers

## 📊 **Compatibility**

- **Python**: 3.8+
- **LlamaIndex**: 0.10.0+
- **Backward Compatible**: 100% with existing configurations
- **Dependencies**: No new dependencies required

## 🤝 **Contributing**

This release maintains the clean, extensible architecture that makes it easy to add new providers. See `docs/embedding_providers.md` for extension guidelines.

---

**Full Changelog**: See `CHANGELOG.md` for detailed technical changes.

**Questions?** Check the troubleshooting guide in `docs/embedding_providers.md` or review the examples in `example_embedding_usage.py`.