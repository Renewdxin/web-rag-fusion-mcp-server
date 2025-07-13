# Configuration Guide

This document provides comprehensive configuration options for the RAG MCP Server system.

## Overview

The RAG MCP Server uses a centralized configuration system that supports environment variables, dotenv files, and validation. The configuration is managed through the `Config` class which implements a singleton pattern for consistent access across all components.

## Configuration File Structure

### Core Configuration (`config/settings.py`)

```python
from config import config

# Access configuration values
api_key = config.OPENAI_API_KEY
vector_path = config.VECTOR_STORE_PATH
log_level = config.LOG_LEVEL
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | `sk-proj-...` |
| `TAVILY_API_KEY` | Tavily API key for web search | `tvly-...` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Runtime environment (development/production) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `VECTOR_STORE_PATH` | `./data` | Directory for vector database storage |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score for search results |
| `MAX_RETRIES` | `3` | Maximum retry attempts for API calls |
| `TIMEOUT_SECONDS` | `30` | Request timeout in seconds |
| `WEB_SEARCH_TIMEOUT` | `30` | Web search specific timeout |
| `TAVILY_QUOTA_LIMIT` | `None` | Daily API quota limit for Tavily |

## Configuration Setup

### 1. Environment File (.env)

Create a `.env` file in the project root:

```bash
# API Keys (Required)
OPENAI_API_KEY=sk-proj-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# Environment Settings
ENVIRONMENT=development
LOG_LEVEL=INFO

# Vector Store Configuration
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=rag_documents
SIMILARITY_THRESHOLD=0.75

# Performance Settings
MAX_RETRIES=3
TIMEOUT_SECONDS=30
WEB_SEARCH_TIMEOUT=45
TAVILY_QUOTA_LIMIT=1000

# Server Settings
SERVER_NAME=rag-mcp-server
SERVER_VERSION=1.0.0
```

### 2. Production Configuration

For production environments:

```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Optimized paths
VECTOR_STORE_PATH=/opt/rag-server/data
COLLECTION_NAME=production_documents

# Performance tuning
MAX_RETRIES=5
TIMEOUT_SECONDS=60
SIMILARITY_THRESHOLD=0.8

# Higher quotas for production
TAVILY_QUOTA_LIMIT=10000
```

## Configuration Validation

The system performs comprehensive validation on startup:

### Validation Rules

1. **Required Fields**: `OPENAI_API_KEY` and `TAVILY_API_KEY` must be present
2. **Numeric Ranges**: 
   - `SIMILARITY_THRESHOLD`: 0.0 - 1.0
   - `MAX_RETRIES`: 1 - 10
   - `TIMEOUT_SECONDS`: 5 - 300
3. **Path Validation**: Vector store path must be writable
4. **Log Level**: Must be valid Python logging level

### Validation Examples

```python
# Valid configuration
config.SIMILARITY_THRESHOLD = 0.75  # ✅ Valid
config.MAX_RETRIES = 3              # ✅ Valid
config.LOG_LEVEL = "INFO"           # ✅ Valid

# Invalid configuration
config.SIMILARITY_THRESHOLD = 1.5   # ❌ > 1.0
config.MAX_RETRIES = 0              # ❌ < 1
config.LOG_LEVEL = "INVALID"        # ❌ Not a valid level
```

## Component-Specific Configuration

### Vector Store Manager

```python
# Vector store specific settings
VECTOR_STORE_PATH=./data/chroma_db
COLLECTION_NAME=knowledge_base
EMBEDDING_CACHE_DIR=./cache/embeddings
DEDUPLICATION_ENABLED=true
CONNECTION_RETRIES=3
```

### Web Search Manager

```python
# Web search specific settings
TAVILY_API_KEY=tvly-your-key
WEB_SEARCH_TIMEOUT=30
SEARCH_CACHE_TTL_HOURS=1
SEARCH_CACHE_MAX_SIZE=1000
CONTENT_QUALITY_THRESHOLD=0.5
QUERY_OPTIMIZATION_ENABLED=true
```

### Document Processor

```python
# Document processing settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONCURRENCY=5
PROCESSING_CACHE_DIR=./cache/documents
SUPPORTED_FORMATS=pdf,txt,md,docx,html
```

## Advanced Configuration

### Custom Configuration Class

You can extend the configuration for custom needs:

```python
class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.CUSTOM_FEATURE_ENABLED = self._get_bool_env('CUSTOM_FEATURE_ENABLED', False)
        self.CUSTOM_API_ENDPOINT = self._get_env('CUSTOM_API_ENDPOINT', 'https://api.example.com')
    
    def validate_custom(self):
        """Add custom validation rules."""
        if self.CUSTOM_FEATURE_ENABLED and not self.CUSTOM_API_ENDPOINT:
            raise ConfigurationError("Custom API endpoint required when feature is enabled")
```

### Configuration Profiles

Create different configuration profiles for different environments:

```python
# config/profiles/development.py
class DevelopmentConfig(Config):
    LOG_LEVEL = "DEBUG"
    VECTOR_STORE_PATH = "./dev_data"
    SIMILARITY_THRESHOLD = 0.6  # Lower for testing

# config/profiles/production.py
class ProductionConfig(Config):
    LOG_LEVEL = "WARNING"
    VECTOR_STORE_PATH = "/opt/data"
    SIMILARITY_THRESHOLD = 0.8  # Higher for production
```

## Configuration Best Practices

### Security

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Restrict file permissions** on configuration files
4. **Use separate keys** for development and production

### Performance

1. **Tune similarity threshold** based on your use case
2. **Adjust timeout values** for your network conditions
3. **Configure appropriate cache sizes** for your memory constraints
4. **Set reasonable retry limits** to avoid excessive API calls

### Monitoring

1. **Enable appropriate logging** for your environment
2. **Monitor API quota usage** to avoid service interruptions
3. **Track configuration changes** in production
4. **Set up alerts** for configuration validation failures

## Troubleshooting

### Common Configuration Issues

#### Missing API Keys
```bash
Error: OPENAI_API_KEY environment variable not set
Solution: Set the required API key in your .env file
```

#### Invalid Path Configuration
```bash
Error: Vector store path not accessible: /invalid/path
Solution: Ensure the path exists and is writable
```

#### Validation Failures
```bash
Error: SIMILARITY_THRESHOLD must be between 0.0 and 1.0
Solution: Check your environment variable values
```

### Configuration Debugging

Enable debug logging to troubleshoot configuration issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from config import config
config.validate()  # Will show detailed validation info
```

### Environment Variable Loading

Check if environment variables are loaded correctly:

```python
import os
from config import config

print("Environment variables:")
for key in ['OPENAI_API_KEY', 'TAVILY_API_KEY', 'LOG_LEVEL']:
    print(f"{key}: {'✓ Set' if getattr(config, key, None) else '✗ Missing'}")
```

## Migration Guide

### Upgrading Configuration

When upgrading to newer versions:

1. **Check new required variables** in the changelog
2. **Update your .env file** with new options
3. **Run validation** to ensure compatibility
4. **Test in development** before deploying to production

### Backward Compatibility

The configuration system maintains backward compatibility:

- Old variable names are supported with deprecation warnings
- Default values ensure existing installations continue working
- Migration helpers assist with configuration updates

## Examples

### Development Setup
```bash
cp .env.example .env
# Edit .env with your API keys
python -c "from config import config; config.validate()"
```

### Docker Configuration
```dockerfile
ENV OPENAI_API_KEY=sk-proj-your-key
ENV TAVILY_API_KEY=tvly-your-key
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV VECTOR_STORE_PATH=/app/data
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-server-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  VECTOR_STORE_PATH: "/data"
  SIMILARITY_THRESHOLD: "0.8"
```