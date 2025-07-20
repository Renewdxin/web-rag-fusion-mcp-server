# Simplified Docker build for RAG MCP Server
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set Chinese mirror proxies for faster package installation
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Configure APT to use Chinese mirrors for faster package downloads
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# Install minimal system dependencies including build tools
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/ 
COPY *.md ./

# Skip tests in build stage since tests directory may not exist
# Tests will be run in CI/CD pipeline instead

# Production stage
FROM python:3.12-slim as production

# Set metadata labels
LABEL maintainer="RAG MCP Server Team" \
      org.opencontainers.image.title="RAG MCP Server" \
      org.opencontainers.image.description="Production-ready Model Context Protocol server with RAG capabilities" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/your-org/rag-mcp-server"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    VECTOR_STORE_PATH=/app/data \
    COLLECTION_NAME=rag_documents \
    SIMILARITY_THRESHOLD=0.75

# Configure APT to use Chinese mirrors for production stage
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Create app directory and data directories
WORKDIR /app
RUN mkdir -p /app/data /app/logs && \
    chown -R raguser:raguser /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=raguser:raguser src/ ./src/
COPY --chown=raguser:raguser config/ ./config/
COPY --chown=raguser:raguser *.md ./
COPY --chown=raguser:raguser requirements.txt ./

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Function to log with timestamp to stderr
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Health check function - redirect output to stderr
health_check() {
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from config.settings import config
    config.validate()
    print('Configuration validation passed', file=sys.stderr)
except Exception as e:
    print(f'Configuration validation failed: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Initialize application - all logs to stderr
log "🚀 Starting RAG MCP Server..."
log "Environment: ${ENVIRONMENT}"
log "Log Level: ${LOG_LEVEL}"
log "Vector Store Path: ${VECTOR_STORE_PATH}"
log "Python Version: $(python --version)"

# Validate configuration
log "🔧 Validating configuration..."
health_check

# Ensure data directory exists with proper permissions
mkdir -p "${VECTOR_STORE_PATH}" 2>&1
chown raguser:raguser "${VECTOR_STORE_PATH}" 2>&1

# Start the server
log "🌟 Launching MCP server..."
exec python -m src.mcp_server "$@"
EOF

RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from config.settings import config; config.validate()" || exit 1

# Expose port (if needed for future HTTP interface)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

# Development stage (for local development)
FROM production as development

USER root

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-mock black isort mypy flake8

# Switch back to raguser
USER raguser

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]