# CI/CD and Deployment Guide

## 🚀 CI/CD Pipeline Overview

This project includes a comprehensive CI/CD pipeline using GitHub Actions that provides:

### ✅ **Pipeline Jobs**

1. **Code Quality & Security**
   - Code formatting (Black)
   - Import sorting (isort)
   - Linting (flake8)
   - Type checking (mypy)
   - Security scanning (bandit, safety)

2. **Testing Suite**
   - Unit tests with coverage (pytest)
   - Integration tests
   - Multi-Python version testing (3.8-3.11)
   - Coverage reporting to Codecov

3. **Performance Testing**
   - Benchmark tests (pytest-benchmark)
   - Load testing (locust)
   - Performance regression detection

4. **Build & Package**
   - Python package building
   - Docker multi-platform builds
   - Artifact management

5. **Documentation**
   - Sphinx documentation build
   - GitHub Pages deployment
   - API documentation generation

6. **Deployment**
   - PyPI package publishing
   - Docker Hub image publishing
   - Release automation

7. **Notifications**
   - Slack integration for CI/CD status
   - Email notifications for failures

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd rag-mcp-server

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run with Docker Compose
docker-compose up rag-mcp-server
```

### Docker Compose Profiles

```bash
# Development environment
docker-compose --profile dev up

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up

# With caching (Redis)
docker-compose --profile cache up

# With database (PostgreSQL)
docker-compose --profile database up

# All services
docker-compose --profile dev --profile monitoring --profile cache up
```

### Production Deployment

```bash
# Production configuration
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export OPENAI_API_KEY=your-key
export TAVILY_API_KEY=your-key

# Deploy to production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🛠️ Development Workflow

### Local Development Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd rag-mcp-server
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 2. Install dependencies (requires Python 3.12+)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 4. Run tests
pytest tests/ -v

# 5. Start development server
python src/mcp_server.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
safety check
```

## 🔧 Configuration

### Environment Variables

Required variables for production:
```bash
OPENAI_API_KEY=sk-proj-your-key
TAVILY_API_KEY=tvly-your-key
ENVIRONMENT=production
LOG_LEVEL=WARNING
```

Optional production optimizations:
```bash
SIMILARITY_THRESHOLD=0.8
MAX_CONCURRENCY=10
TAVILY_QUOTA_LIMIT=10000
ENABLE_CACHING=true
```

### GitHub Secrets

Configure these secrets in your GitHub repository:

```bash
# Required for CI/CD
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-password
PYPI_API_TOKEN=your-pypi-token

# Optional for notifications
SLACK_WEBHOOK_URL=your-slack-webhook
CODECOV_TOKEN=your-codecov-token
```

## 📊 Monitoring and Observability

### Prometheus Metrics

Access metrics at `http://localhost:9090` when using monitoring profile:

- Request latency and throughput
- Cache hit rates
- Error rates by component
- Resource utilization

### Grafana Dashboards

Access dashboards at `http://localhost:3000` (admin/admin):

- System performance overview
- API usage analytics
- Search result quality metrics
- Resource consumption trends

### Health Checks

```bash
# Docker health check
docker exec rag-mcp-server python -c "from src.config import config; config.validate()"

# Manual health check
curl -f http://localhost:8000/health || exit 1
```

## 🚀 Deployment Strategies

### 1. Development Deployment

```yaml
# .github/workflows/deploy-dev.yml
on:
  push:
    branches: [develop]

jobs:
  deploy-dev:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to development
        run: echo "Deploying to dev environment"
```

### 2. Staging Deployment

```yaml
# .github/workflows/deploy-staging.yml
on:
  push:
    branches: [main]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: echo "Deploying to staging environment"
```

### 3. Production Deployment

```yaml
# .github/workflows/deploy-prod.yml
on:
  release:
    types: [published]

jobs:
  deploy-prod:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: echo "Deploying to production"
```

## 🔄 Release Process

### Automated Release

1. Create a new tag: `git tag v1.0.0`
2. Push the tag: `git push origin v1.0.0`
3. Create a GitHub release
4. CI/CD automatically builds and deploys

### Manual Release

```bash
# 1. Build package
python -m build

# 2. Upload to PyPI
twine upload dist/*

# 3. Build Docker image
docker build -t ragmcp/rag-mcp-server:v1.0.0 .

# 4. Push to Docker Hub
docker push ragmcp/rag-mcp-server:v1.0.0
```

## 🧪 Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Component isolation
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (`tests/integration/`)
   - End-to-end workflows
   - Real API interactions
   - Database integration

3. **Performance Tests** (`tests/performance/`)
   - Benchmark critical paths
   - Load testing scenarios
   - Memory usage profiling

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Performance tests
pytest tests/performance/ --benchmark-only
```

## 📈 Performance Optimization

### Production Optimizations

1. **Caching Strategy**
   ```bash
   ENABLE_CACHING=true
   CACHE_TTL_HOURS=24
   MAX_CACHE_SIZE_MB=1000
   ```

2. **Concurrency Settings**
   ```bash
   MAX_CONCURRENCY=10
   RATE_LIMIT_REQUESTS=200
   ```

3. **Resource Limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

### Scaling Strategies

1. **Horizontal Scaling**
   - Load balancer + multiple instances
   - Shared cache (Redis)
   - External database

2. **Vertical Scaling**
   - Increase CPU/memory allocation
   - Optimize batch sizes
   - Tune concurrency limits

## 🔒 Security Considerations

### Production Security

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Container Security**
   - Non-root user
   - Minimal base image
   - Security scanning

3. **Network Security**
   - Internal network isolation
   - TLS encryption
   - Access controls

### Security Scanning

```bash
# Container security scan
docker scan ragmcp/rag-mcp-server:latest

# Dependency vulnerability scan
safety check

# Code security scan
bandit -r src/
```

This comprehensive CI/CD setup provides enterprise-grade deployment capabilities with automated testing, security scanning, performance monitoring, and multi-environment support.