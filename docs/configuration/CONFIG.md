# Project Configuration Guide

This document provides instructions on how to configure and run the RAG MCP Server project using environment variables and Docker.

## Quick Start

1.  **Copy the environment file**: Start by creating a local configuration file from the example template.

    ```bash
    cp .env.example .env
    ```

2.  **Edit `.env`**: Open the newly created `.env` file and fill in the required API keys:

    - `OPENAI_API_KEY`: Your API key for OpenAI services.
    - `TAVILY_API_KEY`: Your API key for the Tavily search service.

3.  **Run with Docker**: Use Docker Compose to build and run the application. For development, use the `dev` profile:

    ```bash
    docker-compose --profile dev up --build
    ```

    For production:

    ```bash
    docker-compose up --build -d
    ```

## Environment Variables

The `.env` file is the central place for managing all environment variables. Below is a description of each variable.

### Required API Keys

- `OPENAI_API_KEY`: **Required**. Used for accessing OpenAI's embedding and language models.
- `TAVILY_API_KEY`: **Required**. Used for the web search functionality provided by Tavily.

### Server Configuration

- `ENVIRONMENT`: The application environment. Can be `development`, `staging`, or `production`. Defaults to `development`.
- `LOG_LEVEL`: The logging level for the application. Can be `DEBUG`, `INFO`, `WARNING`, or `ERROR`. Defaults to `INFO`.

### Storage Configuration

- `VECTOR_STORE_PATH`: The file system path where the vector database will be stored. Defaults to `./data`.
- `COLLECTION_NAME`: The name of the collection within ChromaDB. Defaults to `rag_documents`.

### Search Parameters

- `SIMILARITY_THRESHOLD`: The minimum similarity score for search results, ranging from 0.0 to 1.0. Defaults to `0.75`.
- `MAX_RESULTS_DEFAULT`: The default number of search results to return. Defaults to `10`.

### Performance Settings

- `MAX_RETRIES`: The maximum number of times to retry failed API calls. Defaults to `3`.
- `TIMEOUT_SECONDS`: The general timeout for requests in seconds. Defaults to `30`.
- `WEB_SEARCH_TIMEOUT`: A specific timeout for web search operations. Defaults to `45`.
- `MAX_CONCURRENCY`: The maximum number of documents to process concurrently. Defaults to `5`.

### API Quotas and Limits

- `TAVILY_QUOTA_LIMIT`: The daily limit for Tavily API calls. Defaults to `1000`.
- `RATE_LIMIT_REQUESTS`: The maximum number of requests allowed per time window for rate limiting. Defaults to `100`.
- `RATE_LIMIT_WINDOW`: The time window in seconds for rate limiting. Defaults to `60`.

### Docker-Specific Settings

- `BUILD_DATE`: The build date for the Docker image. Automatically set to the current UTC time.
- `VERSION`: The version of the Docker image. Defaults to `latest`.
- `VCS_REF`: The Git commit hash for the Docker image. Automatically set to the current HEAD.
- `GRAFANA_USER`: The admin username for Grafana. Defaults to `admin`.
- `GRAFANA_PASSWORD`: The admin password for Grafana. Defaults to `admin`.

## Docker Usage

The `docker-compose.yml` file defines two main services: `rag-mcp-server` for production and `rag-mcp-dev` for development.

### Development Environment

To run the development environment, use the `dev` profile:

```bash
docker-compose --profile dev up --build
```

This will start a container with the source code mounted, allowing for live code reloading. You can access the container with:

```bash
docker-compose exec rag-mcp-dev /bin/bash
```

### Production Environment

To run the production environment:

```bash
docker-compose up --build -d
```

This will start the server in detached mode.

### Monitoring

The `docker-compose.yml` also includes optional services for monitoring with Prometheus and Grafana. To start them, use the `monitoring` profile:

```bash
docker-compose --profile monitoring up --build -d
```
