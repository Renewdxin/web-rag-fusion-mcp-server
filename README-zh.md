# RAG MCP 服务器

RAG MCP 服务器是一个功能强大、生产就绪的服务器，可通过先进的检索增强生成（RAG）能力来强化语言模型。它智能地结合了跨本地文档的语义搜索和实时网络搜索，为您的模型提供全面而准确的上下文。

## 核心功能

- **知识库搜索**: 在您的本地文档中查找相关信息。
- **网络搜索**: 从互联网获取最新信息。
- **智能搜索**: 一种混合方法，同时使用知识库和网络搜索以获得最相关的结果。

## 快速上手

### 环境要求

- Python 3.9+
- Docker (推荐，便于快速部署)
- 一个 [OpenAI API 密钥](https://platform.openai.com/api-keys)
- 一个 [Tavily API 密钥](https://tavily.com/#api)

### 安装与运行

#### Docker (推荐)

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/rag-mcp-server.git
    cd rag-mcp-server
    ```

2.  **创建您的环境文件:**
    ```bash
    cp .env.example .env
    ```
    打开 `.env` 文件并填入您的 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`。

3.  **使用 Docker Compose 构建并运行:**
    ```bash
    docker-compose up --build
    ```
    服务器将在 `http://localhost:8000` 上可用。

#### 本地安装

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/rag-mcp-server.git
    cd rag-mcp-server
    ```

2.  **创建虚拟环境并安装依赖:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **创建您的环境文件:**
    ```bash
    cp .env.example .env
    ```
    打开 `.env` 文件并填入您的 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`。

4.  **运行服务器:**
    ```bash
    python3 src/mcp_server.py
    ```
    服务器将在 `http://localhost:8000` 上可用。

## 如何使用

您可以使用任何 HTTP 客户端（如 `curl` 或 Postman）与服务器进行交互。服务器通过单个 `/mcp` 端点接收 POST 请求。

### 请求格式

请求体应为一个 JSON 对象，其结构如下:

```json
{
  "tool_name": "要使用的工具名称",
  "parameters": {
    "参数1": "值1",
    "参数2": "值2"
  }
}
```

### 可用工具

#### 1. `search_knowledge_base`

在您的本地知识库中搜索文档。

**示例:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "search_knowledge_base",
           "parameters": {
             "query": "AI领域的最新进展是什么？",
             "top_k": 5
           }
         }'
```

#### 2. `web_search`

使用 Tavily API 执行网络搜索。

**示例:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "web_search",
           "parameters": {
             "query": "旧金山今天的天气怎么样？",
             "max_results": 3
           }
         }'
```

#### 3. `smart_search`

结合本地知识库和网络搜索，提供全面的结果。

**示例:**

```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{
           "tool_name": "smart_search",
           "parameters": {
             "query": "对比我们的内部销售数据和公开市场趋势。"
           }
         }'
```

## 配置

服务器通过 `.env` 文件进行配置。以下是一些关键设置：

- `OPENAI_API_KEY`: **(必需)** 您的 OpenAI API 密钥，用于文档嵌入。
- `TAVILY_API_KEY`: **(必需)** 您的 Tavily API 密钥，用于网络搜索。
- `VECTOR_STORE_PATH`: 存储知识库向量的本地目录。默认为 `./vector_store`。
- `LOG_LEVEL`: 服务器的日志级别。默认为 `INFO`。

更多高级配置选项，请参阅 `config/settings.py` 文件。