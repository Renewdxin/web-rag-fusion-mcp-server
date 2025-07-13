# MCP智能RAG系统

一个符合标准的**模型上下文协议（MCP）服务器**，提供智能的检索增强生成（RAG）功能。该服务器可与任何MCP兼容的客户端（如Claude Desktop）一起使用，提供本地优先的知识搜索和智能网络搜索回退功能。

## 🔌 什么是MCP？

[模型上下文协议（MCP）](https://modelcontextprotocol.io/)是一个开放标准，使AI应用程序能够安全地连接到外部数据源和工具。与传统的API方法不同，MCP提供：

- **标准化通信**：基于JSON-RPC的AI工具交互协议
- **客户端无关**：适用于任何MCP兼容客户端
- **工具发现**：自动能力发现和模式验证
- **安全性**：对敏感数据和操作的受控访问

## 🚀 特性

- **标准兼容**：实现官方MCP规范
- **智能RAG搜索**：本地知识库搜索，带有相似度评分
- **自适应网络搜索**：当本地知识不足时自动回退到网络搜索
- **可配置阈值**：可自定义搜索决策的相似度阈值
- **多源信息整合**：智能组合本地和网络信息
- **来源归属**：清晰的引用和置信度分数
- **易于集成**：适用于Claude Desktop和其他MCP客户端

## 🏗️ 架构

```
MCP客户端 (Claude Desktop) ←→ MCP协议 ←→ RAG服务器
                                           ↓
                                    工具注册表
                                           ↓
                        ┌─────────────────────────────┐
                        │  search_knowledge_base      │
                        │  web_search                │
                        │  smart_search             │
                        └─────────────────────────────┘
                                           ↓
                        ┌─────────────────────────────┐
                        │  ChromaDB向量存储           │
                        │  Tavily网络搜索            │
                        └─────────────────────────────┘
```

## 📋 前置要求

- Python 3.9+
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- ChromaDB用于向量存储
- Tavily API密钥（用于网络搜索）
- OpenAI API密钥（用于嵌入向量，可选）

## 🛠️ 安装

### 1. 克隆和设置

```bash
git clone https://github.com/yourusername/mcp-rag-server.git
cd mcp-rag-server

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装MCP SDK

```bash
pip install mcp
```

### 3. 配置环境

```bash
cp .env.example .env
# 编辑.env文件配置您的设置
```

## ⚙️ 配置

### 环境变量

创建 `.env` 文件：

```env
# 必需：网络搜索API
TAVILY_API_KEY=tvly-your-tavily-api-key

# 可选：嵌入向量（如果使用OpenAI嵌入）
OPENAI_API_KEY=sk-your-openai-key

# 向量存储配置
VECTOR_STORE_PATH=./vector_store
COLLECTION_NAME=knowledge_base

# MCP服务器设置
MCP_SERVER_NAME=rag-agent
SIMILARITY_THRESHOLD=0.75

# 日志记录
LOG_LEVEL=INFO
```

### 关键配置参数

- `SIMILARITY_THRESHOLD`：触发网络搜索的分数阈值（0-1，默认：0.75）
- `VECTOR_STORE_PATH`：ChromaDB存储目录路径
- `TAVILY_API_KEY`：网络搜索功能必需

## 🚀 快速开始

### 1. 初始化知识库

```python
# scripts/init_knowledge_base.py
from vector_store import VectorStoreManager
from document_loader import load_documents

# 初始化向量存储
vector_manager = VectorStoreManager("./vector_store")
await vector_manager.initialize_collection("knowledge_base")

# 加载您的文档
documents = load_documents("./data/")
await vector_manager.add_documents(documents)
```

运行初始化：

```bash
python scripts/init_knowledge_base.py
```

### 2. 测试MCP服务器

```bash
# 测试服务器功能
python mcp_server.py
```

### 3. 配置MCP客户端

对于**Claude Desktop**，添加到您的 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/mcp-rag-server",
      "env": {
        "TAVILY_API_KEY": "your-tavily-key"
      }
    }
  }
}
```

**Claude Desktop配置文件位置：**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## 🔧 可用工具

### 1. `search_knowledge_base`
使用相似度评分搜索本地向量数据库。

**输入：**
```json
{
  "query": "string (必需)",
  "top_k": "integer (可选, 默认: 5)"
}
```

**在Claude中的使用示例：**
```
请搜索我们的知识库以获取API文档信息。
```

### 2. `web_search`
当本地知识不足时使用Tavily API搜索网络。

**输入：**
```json
{
  "query": "string (必需)",
  "max_results": "integer (可选, 默认: 5)"
}
```

### 3. `smart_search`（推荐）
智能搜索，首先尝试本地知识，然后在需要时进行网络搜索。

**输入：**
```json
{
  "query": "string (必需)",
  "similarity_threshold": "number (可选, 默认: 0.75)",
  "local_top_k": "integer (可选, 默认: 5)",
  "web_max_results": "integer (可选, 默认: 5)"
}
```

**在Claude中的使用示例：**
```
我需要关于我们产品定价的全面信息。使用智能搜索首先检查内部文档，然后在需要时搜索网络。
```

## 📚 使用示例

### 基础知识搜索

```
人类：我们公司的核心价值观是什么？

Claude：我将搜索您的知识库以获取公司核心价值观的信息。

[使用search_knowledge_base工具]

根据您的内部文档，您公司的核心价值观是：
1. 客户至上 - 在所有决策中优先考虑客户需求
2. 创新 - 持续改进并拥抱新技术
3. 诚信 - 在所有商业实践中维护道德标准
...

来源：employee_handbook.pdf（相似度：0.94）
```

### 智能搜索回退

```
人类：2024年AI发展的最新趋势是什么？

Claude：我将使用智能搜索首先检查您的知识库，然后在需要时补充当前的网络信息。

[使用smart_search工具]

本地知识结果（最高分数：0.45）
- 找到一些通用AI信息，但分数低于阈值（0.75）

网络搜索结果：
1. "2024年顶级AI趋势" - TechCrunch
   - 生成式AI在企业中的采用
   - 多模态AI系统
   - AI安全和监管发展
...

由于本地知识不包含当前AI趋势信息，我提供了来自网络来源的最新信息。
```

## 🧪 测试

### 运行单元测试

```bash
pytest tests/unit/ -v
```

### 测试MCP协议合规性

```bash
python tests/test_mcp_protocol.py
```

### 集成测试

```bash
# 使用实际MCP客户端测试
python tests/integration/test_claude_desktop.py
```

## 🐳 Docker部署

### 构建和运行

```bash
# 构建镜像
docker build -t mcp-rag-server .

# 使用环境文件运行
docker run --env-file .env -v $(pwd)/vector_store:/app/vector_store mcp-rag-server
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-rag-server:
    build: .
    volumes:
      - ./vector_store:/app/vector_store
      - ./data:/app/data
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

运行：
```bash
docker-compose up -d
```

## 📊 监控和调试

### 启用调试日志

```bash
export LOG_LEVEL=DEBUG
python mcp_server.py
```

### 监控工具使用

服务器记录所有工具调用及其结果：

```
2024-01-15 10:30:15 - INFO - 工具调用: smart_search
2024-01-15 10:30:15 - INFO - 查询: "公司Q1收入"
2024-01-15 10:30:16 - INFO - 本地搜索最高分数: 0.92
2024-01-15 10:30:16 - INFO - 决策: 本地知识足够
```

### 性能指标

检查服务器性能：

```python
# 在您的客户端中
# 监控响应时间和成功率
```

## 🔍 故障排除

### 常见问题

1. **"未找到MCP服务器"**
   ```bash
   # 检查Claude Desktop配置路径
   # 确保使用绝对路径
   # 验证Python环境
   ```

2. **"ChromaDB未初始化"**
   ```bash
   python scripts/init_knowledge_base.py
   ```

3. **"Tavily API密钥无效"**
   ```bash
   # 检查.env文件
   # 验证API密钥格式：tvly-...
   ```

4. **"无相似性结果"**
   ```bash
   # 检查文档是否正确索引
   # 验证嵌入模型兼容性
   # 临时降低相似度阈值
   ```

### 调试模式

```bash
# 在调试模式下运行
export LOG_LEVEL=DEBUG
export MCP_DEBUG=true
python mcp_server.py
```

### 检查MCP客户端连接

对于Claude Desktop，检查日志：
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%LOCALAPPDATA%\Claude\logs\`

## 📈 高级用法

### 自定义嵌入模型

```python
# custom_embeddings.py
from chromadb.utils import embedding_functions

# 使用自定义嵌入函数
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

### 多集合支持

```python
# 支持多个知识库
collections = {
    "technical_docs": "技术文档",
    "company_policies": "公司政策",
    "product_specs": "产品规格"
}
```

### 按主题自定义相似度阈值

```python
# 动态阈值调整
topic_thresholds = {
    "technical": 0.80,      # 技术查询的更高阈值
    "general": 0.70,        # 一般查询的较低阈值
    "current_events": 0.60  # 时效性查询的更低阈值
}
```

## 🤝 贡献

### 开发设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit钩子
pre-commit install

# 运行代码格式化
black .
isort .
```

### 添加新工具

1. 在`tool_schemas.py`中定义工具模式
2. 在`mcp_server.py`中实现工具处理程序
3. 在`tests/`中添加测试
4. 更新文档

### 提交更改

1. Fork仓库
2. 创建功能分支：`git checkout -b feature-name`
3. 进行更改并添加测试
4. 运行测试套件：`pytest`
5. 提交拉取请求

## 📄 许可证

该项目基于MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 🙏 致谢

- [模型上下文协议](https://modelcontextprotocol.io/)提供的开放标准
- [ChromaDB](https://www.trychroma.com/)提供向量存储
- [Tavily](https://tavily.com/)提供网络搜索功能
- [Anthropic](https://www.anthropic.com/)开发Claude和MCP

## 📞 支持

- **问题**：[GitHub Issues](https://github.com/yourusername/mcp-rag-server/issues)
- **讨论**：[GitHub Discussions](https://github.com/yourusername/mcp-rag-server/discussions)
- **MCP文档**：[https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)

## 🗺️ 路线图

- [ ] **多客户端支持**：支持除Claude Desktop之外的更多MCP客户端
- [ ] **高级向量存储**：Pinecone、Weaviate、Qdrant集成
- [ ] **文档处理管道**：自动化文档摄取和处理
- [ ] **语义缓存**：搜索结果的智能缓存
- [ ] **多语言支持**：支持非英语知识库
- [ ] **图RAG**：与知识图谱的集成
- [ ] **实时更新**：实时文档同步
- [ ] **分析仪表板**：使用分析和性能监控

## 🔧 requirements.txt

```txt
# 核心MCP依赖
mcp>=1.0.0

# 向量存储和搜索
chromadb>=0.4.15
sentence-transformers>=2.2.2

# 网络搜索
tavily-python>=0.3.0
requests>=2.31.0

# 文档处理
langchain>=0.1.0
langchain-community>=0.0.10

# 嵌入和AI
openai>=1.3.0
tiktoken>=0.5.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0

# 配置和环境
python-dotenv>=1.0.0
pydantic>=2.0.0

# 日志和监控
structlog>=23.1.0
prometheus-client>=0.17.0

# 测试（开发依赖）
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
pre-commit>=3.3.0

# 可选：缓存
redis>=4.6.0

# 可选：数据库
sqlite3
```

## 📝 安装脚本

```bash
#!/bin/bash
# install.sh

echo "🚀 安装MCP RAG服务器..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 需要Python 3.9+，当前版本：$python_version"
    exit 1
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装依赖
echo "⬇️  安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建配置文件
if [ ! -f .env ]; then
    echo "⚙️  创建配置文件..."
    cp .env.example .env
    echo "✏️  请编辑 .env 文件并添加您的API密钥"
fi

# 创建向量存储目录
mkdir -p vector_store
mkdir -p data
mkdir -p logs

echo "✅ 安装完成！"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件并添加您的API密钥"
echo "2. 将您的文档放在 ./data/ 目录中"
echo "3. 运行: python scripts/init_knowledge_base.py"
echo "4. 启动服务器: python mcp_server.py"
echo ""
echo "📖 更多信息请查看 README.md"
```

---

**使用模型上下文协议构建 ❤️**
