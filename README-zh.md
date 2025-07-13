# RAG MCP 服务器

一个生产就绪的模型上下文协议 (MCP) 服务器，通过结合本地向量数据库搜索和智能网络搜索，提供智能的检索增强生成能力。

## 🚀 特性

### 🔍 **三重搜索策略**
- **知识库搜索**: 跨本地文档的语义相似性搜索
- **网络搜索**: 带内容过滤和优化的实时互联网搜索
- **智能搜索**: 结合两种来源的智能混合搜索，获得全面结果

### 🎯 **高级能力**
- **多格式文档处理**: PDF、TXT、MD、DOCX、HTML，支持智能分块
- **语义搜索**: OpenAI 嵌入与 ChromaDB，实现准确的内容检索
- **内容智能**: 自动广告过滤、质量评分和相关性排名
- **性能优化**: 多级缓存、连接池和异步操作
- **生产就绪**: 全面的错误处理、监控和安全特性

### 🛠️ **开发者体验**
- **MCP 协议**: 完整的模型上下文协议兼容性，无缝集成
- **丰富格式**: 美观的搜索结果，支持语法高亮和元数据
- **进度跟踪**: 长时间运行操作的实时进度
- **全面日志**: 结构化日志记录，支持请求追踪和性能指标

## 📋 快速开始

### 前置要求
- Python 3.12+
- OpenAI API 密钥（用于嵌入）
- Tavily API 密钥（用于网络搜索）

### 安装

1. **克隆仓库**
```bash
git clone <repository-url>
cd rag-mcp-server
```

2. **安装依赖**
```bash
pip install -r requirements.txt

# 可选：安装其他格式支持
pip install python-docx beautifulsoup4 PyPDF2
```

3. **配置环境**
```bash
# 创建 .env 文件
cat > .env << EOF
# 必需的 API 密钥
OPENAI_API_KEY=sk-proj-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# 可选配置
ENVIRONMENT=development
LOG_LEVEL=INFO
VECTOR_STORE_PATH=./data
SIMILARITY_THRESHOLD=0.75
EOF
```

4. **运行服务器**
```bash
python src/mcp_server.py
```

### MCP 客户端集成

#### Claude Desktop
添加到您的 Claude Desktop 配置：
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp-server/src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "TAVILY_API_KEY": "your-tavily-key"
      }
    }
  }
}
```

## 🔧 可用工具

### 1. 🔍 search_knowledge_base
使用语义相似性搜索您的本地文档集合。

**参数：**
- `query`（必需）：搜索查询字符串（1-1000 字符）
- `top_k`（可选）：返回结果数量（1-20，默认：5）
- `filter_dict`（可选）：用于精细搜索的元数据过滤器
- `include_metadata`（可选）：包含文档元数据（默认：true）

**示例：**
```json
{
  "query": "机器学习算法",
  "top_k": 10,
  "filter_dict": {"file_type": "pdf"},
  "include_metadata": true
}
```

**响应特性：**
- 🟢🟡🔴 颜色编码的相似度分数
- **粗体关键词高亮**
- 📁 可点击的源文件路径
- ⏱️ 搜索执行时间
- 📊 全面的元数据显示

### 2. 🌐 web_search
使用智能内容过滤搜索互联网。

**参数：**
- `query`（必需）：搜索查询字符串（1-400 字符）
- `max_results`（可选）：结果数量（1-20，默认：5）
- `search_depth`（可选）："basic" 或 "advanced"（默认："basic"）
- `include_answer`（可选）：包含 AI 生成的摘要（默认：true）
- `include_raw_content`（可选）：包含原始网页内容（默认：false）
- `exclude_domains`（可选）：要排除的域名列表

**示例：**
```json
{
  "query": "2024年最新AI发展",
  "max_results": 8,
  "search_depth": "advanced",
  "exclude_domains": ["example.com"]
}
```

**高级特性：**
- 🚫 自动广告内容过滤
- ✅ 内容质量评分（0.0-1.0）
- 📋 1小时 TTL 结果缓存
- 🎯 查询优化，包含停用词移除
- 📊 API 配额跟踪和管理

### 3. 🧠 smart_search
结合本地知识和网络搜索的智能混合搜索。

**参数：**
- `query`（必需）：搜索查询字符串
- `local_max_results`（可选）：最大本地结果数（1-20，默认：5）
- `web_max_results`（可选）：最大网络结果数（0-10，默认：3）
- `local_threshold`（可选）：本地相似度阈值（0.0-1.0，默认：0.7）
- `min_local_results`（可选）：网络搜索前的最小本地结果数（0-10，默认：2）
- `combine_strategy`（可选）："interleave"、"local_first" 或 "relevance_score"
- `include_sources`（可选）：包含源信息（默认：true）

**工作原理：**
1. 🔍 首先搜索本地知识库
2. 📊 评估结果质量和覆盖度
3. 🌐 根据需要补充网络搜索
4. 🧠 智能组合和排名结果
5. 📈 返回综合排名结果

## 📁 文档处理

### 支持的格式
| 格式 | 扩展名 | 特性 |
|--------|------------|------------|
| **PDF** | `.pdf` | 文本提取、元数据解析、多页支持 |
| **文本** | `.txt`, `.md` | 编码检测、结构保持 |
| **Word** | `.docx` | 内容提取、文档属性 |
| **HTML** | `.html`, `.htm` | 清洁文本提取、元数据提取 |

### 添加文档

#### 方法 1：直接处理
```python
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

# 初始化组件
processor = DocumentProcessor(chunk_size=1000, overlap=200)
vector_store = VectorStoreManager()

# 处理并添加文档
processed_docs = await processor.process_file(Path("document.pdf"))
documents = processor.convert_to_documents(processed_docs)
await vector_store.add_documents(documents)
```

#### 方法 2：批量目录处理
```python
# 处理整个目录，带进度跟踪
processed_docs, stats = await processor.process_directory(
    Path("./documents"),
    recursive=True,
    progress_callback=lambda current, total, status: print(f"{current}/{total}: {status}")
)

print(f"已处理 {stats.processed_files} 个文件，{stats.total_chunks} 个块")
```

## ⚙️ 配置

### 环境变量

#### 必需
```bash
OPENAI_API_KEY=sk-proj-your-openai-key    # OpenAI API 密钥用于嵌入
TAVILY_API_KEY=tvly-your-tavily-key       # Tavily API 密钥用于网络搜索
```

#### 可选
```bash
# 服务器配置
ENVIRONMENT=development                    # development, staging, production
LOG_LEVEL=INFO                            # DEBUG, INFO, WARNING, ERROR

# 存储和数据库
VECTOR_STORE_PATH=./data                  # 向量数据库存储路径
COLLECTION_NAME=rag_documents             # ChromaDB 集合名称

# 搜索参数
SIMILARITY_THRESHOLD=0.75                 # 最小相似度分数（0.0-1.0）
MAX_RESULTS_DEFAULT=10                    # 默认搜索结果数

# 性能设置
MAX_RETRIES=3                             # API 重试次数
TIMEOUT_SECONDS=30                        # 通用请求超时
WEB_SEARCH_TIMEOUT=45                     # 网络搜索特定超时
MAX_CONCURRENCY=5                         # 文档处理并发数

# API 配额
TAVILY_QUOTA_LIMIT=1000                   # 每日 Tavily API 配额限制
```

### 开发环境 vs 生产环境

**开发环境：**
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
SIMILARITY_THRESHOLD=0.6  # 测试时较低
VECTOR_STORE_PATH=./dev_data
```

**生产环境：**
```bash
ENVIRONMENT=production
LOG_LEVEL=WARNING
SIMILARITY_THRESHOLD=0.8  # 质量要求更高
VECTOR_STORE_PATH=/opt/rag-server/data
TAVILY_QUOTA_LIMIT=10000  # 更高配额
```

## 🏗️ 架构

### 核心组件

```
RAGMCPServer（主要协调器）
├── VectorStoreManager（本地搜索）
│   ├── ChromaDB（向量数据库）
│   ├── OpenAI Embeddings（文本→向量）
│   ├── EmbeddingCache（性能）
│   └── DocumentProcessor（内容验证）
├── WebSearchManager（互联网搜索）
│   ├── Tavily API（搜索服务）
│   ├── QueryOptimizer（查询增强）
│   ├── ContentFilter（质量控制）
│   ├── SearchCache（1小时 TTL）
│   └── UsageTracker（配额管理）
└── DocumentProcessor（多格式支持）
    ├── PDFLoader（PDF 处理）
    ├── TextLoader（文本/Markdown）
    ├── DocxLoader（Word 文档）
    ├── HTMLLoader（网页内容）
    ├── TextChunker（智能分割）
    ├── MetadataExtractor（文件信息）
    └── ProcessingCache（避免重复处理）
```

### 关键设计模式
- **单例模式**: 配置管理
- **工厂模式**: 不同格式的文档加载器
- **策略模式**: 可插拔的搜索算法
- **观察者模式**: 进度跟踪和通知
- **适配器模式**: 外部 API 集成

## 🚀 性能特性

### 缓存策略
- **L1 内存缓存**: 最近的查询和结果
- **L2 SQLite 缓存**: 带 TTL 的持久化缓存
- **L3 文件缓存**: 文档处理结果

### 优化技术
- **异步/等待**: 非阻塞 I/O 操作
- **连接池**: 高效的数据库连接
- **批处理**: 同时处理多个文档
- **内容去重**: 基于 SHA-256 哈希的重复检测
- **查询优化**: 停用词移除和关键短语提取

### 性能基准
- **本地搜索**: ~50ms 平均响应时间
- **网络搜索**: ~1.2s 平均响应时间
- **缓存命中**: <10ms 响应时间
- **文档处理**: 5-10 个文件并发处理
- **缓存命中率**: 重复操作 >85%

## 🔒 安全与可靠性

### 安全特性
- **API 密钥管理**: 基于环境变量，无硬编码凭据
- **输入验证**: 所有工具输入的 JSON Schema 验证
- **速率限制**: 令牌桶算法，可配置限制
- **SQL 注入防护**: 全程参数化查询
- **内容净化**: 安全处理用户提供的内容

### 错误处理
- **指数退避**: API 失败的智能重试逻辑
- **优雅降级**: 服务不可用时的后备策略
- **全面日志**: 带请求追踪的结构化日志
- **用户友好消息**: 将技术错误转换为可操作的反馈

### 监控
- **性能指标**: 请求延迟、缓存命中率、错误率
- **资源监控**: 内存使用、连接数、队列大小
- **API 使用跟踪**: 配额管理和使用分析
- **健康检查**: 自动系统健康验证

## 🛠️ 开发

### 项目结构
```
rag-mcp-server/
├── src/
│   ├── mcp_server.py          # 主 MCP 服务器实现
│   ├── vector_store.py        # 向量数据库管理
│   ├── web_search.py          # Tavily API 网络搜索
│   ├── document_processor.py  # 多格式文档处理
│   └── config/
│       └── settings.py        # 配置管理
├── docs/                      # 全面文档
├── tests/                     # 测试套件
├── requirements.txt           # Python 依赖
├── .env.example              # 环境模板
└── README.md                 # 本文件
```

### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-asyncio pytest-mock

# 运行测试套件
pytest tests/ -v

# 运行带覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 开发设置
```bash
# 克隆和设置
git clone <repository-url>
cd rag-mcp-server
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# 配置环境
cp .env.example .env
# 编辑 .env 文件添加您的 API 密钥

# 验证安装
python -c "from config import config; config.validate()"
```

## 📊 使用示例

### 搜索结果格式

**知识库搜索：**
```
🔍 发现 3 个关于"机器学习算法"的结果

**1. 🟢 相似度: 0.892**
📂 来源: [研究论文.pdf](./docs/研究论文.pdf)

📖 内容:
**机器学习** **算法** 是能够使系统从数据中学习模式的计算方法...

ℹ️ 元数据:
📄 来源: ./docs/研究论文.pdf
📄 文件名: 研究论文.pdf  
📄 文件类型: pdf

⏱️ 搜索完成于 0.234 秒
🎯 使用的关键词: 机器, 学习, 算法
```

**网络搜索：**
```
🌐 发现 5 个关于"2024年最新AI发展"的网络结果

**1. 🟢 分数: 0.945**
📰 标题: 突破性AI模型改变行业
🌐 来源: [techcrunch.com](https://techcrunch.com/article)

📄 内容:
2024年的主要 **AI** 突破包括先进的语言模型...

✅ 质量分数: 0.89

⏱️ 搜索完成于 1.234 秒
🔄 来自网络的新鲜结果
📊 API 使用: 每日配额的 15.2%
```

## 🔧 故障排除

### 常见问题

**配置错误：**
```bash
错误：未找到 OPENAI_API_KEY
解决方案：设置环境变量或添加到 .env 文件
```

**API 配额超出：**
```bash
错误：每日配额已超出
解决方案：等待重置（UTC午夜）或升级 API 计划
```

**向量存储连接：**
```bash
错误：ChromaDB 连接失败
解决方案：检查权限并确保 ./data 目录存在
```

**文档处理：**
```bash
错误：不支持的文件格式
解决方案：使用 processor.get_supported_formats() 检查支持的格式
```

### 调试模式
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python src/mcp_server.py 2>&1 | tee debug.log
```

## 📚 文档

- **[用户指南](docs/USER_GUIDE.md)**: 完整的用户文档和示例
- **[API 参考](docs/API.md)**: 所有组件的详细 API 文档
- **[架构指南](docs/ARCHITECTURE.md)**: 系统设计和组件关系
- **[配置指南](docs/CONFIG.md)**: 全面的配置选项
- **[技术文档](TECH.md)**: 深入的技术实现细节

## 🤝 贡献

我们欢迎贡献！请查看我们的贡献指南：

1. Fork 仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交您的更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 打开 Pull Request

### 开发指南
- 遵循 Python PEP 8 风格指南
- 为新功能添加全面测试
- 为 API 更改更新文档
- 确保所有测试在提交 PR 前通过

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **模型上下文协议 (MCP)**: 提供基础协议
- **ChromaDB**: 高效的向量数据库能力
- **OpenAI**: 强大的嵌入模型
- **Tavily**: 智能网络搜索 API
- **Python 社区**: 出色的异步和数据处理库

## 📞 支持

- **GitHub Issues**: [报告错误和请求功能](https://github.com/your-repo/issues)
- **讨论**: [提问和分享想法](https://github.com/your-repo/discussions)
- **文档**: [全面指南和 API 参考](docs/)

---

**准备好增强您的搜索能力了吗？** 🚀

立即开始使用 RAG MCP 服务器，体验智能混合搜索的强大功能，结合本地知识和实时网络信息的最佳效果！
