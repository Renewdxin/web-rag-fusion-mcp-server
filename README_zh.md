# 多提供者 RAG 服务器

[![Release](https://img.shields.io/github/v/release/Renewdxin/mcp)](https://github.com/Renewdxin/mcp/releases)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 中文文档 | [English](README.md)

一个功能强大的 RAG（检索增强生成）服务器，通过模型上下文协议（MCP）支持**动态嵌入提供者**。无需修改代码即可在 OpenAI 和 DashScope/通义千问提供者之间实时切换。

## ✨ 核心特性

- 🔄 **动态提供者切换** - OpenAI 和 DashScope 之间运行时切换
- 🏗️ **多索引支持** - 不同文档集合使用不同提供者
- 🛡️ **强大错误处理** - 自动降级和全面错误恢复
- 🌐 **网络搜索集成** - 通过 Perplexity/Exa API 增强搜索
- ⚙️ **环境变量配置** - 零代码配置变更
- 🚀 **生产就绪** - 限流、监控和度量

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/Renewdxin/multi-provider-rag.git
cd multi-provider-rag
pip install -r requirements.txt
```

### 2. 配置

复制并配置环境变量：

```bash
cp .env.example .env
# 编辑 .env 文件填入你的 API 密钥
```

**基础 OpenAI 配置：**
```bash
EMBED_PROVIDER=openai
OPENAI_API_KEY=你的_openai_密钥
EMBEDDING_MODEL=text-embedding-3-small
```

**DashScope/通义千问配置：**
```bash
EMBED_PROVIDER=dashscope
DASHSCOPE_API_KEY=你的_dashscope_密钥
EMBEDDING_MODEL=text-embedding-v1
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 运行服务器

```bash
python -m src.mcp_server
```

## 🔧 使用示例

### 动态提供者选择

```python
from src.embedding_provider import get_embed_model

# 使用 OpenAI
openai_model = get_embed_model("openai")

# 使用 DashScope
dashscope_model = get_embed_model("dashscope", model="text-embedding-v1")

# 基于环境变量选择
embed_model = get_embed_model_from_env()  # 使用 EMBED_PROVIDER
```

### 多提供者索引

```python
from src.embedding_provider import create_index_with_provider

# 创建专用索引
docs_index = create_index_with_provider("openai", documents)
code_index = create_index_with_provider("dashscope", code_docs)
```

## 🌐 支持的提供者

| 提供者 | 模型 | 端点 | 特性 |
|--------|------|------|------|
| **OpenAI** | `text-embedding-ada-002`<br>`text-embedding-3-small`<br>`text-embedding-3-large` | `https://api.openai.com/v1` | 高质量，全球可用 |
| **DashScope** | `text-embedding-v1`<br>`text-embedding-v2` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 中国优化，成本效益 |

## ⚙️ 配置

### 环境变量

| 变量 | 描述 | 默认值 | 必需 |
|------|------|--------|------|
| `EMBED_PROVIDER` | 嵌入提供者 (`openai`/`dashscope`) | `openai` | 否 |
| `EMBEDDING_MODEL` | 模型名称（提供者特定） | `text-embedding-3-small` | 否 |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - | 是（OpenAI） |
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | - | 是（DashScope） |
| `SEARCH_API_KEY` | Perplexity/Exa API 密钥 | - | 可选 |
| `VECTOR_STORE_PATH` | 向量数据库路径 | `./data/vector_store.db` | 否 |

### 提供者切换

通过更新环境变量即时切换提供者：

```bash
# 切换到 DashScope
export EMBED_PROVIDER=dashscope
export DASHSCOPE_API_KEY=你的密钥

# 切换到 OpenAI  
export EMBED_PROVIDER=openai
export OPENAI_API_KEY=你的密钥
```

## 🐳 Docker 部署

```bash
# 构建并运行
docker-compose up -d

# 使用自定义配置
docker-compose -f docker-compose.yml up -d
```

## 📖 API 参考

### MCP 工具

- **`search_knowledge_base`** - 搜索本地向量数据库
- **`web_search`** - 通过 Perplexity/Exa 搜索网络
- **`smart_search`** - 混合本地 + 网络搜索
- **`add_document`** - 向知识库添加文档

### Python API

```python
# 核心嵌入函数
from src.embedding_provider import (
    get_embed_model,
    get_embed_model_from_env,
    create_index_with_provider,
    validate_provider_config
)

# RAG 引擎
from src.llamaindex_processor import RAGEngine

# MCP 服务器
from src.mcp_server import RAGMCPServer
```

## 🔍 提供者验证

检查提供者配置：

```python
from src.embedding_provider import validate_provider_config

# 验证 OpenAI 设置
openai_status = validate_provider_config("openai")
print(f"OpenAI 就绪: {openai_status['valid']}")

# 验证 DashScope 设置  
dashscope_status = validate_provider_config("dashscope")
print(f"DashScope 就绪: {dashscope_status['valid']}")
```

## 🚀 生产特性

- **限流** - 可配置的请求节流
- **监控** - Prometheus 度量集成
- **日志** - 使用 loguru 的结构化日志
- **错误恢复** - 自动提供者降级
- **健康检查** - 内置验证端点

## 📊 性能

- **提供者切换** - 零停机时间切换
- **缓存** - 智能查询引擎缓存
- **批处理** - 优化的批量操作
- **内存高效** - 懒加载和清理

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 打开 Pull Request

## 📄 许可证

该项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 链接

- **文档**: [完整文档](docs/embedding_providers.md)
- **发布**: [GitHub Releases](https://github.com/Renewdxin/mcp/releases)
- **问题**: [错误报告和功能请求](https://github.com/Renewdxin/mcp/issues)

## ⭐ Star 历史

如果这个项目对你有帮助，请考虑给它一个 star！⭐

---

**用** ❤️ **为 AI 社区构建**