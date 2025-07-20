# RAG MCP Server 

一个功能强大的Model Context Protocol (MCP)服务器，为语言模型提供检索增强生成(RAG)能力。它结合了本地知识库搜索和实时网络搜索，为你的AI助手提供全面准确的上下文信息。

## 🚀 核心功能

- **📚 知识库搜索**: 在本地文档中快速找到相关信息
- **🌐 网络搜索**: 获取最新的网络信息
- **🧠 智能搜索**: 结合知识库和网络搜索的混合方法
- **🔌 MCP兼容**: 与Claude Code、Claude Desktop等MCP客户端完美集成

## 📚 文档导航

### 🚀 快速开始
- **[快速设置指南](docs/installation/QUICKSTART.md)** - 3步快速部署 ⭐
- **[Docker部署指南](docs/installation/docker-setup.md)** - 推荐的容器化部署
- **[本地安装指南](docs/installation/local-setup.md)** - 开发环境设置

### 📖 使用指南  
- **[MCP客户端集成](docs/usage/mcp-integration.md)** - Claude Code/Desktop配置
- **[工具参考手册](docs/usage/tools-reference.md)** - 所有工具的详细说明
- **[配置参数说明](docs/configuration/CONFIG.md)** - 完整配置选项

### 🛠️ 问题解决
- **[常见问题解决](docs/troubleshooting/common-issues.md)** - 故障排除指南
- **[开发者指南](docs/development/DEVELOPER_GUIDE.md)** - 开发和扩展

> 💡 **新用户建议**: 直接查看 [快速设置指南](docs/installation/QUICKSTART.md) 快速上手！

## 📋 前置要求

- Python 3.9+ 或 Docker
- [OpenAI API Key](https://platform.openai.com/api-keys)
- 搜索服务API Key：
  - [Perplexity API Key](https://www.perplexity.ai/settings/api) (推荐)
  - 或 [Exa API Key](https://exa.ai/)

## ⚡ 快速开始

### 3步快速部署

1. **克隆并配置**
   ```bash
   git clone <your-repo-url>
   cd rag-mcp-server
   cp .env.example .env
   # 编辑 .env 文件，填入你的 OpenAI 和搜索 API 密钥
   ```

2. **启动服务**
   ```bash
   # Docker部署 (推荐)
   docker-compose up rag-mcp-server --build -d
   
   # 或本地部署
   pip install -r requirements.txt && python src/mcp_server.py
   ```

3. **集成到客户端**
   配置Claude Code或Claude Desktop - 详见 [MCP集成指南](docs/usage/mcp-integration.md)

✅ **完成！** 现在你可以在Claude中使用强大的RAG功能了！

> 📖 **详细步骤**: 查看 [快速设置指南](docs/installation/QUICKSTART.md) 获取完整说明

## 🔧 客户端集成

### 支持的客户端
- **Claude Code** - 代码编辑器中的AI助手
- **Claude Desktop** - 桌面应用
- 其他MCP兼容客户端

### 配置示例

**两种配置方式任选其一:**

**方式1: 使用.env文件 (推荐)**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /path/to/rag-mcp-server && source venv/bin/activate && source .env && python src/mcp_server.py"]
    }
  }
}
```

**方式2: 直接设置环境变量**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /path/to/rag-mcp-server && source venv/bin/activate && python src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "SEARCH_API_KEY": "your-search-api-key"
      }
    }
  }
}
```

> 💡 **说明**: `.env`文件和MCP配置中的`env`字段作用相同，选择其中一种即可！

📖 **详细配置指南**: [MCP客户端集成](docs/usage/mcp-integration.md)

## 📖 使用方法

### 🔍 可用工具

1. **`search_knowledge_base`** - 搜索本地文档知识库
   ```
   搜索我们的API文档中关于认证的内容
   ```

2. **`web_search`** - 实时网络搜索
   ```
   搜索最新的Python最佳实践
   ```

3. **`smart_search`** - 智能混合搜索
   ```
   比较我们的技术架构与业界最佳实践
   ```

📖 **详细用法**: [工具参考手册](docs/usage/tools-reference.md)

## 📁 文档管理

### 添加文档
```bash
# 创建文档目录并添加文件
mkdir -p documents
cp your-documents.pdf documents/
# 支持格式：PDF、TXT、DOCX、MD
```

服务器会自动检测并处理新文档！

## ⚙️ 配置

### 核心环境变量
```bash
# 必需配置
OPENAI_API_KEY=sk-your-key              # OpenAI API密钥
SEARCH_API_KEY=your-search-key          # 搜索API密钥
SEARCH_BACKEND=perplexity               # perplexity 或 exa

# 可选配置
SIMILARITY_THRESHOLD=0.75               # 搜索相似度阈值
LOG_LEVEL=INFO                          # 日志级别
ENVIRONMENT=prod                        # 运行环境
```

📖 **完整配置**: [配置参数说明](docs/configuration/CONFIG.md)

## 🛠️ 故障排除

### 快速检查
```bash
# 验证API密钥
cat .env | grep API_KEY

# 查看服务日志
docker-compose logs rag-mcp-server

# 检查文档加载
ls -la documents/
```

### 常见问题
- **无法启动**: 检查API密钥配置
- **搜索无结果**: 降低相似度阈值 (`SIMILARITY_THRESHOLD=0.5`)
- **MCP连接失败**: 验证配置文件路径和格式

🔧 **详细解决方案**: [常见问题解决](docs/troubleshooting/common-issues.md)

## 🔒 安全与维护

### 安全要点
- 保护API密钥，不要提交到版本控制
- 定期轮换密钥
- 监控使用日志

### 数据备份
```bash
# 备份向量数据库
docker run --rm -v mcp_rag_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/rag_backup.tar.gz -C /data .
```

### 监控
```bash
# 启用监控面板
docker-compose --profile monitoring up
# 访问 http://localhost:3000 (Grafana)
```

## 🤝 支持与贡献

### 获取帮助
- 📖 查看[完整文档](docs/README.md)
- 🐛 [报告问题](https://github.com/your-repo/issues)
- 💬 参与讨论和改进

### 贡献代码
欢迎提交Pull Request！请先阅读[开发者指南](docs/development/DEVELOPER_GUIDE.md)。

## 📄 许可证

MIT License - 详见[LICENSE](LICENSE)文件

---

⭐ **喜欢这个项目？** 给个Star支持一下！