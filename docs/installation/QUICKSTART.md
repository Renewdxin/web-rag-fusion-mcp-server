# 🚀 RAG MCP Server 快速设置指南

只需3步，快速设置你的RAG MCP服务器！

## 📋 准备工作

确保你有以下API密钥：
- [OpenAI API Key](https://platform.openai.com/api-keys) 
- [Perplexity API Key](https://www.perplexity.ai/settings/api) (推荐) 或 [Exa API Key](https://exa.ai/)

## 🐳 方法1: Docker快速部署 (推荐)

### 第1步: 下载并配置
```bash
# 克隆项目
git clone <your-repo-url>
cd rag-mcp-server

# 复制并编辑配置文件
cp .env.example .env
```

### 第2步: 填入你的API密钥
编辑 `.env` 文件，只需要修改这两行：
```bash
OPENAI_API_KEY=sk-your-actual-openai-key-here
SEARCH_API_KEY=your-actual-perplexity-key-here
```

### 第3步: 启动服务
```bash
docker-compose up rag-mcp-server --build -d
```

完成！✨

## 💻 方法2: 本地部署

### 第1步: 安装依赖
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 第2步: 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 第3步: 运行服务
```bash
python src/mcp_server.py
```

## 🔌 接入Claude Code

### 第1步: 找到配置文件
- **macOS**: `~/Library/Application Support/Claude Code/settings.json`
- **Windows**: `%APPDATA%/Claude Code/settings.json`
- **Linux**: `~/.config/claude-code/settings.json`

### 第2步: 添加MCP服务器配置

**本地部署版本：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/完整路径/到/rag-mcp-server/src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "你的OpenAI密钥",
        "SEARCH_API_KEY": "你的搜索API密钥",
        "SEARCH_BACKEND": "perplexity"
      }
    }
  }
}
```

**Docker版本：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/完整路径/到/rag-mcp-server/.env",
        "rag-mcp-server:production"
      ]
    }
  }
}
```

### 第3步: 重启Claude Code

## 🔌 接入Claude Desktop

### 配置文件位置
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

### 添加配置
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/完整路径/到/rag-mcp-server/src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "你的OpenAI密钥",
        "SEARCH_API_KEY": "你的搜索API密钥",
        "SEARCH_BACKEND": "perplexity"
      }
    }
  }
}
```

## 📁 添加你的文档

```bash
# 创建文档目录
mkdir documents

# 添加你的文档（支持PDF、TXT、DOCX、MD）
cp 你的文档.pdf documents/
cp 你的手册.md documents/
```

服务器会自动处理新文档！

## ✅ 测试一下

在Claude Code或Claude Desktop中试试这些命令：

1. **搜索知识库**：
   ```
   请搜索知识库中关于"项目架构"的信息
   ```

2. **网络搜索**：
   ```
   搜索最新的Python最佳实践
   ```

3. **智能搜索**：
   ```
   比较我们的技术栈与当前流行的技术趋势
   ```

## 🆘 遇到问题？

### 常见解决方案

**服务器无法启动**
```bash
# 检查API密钥
grep API_KEY .env

# 查看错误日志
docker-compose logs rag-mcp-server
```

**MCP连接失败**
- 确保配置文件路径正确
- 重启Claude应用
- 检查API密钥是否有效

**搜索结果为空**
```bash
# 检查文档目录
ls -la documents/

# 降低相似度阈值
echo "SIMILARITY_THRESHOLD=0.5" >> .env
```

### 获取更多帮助
- 查看完整文档：[README.md](../README.md)
- 开发模式调试：`docker-compose --profile dev up rag-mcp-dev`

## 🎉 完成！

现在你就可以在Claude中使用强大的RAG功能了！你的AI助手现在可以：
- 搜索你的本地文档
- 获取最新的网络信息  
- 提供更准确、更相关的回答

享受你的增强版AI助手吧！ 🤖✨