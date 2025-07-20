# MCP 客户端集成指南

## 🔌 支持的客户端

- Claude Code
- Claude Desktop
- 其他MCP兼容客户端

## Claude Code 集成

### 配置位置
- **macOS**: `~/Library/Application Support/Claude Code/settings.json`
- **Linux**: `~/.config/claude-code/settings.json`  
- **Windows**: `%APPDATA%/Claude Code/settings.json`

### 配置方式选择

> 💡 **重要说明**: `.env`文件和MCP配置中的`env`字段作用相同，只需选择其中一种方式！

#### 方式1: 使用.env文件 (推荐)

**优点**:
- 集中管理所有环境变量
- 便于版本控制和团队协作
- 与Docker完美配合
- 开发和生产环境易于切换

**适用场景**: 
- 多环境部署
- 团队开发
- Docker部署

**本地部署：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": [
        "-c", 
        "cd /absolute/path/to/rag-mcp-server && source venv/bin/activate && source .env && python src/mcp_server.py"
      ]
    }
  }
}
```

**Docker部署：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/absolute/path/to/rag-mcp-server/.env",
        "rag-mcp-server:production"
      ]
    }
  }
}
```

#### 方式2: 直接在MCP配置中设置环境变量

**优点**:
- 配置集中在一个文件
- 不需要额外的.env文件
- 适合简单场景

**缺点**:
- API密钥暴露在配置文件中
- 不利于版本控制
- 多环境管理复杂

**适用场景**:
- 个人使用
- 简单部署
- 测试环境

**本地部署：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": [
        "-c",
        "cd /absolute/path/to/rag-mcp-server && source venv/bin/activate && python src/mcp_server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-openai-key",
        "SEARCH_API_KEY": "your-actual-search-key",
        "SEARCH_BACKEND": "perplexity",
        "ENVIRONMENT": "prod"
      }
    }
  }
}
```

### 配置选择建议

| 场景 | 推荐方式 | 理由 |
|------|----------|------|
| 🐳 Docker部署 | 方式1 (.env文件) | 完美支持，安全性好 |
| 👥 团队开发 | 方式1 (.env文件) | 便于协作和环境管理 |
| 🏠 个人使用 | 方式2 (MCP配置) | 简单直接 |
| 🔒 生产环境 | 方式1 (.env文件) | 安全性更好 |
| 🧪 测试调试 | 方式2 (MCP配置) | 快速修改配置 |

> ⚠️ **安全提醒**: 
> - 使用方式1时：不要将`.env`文件提交到版本控制
> - 使用方式2时：不要将包含API密钥的配置文件分享给他人

### 高级配置

**带调试日志：**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp-server/src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key",
        "SEARCH_API_KEY": "your-key",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "dev"
      }
    }
  },
  "logging": {
    "level": "debug"
  }
}
```

## Claude Desktop 集成

### 配置位置
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

### 配置示例

**方式1: 使用.env文件**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /absolute/path/to/rag-mcp-server && source venv/bin/activate && source .env && python src/mcp_server.py"]
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
      "args": ["-c", "cd /absolute/path/to/rag-mcp-server && source venv/bin/activate && python src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-api-key",
        "SEARCH_API_KEY": "your-search-api-key",
        "SEARCH_BACKEND": "perplexity"
      }
    }
  }
}
```

## 验证连接

### 检查服务器状态
在Claude客户端中发送：
```
检查MCP服务器连接状态
```

### 测试工具
```
# 测试知识库搜索
搜索知识库中关于"测试"的内容

# 测试网络搜索  
搜索最新的技术新闻

# 测试智能搜索
综合搜索关于AI发展的信息
```

## 故障排除

### 连接失败
1. **检查配置文件路径**
   ```bash
   # 验证路径是否正确
   ls -la ~/Library/Application\ Support/Claude\ Code/settings.json
   ```

2. **验证JSON格式**
   ```bash
   # 使用jq验证JSON
   cat settings.json | jq .
   ```

3. **检查权限**
   ```bash
   # 确保文件可读
   chmod 644 settings.json
   ```

### 服务器启动失败
1. **手动测试启动**
   ```bash
   python src/mcp_server.py
   ```

2. **检查环境变量**
   ```bash
   env | grep -E "(OPENAI|SEARCH)_API_KEY"
   ```

3. **查看错误日志**
   - Claude Code: 开发者工具控制台
   - Claude Desktop: 应用程序日志

### API密钥问题
```bash
# 测试OpenAI连接
python -c "
import openai
client = openai.OpenAI()
models = client.models.list()
print('OpenAI连接成功')
"

# 测试搜索API
python -c "
import requests
headers = {'Authorization': 'Bearer your-perplexity-key'}
response = requests.get('https://api.perplexity.ai/chat/completions', headers=headers)
print(f'搜索API状态: {response.status_code}')
"
```

## 多客户端配置

### 同时使用多个客户端
每个客户端都需要独立配置，但可以共享同一个MCP服务器实例。

### 负载均衡
对于高负载场景，可以启动多个服务器实例：

**Claude Code (实例1):**
```json
{
  "mcpServers": {
    "rag-server-1": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-p", "8001:8000", "rag-mcp-server:production"]
    }
  }
}
```

**Claude Desktop (实例2):**
```json
{
  "mcpServers": {
    "rag-server-2": {
      "command": "docker", 
      "args": ["run", "--rm", "-i", "-p", "8002:8000", "rag-mcp-server:production"]
    }
  }
}
```

## 安全考虑

### API密钥保护
- 不要在配置文件中硬编码密钥
- 使用环境变量
- 定期轮换密钥

### 网络安全
- 本地部署避免暴露端口
- 使用HTTPS（如果需要远程访问）
- 启用防火墙规则

### 日志安全
- 避免在日志中记录敏感信息
- 定期清理日志文件
- 设置适当的日志级别