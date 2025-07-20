# 环境变量配置详解

## 🔧 配置方式对比

### .env文件 vs MCP配置

你可能会疑惑：既然有`.env`文件，为什么MCP配置中还要设置`env`字段？

**答案是：这两种方式作用相同，只需选择其中一种！**

## 📋 配置方式详解

### 方式1: 使用.env文件 (推荐)

**工作原理**:
```bash
# .env文件内容
OPENAI_API_KEY=sk-your-key
SEARCH_API_KEY=your-key
SEARCH_BACKEND=perplexity
```

**MCP配置**:
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

**执行流程**:
1. MCP客户端执行bash命令
2. `cd`到项目目录
3. `source venv/bin/activate`激活虚拟环境
4. `source .env`加载环境变量
5. `python src/mcp_server.py`启动服务器

### 方式2: MCP配置中直接设置

**MCP配置**:
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /path/to/rag-mcp-server && source venv/bin/activate && python src/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key",
        "SEARCH_API_KEY": "your-key",
        "SEARCH_BACKEND": "perplexity"
      }
    }
  }
}
```

**执行流程**:
1. MCP客户端设置环境变量
2. 执行bash命令切换到项目目录
3. 激活虚拟环境
4. 直接执行python命令启动服务器

## 🎯 选择建议

### 推荐使用.env文件的场景

✅ **Docker部署**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--env-file", "/path/to/.env", "rag-mcp-server:production"]
    }
  }
}
```

✅ **团队开发**
- 每个开发者有自己的`.env`文件
- `.env.example`作为模板
- `.env`加入`.gitignore`

✅ **多环境部署**
```bash
# 开发环境
cp .env.dev .env

# 生产环境  
cp .env.prod .env
```

### 推荐直接在MCP配置的场景

✅ **个人使用**
- 只有一套环境
- 配置简单固定

✅ **快速测试**
- 临时修改配置
- 调试不同参数

## ⚠️ 常见错误

### 错误1: 同时使用两种方式
```json
// ❌ 错误：重复配置
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "source .env && python src/mcp_server.py"],  // 使用了.env
      "env": {
        "OPENAI_API_KEY": "sk-another-key"  // 又设置了env
      }
    }
  }
}
```

**问题**: 环境变量会被覆盖，导致配置混乱。

### 错误2: 忘记激活虚拟环境
```json
// ❌ 错误：没有激活虚拟环境
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /path/to/rag-mcp-server && source .env && python src/mcp_server.py"]
    }
  }
}
```

**解决**: 激活虚拟环境
```json
// ✅ 正确 (venv)
{
  "mcpServers": {
    "rag-server": {
      "command": "bash", 
      "args": ["-c", "cd /absolute/path/to/rag-mcp-server && source venv/bin/activate && source .env && python src/mcp_server.py"]
    }
  }
}

// ✅ 正确 (conda)
{
  "mcpServers": {
    "rag-server": {
      "command": "bash", 
      "args": ["-c", "cd /absolute/path/to/rag-mcp-server && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rag-mcp && source .env && python src/mcp_server.py"]
    }
  }
}
```

### 错误4: 环境变量格式错误
```bash
# ❌ 错误的.env格式
OPENAI_API_KEY = sk-your-key    # 等号两边不能有空格
SEARCH_API_KEY="your-key"       # 不需要引号（除非值中包含空格）
```

```bash
# ✅ 正确的.env格式
OPENAI_API_KEY=sk-your-key
SEARCH_API_KEY=your-key
```

## 🔍 调试配置

### 验证环境变量加载

**方法1: 检查虚拟环境和.env文件加载**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "bash",
      "args": ["-c", "cd /path/to/rag-mcp-server && source venv/bin/activate && which python && source .env && env | grep API_KEY && python src/mcp_server.py"]
    }
  }
}
```

**方法2: 在Python中打印环境变量**
```python
# 在mcp_server.py开头添加
import os
print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'NOT SET')}")
print(f"SEARCH_API_KEY: {os.environ.get('SEARCH_API_KEY', 'NOT SET')}")
```

### 常见调试命令

```bash
# 检查虚拟环境
which python
python --version
pip list | grep -E "(openai|requests)"

# 检查.env文件内容
cat .env

# 验证完整启动流程
cd /path/to/rag-mcp-server
source venv/bin/activate
source .env
env | grep API_KEY
python src/mcp_server.py

# Conda环境调试
conda activate rag-mcp
which python
python --version

# Docker环境变量检查
docker run --rm --env-file .env alpine env | grep API_KEY
```

## 📝 最佳实践

### 1. 使用.env文件模板
```bash
# 创建模板
cp .env.example .env

# 编辑实际值
nano .env
```

### 2. 版本控制配置
```bash
# .gitignore
.env
.env.local
.env.*.local
```

### 3. 安全性考虑
- 使用最小权限原则
- 定期轮换API密钥
- 监控密钥使用情况

### 4. 文档化
在README中明确说明：
- 需要哪些环境变量
- 如何获取API密钥
- 配置示例

## 🚀 推荐配置流程

1. **安装依赖并创建虚拟环境**
   ```bash
   cd rag-mcp-server
   python3 -m venv venv
   source venv/bin/activate  # 或 conda activate rag-mcp
   pip install -r requirements.txt
   ```

2. **复制环境文件**
   ```bash
   cp .env.example .env
   ```

3. **获取API密钥**
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Perplexity API Key](https://www.perplexity.ai/settings/api)

4. **编辑.env文件**
   ```bash
   OPENAI_API_KEY=sk-your-actual-key
   SEARCH_API_KEY=your-actual-key
   SEARCH_BACKEND=perplexity
   ```

5. **配置MCP客户端**
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

6. **测试连接**
   在Claude中测试工具是否可用

这样配置简单、安全、易维护！