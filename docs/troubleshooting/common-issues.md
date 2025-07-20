# 常见问题解决方案

## 🚨 启动问题

### 问题：服务器无法启动

#### 症状
```
Configuration validation failed:
- OPENAI_API_KEY is required but not set or empty
- SEARCH_API_KEY is required but not set or empty
```

#### 解决方案
1. **检查环境变量**
   ```bash
   # 查看当前环境变量
   cat .env | grep API_KEY
   
   # 验证API密钥格式
   echo $OPENAI_API_KEY | wc -c  # OpenAI密钥通常51字符
   ```

2. **验证API密钥有效性**
   ```bash
   # 测试OpenAI API
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   
   # 测试Perplexity API
   curl -H "Authorization: Bearer $SEARCH_API_KEY" \
        https://api.perplexity.ai/chat/completions
   ```

3. **重新配置环境**
   ```bash
   # 删除并重新创建配置
   rm .env
   cp .env.example .env
   # 重新填入正确的API密钥
   ```

### 问题：Docker构建失败

#### 症状
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" 
did not complete successfully: exit code: 1
```

#### 解决方案
1. **清理Docker缓存**
   ```bash
   docker system prune -f
   docker-compose build --no-cache
   ```

2. **检查网络连接**
   ```bash
   # 测试网络连接
   curl -I https://pypi.org/simple/
   
   # 使用国内镜像
   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ .
   ```

3. **分步构建调试**
   ```bash
   # 单独测试依赖安装
   docker run --rm python:3.12-slim pip install -r requirements.txt
   ```

## 🔍 搜索问题

### 问题：搜索结果为空

#### 症状
- 知识库搜索返回空结果
- 所有查询都没有匹配

#### 解决方案
1. **检查文档加载**
   ```bash
   # 检查documents目录
   ls -la documents/
   
   # 检查向量数据库
   ls -la data/vector_store.db
   
   # 查看数据库大小
   du -h data/vector_store.db
   ```

2. **降低相似度阈值**
   ```bash
   # 在.env中设置更低的阈值
   echo "SIMILARITY_THRESHOLD=0.3" >> .env
   echo "SIMILARITY_CUTOFF=0.2" >> .env
   ```

3. **重建索引**
   ```bash
   # 删除现有索引
   rm -rf data/vector_store.db
   
   # 重启服务重建索引
   docker-compose restart rag-mcp-server
   ```

4. **检查文档格式**
   ```bash
   # 支持的格式：PDF, TXT, DOCX, MD
   file documents/*
   
   # 检查文档是否可读
   head -n 5 documents/sample.txt
   ```

### 问题：网络搜索失败

#### 症状
```
HTTP 401: Unauthorized
HTTP 429: Too Many Requests
```

#### 解决方案
1. **验证API密钥**
   ```bash
   # 检查密钥格式
   echo $SEARCH_API_KEY | grep -E '^[a-zA-Z0-9_-]+$'
   
   # 测试API连接
   python -c "
   import requests
   headers = {'Authorization': 'Bearer $SEARCH_API_KEY'}
   response = requests.get('https://api.perplexity.ai/models', headers=headers)
   print(f'状态码: {response.status_code}')
   "
   ```

2. **检查API配额**
   - 登录搜索服务提供商控制台
   - 查看剩余配额和使用情况
   - 检查计费状态

3. **切换搜索后端**
   ```bash
   # 在.env中切换到Exa
   sed -i 's/SEARCH_BACKEND=perplexity/SEARCH_BACKEND=exa/' .env
   ```

## 🔌 MCP连接问题

### 问题：Claude客户端无法连接MCP服务器

#### 症状
- Claude Code显示"MCP服务器连接失败"
- 工具不可用

#### 解决方案
1. **验证配置文件路径**
   ```bash
   # macOS
   ls -la ~/Library/Application\ Support/Claude\ Code/settings.json
   
   # 检查配置文件格式
   cat ~/Library/Application\ Support/Claude\ Code/settings.json | jq .
   ```

2. **手动测试MCP服务器**
   ```bash
   # 直接运行服务器
   python src/mcp_server.py
   
   # 检查输出是否正常
   echo '{"method": "initialize", "params": {}}' | python src/mcp_server.py
   ```

3. **检查文件权限**
   ```bash
   # 确保配置文件可读写
   chmod 644 ~/Library/Application\ Support/Claude\ Code/settings.json
   
   # 确保脚本可执行
   chmod +x src/mcp_server.py
   ```

4. **查看客户端日志**
   - Claude Code: 开发者工具 → 控制台
   - Claude Desktop: 应用菜单 → 查看日志

### 问题：Docker MCP连接问题

#### 症状
- Docker版本的MCP服务器无法启动
- 容器立即退出

#### 解决方案
1. **检查容器状态**
   ```bash
   # 查看容器日志
   docker-compose logs rag-mcp-server
   
   # 检查容器退出码
   docker-compose ps
   ```

2. **交互式调试**
   ```bash
   # 进入容器调试
   docker run --rm -it --env-file .env rag-mcp-server:production bash
   
   # 手动运行服务器
   python src/mcp_server.py
   ```

3. **修复配置文件路径**
   ```json
   {
     "mcpServers": {
       "rag-server": {
         "command": "docker",
         "args": [
           "run", "--rm", "-i",
           "--env-file", "/absolute/path/to/.env",
           "rag-mcp-server:production"
         ]
       }
     }
   }
   ```

## 🐛 性能问题

### 问题：响应时间过长

#### 症状
- 搜索请求超时
- 客户端等待时间过长

#### 解决方案
1. **优化搜索参数**
   ```bash
   # 减少返回结果数量
   echo "SIMILARITY_TOP_K=5" >> .env
   
   # 增加超时时间
   echo "TIMEOUT_SECONDS=60" >> .env
   ```

2. **启用缓存**
   ```bash
   # 启用结果缓存
   echo "ENABLE_CACHE=true" >> .env
   echo "CACHE_TTL=3600" >> .env
   ```

3. **资源监控**
   ```bash
   # 监控容器资源使用
   docker stats rag-mcp-server
   
   # 查看内存使用
   docker exec rag-mcp-server free -h
   ```

### 问题：内存使用过高

#### 症状
- 系统变慢
- 容器被杀死

#### 解决方案
1. **限制Docker资源**
   ```yaml
   # 在docker-compose.yml中添加
   services:
     rag-mcp-server:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   ```

2. **优化向量数据库**
   ```bash
   # 压缩数据库
   sqlite3 data/vector_store.db "VACUUM;"
   
   # 检查数据库大小
   du -h data/vector_store.db
   ```

3. **调整文档处理**
   ```bash
   # 减少块大小
   echo "CHUNK_SIZE=512" >> .env
   echo "CHUNK_OVERLAP=50" >> .env
   ```

## 🔧 调试技巧

### 启用详细日志
```bash
# 设置调试级别
echo "LOG_LEVEL=DEBUG" >> .env

# 查看实时日志
docker-compose logs rag-mcp-server -f

# 保存日志到文件
docker-compose logs rag-mcp-server > debug.log 2>&1
```

### 健康检查
```bash
# 检查服务健康状态
docker-compose exec rag-mcp-server python -c "
import sys
sys.path.insert(0, '/app')
from config.settings import config
try:
    config.validate()
    print('✅ 配置验证成功')
except Exception as e:
    print(f'❌ 配置验证失败: {e}')
"
```

### 网络诊断
```bash
# 测试网络连接
docker-compose exec rag-mcp-server curl -I https://api.openai.com
docker-compose exec rag-mcp-server curl -I https://api.perplexity.ai

# 检查DNS解析
docker-compose exec rag-mcp-server nslookup api.openai.com
```

## 📞 获取帮助

### 收集诊断信息
运行以下命令收集系统信息：
```bash
#!/bin/bash
echo "=== 系统信息 ===" > diagnostic.log
uname -a >> diagnostic.log
docker --version >> diagnostic.log
docker-compose --version >> diagnostic.log

echo "=== 环境变量 ===" >> diagnostic.log
env | grep -E "(OPENAI|SEARCH)" >> diagnostic.log

echo "=== 文件结构 ===" >> diagnostic.log
ls -la documents/ >> diagnostic.log
ls -la data/ >> diagnostic.log

echo "=== 容器状态 ===" >> diagnostic.log
docker-compose ps >> diagnostic.log

echo "=== 最近日志 ===" >> diagnostic.log
docker-compose logs rag-mcp-server --tail 50 >> diagnostic.log

echo "诊断信息已保存到 diagnostic.log"
```

### 社区支持
- GitHub Issues: 报告bug和功能请求
- 讨论区: 技术讨论和经验分享
- 文档贡献: 改进文档内容