# Docker 部署指南

## 🐳 Docker 安装

### 前置要求
- Docker Desktop 或 Docker Engine
- Docker Compose v2+

### 快速部署

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd rag-mcp-server
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的API密钥
   ```

3. **构建和启动**
   ```bash
   # 生产环境
   docker-compose up rag-mcp-server --build -d
   
   # 开发环境  
   docker-compose --profile dev up rag-mcp-dev
   
   # 监控环境
   docker-compose --profile monitoring up
   ```

### 环境说明

#### 生产环境 (production)
- 最小化镜像
- 优化性能
- 安全配置
- 健康检查

#### 开发环境 (development)
- 源码挂载
- 调试工具
- 交互式shell
- 热重载

#### 监控环境 (monitoring)
- Prometheus metrics
- Grafana dashboard
- 性能监控

### 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs rag-mcp-server -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart rag-mcp-server

# 进入容器
docker-compose exec rag-mcp-server bash

# 数据备份
docker run --rm -v mcp_rag_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz -C /data .
```

### 故障排除

#### 构建失败
```bash
# 清理缓存重新构建
docker-compose build --no-cache rag-mcp-server

# 查看构建日志
docker-compose build rag-mcp-server 2>&1 | tee build.log
```

#### 容器无法启动
```bash
# 检查配置
docker-compose config

# 查看详细错误
docker-compose up rag-mcp-server --no-deps
```

#### 网络问题
```bash
# 重建网络
docker-compose down
docker network prune -f
docker-compose up -d
```