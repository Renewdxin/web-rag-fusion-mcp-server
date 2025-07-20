# 本地部署指南

## 🖥️ 本地安装

### 前置要求
- Python 3.9+
- pip 或 conda
- Git

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd rag-mcp-server
   ```

2. **创建虚拟环境**
   ```bash
   # 使用 venv
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # 或使用 conda
   conda create -n rag-mcp python=3.11
   conda activate rag-mcp
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的API密钥
   ```

5. **初始化知识库** (可选)
   ```bash
   python scripts/init_knowledge_base.py
   ```

6. **启动服务器**
   ```bash
   python src/mcp_server.py
   ```

### 开发环境设置

#### 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

#### 代码质量工具
```bash
# 代码格式化
black src/
isort src/

# 类型检查
mypy src/

# 代码检查
flake8 src/

# 测试
pytest tests/
```

#### 预提交钩子
```bash
pip install pre-commit
pre-commit install
```

### 配置IDE

#### VS Code
推荐插件：
- Python
- Pylance  
- Black Formatter
- isort

配置文件 `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.sortImports.provider": "isort",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true
}
```

#### PyCharm
1. 设置Python解释器为虚拟环境
2. 配置代码风格为Black
3. 启用类型检查

### 常见问题

#### 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装特定版本
pip install package==version
```

#### 权限问题
```bash
# macOS/Linux
sudo chown -R $USER:$USER ./
chmod +x scripts/*.py

# Windows (以管理员身份运行)
icacls . /grant %USERNAME%:F /T
```

#### 环境变量问题
```bash
# 检查环境变量
python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'Not Set'))"

# 临时设置
export OPENAI_API_KEY="your-key"

# 永久设置 (添加到 ~/.bashrc 或 ~/.zshrc)
echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
```