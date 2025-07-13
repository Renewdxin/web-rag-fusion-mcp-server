#!/bin/bash

# MCP RAG Server 安装脚本
# 适用于 macOS, Linux, WSL

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出函数
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查 Python 版本
check_python_version() {
    print_status "检查 Python 版本..."
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        required_version="3.9"
        
        if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
            print_error "需要 Python 3.9+，当前版本：$python_version"
            exit 1
        else
            print_success "Python 版本检查通过：$python_version"
        fi
    else
        print_error "未找到 Python3，请先安装 Python 3.9+"
        exit 1
    fi
}

# 创建虚拟环境
create_virtual_env() {
    print_status "创建 Python 虚拟环境..."
    
    if [ -d "venv" ]; then
        print_warning "虚拟环境已存在，删除旧环境..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级 pip
    print_status "升级 pip..."
    pip install --upgrade pip
    
    print_success "虚拟环境创建完成"
}

# 安装依赖
install_dependencies() {
    print_status "安装 Python 依赖包..."
    
    # 确保虚拟环境已激活
    source venv/bin/activate
    
    # 安装基础依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "基础依赖安装完成"
    else
        print_error "未找到 requirements.txt 文件"
        exit 1
    fi
    
    # 询问是否安装开发依赖
    read -p "是否安装开发依赖？(y/N): " install_dev
    if [[ $install_dev =~ ^[Yy]$ ]]; then
        if [ -f "requirements-dev.txt" ]; then
            pip install -r requirements-dev.txt
            print_success "开发依赖安装完成"
        else
            print_warning "未找到 requirements-dev.txt 文件，跳过开发依赖安装"
        fi
    fi
}

# 创建配置文件
setup_config() {
    print_status "设置配置文件..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "已创建 .env 配置文件"
        else
            # 创建基础 .env 文件
            cat > .env << EOF
# MCP RAG Server 配置文件

# ==========================================
# 必需配置
# ==========================================

# Tavily API Key (网络搜索)
TAVILY_API_KEY=

# OpenAI API Key (嵌入向量，可选)
OPENAI_API_KEY=

# ==========================================
# 向量存储配置
# ==========================================
VECTOR_STORE_PATH=./vector_store
COLLECTION_NAME=knowledge_base

# ==========================================
# MCP 服务器配置
# ==========================================
MCP_SERVER_NAME=rag-agent
SIMILARITY_THRESHOLD=0.75

# ==========================================
# 搜索配置
# ==========================================
RAG_TOP_K=5
WEB_SEARCH_MAX_RESULTS=5

# ==========================================
# 日志配置
# ==========================================
LOG_LEVEL=INFO

# ==========================================
# 可选：缓存配置
# ==========================================
# REDIS_HOST=localhost
# REDIS_PORT=6379
EOF
            print_success "已创建默认 .env 配置文件"
        fi
        print_warning "请编辑 .env 文件并添加您的 API 密钥"
    else
        print_warning ".env 文件已存在，跳过创建"
    fi
}

# 创建目录结构
create_directories() {
    print_status "创建项目目录结构..."
    
    directories=(
        "vector_store"
        "data"
        "logs"
        "scripts"
        "tests"
        "tests/unit"
        "tests/integration"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "创建目录：$dir"
        fi
    done
}

# 创建示例脚本
create_sample_scripts() {
    print_status "创建示例脚本..."
    
    # 创建知识库初始化脚本
    if [ ! -f "scripts/init_knowledge_base.py" ]; then
        cat > scripts/init_knowledge_base.py << 'EOF'
#!/usr/bin/env python3
"""
初始化知识库脚本
将 ./data/ 目录中的文档加载到向量存储中
"""

import asyncio
import os
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vector_store import VectorStoreManager
from document_loader import load_documents

async def main():
    """主函数"""
    data_dir = Path("./data")
    if not data_dir.exists():
        print("❌ ./data/ 目录不存在，请创建并添加文档")
        return

    if not any(data_dir.iterdir()):
        print("⚠️  ./data/ 目录为空，请添加要索引的文档")
        return

    print("🚀 开始初始化知识库...")

    # 初始化向量存储
    vector_manager = VectorStoreManager("./vector_store")
    await vector_manager.initialize_collection("knowledge_base")

    # 加载文档
    documents = load_documents("./data/")
    if documents:
        await vector_manager.add_documents(documents)
        print(f"✅ 成功加载 {len(documents)} 个文档到知识库")
    else:
        print("⚠️  未找到可加载的文档")

if __name__ == "__main__":
    asyncio.run(main())
EOF
        chmod +x scripts/init_knowledge_base.py
        print_success "创建知识库初始化脚本"
    fi

    # 创建测试脚本
    if [ ! -f "scripts/test_server.py" ]; then
        cat > scripts/test_server.py << 'EOF'
#!/usr/bin/env python3
"""
测试 MCP 服务器脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from mcp_server import RAGMCPServer

async def main():
    """测试服务器基本功能"""
    print("🧪 测试 MCP RAG 服务器...")

    try:
        server = RAGMCPServer()
        server.config.validate()
        print("✅ 服务器配置验证通过")

        # 这里可以添加更多测试
        print("✅ 所有测试通过")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF
        chmod +x scripts/test_server.py
        print_success "创建服务器测试脚本"
    fi
}

# 检查系统依赖
check_system_dependencies() {
    print_status "检查系统依赖..."

    # 检查 git
    if ! command -v git &> /dev/null; then
        print_warning "未安装 Git，建议安装用于版本控制"
    fi

    # 检查 curl
    if ! command -v curl &> /dev/null; then
        print_warning "未安装 curl，可能影响网络功能"
    fi
}

# 显示下一步提示
show_next_steps() {
    print_success "🎉 MCP RAG Server 安装完成！"
    echo ""
    echo -e "${BLUE}📋 下一步操作：${NC}"
    echo ""
    echo "1. 配置 API 密钥："
    echo "   编辑 .env 文件，添加您的 Tavily 和 OpenAI API 密钥"
    echo ""
    echo "2. 准备文档："
    echo "   将要索引的文档放入 ./data/ 目录"
    echo ""
    echo "3. 初始化知识库："
    echo "   source venv/bin/activate"
    echo "   python scripts/init_knowledge_base.py"
    echo ""
    echo "4. 测试服务器："
    echo "   python scripts/test_server.py"
    echo ""
    echo "5. 启动服务器："
    echo "   python mcp_server.py"
    echo ""
    echo "6. 配置 Claude Desktop："
    echo "   参考 README.md 中的 MCP 客户端配置部分"
    echo ""
    echo -e "${YELLOW}💡 提示：运行前请确保激活虚拟环境${NC}"
    echo "   source venv/bin/activate"
    echo ""
}

# 主函数
main() {
    echo -e "${GREEN}"
    echo "=================================================="
    echo "     MCP RAG Server 自动安装脚本"
    echo "=================================================="
    echo -e "${NC}"

    check_python_version
    check_system_dependencies
    create_virtual_env
    install_dependencies
    setup_config
    create_directories
    create_sample_scripts
    show_next_steps
}

# 运行主函数
main "$@"