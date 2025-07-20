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
