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
