#!/usr/bin/env python3
"""
Qdrant向量数据库设置脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.vector_store import vector_store
from src.utils.logger import logger


def setup_qdrant():
    """设置Qdrant向量数据库"""
    logger.info("开始设置Qdrant向量数据库...")

    try:
        # 创建集合
        vector_store.create_collection()

        # 获取集合信息
        info = vector_store.get_collection_info()

        logger.info(f"Qdrant设置完成")
        logger.info(f"集合名称: {info.get('name', 'N/A')}")
        logger.info(f"向量数量: {info.get('vectors_count', 0)}")
        logger.info(f"状态: {info.get('status', 'N/A')}")

        return True

    except Exception as e:
        logger.error(f"设置Qdrant失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = setup_qdrant()
    sys.exit(0 if success else 1)