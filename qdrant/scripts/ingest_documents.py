#!/usr/bin/env python3
"""
文档摄入脚本
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.loader import DocumentLoader
from utils.splitter import DocumentSplitter
from src.core.vector_store import vector_store
from src.utils.logger import logger


def ingest_documents(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """摄入文档到向量库"""
    logger.info(f"开始摄入文档: {directory_path}")

    # 检查目录是否存在
    if not Path(directory_path).exists():
        logger.error(f"目录不存在: {directory_path}")
        return False

    try:
        # 加载文档
        documents = DocumentLoader.load_directory(directory_path)

        if not documents:
            logger.warning(f"目录中没有找到可处理的文档: {directory_path}")
            return False

        logger.info(f"加载了 {len(documents)} 个文档")

        # 分割文档
        splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        logger.info(f"文档分割为 {len(chunks)} 个块")

        # 确保集合存在
        vector_store.create_collection()

        # 添加到向量库
        vector_store.add_documents(chunks)

        # 获取集合信息
        info = vector_store.get_collection_info()

        logger.info("文档摄入完成")
        logger.info(f"当前向量库中有 {info.get('vectors_count', 0)} 个向量")

        return True

    except Exception as e:
        logger.error(f"文档摄入失败: {str(e)}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档摄入到向量库")
    parser.add_argument("directory", help="文档目录路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="分块重叠")

    args = parser.parse_args()

    success = ingest_documents(
        args.directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()