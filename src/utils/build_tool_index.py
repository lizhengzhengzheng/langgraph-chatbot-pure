# scripts/build_tool_index_pure.py
"""
MCP工具向量索引构建脚本 (纯净版)
唯一职责：通过项目的标准MCP客户端获取工具，并为其构建向量索引。
不包含任何硬编码的工具数据。
"""
import asyncio
import sys
from pathlib import Path
import json

from config.settings import settings

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.mcp_client import mcp_client  # 使用项目中唯一的权威客户端
from src.core.vector_store import vector_store
from src.utils.logger import logger
from sentence_transformers import SentenceTransformer


async def build_index():
    """构建工具向量索引的唯一正确流程"""
    logger.info("开始构建MCP工具向量索引 (纯净版)...")

    # 1. 从标准MCP客户端获取工具列表
    logger.info("正在通过 mcp_client.fetch_all_tools() 获取工具列表...")
    try:
        tools = await mcp_client.fetch_all_tools()
        actual_count = len(tools)
        logger.info(f"客户端返回了 {actual_count} 个工具。")

        if actual_count == 0:
            logger.error("警告：获取到的工具列表为空。这可能是因为：")
            logger.error("  1. MCP服务器未运行或无法连接")
            logger.error("  2. mcp_client 配置的服务器地址或端口错误")
            logger.error("  3. MCP服务器尚未注册任何工具")
            logger.error("索引构建失败。")
            return False

        # 简单打印工具名用于验证
        tool_names = [t.get('name', 'N/A') for t in tools]
        logger.info(f"工具列表: {', '.join(tool_names)}")

    except Exception as e:
        logger.error(f"通过客户端获取工具列表失败: {e}")
        logger.error("请检查MCP服务器状态和客户端配置。")
        return False

    # 2. 准备文档 - 完全使用MCP服务器返回的原始数据
    documents = []
    for tool in tools:
        name = tool.get('name', 'unnamed_tool')
        description = tool.get('description', '')
        parameters = tool.get('parameters', {})

        # 构建检索文本：简单拼接已有字段，不做任何“优化”或“丰富”
        # 这是唯一正确的做法，语义质量应由MCP服务器的工具定义保证
        text_parts = [f"工具{name}"]
        if description:
            text_parts.append(description)
        if parameters:
            # 将参数信息转换为字符串，用于检索
            try:
                params_str = json.dumps(parameters, ensure_ascii=False)
                text_parts.append(f"参数定义:{params_str}")
            except:
                text_parts.append("需要参数")

        tool_text = " | ".join(text_parts)

        documents.append({
            "text": tool_text,
            "metadata": {
                "name": name,
                "description": description,
                "full_schema": tool,  # 保存完整Schema，供后续调用使用
                "source": "mcp-server"
            },
            "source": f"mcp:{name}",
            "title": name
        })
        logger.debug(f"已处理工具 '{name}' -> 文本长度: {len(tool_text)}")

    # 3. 生成嵌入向量
    logger.info(f"正在为 {len(documents)} 个工具生成向量...")
    embedding_model = SentenceTransformer(settings.embedding_model)
    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts)

    # 4. 存储到向量库
    collection_name = "mcp_tools_index"

    # 清理旧集合
    try:
        vector_store.client.delete_collection(collection_name)
        logger.info(f"已删除旧集合: {collection_name}")
    except Exception as e:
        logger.info(f"无需删除集合: {e}")

    # 创建新集合
    vector_store.create_collection(
        collection_name=collection_name,
        vector_size=embeddings.shape[1]
    )
    logger.info(f"已创建集合 '{collection_name}', 向量维度: {embeddings.shape[1]}")

    # 批量插入
    from qdrant_client import models
    points = [
        models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=doc
        )
        for i, (doc, embedding) in enumerate(zip(documents, embeddings))
    ]

    vector_store.client.upsert(
        collection_name=collection_name,
        points=points
    )

    # 5. 最终验证
    try:
        coll_info = vector_store.client.get_collection(collection_name)
        # 使用兼容方式获取向量数量
        vec_count = getattr(coll_info, 'vectors_count', getattr(coll_info, 'points_count', 0))
        logger.info(f"✅ MCP工具向量索引构建完成！")
        logger.info(f"   集合: {collection_name}")
        logger.info(f"   索引内工具数量: {vec_count}")

        # 提供一个快速验证命令
        logger.info(f"   验证命令: curl http://localhost:6333/collections/{collection_name}")

        return True
    except Exception as e:
        logger.error(f"索引构建后验证失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(build_index())
    sys.exit(0 if success else 1)