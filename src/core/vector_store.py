import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config.settings import settings
from src.core.embeddings import embeddings
from src.utils.logger import logger


# 向量 + 原文块 → 存入数据库的记录
# 负责连接、操作本地的Qdrant向量数据库，实现知识的存储（add_documents）和基于语义的检索（search）。
# 将向量和原文作为一个整体，高效地存储到专用数据库（Qdrant）中，并提供检索接口。
class QdrantVectorStore:
    """Qdrant向量存储"""

    def __init__(self):
        self.host = settings.qdrant_host
        self.port = settings.qdrant_port
        self.collection_name = settings.qdrant_collection_name
        self.client = None
        self._connect()

    def _connect(self):
        """连接Qdrant"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"已连接到Qdrant: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Qdrant失败: {str(e)}")
            raise

    def create_collection(self, collection_name: str = None, vector_size: int = None):
        """创建集合"""
        if collection_name is None:
            collection_name = self.collection_name

        if vector_size is None:
            vector_size = embeddings.get_embedding_dimension()

        try:
            # 检查集合是否存在 - 使用更安全的方式
            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"集合已存在: {collection_name}")
                return
            except Exception:
                # 集合不存在，继续创建
                pass

            # 创建新集合
            from qdrant_client import models
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"创建集合成功: {collection_name}")

        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """添加文档到向量库"""
        ids = []
        points = []

        for doc in documents:
            # 生成ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            # 生成嵌入
            text = doc.get("text", "")
            if not text:
                continue

            embedding = embeddings.embed_query(text)

            # 创建点
            point = qmodels.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source", ""),
                    "title": doc.get("title", "")
                }
            )
            points.append(point)

        # 批量插入
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"添加了 {len(points)} 个文档到向量库")

        return ids

    def search(
            self,
            query: str,
            top_k: int = 5,
            score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 生成查询嵌入
            query_embedding = embeddings.embed_query(query)

            # 搜索
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,  # 关键：参数名改为 query
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True  # 确保返回payload（文档内容）
            ).points  # 注意：query_points 返回的对象需要访问 .points 属性

            # 格式化结果
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "source": hit.payload.get("source", ""),
                    "title": hit.payload.get("title", "")
                })

            return results

        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def delete_collection(self):
        """删除集合"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"删除集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            # 使用更兼容的方式获取信息
            return {
                "name": self.collection_name,  # 直接使用我们知道的名字
                "vectors_count": getattr(info, 'vectors_count', getattr(info, 'points_count', 0)),
                "status": getattr(info, 'status', 'unknown')
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {}


vector_store = QdrantVectorStore()