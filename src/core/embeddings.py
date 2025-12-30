from typing import List

from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.utils.logger import logger
# 文本块 → 向量（数字列表）
# 语义理解器。将任何文本（用户问题、你的文档）转换为蕴含语义的数字向量（嵌入），这是实现语义搜索的数学基础。
class LocalEmbeddings:
    """本地嵌入模型"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(settings.embedding_model)
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入"""
        if not self.model:
            self._load_model()

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        if not self.model:
            self._load_model()

        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()

    def get_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        if not self.model:
            self._load_model()

        # 通过一个小样本来获取维度
        sample_embedding = self.model.encode(["sample text"])
        return sample_embedding.shape[1]


embeddings = LocalEmbeddings()