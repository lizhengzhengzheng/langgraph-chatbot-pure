from typing import List, Dict, Any
import re
from src.utils.logger import logger

# 长文本字符串 → 文本块列表
# 根据语义、长度等规则，将长文本加工成大小合适、语义完整的片段。就像把一本厚书拆分成一篇篇独立的文章或章节
class DocumentSplitter:
    """文档分割器"""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        if not text:
            return []

        # 按分隔符分割
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = [text]

        # 合并小块
        chunks = []
        current_chunk = ""

        for split in splits:
            # 如果当前块加上新的分割会超过大小限制
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果单个分割就超过大小限制，需要进一步分割
                if len(split) > self.chunk_size:
                    # 递归分割
                    sub_chunks = self._split_large_text(split)
                    chunks.extend(sub_chunks[:-1])  # 添加除最后一个外的所有块
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """分割大文本"""
        # 按句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果单个句子就超过大小限制
                if len(sentence) > self.chunk_size:
                    # 按字符分割
                    for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                        chunk = sentence[i:i + self.chunk_size]
                        if chunk:
                            chunks.append(chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分割文档"""
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source = doc.get("source", "")
            title = doc.get("title", "")

            chunks = self.split_text(text)

            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

                all_chunks.append({
                    "text": chunk,
                    "metadata": chunk_metadata,
                    "source": source,
                    "title": f"{title}_chunk_{i + 1}"
                })

        logger.info(f"文档分割完成: {len(documents)} -> {len(all_chunks)} 个块")
        return all_chunks