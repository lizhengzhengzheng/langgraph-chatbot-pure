# file: src/core/advanced_retriever.py - 增强版
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.core.llm_client import llm_client
from src.core.session import session_manager
from src.core.vector_store import vector_store


class QueryRewriter:
    """查询重写器（支持会话上下文）"""

    def rewrite(self, original_query: str, session_context: Optional[Dict] = None) -> str:
        prompt = f"""
        你是一个查询优化助手。请将用户的原始问题，优化成一个更适合用于知识库语义检索的查询语句。

        原始问题：{original_query}

        """

        # 如果有会话上下文，添加相关信息
        if session_context:
            topics = session_context.get("topics", [])
            if topics:
                prompt += f"\n当前对话主题：{', '.join(topics[:3])}"

        prompt += """
        要求：保持原意，消除歧义，可以适当补充相关的同义词或上位词，使其更完整、更正式。
        只需输出优化后的查询语句。

        优化后的查询：
        """

        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            return response["content"].strip()
        except:
            return original_query


class HybridRetriever:
    """混合检索器（支持会话偏好）"""

    def __init__(self):
        self.rewriter = QueryRewriter()

    def retrieve(self, query: str, session_id: str,
                 preferred_doc_ids: Optional[List[str]] = None,
                 top_k: int = 15) -> List[Dict[str, Any]]:
        """检索文档（考虑会话偏好）"""
        # 获取会话上下文
        session_context = session_manager.get_session(session_id, auto_create=True)
        session_analysis = session_context.conversation_analysis

        # 1. 查询重写（考虑会话上下文）
        rewritten_query = self.rewriter.rewrite(query, session_analysis)

        all_results = []

        # 2. 多路召回策略
        # 策略A: 用原查询进行向量搜索
        results_a = self._semantic_search(query, top_k=top_k)
        all_results.extend(results_a)

        # 策略B: 用重写后的查询进行向量搜索
        results_b = self._semantic_search(rewritten_query, top_k=top_k)
        all_results.extend([r for r in results_b if r['id'] not in [a['id'] for a in results_a]])

        # 策略C: 如果有偏好文档，优先获取它们
        if preferred_doc_ids:
            results_c = self._search_preferred_docs(preferred_doc_ids)
            all_results.extend(results_c)

        # 3. 去重
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)

        final_results = unique_results[:top_k * 2]

        # 4. 标记偏好文档
        if preferred_doc_ids:
            for result in final_results:
                if result['id'] in preferred_doc_ids:
                    result['preferred'] = True

        return final_results

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """语义搜索"""
        try:
            return vector_store.search(query, top_k=top_k)
        except Exception as e:
            print(f"语义搜索失败: {e}")
            return []

    def _search_preferred_docs(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """搜索偏好文档"""
        results = []
        for doc_id in doc_ids[:5]:  # 最多5个偏好文档
            try:
                # 这里需要实现根据ID获取文档的逻辑
                # 假设 vector_store 有 get_document_by_id 方法
                doc = vector_store.get_document_by_id(doc_id)
                if doc:
                    doc['preferred'] = True
                    results.append(doc)
            except:
                continue
        return results


class Reranker:
    """重排器（支持多种策略）"""

    def __init__(self):
        # 模型应专门用于 (query, passage) 相关性评分
        try:
            self.model = CrossEncoder(settings.embedding_model)
            self.use_cross_encoder = True
        except:
            self.use_cross_encoder = False

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """重排候选文档"""
        if not candidates:
            return []

        # 如果可用，使用交叉编码器
        if self.use_cross_encoder and len(candidates) > 1:
            return self._rerank_with_cross_encoder(query, candidates, top_k)
        else:
            return self._rerank_simple(query, candidates, top_k)

    def _rerank_with_cross_encoder(self, query: str, candidates: List[Dict[str, Any]],
                                   top_k: int) -> List[Dict[str, Any]]:
        """使用交叉编码器重排"""
        # 准备模型输入
        model_inputs = [(query, cand['text'][:500]) for cand in candidates]

        # 获取相关性分数
        try:
            scores = self.model.predict(model_inputs)
        except:
            return self._rerank_simple(query, candidates, top_k)

        # 将分数附加到候选项
        for cand, score in zip(candidates, scores):
            cand['relevance_score'] = float(score)

        # 提升偏好文档的分数
        for cand in candidates:
            if cand.get('preferred', False):
                cand['relevance_score'] = cand.get('relevance_score', 0.5) * 1.2

        # 按分数降序排序
        candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return candidates[:top_k]

    def _rerank_simple(self, query: str, candidates: List[Dict[str, Any]],
                       top_k: int) -> List[Dict[str, Any]]:
        """简单重排（基于关键词匹配）"""
        query_words = set(query.lower().split())

        for cand in candidates:
            # 计算简单匹配分数
            text = cand.get('text', '').lower()
            cand_words = set(text.split()[:100])  # 只考虑前100个词

            # Jaccard相似度
            if query_words and cand_words:
                intersection = len(query_words & cand_words)
                union = len(query_words | cand_words)
                score = intersection / union if union > 0 else 0
            else:
                score = 0

            # 提升偏好文档
            if cand.get('preferred', False):
                score *= 1.2

            cand['relevance_score'] = score

        # 排序
        candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return candidates[:top_k]