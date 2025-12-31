# src/core/advanced_retriever.py
from core.session import session_manager
from src.core.llm_client import llm_client

# 高级检索引擎 (集成查询重写、混合搜索、重排)
class QueryRewriter:
    def rewrite(self, original_query: str, chat_history: list = None) -> str:
        prompt = f"""
        你是一个查询优化助手。请将用户的原始问题，优化成一个更适合用于知识库语义检索的查询语句。
        要求：保持原意，消除歧义，可以适当补充相关的同义词或上位词，使其更完整、更正式。
        只需输出优化后的查询语句。

        原始问题：{original_query}
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
            return original_query  # 失败时回退到原查询


# 在同一个文件中继续
from src.core.vector_store import vector_store
from typing import List, Dict, Any


class HybridRetriever:
    def __init__(self):
        self.rewriter = QueryRewriter()

    def retrieve(self, query: str, session_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. 获取会话上下文（自动创建不存在的会话）
        session_context = session_manager.get_session(session_id, auto_create=True)
        # 2. 优先从会话缓存获取（可选，提升性能）
        cache_key = f"retrieve_{hash(query)}_{top_k}"
        if cache_key in session_context.retriever_cache:
            return session_context.retriever_cache[cache_key]
        # 1. 查询重写
        rewritten_query = self.rewriter.rewrite(query)

        all_results = []

        # 2. 多路召回策略
        # 策略A: 用原查询进行向量搜索 (保留原始意图)
        results_a = vector_store._semantic_search(query, top_k=top_k)
        all_results.extend(results_a)

        # 策略B: 用重写后的查询进行向量搜索 (增强的语义)
        results_b = vector_store._semantic_search(rewritten_query, top_k=top_k)
        all_results.extend([r for r in results_b if r['id'] not in [a['id'] for a in results_a]])

        # 策略C: 关键词搜索 (用于精确匹配，如产品型号)
        # 这里需要你的向量库支持或结合其他库，例如：
        # results_c = self._keyword_search(rewritten_query, top_k=top_k)
        # all_results.extend(results_c)

        # 3. 去重 (基于文档ID)
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)

        final_results = unique_results[:top_k * 2]

        # 4. 缓存到当前会话（避免重复检索）
        session_context.retriever_cache[cache_key] = final_results

        return final_results

# 安装 sentence-transformers 的交叉编码器
# pip install sentence-transformers[cross-encoder]

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        # 这个模型专门用于 (query, passage) 相关性评分
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # 准备模型输入: [(query, text1), (query, text2), ...]
        model_inputs = [(query, cand['text'][:500]) for cand in candidates]  # 截断文本

        # 获取相关性分数
        scores = self.model.predict(model_inputs)

        # 将分数附加到候选项
        for cand, score in zip(candidates, scores):
            cand['relevance_score'] = float(score)

        # 按分数降序排序
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        return candidates[:top_k]