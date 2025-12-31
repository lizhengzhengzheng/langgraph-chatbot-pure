# file: src/graph/nodes.py
from typing import Dict, Any

from src.core.advanced_retriever import HybridRetriever, Reranker
from src.core.context import RequestContextAware
from src.core.llm_client import llm_client
from src.core.vector_store import vector_store
from src.graph.state import AgentState
from src.utils.logger import logger


class BaseNode(RequestContextAware):
    """节点基类（支持请求上下文）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class RouteQueryNode(BaseNode):
    """路由查询节点"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("路由查询")

        user_input = state["user_input"]

        # 判断是否需要检索文档
        retrieval_keywords = ["什么", "如何", "为什么", "怎样", "解释", "介绍", "说明"]
        should_retrieve = any(keyword in user_input for keyword in retrieval_keywords)

        if not should_retrieve and len(user_input) > 10 and "?" in user_input:
            should_retrieve = True

        return {
            "should_retrieve": should_retrieve,
            "current_step": "route_query"
        }

class AdvancedRetrievalNode(BaseNode):
    """高级检索节点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("执行高级检索流程")

        user_input = state["user_input"]

        try:
            # 1. 混合检索
            candidates = self.hybrid_retriever.retrieve(user_input, top_k=15)

            # 2. 重排
            reranked_results = self.reranker.rerank(user_input, candidates, top_k=5)

            # 3. 构建上下文
            context = "\n\n".join([
                f"[参考文档 {i + 1}，相关性：{res['relevance_score']:.3f}]: {res['text']}"
                for i, res in enumerate(reranked_results)
            ])

            sources = [{
                "title": res.get("title", f"文档{i + 1}"),
                "source": res.get("source", ""),
                "score": res["relevance_score"]
            } for i, res in enumerate(reranked_results)]

            return {
                "retrieved_documents": reranked_results,
                "context": context,
                "sources": sources,
                "current_step": "advanced_retrieval"
            }

        except Exception as e:
            logger.error(f"高级检索失败: {e}")
            # 降级到基础检索
            retrieved_docs = vector_store.search(user_input, top_k=5)
            context = "\n".join([f"[文档{i + 1}]: {doc['text']}" for i, doc in enumerate(retrieved_docs)])

            return {
                "retrieved_documents": retrieved_docs,
                "context": context,
                "sources": [],
                "current_step": "advanced_retrieval",
                "retrieval_error": str(e)
            }

class GenerateResponseNode(BaseNode):
    """生成响应节点"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        user_input = state["user_input"]
        chat_history = state["chat_history"]
        context = state.get("context", "")

        # 如果有工具执行结果，整合到上下文中
        tool_result = state.get("tool_result")
        tool_used = state.get("tool_name")

        system_message = """你是一个智能助手，基于用户的问题和提供的上下文信息来回答问题。"""

        if tool_result and not tool_result.get("error"):
            tool_context = f"\n[工具调用结果] 使用工具 '{tool_used}' 得到：{tool_result}"
            system_message += tool_context

        # 构建消息列表
        messages = [{"role": "system", "content": system_message}]

        if context:
            context_prompt = f"相关上下文：\n{context}\n请基于以上信息回答。"
            messages.append({"role": "system", "content": context_prompt})

        for msg in chat_history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        # 调用大模型
        response = await llm_client.async_chat_completion(messages=messages, temperature=0.7)

        # 更新历史（包含工具信息）
        tool_info = f" (使用工具: {tool_used})" if tool_used else ""
        updated_history = chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response["content"] + tool_info}
        ]

        return {
            "response": response["content"],
            "chat_history": updated_history,
            "current_step": "generate_response",
            "should_end": True
        }


class DirectResponseNode(BaseNode):
    """直接响应节点"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("直接响应")

        user_input = state["user_input"]
        chat_history = state["chat_history"]

        # 直接调用大模型
        messages = [
            {"role": "system", "content": "你是一个友好的助手，请直接回答用户的问题。"}
        ]

        # 添加历史
        for msg in chat_history[-4:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        response = await llm_client.async_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        # 更新历史
        updated_history = chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response["content"]}
        ]

        return {
            "response": response["content"],
            "chat_history": updated_history,
            "current_step": "direct_response",
            "should_end": True,
            "sources": []
        }

# 创建节点实例的函数（用于依赖注入）
def create_nodes(context_manager=None):
    """创建所有节点实例"""
    return {
        "route_query": RouteQueryNode(context_manager=context_manager),
        "advanced_retrieval": AdvancedRetrievalNode(context_manager=context_manager),
        "generate_response": GenerateResponseNode(context_manager=context_manager),
        "direct_response": DirectResponseNode(context_manager=context_manager)
    }
