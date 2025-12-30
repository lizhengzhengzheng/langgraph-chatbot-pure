from typing import Dict, Any
from src.graph.state import AgentState
from src.core.vector_store import vector_store
from src.core.llm_client import llm_client
from src.utils.logger import logger


def retrieve_documents(state: AgentState) -> Dict[str, Any]:
    """检索相关文档节点"""
    logger.info("执行文档检索")

    user_input = state["user_input"]
    chat_history = state["chat_history"]

    # 构建增强查询
    enhanced_query = user_input
    if chat_history:
        # 添加最近的历史上下文
        recent_history = chat_history[-3:]  # 最近3轮对话
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        enhanced_query = f"基于以下对话历史回答问题:\n{history_text}\n\n问题: {user_input}"

    # 检索文档
    retrieved_docs = vector_store.search(enhanced_query, top_k=5)

    # 构建上下文
    context = ""
    sources = []

    for i, doc in enumerate(retrieved_docs):
        context += f"[文档{i + 1}]: {doc['text']}\n\n"
        sources.append({
            "title": doc.get("title", f"文档{i + 1}"),
            "source": doc.get("source", ""),
            "score": doc["score"]
        })

    return {
        "retrieved_documents": retrieved_docs,
        "context": context,
        "sources": sources,
        "current_step": "retrieve_documents"
    }


def generate_response(state: AgentState) -> Dict[str, Any]:
    """生成响应节点 - 增强版，能处理工具结果"""

    user_input = state["user_input"]
    chat_history = state["chat_history"]
    context = state.get("context", "")

    # 如果有工具执行结果，整合到上下文中
    tool_result = state.get("tool_result")
    tool_used = state.get("tool_used")

    system_message = """你是一个智能助手，基于用户的问题和提供的上下文信息来回答问题。"""

    if tool_result and not tool_result.get("error"):
        # 将工具结果作为额外上下文
        tool_context = f"\n[工具调用结果] 使用工具 '{tool_used}' 得到：{tool_result}"
        system_message += tool_context

    # 构建消息列表（原有逻辑）
    messages = [{"role": "system", "content": system_message}]

    if context:
        context_prompt = f"相关上下文：\n{context}\n请基于以上信息回答。"
        messages.append({"role": "system", "content": context_prompt})

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_input})

    # 调用大模型
    response = llm_client.chat_completion(messages=messages, temperature=0.7)

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


def route_query(state: AgentState) -> Dict[str, Any]:
    """路由查询节点 - 决定是否需要检索文档"""
    logger.info("路由查询")

    user_input = state["user_input"]

    # 判断是否需要检索文档
    # 这里可以根据查询内容来决定是否需要检索
    # 例如：如果是一般性对话，可能不需要检索；如果是具体问题，需要检索

    # 简单的关键词判断
    retrieval_keywords = ["什么", "如何", "为什么", "怎样", "解释", "介绍", "说明"]
    should_retrieve = any(keyword in user_input for keyword in retrieval_keywords)

    # 如果没有明确关键词，但也可能是问题
    if not should_retrieve and len(user_input) > 10 and "?" in user_input:
        should_retrieve = True

    return {
        "should_retrieve": should_retrieve,
        "current_step": "route_query"
    }


def direct_response(state: AgentState) -> Dict[str, Any]:
    """直接响应节点 - 当不需要检索时直接回答"""
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

    response = llm_client.chat_completion(
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
        "sources": []  # 直接响应没有来源
    }


from src.core.advanced_retriever import HybridRetriever, Reranker


class AdvancedRetrievalNode:
    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("执行高级检索流程")

        user_input = state["user_input"]

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