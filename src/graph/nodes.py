# file: src/graph/nodes.py - 增强版
from typing import Dict, Any, List

from src.core.advanced_retriever import HybridRetriever, Reranker
from src.core.context import RequestContextAware
from src.core.llm_client import llm_client
from src.core.session import session_manager
from src.graph.state import AgentState
from src.utils.logger import logger


class BaseNode(RequestContextAware):
    """节点基类（支持会话增强）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RouteQueryNode(BaseNode):
    """路由查询节点（增强版）"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("路由查询（会话增强）")

        user_input = state["user_input"]
        session_id = self.session_id

        # 获取会话分析
        session_context = session_manager.get_session(session_id)
        session_analysis = session_context.conversation_analysis

        # 基于会话历史的智能判断
        should_retrieve = self._should_retrieve_with_context(
            user_input, session_analysis
        )

        # 检查是否需要会话增强
        use_session_enhancement = self._should_use_session_enhancement(
            user_input, session_analysis
        )

        return {
            "should_retrieve": should_retrieve,
            "use_session_enhancement": use_session_enhancement,
            "session_id": session_id,  # 传递会话ID
            "current_step": "route_query"
        }

    def _should_retrieve_with_context(self, query: str, session_analysis: Dict[str, Any]) -> bool:
        """基于会话上下文判断是否需要检索"""
        # 基础关键词判断
        retrieval_keywords = ["什么", "如何", "为什么", "怎样", "解释", "介绍", "说明"]
        basic_retrieve = any(keyword in query for keyword in retrieval_keywords)

        # 基于会话历史的判断
        topics = session_analysis.get("topics", [])
        if topics and any(topic in query for topic in topics[:3]):
            # 如果查询与会话主题相关，很可能需要检索
            return True

        # 如果是技术性问题且用户偏好详细回答
        tech_keywords = ["代码", "实现", "原理", "架构", "配置"]
        if any(keyword in query for keyword in tech_keywords):
            return True

        return basic_retrieve or len(query) > 10 and "?" in query

    def _should_use_session_enhancement(self, query: str, session_analysis: Dict[str, Any]) -> bool:
        """判断是否需要会话增强"""
        total_turns = session_analysis.get("total_turns", 0)
        topics = session_analysis.get("topics", [])

        # 如果对话轮次多且有明确主题，使用会话增强
        if total_turns > 3 and len(topics) > 0:
            return True

        # 如果查询很短，可能需要会话上下文
        if len(query) < 6:
            return True

        return False


class SessionEnhancedRetrievalNode(BaseNode):
    """会话增强检索节点（新增）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("执行会话增强检索")

        user_input = state["user_input"]
        session_id = state.get("session_id", self.session_id)
        use_session_enhancement = state.get("use_session_enhancement", True)

        try:
            # 1. 获取会话上下文
            session_context = session_manager.get_session(session_id)

            # 2. 会话增强查询
            if use_session_enhancement:
                enhanced_query = session_manager.enhance_query_with_session(
                    session_id, user_input
                )
                logger.info(f"[{session_id[:8]}] 增强查询: {user_input} -> {enhanced_query}")
            else:
                enhanced_query = user_input

            # 3. 获取相关重要事实
            relevant_facts = session_context.get_relevant_facts(user_input, limit=3)

            # 4. 获取偏好文档
            preferred_doc_ids = session_context.get_preferred_doc_ids(limit=3)

            # 5. 执行混合检索（考虑会话偏好）
            candidates = self.hybrid_retriever.retrieve(
                query=enhanced_query,
                session_id=session_id,
                preferred_doc_ids=preferred_doc_ids,
                top_k=20  # 召回更多结果用于重排
            )

            # 6. 重排（考虑会话相关性）
            reranked_results = self._rerank_with_session_context(
                enhanced_query, candidates, session_context, top_k=5
            )

            # 7. 更新文档偏好
            for result in reranked_results:
                if "id" in result:
                    session_context.update_document_preference(
                        result["id"], result.get("relevance_score", 0.5)
                    )

            # 8. 构建上下文
            context = self._build_enhanced_context(
                reranked_results, relevant_facts, session_context
            )

            sources = [{
                "title": res.get("title", f"文档{i + 1}"),
                "source": res.get("source", ""),
                "score": res.get("relevance_score", 0.0),
                "session_relevant": res.get("session_relevant", False)
            } for i, res in enumerate(reranked_results)]

            # 9. 保存更新后的会话
            session_manager._save_session(session_context)

            return {
                "retrieved_documents": reranked_results,
                "context": context,
                "sources": sources,
                "relevant_facts": relevant_facts,
                "enhanced_query": enhanced_query if use_session_enhancement else None,
                "session_context": {
                    "topics": session_context.conversation_analysis.get("topics", []),
                    "total_turns": len(session_context.chat_history)
                },
                "current_step": "advanced_retrieval"
            }

        except Exception as e:
            logger.error(f"会话增强检索失败: {e}")
            # 降级到基础检索
            return await self._fallback_retrieval(user_input, session_id)

    def _rerank_with_session_context(self, query: str, candidates: List[Dict],
                                     session_context: Any, top_k: int = 5) -> List[Dict]:
        """考虑会话上下文的重排"""
        if not candidates:
            return []

        # 基础重排
        reranked = self.reranker.rerank(query, candidates, top_k=top_k * 2)

        # 应用会话偏好
        for result in reranked:
            doc_id = result.get("id")
            if doc_id in session_context.document_preferences:
                # 提升偏好文档的分数
                preference_weight = session_context.document_preferences[doc_id]
                result["relevance_score"] = result.get("relevance_score", 0.5) * (1 + preference_weight)
                result["session_relevant"] = True

        # 重新排序
        reranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return reranked[:top_k]

    def _build_enhanced_context(self, documents: List[Dict], relevant_facts: List[str],
                                session_context: Any) -> str:
        """构建增强的上下文"""
        context_parts = []

        # 1. 相关重要事实
        if relevant_facts:
            context_parts.append("【会话记忆】")
            for i, fact in enumerate(relevant_facts, 1):
                context_parts.append(f"{i}. {fact}")

        # 2. 检索到的文档
        if documents:
            context_parts.append("\n【参考信息】")
            for i, doc in enumerate(documents, 1):
                score = doc.get("relevance_score", 0)
                session_flag = "★" if doc.get("session_relevant") else ""
                context_parts.append(
                    f"{session_flag}参考{i} (相关性:{score:.2f}): {doc.get('text', '')[:300]}"
                )

        # 3. 会话主题提示
        topics = session_context.conversation_analysis.get("topics", [])
        if topics:
            context_parts.append(f"\n【当前会话主题】: {', '.join(topics[:3])}")

        return "\n".join(context_parts)

    async def _fallback_retrieval(self, query: str, session_id: str) -> Dict[str, Any]:
        """降级检索"""
        from src.core.vector_store import vector_store

        retrieved_docs = vector_store.search(query, top_k=5)
        context = "\n".join([f"[文档{i + 1}]: {doc['text'][:200]}"
                             for i, doc in enumerate(retrieved_docs)])

        return {
            "retrieved_documents": retrieved_docs,
            "context": context,
            "sources": [],
            "current_step": "advanced_retrieval",
            "retrieval_error": "降级到基础检索"
        }


class GenerateResponseNode(BaseNode):
    """生成响应节点（会话增强版）"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        user_input = state["user_input"]
        chat_history = state["chat_history"]
        context = state.get("context", "")
        session_id = state.get("session_id", self.session_id)

        # 获取会话信息
        session_context = session_manager.get_session(session_id)

        # 构建个性化系统消息
        system_message = self._build_personalized_system_message(
            session_context, state
        )

        # 如果有工具执行结果，整合到上下文中
        tool_result = state.get("tool_result")
        tool_used = state.get("tool_name")

        if tool_result and not tool_result.get("error"):
            tool_context = f"\n[工具调用结果] 使用工具 '{tool_used}' 得到：{tool_result}"
            system_message += tool_context

        # 构建消息列表（考虑会话历史）
        messages = [{"role": "system", "content": system_message}]

        # 添加上下文
        if context:
            messages.append({"role": "system", "content": f"相关上下文：\n{context}"})

        # 添加会话历史（智能选择）
        relevant_history = self._select_relevant_history(
            chat_history, user_input, session_context
        )
        for msg in relevant_history:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        # 调用大模型
        response = await llm_client.async_chat_completion(
            messages=messages,
            temperature=self._get_temperature(session_context),
            max_tokens=self._get_max_tokens(session_context)
        )

        # 提取可能的重要事实
        self._extract_and_save_facts(
            session_context, user_input, response["content"]
        )

        # 更新历史
        tool_info = f" (使用工具: {tool_used})" if tool_used else ""
        updated_history = chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response["content"] + tool_info}
        ]

        # 保存更新
        session_manager.update_chat_history(
            session_id, user_input, response["content"]
        )

        return {
            "response": response["content"],
            "chat_history": updated_history,
            "current_step": "generate_response",
            "should_end": True
        }

    def _build_personalized_system_message(self, session_context: Any, state: AgentState) -> str:
        """构建个性化系统消息"""
        profile = session_context.user_profile
        analysis = session_context.conversation_analysis

        base_message = """你是一个智能助手，基于用户的问题和提供的上下文信息来回答问题。"""

        # 个性化设置
        personalized_parts = []

        # 响应风格
        style = profile.get("response_style", "balanced")
        if style == "concise":
            personalized_parts.append("请保持回答简洁明了，直接回答问题核心。")
        elif style == "detailed":
            personalized_parts.append("请提供详细全面的回答，包含必要的解释和示例。")

        # 专业程度
        tech_level = profile.get("technical_level", "intermediate")
        if tech_level == "beginner":
            personalized_parts.append("请用通俗易懂的语言解释，避免专业术语。")
        elif tech_level == "advanced":
            personalized_parts.append("可以深入讨论技术细节，使用专业术语。")

        # 会话连续性
        topics = analysis.get("topics", [])
        if topics:
            personalized_parts.append(
                f"注意当前会话主题包括：{', '.join(topics[:3])}。"
            )

        # 重要事实提醒
        relevant_facts = state.get("relevant_facts", [])
        if relevant_facts:
            personalized_parts.append("以下信息可能相关：")
            for fact in relevant_facts[:2]:
                personalized_parts.append(f"- {fact}")

        if personalized_parts:
            base_message += "\n\n个性化要求：\n" + "\n".join(personalized_parts)

        return base_message

    def _select_relevant_history(self, chat_history: List[Dict],
                                 current_query: str, session_context: Any) -> List[Dict]:
        """智能选择相关历史"""
        if not chat_history:
            return []

        # 如果历史很短，返回全部
        if len(chat_history) <= 6:
            return chat_history[-6:]

        # 分析当前查询
        query_keywords = set(current_query.lower().split()[:5])

        # 计算每条历史消息的相关性
        relevant_messages = []
        for msg in chat_history[-10:]:  # 检查最近10条
            if msg["role"] == "user":
                content = msg["content"].lower()
                # 简单关键词匹配
                if any(keyword in content for keyword in query_keywords):
                    relevant_messages.append(msg)

        # 如果找到相关消息，返回它们
        if relevant_messages:
            return relevant_messages[-4:]  # 最多4条相关消息

        # 否则返回最近的消息
        return chat_history[-4:]

    def _get_temperature(self, session_context: Any) -> float:
        """根据会话获取温度参数"""
        tone = session_context.conversation_analysis.get("conversation_tone", "neutral")
        if tone == "formal":
            return 0.3
        elif tone == "casual":
            return 0.8
        return 0.7

    def _get_max_tokens(self, session_context: Any) -> int:
        """根据用户偏好获取最大token数"""
        style = session_context.user_profile.get("response_style", "balanced")
        if style == "concise":
            return 500
        elif style == "detailed":
            return 1500
        return 1000

    def _extract_and_save_facts(self, session_context: Any,
                                user_input: str, response: str):
        """提取并保存重要事实"""
        # 检查是否是重要信息
        important_indicators = [
            "是", "有", "可以", "应该", "需要", "必须", "建议",
            "记住", "重要", "关键", "注意", "原则", "规则"
        ]

        sentences = response.split('。')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence for indicator in important_indicators):
                # 提取简洁的事实
                fact = sentence[:100]  # 限制长度
                if len(fact) > 20:  # 足够长
                    session_context.add_important_fact(fact)


class DirectResponseNode(BaseNode):
    """直接响应节点（会话感知版）"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("直接响应（会话感知）")

        user_input = state["user_input"]
        chat_history = state["chat_history"]
        session_id = self.session_id

        # 获取会话信息
        session_context = session_manager.get_session(session_id)

        # 个性化系统消息
        system_message = self._build_friendly_message(session_context)

        # 构建消息
        messages = [{"role": "system", "content": system_message}]

        # 添加最近历史
        for msg in chat_history[-4:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        # 调用大模型
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

        # 保存到会话
        session_manager.update_chat_history(session_id, user_input, response["content"])

        return {
            "response": response["content"],
            "chat_history": updated_history,
            "current_step": "direct_response",
            "should_end": True,
            "sources": []
        }

    def _build_friendly_message(self, session_context: Any) -> str:
        """构建友好消息"""
        name = session_context.user_profile.get("name", "朋友")
        tone = session_context.conversation_analysis.get("conversation_tone", "neutral")

        if tone == "polite":
            return f"你是一个友好、礼貌的助手，正在与{name}对话。请热情但专业地回答。"
        else:
            return f"你是一个友好的助手，正在与{name}对话。请直接、清晰地回答问题。"


# 创建节点实例的函数
def create_nodes(context_manager=None):
    """创建所有节点实例（增强版）"""
    return {
        "route_query": RouteQueryNode(context_manager=context_manager),
        "advanced_retrieval": SessionEnhancedRetrievalNode(context_manager=context_manager),  # 替换为增强版
        "generate_response": GenerateResponseNode(context_manager=context_manager),
        "direct_response": DirectResponseNode(context_manager=context_manager)
    }