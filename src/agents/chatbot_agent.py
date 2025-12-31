import json
from datetime import datetime
from typing import Dict, Any

from core.session import session_manager, redis_client
from src.graph import chatbot_graph
from src.utils.logger import logger


class ChatbotAgent:
    """聊天机器人智能体"""

    async def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """处理用户输入"""
        logger.info(f"处理用户输入: {user_input[:50]}... | session_id: {session_id}")

        # 1. 从Redis加载该session_id的会话上下文（含聊天历史）
        session_context = session_manager.get_session(session_id)
        chat_history = session_context.chat_history  # 获取该会话的专属历史

        # 调用LangGraph流程
        result = await chatbot_graph.ainvoke(user_input, chat_history)

        # 更新聊天历史
        session_context.chat_history = result["chat_history"]

        # 限制历史长度
        if len(session_context.chat_history) > 20:
            session_context.chat_history = session_context.chat_history[-20:]

        # 4. 将更新后的历史写回Redis（关键：持久化多轮记忆）
        session_key = session_manager._get_session_key(session_id)
        # 重新序列化整个会话数据
        session_data = {
            "session_id": session_context.session_id,
            "user_id": session_context.user_id,
            "chat_history": session_context.chat_history,
            "retriever_cache": session_context.retriever_cache,
            "tool_call_state": session_context.tool_call_state,
            "create_time": session_context.create_time.isoformat(),
            "last_active_time": datetime.now().isoformat()  # 更新最后活跃时间
        }
        redis_client.setex(
            name=session_key,
            time=86400,
            value=json.dumps(session_data)
        )
        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "conversation_id": session_id
        }