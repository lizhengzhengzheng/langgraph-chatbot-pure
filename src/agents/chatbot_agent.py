# file: src/agents/chatbot_agent.py
import json
from datetime import datetime
from typing import Dict, Any

from core.session import session_manager, redis_client
from src.core.context import request_context_manager
from src.graph import chatbot_graph_factory
from src.utils.logger import logger


class ChatbotAgent:
    """聊天机器人智能体（支持请求上下文）"""

    async def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """处理用户输入"""
        logger.info(f"[{session_id[:8]}] 处理用户输入: {user_input[:50]}...")

        # 1. 从Redis加载该session_id的会话上下文
        session_context = session_manager.get_session(session_id)
        chat_history = session_context.chat_history

        # 2. 在请求上下文中执行
        async with request_context_manager.async_context(session_id):
            # 3. 创建图实例并执行
            graph = chatbot_graph_factory.get_graph(session_id)
            result = await graph.ainvoke(user_input, chat_history)

            # 4. 更新聊天历史
            session_context.chat_history = result["chat_history"]

            # 5. 限制历史长度
            if len(session_context.chat_history) > 20:
                session_context.chat_history = session_context.chat_history[-20:]

            # 6. 将更新后的历史写回Redis
            session_key = session_manager._get_session_key(session_id)
            session_data = {
                "session_id": session_context.session_id,
                "user_id": session_context.user_id,
                "chat_history": session_context.chat_history,
                "retriever_cache": session_context.retriever_cache,
                "tool_call_state": session_context.tool_call_state,
                "create_time": session_context.create_time.isoformat(),
                "last_active_time": datetime.now().isoformat()
            }
            redis_client.setex(
                name=session_key,
                time=86400,
                value=json.dumps(session_data)
            )

            return {
                "response": result["response"],
                "sources": result.get("sources", []),
                "conversation_id": session_id,
                "request_id": result.get("request_id", ""),
                "current_step": result.get("current_step", ""),
                "tool_used": result.get("tool_used")
            }