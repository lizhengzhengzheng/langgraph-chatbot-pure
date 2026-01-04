# file: src/agents/chatbot_agent.py - 增强版
from typing import Dict, Any

from src.core.context import request_context_manager
from src.core.session import session_manager
from src.graph import chatbot_graph_factory
from src.utils.logger import logger


class ChatbotAgent:
    """聊天机器人智能体（会话增强版）"""

    async def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """处理用户输入（会话感知）"""
        logger.info(f"[{session_id[:8]}] 处理用户输入: {user_input[:50]}...")

        # 1. 获取会话上下文（增强版）
        session_context = session_manager.get_session(session_id)
        chat_history = session_context.chat_history

        # 2. 在请求上下文中执行
        async with request_context_manager.async_context(session_id):
            # 3. 创建图实例并执行
            graph = chatbot_graph_factory.get_graph(session_id)
            result = await graph.ainvoke(user_input, chat_history)

            # 4. 返回结果（不再需要手动更新历史，节点中已处理）
            return {
                "response": result["response"],
                "sources": result.get("sources", []),
                "conversation_id": session_id,
                "request_id": result.get("request_id", ""),
                "current_step": result.get("current_step", ""),
                "tool_used": result.get("tool_used"),

                # 新增：会话增强信息
                "session_enhanced": result.get("session_enhanced", False),
                "session_topics": result.get("session_topics", []),
                "relevant_facts_used": result.get("relevant_facts_used", 0)
            }

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息（新增API）"""
        try:
            session_context = session_manager.get_session(session_id)
            summary = session_manager.get_session_summary(session_id)

            return {
                "success": True,
                "session_id": session_id,
                "session_info": {
                    "total_turns": summary["total_turns"],
                    "topics": summary["topics"],
                    "user_preferences": summary["user_preferences"],
                    "important_facts": summary["important_facts"],
                    "preferred_documents": summary["preferred_documents"],
                    "recent_tools": summary["recent_tools"],
                    "create_time": session_context.create_time.isoformat(),
                    "last_active": session_context.last_active_time.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"获取会话信息失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_user_preferences(self, session_id: str,
                                      preferences: Dict[str, Any]) -> Dict[str, Any]:
        """更新用户偏好（新增API）"""
        try:
            session_context = session_manager.get_session(session_id)

            # 更新用户偏好
            if "response_style" in preferences:
                session_context.user_profile["response_style"] = preferences["response_style"]
            if "technical_level" in preferences:
                session_context.user_profile["technical_level"] = preferences["technical_level"]
            if "preferred_language" in preferences:
                session_context.user_profile["preferred_language"] = preferences["preferred_language"]
            if "interests" in preferences:
                session_context.user_profile["interests"] = preferences["interests"]

            # 保存
            session_manager._save_session(session_context)

            return {
                "success": True,
                "message": "用户偏好已更新",
                "updated_preferences": session_context.user_profile
            }
        except Exception as e:
            logger.error(f"更新用户偏好失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }