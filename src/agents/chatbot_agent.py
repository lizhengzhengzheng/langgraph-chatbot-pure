from typing import Dict, Any, List
from src.graph import chatbot_graph
from src.utils.logger import logger


class ChatbotAgent:
    """聊天机器人智能体"""

    def __init__(self):
        self.chat_history = []

    async def chat(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入"""
        logger.info(f"处理用户输入: {user_input[:50]}...")

        # 调用LangGraph流程
        result = await chatbot_graph.ainvoke(user_input, self.chat_history)

        # 更新聊天历史
        self.chat_history = result["chat_history"]

        # 限制历史长度
        if len(self.chat_history) > 20:  # 最多保存10轮对话
            self.chat_history = self.chat_history[-20:]

        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "chat_history": self.chat_history
        }

    def clear_history(self):
        """清空聊天历史"""
        self.chat_history = []
        logger.info("聊天历史已清空")

    def get_history(self) -> List[Dict[str, str]]:
        """获取聊天历史"""
        return self.chat_history