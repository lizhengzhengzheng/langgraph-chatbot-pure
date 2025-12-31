# file: src/graph/mcp_nodes.py
from typing import Dict, Any

from src.core.context import RequestContextAware
from src.core.mcp_client import mcp_client
from src.graph.state import AgentState
from src.utils.logger import logger


class MCPRoutingNode(RequestContextAware):
    """MCP智能路由节点"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("MCP智能路由节点执行")

        user_input = state["user_input"]
        chat_history = state.get("chat_history", [])

        # 构建上下文
        context = ""
        if chat_history:
            recent = chat_history[-3:]
            context = "对话历史:\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in recent]
            )

        # 调用智能工具调用
        try:
            tool_result = await mcp_client.intelligent_tool_call(user_input, context)
        except Exception as e:
            logger.error(f"智能工具调用失败: {e}")
            return {
                "should_use_tool": False,
                "current_step": "mcp_routing",
                "tool_error": str(e)
            }

        if "error" in tool_result:
            logger.warning(f"工具调用失败: {tool_result['error']}")
            return {
                "should_use_tool": False,
                "current_step": "mcp_routing",
                "tool_error": tool_result["error"]
            }
        else:
            return {
                "should_use_tool": True,
                "tool_name": tool_result.get("tool_used"),
                "tool_result": tool_result.get("tool_result"),
                "tool_reasoning": tool_result.get("reasoning", ""),
                "current_step": "mcp_routing"
            }


class MCPExecutionNode(RequestContextAware):
    """MCP工具执行后处理节点"""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("MCP工具执行后处理")

        tool_result = state.get("tool_result", {})
        tool_name = state.get("tool_name", "未知工具")

        # 准备工具结果作为上下文
        tool_context = f"\n[工具调用结果] 使用工具 '{tool_name}' 得到: {tool_result}"

        return {
            "tool_context": tool_context,
            "current_step": "mcp_execution"
        }


def create_mcp_nodes(context_manager=None):
    """创建MCP节点实例"""
    return {
        "mcp_routing": MCPRoutingNode(context_manager=context_manager),
        "mcp_execution": MCPExecutionNode(context_manager=context_manager)
    }

