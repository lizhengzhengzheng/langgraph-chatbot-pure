# file: src/graph/__init__.py - 增强版
import time
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.core.context import request_context_manager, RequestContextAware
from src.core.session import session_manager
from src.graph.mcp_nodes import create_mcp_nodes
from src.graph.nodes import create_nodes
from src.graph.state import AgentState
from src.utils.logger import logger


class ChatbotGraph(RequestContextAware):
    """聊天机器人图（支持会话增强）"""

    def __init__(self, context_manager=None):
        super().__init__(context_manager)
        self.graph = None
        self.compiled_graph = None
        self.nodes = {}

        # 构建图
        self._build_graph()
        logger.info("✅ ChatbotGraph初始化完成（会话增强版）")

    def _build_graph(self):
        """构建图"""
        logger.info("构建LangGraph (会话增强模式)")

        # 创建所有节点
        self.nodes.update(create_nodes(self.context_manager))
        self.nodes.update(create_mcp_nodes(self.context_manager))

        # 创建图
        workflow = StateGraph(AgentState)

        # 添加所有节点
        for name, node in self.nodes.items():
            workflow.add_node(name, node)

        # 设置入口点
        workflow.set_entry_point("route_query")

        # 扩展路由逻辑
        workflow.add_conditional_edges(
            "route_query",
            self._enhanced_router,
            {
                "retrieve": "advanced_retrieval",
                "direct": "direct_response",
                "mcp": "mcp_routing"
            }
        )

        # MCP路由后的流程
        workflow.add_conditional_edges(
            "mcp_routing",
            lambda s: "mcp_execution" if s.get("should_use_tool") else "generate_response",
            {
                "mcp_execution": "mcp_execution",
                "generate_response": "generate_response"
            }
        )

        # 连接边
        workflow.add_edge("mcp_execution", "generate_response")
        workflow.add_edge("advanced_retrieval", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("direct_response", END)

        # 编译图
        self.graph = workflow
        self.compiled_graph = workflow.compile()

        logger.info("LangGraph构建完成（会话增强）")
        logger.info(f"节点列表: {list(self.compiled_graph.nodes.keys())}")

    @staticmethod
    def _enhanced_router(agent_state: AgentState) -> str:
        """增强的路由逻辑（考虑会话）"""
        user_input = agent_state["user_input"].lower()

        # MCP工具关键词
        mcp_keywords = [
            "天气", "温度", "湿度", "气象", "下雨", "晴天", "多云",
            "计算", "算一下", "等于多少", "加减", "乘除", "面积", "周长",
            "换算", "转换", "公里", "英里", "千克", "磅", "厘米", "英寸",
            "时间", "日期", "现在几点", "今天星期几", "现在时间", "当前日期",
        ]

        # 检查是否触发MCP工具
        needs_mcp = any(keyword in user_input for keyword in mcp_keywords)

        logger.debug(f"路由判断 - 输入: '{user_input}', 需要MCP: {needs_mcp}")

        if needs_mcp:
            logger.info(f"路由到MCP: {user_input[:30]}...")
            return "mcp"
        elif agent_state.get("should_retrieve", False):
            logger.info(f"路由到检索: {user_input[:30]}...")
            return "retrieve"
        else:
            logger.info(f"路由到直接响应: {user_input[:30]}...")
            return "direct"

    async def ainvoke(self, user_input: str, chat_history: list = None) -> Dict[str, Any]:
        """执行图（会话增强）"""
        if chat_history is None:
            chat_history = []

        # 验证当前上下文
        if not self.current_context:
            raise RuntimeError("No request context available. Call within a request context.")

        session_id = self.session_id

        # 准备初始状态（包含会话信息）
        initial_state = {
            "user_input": user_input,
            "chat_history": chat_history,
            "session_id": session_id,  # 新增：传递会话ID

            # 其他字段
            "retrieved_documents": [],
            "context": "",
            "reasoning": "",
            "response": "",
            "sources": [],
            "should_retrieve": False,
            "should_end": False,
            "current_step": "",
            "should_use_tool": False,
            "tool_name": None,
            "tool_result": None,
            "tool_context": None,
            "tool_error": None,

            # 会话增强相关
            "use_session_enhancement": True,
        }

        try:
            # 执行图
            ctx = self.current_context
            logger.info(f"[{ctx.request_id}] 开始执行图，会话: {session_id[:8]}, 输入: {user_input[:50]}...")

            # 获取会话摘要（用于日志）
            session_summary = session_manager.get_session_summary(session_id)
            logger.info(
                f"[{ctx.request_id}] 会话摘要: {session_summary['topics']}, 轮次: {session_summary['total_turns']}")

            result = await self.compiled_graph.ainvoke(initial_state)

            # 构建响应
            response = {
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "chat_history": result.get("chat_history", chat_history),
                "current_step": result.get("current_step", ""),
                "tool_used": result.get("tool_name", None),
                "tool_result": result.get("tool_result", None),
                "tool_context": result.get("tool_context", ""),
                "request_id": ctx.request_id,
                "session_id": session_id,

                # 新增：会话相关信息
                "session_enhanced": result.get("enhanced_query") is not None,
                "relevant_facts_used": len(result.get("relevant_facts", [])),
                "session_topics": result.get("session_context", {}).get("topics", [])
            }

            logger.info(f"[{ctx.request_id}] 图执行完成，步骤: {response['current_step']}")
            logger.info(
                f"[{ctx.request_id}] 会话增强: {response['session_enhanced']}, 使用事实: {response['relevant_facts_used']}")

            if response.get('tool_used'):
                logger.info(f"[{ctx.request_id}] 使用的工具: {response['tool_used']}")

            return response

        except Exception as e:
            ctx = self.current_context
            logger.error(f"[{ctx.request_id if ctx else 'NO_CTX'}] 图执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                "response": f"抱歉，处理您的请求时出现错误: {str(e)}",
                "sources": [],
                "chat_history": chat_history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"处理错误: {str(e)}"}
                ],
                "current_step": "error",
                "error": str(e)
            }


# ChatbotGraphFactory 保持不变（或可以添加会话感知的缓存策略）
class ChatbotGraphFactory:
    """聊天机器人图工厂（可添加会话感知缓存）"""

    def __init__(self):
        self.graph_cache = {}

    def get_graph(self, session_id: str = None) -> ChatbotGraph:
        """获取或创建图实例"""
        if not session_id:
            return ChatbotGraph(context_manager=request_context_manager)

        # 检查缓存
        if session_id in self.graph_cache:
            cached = self.graph_cache[session_id]
            if time.time() - cached['last_used'] < 3600:
                cached['last_used'] = time.time()
                return cached['graph']
            else:
                del self.graph_cache[session_id]

        # 创建新实例
        graph = ChatbotGraph(context_manager=request_context_manager)
        self.graph_cache[session_id] = {
            'graph': graph,
            'last_used': time.time(),
            'created_at': time.time()
        }

        # 限制缓存大小
        if len(self.graph_cache) > 1000:
            oldest = min(self.graph_cache.items(), key=lambda x: x[1]['last_used'])
            del self.graph_cache[oldest[0]]

        return graph


# 创建工厂实例
chatbot_graph_factory = ChatbotGraphFactory()