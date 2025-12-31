# file: src/graph/__init__.py
import time

from langgraph.graph import StateGraph, END

from src.core.context import request_context_manager, RequestContextAware
from src.graph.mcp_nodes import create_mcp_nodes
from src.graph.nodes import create_nodes
from src.graph.state import AgentState
from src.utils.logger import logger


class ChatbotGraph(RequestContextAware):
    """聊天机器人图（支持请求上下文）"""

    def __init__(self, context_manager=None):
        super().__init__(context_manager)
        self.graph = None
        self.compiled_graph = None

        # 创建节点实例
        self.nodes = {}

        # 构建图
        self._build_graph()
        logger.info("✅ ChatbotGraph初始化完成")

    def _build_graph(self):
        """构建图"""
        logger.info("构建LangGraph (依赖注入模式)")

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
                "retrieve": "advanced_retrieval",  # 使用高级检索
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

        logger.info("LangGraph构建完成")
        logger.info(f"节点列表: {list(self.compiled_graph.nodes.keys())}")

    @staticmethod
    def _enhanced_router(agent_state: AgentState) -> str:
        """增强的路由逻辑，包含MCP判断"""
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

    async def ainvoke(self, user_input: str, chat_history: list = None) -> dict:
        """执行图（依赖上下文中的session_id）"""
        if chat_history is None:
            chat_history = []

        # 验证当前上下文
        if not self.current_context:
            raise RuntimeError("No request context available. Call within a request context.")

        # 准备初始状态
        initial_state = {
            "user_input": user_input,
            "chat_history": chat_history,
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
            "tool_error": None
        }

        try:
            # 执行图
            ctx = self.current_context
            logger.info(f"[{ctx.request_id}] 开始执行图，输入: {user_input[:50]}...")

            result = await self.compiled_graph.ainvoke(initial_state)

            # 返回格式化的结果
            response = {
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "chat_history": result.get("chat_history", chat_history),
                "current_step": result.get("current_step", ""),
                "tool_used": result.get("tool_name", None),
                "tool_result": result.get("tool_result", None),
                "tool_context": result.get("tool_context", ""),
                "request_id": ctx.request_id,
                "session_id": ctx.session_id
            }

            logger.info(f"[{ctx.request_id}] 图执行完成，步骤: {response['current_step']}")
            if response.get('tool_used'):
                logger.info(f"[{ctx.request_id}] 使用的工具: {response['tool_used']}")

            return response

        except Exception as e:
            ctx = self.current_context
            logger.error(f"[{ctx.request_id if ctx else 'NO_CTX'}] 图执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回错误状态
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


class ChatbotGraphFactory:
    """聊天机器人图工厂"""

    def __init__(self):
        self.graph_cache = {}

    def get_graph(self, session_id: str = None) -> ChatbotGraph:
        """获取或创建图实例（带缓存）"""

        if not session_id:
            # 无会话ID时创建新实例
            return ChatbotGraph(context_manager=request_context_manager)

        # 检查缓存
        if session_id in self.graph_cache:
            cached = self.graph_cache[session_id]

            # 检查是否过期（例如会话长时间未使用）
            if time.time() - cached['last_used'] < 3600:  # 1小时过期
                cached['last_used'] = time.time()
                return cached['graph']
            else:
                # 过期删除
                del self.graph_cache[session_id]

        # 创建新实例并缓存
        graph = ChatbotGraph(context_manager=request_context_manager)
        self.graph_cache[session_id] = {
            'graph': graph,
            'last_used': time.time(),
            'created_at': time.time()
        }

        # 限制缓存大小（LRU策略）
        if len(self.graph_cache) > 1000:
            oldest = min(self.graph_cache.items(), key=lambda x: x[1]['last_used'])
            del self.graph_cache[oldest[0]]

        return graph


# 创建工厂实例
chatbot_graph_factory = ChatbotGraphFactory()
