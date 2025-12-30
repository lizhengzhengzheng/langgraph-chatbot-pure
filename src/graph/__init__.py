# src/graph/__init__.py
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    route_query,
    retrieve_documents,
    generate_response,
    direct_response
)
# 导入MCP节点
from src.graph.mcp_nodes import MCPRoutingNode, MCPExecutionNode
from src.graph.edges import should_retrieve, after_retrieve
from src.utils.logger import logger


class ChatbotGraph:
    """聊天机器人图"""

    def __init__(self):
        self.graph = None
        self.compiled_graph = None

        # 初始化MCP节点实例
        self.mcp_routing_node = MCPRoutingNode()
        self.mcp_execution_node = MCPExecutionNode()

        # 构建图
        self._build_graph()
        logger.info("✅ ChatbotGraph初始化完成，包含MCP路由功能")

    def _build_graph(self):
        """构建图"""
        logger.info("构建LangGraph (含MCP路由)")

        # 创建图
        workflow = StateGraph(AgentState)

        # 添加基础节点
        workflow.add_node("route_query", route_query)
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("generate_response", generate_response)
        workflow.add_node("direct_response", direct_response)

        # 添加MCP节点 - 使用实例方法而非类
        workflow.add_node("mcp_routing", self.mcp_routing_node)
        workflow.add_node("mcp_execution", self.mcp_execution_node)

        # 设置入口点
        workflow.set_entry_point("route_query")

        # 扩展路由逻辑：判断是否需要MCP工具
        workflow.add_conditional_edges(
            "route_query",
            self._enhanced_router,
            {
                "retrieve": "retrieve_documents",
                "direct": "direct_response",
                "mcp": "mcp_routing"  # 新增MCP路径
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

        # MCP执行后生成回答
        workflow.add_edge("mcp_execution", "generate_response")

        # 原有边保持不变
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("direct_response", END)

        # 编译图
        self.graph = workflow
        self.compiled_graph = workflow.compile()

        logger.info("LangGraph构建完成")
        logger.info(f"节点列表: {list(self.compiled_graph.nodes.keys())}")

    def _enhanced_router(self, state: AgentState) -> str:
        """增强的路由逻辑，包含MCP判断"""
        user_input = state["user_input"].lower()

        # MCP工具关键词（可根据业务扩展）
        mcp_keywords = [
            # 天气相关
            "天气", "温度", "湿度", "气象", "下雨", "晴天", "多云",
            # 计算相关
            "计算", "算一下", "等于多少", "加减", "乘除", "面积", "周长",
            "求和", "平均数", "百分比",
            # 单位换算
            "换算", "转换", "公里", "英里", "千克", "磅", "厘米", "英寸",
            "升", "加仑", "摄氏度", "华氏度",
            # 时间日期
            "时间", "日期", "现在几点", "今天星期几", "现在时间", "当前日期",
            "时间戳", "时区",
            # 文本处理
            "统计", "分析文本", "字数", "字符数", "摘要", "总结", "文本长度"
        ]

        # 检查是否触发MCP工具
        needs_mcp = any(keyword in user_input for keyword in mcp_keywords)

        logger.debug(f"路由判断 - 输入: '{user_input}', 需要MCP: {needs_mcp}")

        if needs_mcp:
            logger.info(f"路由到MCP: {user_input[:30]}...")
            return "mcp"
        elif state.get("should_retrieve", False):
            logger.info(f"路由到检索: {user_input[:30]}...")
            return "retrieve"
        else:
            logger.info(f"路由到直接响应: {user_input[:30]}...")
            return "direct"

    async def ainvoke(self, user_input: str, chat_history: list = None) -> dict:
        """执行图"""
        if chat_history is None:
            chat_history = []

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
            "current_step": ""
        }

        try:
            # 执行图
            logger.info(f"开始执行图，输入: {user_input[:50]}...")
            result = await self.compiled_graph.ainvoke(initial_state)

            # 返回格式化的结果
            response = {
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "chat_history": result.get("chat_history", chat_history),
                "current_step": result.get("current_step", ""),
                "tool_used": result.get("tool_used", None),
                "tool_result": result.get("tool_result", None),
                "tool_context": result.get("tool_context", "")
            }

            logger.info(f"图执行完成，步骤: {response['current_step']}")
            if response.get('tool_used'):
                logger.info(f"使用的工具: {response['tool_used']}")

            return response

        except Exception as e:
            logger.error(f"图执行失败: {e}")
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

    def visualize(self):
        """可视化图结构"""
        try:
            # 生成Mermaid格式文本
            mermaid_text = self.compiled_graph.get_graph().draw_mermaid()
            print("=== Mermaid 图定义 ===")
            print(mermaid_text)
            print("\n复制上述文本到 https://mermaid.live 查看流程图")

            # 打印节点信息
            print(f"\n=== 图节点 ===")
            for node_name in self.compiled_graph.nodes:
                print(f"  - {node_name}")

        except Exception as e:
            logger.warning(f"无法可视化图: {str(e)}")
            # 打印简单的文本表示
            print(f"图节点: {list(self.compiled_graph.nodes.keys())}")


# 创建全局实例
chatbot_graph = ChatbotGraph()