# file: src/graph/state.py
from typing import List, Dict, Any, TypedDict, Optional, NotRequired


class AgentState(TypedDict):
    """智能体状态（包含会话增强）"""
    # 输入
    user_input: str
    chat_history: List[Dict[str, str]]
    session_id: NotRequired[str]  # 新增：会话ID

    # 会话上下文（新增）
    session_context: NotRequired[Dict[str, Any]]  # 会话分析结果
    relevant_facts: NotRequired[List[str]]  # 相关重要事实
    enhanced_query: NotRequired[str]  # 增强后的查询

    # 处理过程
    retrieved_documents: List[Dict[str, Any]]
    context: str
    reasoning: str

    # 输出
    response: str
    sources: List[Dict[str, Any]]

    # 控制流
    should_retrieve: bool
    should_end: bool
    current_step: str

    # MCP相关
    should_use_tool: bool
    tool_name: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    tool_context: Optional[str]
    tool_error: Optional[str]

    # 会话优化相关（新增）
    use_session_enhancement: NotRequired[bool]  # 是否使用会话增强
    personalized_context: NotRequired[str]  # 个性化上下文