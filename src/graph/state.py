# file: src/graph/state.py
from typing import List, Dict, Any, TypedDict, Optional

class AgentState(TypedDict):
    """智能体状态（纯业务状态）"""
    # 输入
    user_input: str
    chat_history: List[Dict[str, str]]

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