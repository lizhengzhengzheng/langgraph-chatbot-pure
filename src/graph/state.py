from typing import List, Dict, Any, TypedDict

# 定义统一数据格式。用 TypedDict 明确规定在整个流程中传递的“数据包裹”（AgentState）里必须有哪些字段（如user_input, context等）。
class AgentState(TypedDict):
    """智能体状态"""

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