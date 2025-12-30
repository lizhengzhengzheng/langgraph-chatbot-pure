# src/graph/edges.py
from typing import Literal
from src.graph.state import AgentState

def should_retrieve(state: AgentState) -> Literal["retrieve", "direct_response"]:
    """判断是否需要检索文档"""
    return "retrieve" if state.get("should_retrieve", False) else "direct_response"

def after_retrieve(state: AgentState) -> Literal["generate_response"]:
    """检索后执行生成响应"""
    return "generate_response"