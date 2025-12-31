# file: src/core/context.py
"""请求上下文管理模块"""
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RequestContext:
    """请求上下文数据"""
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestContextManager:
    """请求上下文管理器"""

    def __init__(self):
        # 全局上下文变量
        self._current_context = ContextVar('current_request_context', default=None)

    def set_context(self, session_id: str, user_id: Optional[str] = None) -> RequestContext:
        """设置当前请求上下文"""
        context = RequestContext(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            user_id=user_id
        )
        self._current_context.set(context)
        return context

    def get_context(self) -> Optional[RequestContext]:
        """获取当前请求上下文"""
        return self._current_context.get()

    def clear_context(self):
        """清除当前请求上下文"""
        self._current_context.set(None)

    @asynccontextmanager
    async def async_context(self, session_id: str, user_id: Optional[str] = None):
        """异步上下文管理器"""
        context = self.set_context(session_id, user_id)
        try:
            yield context
        finally:
            self.clear_context()

class RequestContextAware:
    """支持请求上下文感知的基类"""

    def __init__(self, context_manager: Optional[RequestContextManager] = None):
        self.context_manager = context_manager or request_context_manager

    @property
    def current_context(self) -> Optional[RequestContext]:
        """获取当前上下文"""
        return self.context_manager.get_context()

    @property
    def session_id(self) -> str:
        """获取当前会话ID"""
        ctx = self.current_context
        if not ctx:
            raise RuntimeError("No request context available")
        return ctx.session_id


# 全局上下文管理器实例
request_context_manager = RequestContextManager()
