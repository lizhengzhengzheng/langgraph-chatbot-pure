# src/core/session.py 完整修复版
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any

import redis
from fastapi import Request, HTTPException

from config.settings import settings


# 单会话上下文结构
@dataclass
class SessionContext:
    session_id: str  # 会话唯一ID
    user_id: str = ""  # 可选：绑定用户
    chat_history: List[Dict[str, Any]] = field(default_factory=list)  # 聊天历史
    retriever_cache: Dict[str, Any] = field(default_factory=dict)  # 检索结果缓存
    tool_call_state: Dict[str, Any] = field(default_factory=dict)  # 工具调用状态
    create_time: datetime = field(default_factory=datetime.now)
    last_active_time: datetime = field(default_factory=datetime.now)


# 初始化Redis客户端（建议通过配置文件管理）
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    decode_responses=True
)


class SessionManager:
    # 会话Key前缀，避免与其他Key冲突
    SESSION_PREFIX = "chat:session:"

    def _get_session_key(self, session_id: str) -> str:
        return f"{self.SESSION_PREFIX}{session_id}"

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def create_session(self, user_id: str = "") -> SessionContext:
        session_id = self.generate_session_id()
        now = datetime.now()
        context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            create_time=now,
            last_active_time=now
        )
        # 序列化并存储为 String 类型
        session_key = self._get_session_key(session_id)
        session_data = {
            **asdict(context),
            "create_time": now.isoformat(),
            "last_active_time": now.isoformat()
        }
        redis_client.setex(
            name=session_key,
            time=86400,
            value=json.dumps(session_data)
        )
        return context

    def get_session(self, session_id: str, auto_create: bool = False) -> SessionContext:
        session_key = self._get_session_key(session_id)
        session_data = redis_client.get(session_key)

        if not session_data:
            if auto_create:
                return self.create_session()
            raise ValueError(f"会话 {session_id} 不存在")

        # 反序列化
        data = json.loads(session_data)
        context = SessionContext(
            session_id=data["session_id"],
            user_id=data.get("user_id", ""),
            chat_history=data.get("chat_history", []),
            retriever_cache=data.get("retriever_cache", {}),
            tool_call_state=data.get("tool_call_state", {}),
            create_time=datetime.fromisoformat(data["create_time"]),
            last_active_time=datetime.fromisoformat(data["last_active_time"])
        )

        # 修复：更新最后活跃时间（重新序列化整个 String）
        context.last_active_time = datetime.now()
        updated_data = {
            **asdict(context),
            "create_time": context.create_time.isoformat(),
            "last_active_time": context.last_active_time.isoformat()
        }
        redis_client.setex(
            name=session_key,
            time=86400,  # 刷新过期时间
            value=json.dumps(updated_data)
        )

        return context

    def clear_session_history(self, session_id: str):
        session_key = self._get_session_key(session_id)
        session_data = redis_client.get(session_key)

        if not session_data:
            raise ValueError(f"会话 {session_id} 不存在")

        # 修复：先获取完整数据，修改后重新存储为 String
        data = json.loads(session_data)
        data["chat_history"] = []
        data["retriever_cache"] = {}
        redis_client.setex(
            name=session_key,
            time=86400,
            value=json.dumps(data)
        )
        return True

    def clean_expired_sessions(self):
        # Redis自动过期，无需手动清理，此方法保留兼容
        return 0

async def verify_session(request: Request):
    """
    会话验证依赖项：
    - 检查请求头或 query 参数中是否有 X-Session-ID
    - 检查用户是否有权限访问该会话
    """
    session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id")

    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 Session ID")

    user_id = request.headers.get("X-User-ID")
    if user_id:
        try:
            session = session_manager.get_session(session_id)
            if session.user_id and session.user_id != user_id:
                raise HTTPException(status_code=403, detail="无权限访问该会话")
        except ValueError:
            raise HTTPException(status_code=404, detail="会话不存在")

    return session_id  # 可以返回 session_id 给路由使用

# 全局会话管理器实例（生产环境建议单例）
session_manager = SessionManager()