# file: src/core/session.py - 增强版
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any

import redis
from fastapi import Request, HTTPException

from src.config.settings import settings


# 增强的会话上下文
@dataclass
class SessionContext:
    """增强的会话上下文，包含个性化信息和状态"""
    # 基础信息
    session_id: str
    user_id: str = ""
    create_time: datetime = field(default_factory=datetime.now)
    last_active_time: datetime = field(default_factory=datetime.now)

    # 对话历史
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    # 个性化信息（新增）
    user_profile: Dict[str, Any] = field(default_factory=lambda: {
        "preferred_language": "zh",
        "response_style": "balanced",  # concise, detailed, balanced
        "technical_level": "intermediate",  # beginner, intermediate, advanced
        "interests": []
    })

    # 会话分析（新增）
    conversation_analysis: Dict[str, Any] = field(default_factory=lambda: {
        "topics": [],  # 会话主题
        "frequent_keywords": [],  # 高频关键词
        "question_types": [],  # 问题类型分布
        "conversation_tone": "neutral",  # 对话语气
        "total_turns": 0  # 总对话轮次
    })

    # 重要记忆（新增）
    important_facts: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    unresolved_issues: List[str] = field(default_factory=list)

    # 检索优化（新增）
    document_preferences: Dict[str, float] = field(default_factory=dict)  # doc_id -> weight
    query_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # 工具状态
    tool_call_state: Dict[str, Any] = field(default_factory=dict)
    recent_tools: List[str] = field(default_factory=list)

    # 添加重要事实
    def add_important_fact(self, fact: str):
        """添加重要事实到会话记忆"""
        if fact and fact not in self.important_facts:
            self.important_facts.append(fact)
            if len(self.important_facts) > 20:  # 限制数量
                self.important_facts = self.important_facts[-20:]

    # 获取相关事实
    def get_relevant_facts(self, query: str, limit: int = 3) -> List[str]:
        """获取与查询相关的重要事实"""
        query_lower = query.lower()
        relevant = []

        for fact in reversed(self.important_facts):  # 从最新的开始
            fact_lower = fact.lower()
            # 简单的相关性判断：共享词汇或语义相似
            if any(word in query_lower for word in fact_lower.split()[:5]):
                relevant.append(fact)
                if len(relevant) >= limit:
                    break

        return relevant

    # 更新文档偏好
    def update_document_preference(self, doc_id: str, relevance_score: float):
        """更新文档偏好权重"""
        current_weight = self.document_preferences.get(doc_id, 0.0)
        # 指数衰减的权重更新
        self.document_preferences[doc_id] = (
                0.7 * current_weight + 0.3 * relevance_score
        )

    # 获取高权重文档
    def get_preferred_doc_ids(self, limit: int = 5) -> List[str]:
        """获取权重最高的文档ID"""
        sorted_docs = sorted(
            self.document_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc_id for doc_id, _ in sorted_docs[:limit]]


# 会话分析器
class SessionAnalyzer:
    """分析会话内容和提取特征"""

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """提取文本关键词（中文友好）"""
        # 简单的中文分词（实际项目应使用jieba等库）
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())

        # 过滤停用词
        stopwords = {"的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
                     "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        words = [w for w in words if w not in stopwords and len(w) > 1]

        # 统计词频
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_k)]

    @staticmethod
    def analyze_conversation_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析对话历史"""
        if not history:
            return {"topics": [], "frequent_keywords": [], "question_types": []}

        # 合并所有用户消息
        user_messages = [
            msg["content"] for msg in history
            if msg.get("role") == "user"
        ]

        all_text = " ".join(user_messages)

        # 提取关键词
        keywords = SessionAnalyzer.extract_keywords(all_text, top_k=15)

        # 分析问题类型
        question_types = []
        for msg in user_messages[-10:]:  # 分析最近10条
            if "什么" in msg or "什么是" in msg:
                question_types.append("definition")
            elif "如何" in msg or "怎样" in msg:
                question_types.append("howto")
            elif "为什么" in msg:
                question_types.append("why")
            elif "?" in msg or "？" in msg:
                question_types.append("general")

        # 统计问题类型
        type_counter = Counter(question_types)
        common_types = [t for t, _ in type_counter.most_common(3)]

        # 推断对话语气
        tone_words = {"请", "谢谢", "麻烦", "不好意思", "紧急", "尽快", "急"}
        tone = "polite" if any(word in all_text for word in tone_words) else "neutral"

        return {
            "topics": keywords[:5],
            "frequent_keywords": keywords,
            "question_types": common_types,
            "conversation_tone": tone,
            "total_turns": len(history)
        }

    @staticmethod
    def enhance_query_with_context(query: str, session_analysis: Dict[str, Any]) -> str:
        """用会话上下文增强查询"""
        topics = session_analysis.get("topics", [])
        keywords = session_analysis.get("frequent_keywords", [])

        if not topics:
            return query

        # 如果查询很短，添加相关主题
        if len(query) < 10 and topics:
            enhanced = f"{query} {' '.join(topics[:2])}"
            return enhanced.strip()

        # 如果查询与当前主题相关，保持原样
        query_lower = query.lower()
        if any(topic in query_lower for topic in topics[:3]):
            return query

        # 否则添加最相关的主题
        for keyword in keywords:
            if keyword in query_lower:
                return query

        # 添加最相关的主题
        return f"{query} {topics[0]}"


# 增强的会话管理器
class EnhancedSessionManager:
    """增强的会话管理器，支持个性化分析"""

    SESSION_PREFIX = "chat:session:"
    ANALYSIS_INTERVAL = 3  # 每3轮对话分析一次

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True
        )
        self.analyzer = SessionAnalyzer()

    def _get_session_key(self, session_id: str) -> str:
        return f"{self.SESSION_PREFIX}{session_id}"

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    def create_session(self, user_id: str = "") -> SessionContext:
        """创建新会话"""
        session_id = self.generate_session_id()
        now = datetime.now()

        context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            create_time=now,
            last_active_time=now
        )

        self._save_session(context)
        return context

    def get_session(self, session_id: str, auto_create: bool = False) -> SessionContext:
        """获取会话（自动分析更新）"""
        session_key = self._get_session_key(session_id)
        session_data = self.redis_client.get(session_key)

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
            user_profile=data.get("user_profile", {}),
            conversation_analysis=data.get("conversation_analysis", {}),
            important_facts=data.get("important_facts", []),
            user_preferences=data.get("user_preferences", {}),
            unresolved_issues=data.get("unresolved_issues", []),
            document_preferences=data.get("document_preferences", {}),
            query_patterns=data.get("query_patterns", []),
            tool_call_state=data.get("tool_call_state", {}),
            recent_tools=data.get("recent_tools", []),
            create_time=datetime.fromisoformat(data["create_time"]),
            last_active_time=datetime.fromisoformat(data["last_active_time"])
        )

        # 自动分析会话（如果需要）
        self._auto_analyze_session(context)

        # 更新最后活跃时间
        context.last_active_time = datetime.now()
        self._save_session(context)

        return context

    def _auto_analyze_session(self, context: SessionContext):
        """自动分析会话（定期触发）"""
        total_turns = len(context.chat_history)

        # 每 ANALYSIS_INTERVAL 轮对话分析一次
        if total_turns % self.ANALYSIS_INTERVAL == 0 and total_turns > 0:
            # 分析对话历史
            analysis = self.analyzer.analyze_conversation_history(
                context.chat_history[-20:]  # 分析最近20条
            )

            # 更新会话分析
            context.conversation_analysis.update(analysis)

            # 自动提取重要事实（从助手回复中）
            self._extract_important_facts(context)

    def _extract_important_facts(self, context: SessionContext):
        """从助手回复中提取重要事实"""
        recent_assistant_messages = [
            msg["content"] for msg in context.chat_history[-10:]
            if msg.get("role") == "assistant"
        ]

        for message in recent_assistant_messages:
            # 提取包含关键信息的句子
            sentences = re.split(r'[。！？!?]', message)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(
                        keyword in sentence for keyword in
                        ["是", "有", "可以", "应该", "需要", "必须", "建议"]
                ):
                    context.add_important_fact(sentence[:100])  # 限制长度

    def _save_session(self, context: SessionContext):
        """保存会话到Redis"""
        session_key = self._get_session_key(context.session_id)

        session_data = {
            **asdict(context),
            "create_time": context.create_time.isoformat(),
            "last_active_time": context.last_active_time.isoformat()
        }

        self.redis_client.setex(
            name=session_key,
            time=86400,  # 24小时过期
            value=json.dumps(session_data, ensure_ascii=False)
        )

    def update_chat_history(self, session_id: str, user_input: str, assistant_response: str):
        """更新聊天历史（并触发分析）"""
        context = self.get_session(session_id)

        # 添加新对话
        context.chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ])

        # 限制历史长度
        if len(context.chat_history) > 50:
            context.chat_history = context.chat_history[-50:]

        # 保存更新
        self._save_session(context)

    def clear_session_history(self, session_id: str):
        """清空会话历史（保留分析信息）"""
        context = self.get_session(session_id)

        # 只清空聊天历史，保留分析结果
        context.chat_history = []

        # 保存
        self._save_session(context)
        return True

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话摘要"""
        context = self.get_session(session_id)

        return {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "total_turns": len(context.chat_history),
            "topics": context.conversation_analysis.get("topics", []),
            "important_facts": context.important_facts[-5:],  # 最近5个重要事实
            "user_preferences": context.user_profile,
            "preferred_documents": len(context.document_preferences),
            "recent_tools": context.recent_tools[-3:]  # 最近3个工具
        }

    def enhance_query_with_session(self, session_id: str, query: str) -> str:
        """用会话信息增强查询"""
        context = self.get_session(session_id)
        return self.analyzer.enhance_query_with_context(
            query,
            context.conversation_analysis
        )


# 依赖项和全局实例
async def verify_session(request: Request):
    """会话验证依赖项"""
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

    return session_id


# 全局会话管理器实例
session_manager = EnhancedSessionManager()