"""多轮对话管理：会话创建、历史读取、上下文改写窗口。

工单要求（5.7）：
- 保存最近 5 轮对话；
- 使用 SQLite 存储 conversation_id / role / content / timestamp；
- 支持新建对话、切换对话、清空对话。

本模块是 SQLite 之上的薄封装，负责：
1. 会话生命周期；
2. 供 Query 理解使用的历史窗口（最近 N 轮）；
3. 供 LLM 使用的消息历史；
4. 首字响应时间等指标的落库。
"""

from __future__ import annotations

import uuid

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.models.schemas import Conversation, Message
from app.storage.sqlite_manager import SQLiteManager, get_sqlite_manager


class ConversationManager:
    """会话管理器。"""

    def __init__(self, store: SQLiteManager | None = None) -> None:
        self.settings = get_settings()
        self.store = store or get_sqlite_manager()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @trace
    def new_conversation(self, title: str = "新对话", doc_id: str | None = None, conversation_id: str | None = None) -> str:
        """新建会话并返回 conversation_id。"""
        cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        self.store.create_conversation(cid, title=title, doc_id=doc_id)
        logger.info("app.core.conversation", "新建会话", conversation_id=cid, doc_id=doc_id)
        return cid

    def get_or_create(self, conversation_id: str | None, doc_id: str | None = None) -> str:
        """存在则返回，不存在则创建。"""
        if conversation_id:
            existing = self.store.get_conversation(conversation_id)
            if existing:
                return existing.conversation_id
            logger.warning(
                "app.core.conversation", "会话不存在，自动新建", requested=conversation_id
            )
        return self.new_conversation(doc_id=doc_id)

    def list_conversations(self) -> list[Conversation]:
        return self.store.list_conversations()

    def get(self, conversation_id: str) -> Conversation | None:
        return self.store.get_conversation(conversation_id)

    @trace
    def clear(self, conversation_id: str) -> int:
        """清空会话消息，返回删除条数。"""
        removed = self.store.clear_conversation(conversation_id)
        logger.info("app.core.conversation", "清空会话", conversation_id=conversation_id, removed=removed)
        return removed

    @trace
    def delete(self, conversation_id: str) -> None:
        self.store.delete_conversation(conversation_id)
        logger.info("app.core.conversation", "删除会话", conversation_id=conversation_id)

    def rename(self, conversation_id: str, title: str) -> None:
        """重命名会话。

        只更新 ``title`` 字段，**不能**走 ``create_conversation``：
        那样会连带重置 ``message_count`` / ``created_at``。
        """
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            logger.warning("app.core.conversation", "重命名失败：会话不存在", conversation_id=conversation_id)
            return
        self.store.set_conversation_title(conversation_id, title)

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    @trace
    def add_user_message(self, conversation_id: str, content: str) -> int:
        message = Message(conversation_id=conversation_id, role="user", content=content)
        return self.store.add_message(message)

    @trace
    def add_assistant_message(self, conversation_id: str, content: str, citations=None, first_token_ms: float = 0.0) -> int:
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            citations=citations or [],
            first_token_ms=first_token_ms,
        )
        return self.store.add_message(message)

    def messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        return self.store.get_messages(conversation_id, limit=limit)

    # ------------------------------------------------------------------
    # 供检索/生成使用的历史窗口
    # ------------------------------------------------------------------
    def history_pairs(self, conversation_id: str) -> list[tuple[str, str]]:
        """返回 ``[(role, content)]``，最多最近 N 轮（工单要求 5 轮）。"""
        rounds = self.settings.conversation.max_history_rounds
        messages = self.store.get_messages(conversation_id, limit=rounds * 2)
        return [(message.role, message.content) for message in messages]

    def auto_title(self, conversation_id: str, question: str) -> None:
        """首轮对话后，用问题前 20 字作为会话标题。

        仅在**首轮**（用户消息数为 1，即助手回复已落库后总数为 2）时命名，
        之后不再改动，避免每轮追问都把标题覆盖成最后一个问题。
        """
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            return
        # 已有实质标题（不是默认值）就说明是首轮定下的，保持不动
        if conversation.title and conversation.title != "新对话":
            return
        title = question.strip()[:20] or "新对话"
        self.rename(conversation_id, title)


def get_conversation_manager(store: SQLiteManager | None = None) -> ConversationManager:
    """工厂函数。"""
    return ConversationManager(store=store)
