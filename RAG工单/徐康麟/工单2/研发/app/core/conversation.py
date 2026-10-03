"""多轮对话管理：会话生命周期、消息落库、最近 N 轮历史（工单 6.5）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 多轮对话（对应 设计/接口设计.md §2.14）

约束：
- ``history_pairs()`` 返回**最近 5 轮**（``conversation.max_history_rounds``）；
- 查询改写只用最近 3 轮（``conversation.rewrite_history_rounds``）；
- 消息与引用全部落 SQLite，保证事后可复盘（配合 retrieval_traces）。
"""

from __future__ import annotations

import threading
import uuid

from app.core.config import get_settings
from app.core.errors import StorageError
from app.core.logging_conf import logger, trace
from app.models.schemas import Citation, Conversation, Message
from app.storage.sqlite_manager import SQLiteManager, get_sqlite_manager


class ConversationManager:
    """会话管理器（唯一入口是 SQLiteManager）。"""

    def __init__(self, store: SQLiteManager | None = None) -> None:
        self._settings = get_settings()
        self._store = store or get_sqlite_manager()
        self._store.init_schema()

    # ------------------------------------------------------------------
    @trace
    def new_conversation(self, title: str = "新对话", doc_id: str | None = None, conversation_id: str | None = None) -> str:
        """新建会话，返回 conversation_id。"""
        try:
            cid = conversation_id or f"conv{uuid.uuid4().hex[:12]}"
            self._store.create_conversation(cid, title=title, doc_id=doc_id)
            logger.info("app.core.conversation", "新建会话", conversation_id=cid, doc_id=doc_id)
            return cid
        except Exception as exc:
            logger.exception("app.core.conversation", "新建会话失败")
            raise StorageError(f"新建会话失败: {exc}") from exc

    def get_or_create(self, conversation_id: str | None, doc_id: str | None = None) -> str:
        """取已有会话或新建（UI 首次提问用）。"""
        try:
            if conversation_id:
                existing = self._store.get_conversation(conversation_id)
                if existing is not None:
                    return conversation_id
            return self.new_conversation(doc_id=doc_id)
        except Exception as exc:
            logger.exception("app.core.conversation", "获取/创建会话失败")
            raise StorageError(f"获取会话失败: {exc}") from exc

    def list_conversations(self) -> list[Conversation]:
        """列出全部会话（更新时间倒序）。"""
        try:
            return self._store.list_conversations()
        except Exception as exc:
            logger.exception("app.core.conversation", "列出会话失败")
            raise StorageError(f"列出会话失败: {exc}") from exc

    def get(self, conversation_id: str) -> Conversation | None:
        """读取单个会话。"""
        try:
            return self._store.get_conversation(conversation_id)
        except Exception as exc:
            logger.exception("app.core.conversation", "读取会话失败", conversation_id=conversation_id)
            raise StorageError(f"读取会话失败: {exc}") from exc

    def clear(self, conversation_id: str) -> int:
        """清空会话消息，返回删除条数。"""
        try:
            count = self._store.clear_conversation(conversation_id)
            logger.info("app.core.conversation", "清空会话", conversation_id=conversation_id, deleted=count)
            return count
        except Exception as exc:
            logger.exception("app.core.conversation", "清空会话失败", conversation_id=conversation_id)
            raise StorageError(f"清空会话失败: {exc}") from exc

    def delete(self, conversation_id: str) -> None:
        """删除会话及其消息。"""
        try:
            self._store.delete_conversation(conversation_id)
            logger.info("app.core.conversation", "删除会话", conversation_id=conversation_id)
        except Exception as exc:
            logger.exception("app.core.conversation", "删除会话失败", conversation_id=conversation_id)
            raise StorageError(f"删除会话失败: {exc}") from exc

    def rename(self, conversation_id: str, title: str) -> None:
        """重命名会话。"""
        try:
            self._store.set_conversation_title(conversation_id, title)
        except Exception as exc:
            logger.exception("app.core.conversation", "重命名会话失败", conversation_id=conversation_id)
            raise StorageError(f"重命名会话失败: {exc}") from exc

    # ------------------------------------------------------------------
    def add_user_message(self, conversation_id: str, content: str) -> int:
        """写入用户消息，返回 message_id。"""
        try:
            return self._store.add_message(
                Message(conversation_id=conversation_id, role="user", content=content)
            )
        except Exception as exc:
            logger.exception("app.core.conversation", "写入用户消息失败", conversation_id=conversation_id)
            raise StorageError(f"写入用户消息失败: {exc}") from exc

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        citations: list[Citation] | None = None,
        first_token_ms: float = 0.0,
    ) -> int:
        """写入助手消息（含结构化引用与首字耗时），返回 message_id。"""
        try:
            return self._store.add_message(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content,
                    citations=citations or [],
                    first_token_ms=float(first_token_ms or 0.0),
                )
            )
        except Exception as exc:
            logger.exception("app.core.conversation", "写入助手消息失败", conversation_id=conversation_id)
            raise StorageError(f"写入助手消息失败: {exc}") from exc

    def messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        """读取消息（时间正序）。"""
        try:
            return self._store.get_messages(conversation_id, limit=limit)
        except Exception as exc:
            logger.exception("app.core.conversation", "读取消息失败", conversation_id=conversation_id)
            raise StorageError(f"读取消息失败: {exc}") from exc

    def history_pairs(self, conversation_id: str) -> list[tuple[str, str]]:
        """返回最近 N 轮的 ``(user, assistant)`` 对（工单要求最近 5 轮）。"""
        try:
            rounds = self._settings.conversation.max_history_rounds
            messages = self._store.get_messages(conversation_id)
            pairs: list[tuple[str, str]] = []
            last_user = ""
            for message in messages:
                if message.role == "user":
                    last_user = message.content
                elif message.role == "assistant" and last_user:
                    pairs.append((last_user, message.content))
                    last_user = ""
            return pairs[-rounds:] if rounds > 0 else []
        except Exception as exc:
            logger.exception("app.core.conversation", "读取历史对失败", conversation_id=conversation_id)
            raise StorageError(f"读取历史对失败: {exc}") from exc

    def auto_title(self, conversation_id: str, question: str) -> None:
        """按首个问题自动命名会话（前 20 字）。"""
        try:
            conversation = self._store.get_conversation(conversation_id)
            if conversation is None or conversation.title not in {"新对话", ""}:
                return
            title = (question or "").strip().replace("\n", " ")[:20] or "新对话"
            self._store.set_conversation_title(conversation_id, title)
            logger.debug("app.core.conversation", "会话自动命名", conversation_id=conversation_id, title=title)
        except Exception:
            logger.exception("app.core.conversation", "会话自动命名失败", conversation_id=conversation_id)


_manager: ConversationManager | None = None
_lock = threading.Lock()


def get_conversation_manager(store: SQLiteManager | None = None) -> ConversationManager:
    """获取进程级会话管理器单例。"""
    global _manager
    with _lock:
        if _manager is None or store is not None:
            _manager = ConversationManager(store=store)
    return _manager
