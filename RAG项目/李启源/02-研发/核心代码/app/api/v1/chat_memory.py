"""Session-memory adapters used by the shared HTTP chat service."""

from __future__ import annotations

import logging
from typing import Any

from app.api.v1 import deps

logger = logging.getLogger(__name__)


def session_identity(session_id: str | None) -> tuple[int, int] | None:
    """Resolve memory scope only from trusted server-side session metadata."""
    if not session_id:
        return None
    manager = deps.get_session_manager()
    if manager is None:
        return None
    try:
        meta = manager.get_session(session_id)
    except Exception as exc:
        logger.warning("读取会话身份失败: %s", exc)
        return None
    if not meta:
        return None
    try:
        return int(meta["user_id"]), int(meta["tenant_id"])
    except (KeyError, TypeError, ValueError):
        logger.warning("会话缺少有效的 user_id/tenant_id: %s", session_id)
        return None


def load_long_term(session_id: str | None, query: str) -> dict[str, Any]:
    identity = session_identity(session_id)
    memory_manager = deps.get_memory_manager()
    if identity is None or memory_manager is None:
        return {}
    user_id, tenant_id = identity
    context = memory_manager.get_memory_context(
        session_id=session_id or "",
        user_id=user_id,
        tenant_id=tenant_id,
        current_query=query,
        enable_long_term=True,
        max_short_turns=1,
    )
    sections: list[str] = []
    if context.user_preferences:
        sections.append("用户偏好：\n" + "\n".join(
            f"- {item.get('text', '')}" for item in context.user_preferences[:3]
        ))
    if context.important_facts:
        sections.append("重要信息：\n" + "\n".join(
            f"- {item.get('text', '')}" for item in context.important_facts[:3]
        ))
    if context.long_term_summaries:
        sections.append("历史对话摘要：\n" + "\n".join(
            f"- {item.get('summary', '')}" for item in context.long_term_summaries[:2]
        ))
    memory = "\n\n".join(section for section in sections if section)
    return {"memory": memory} if memory else {}


def save_long_term(
    session_id: str | None, user_text: str, assistant_text: str
) -> None:
    identity = session_identity(session_id)
    memory_manager = deps.get_memory_manager()
    if identity is None or memory_manager is None or not session_id:
        return
    user_id, tenant_id = identity
    memory_manager.observe_turn(
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )


def load_history(session_id: str | None, max_turns: int) -> list[dict[str, Any]]:
    if not session_id:
        return []
    memory = deps.get_redis_memory()
    if memory is None:
        return []
    try:
        return memory.get_messages(session_id, max_turns=max_turns)
    except Exception as exc:
        logger.warning("读取会话历史失败，按单轮处理: %s", exc)
        return []


def save_turn(session_id: str | None, user_text: str, assistant_text: str) -> None:
    if not session_id:
        return
    memory = deps.get_redis_memory()
    if memory is not None:
        try:
            memory.add_message(session_id, {"role": "user", "content": user_text})
            memory.add_message(
                session_id, {"role": "assistant", "content": assistant_text}
            )
        except Exception as exc:
            logger.warning("写入会话历史失败: %s", exc)
    client = deps.get_mysql_client()
    if client is not None:
        try:
            client.execute(
                "UPDATE sessions SET message_count = message_count + 2, "
                "last_active_at = NOW() WHERE session_id = %s",
                (session_id,),
            )
        except Exception as exc:
            logger.warning("更新会话消息计数失败: %s", exc)
