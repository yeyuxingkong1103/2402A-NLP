# -*- coding: utf-8 -*-
"""
历史记录接口（增量功能）

持久化位置：MySQL（chat_sessions / chat_messages），按 user_id 隔离。
    清除浏览器缓存、更换浏览器都不会丢失 —— 这是「不清除就永远在」的实现方式。

与 Redis 的分工：
    Redis   —— 多轮对话短期上下文（SESSION_TTL_SECONDS 到期失效）
    MySQL   —— 长期历史记录（永久）
    打开历史会话时，把 MySQL 里的最近若干轮**回灌**进 Redis，
    这样「接着昨天的会话继续追问」才有上下文（省略主语的追问才答得准）。

接口：
    GET    /api/history/sessions                      我的会话列表
    GET    /api/history/sessions/{session_id}         打开会话（读历史 + 回灌上下文）
    POST   /api/history/sessions/{session_id}/restore 仅回灌上下文
    DELETE /api/history/sessions/{session_id}         删除整个会话
    DELETE /api/history/messages/{message_id}         删除单条问答

边界：只读写 chat_sessions / chat_messages，不触碰知识库与 RAG 链路。
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.db import mysql, redis_client
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/history", tags=["历史记录"])

SESSION_LIMIT_MAX = 200
MESSAGE_LIMIT_MAX = 500


def _require_user(user_id: str) -> str:
    """所有历史接口都必须带上 user_id：历史记录按用户隔离"""
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="缺少 user_id，请先登录")
    return user_id.strip()


@router.get("/sessions", summary="我的会话列表")
def list_sessions(user_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    列出当前用户的全部会话，按最近问答时间倒序（最新的在最上面）。

    会话标题 = 该会话首个提问的前 12 个字，与前端左侧栏展示一致。
    """
    user_id = _require_user(user_id)
    limit = max(1, min(int(limit or 50), SESSION_LIMIT_MAX))
    sessions = mysql.list_user_sessions(user_id, limit=limit)
    return {"user_id": user_id, "total": len(sessions), "sessions": sessions}


@router.get("/sessions/{session_id}", summary="打开会话（读取历史并回灌上下文）")
def open_session(session_id: str, user_id: str, restore: bool = True) -> Dict[str, Any]:
    """
    打开一个历史会话：返回该会话的全部问答，并默认把最近几轮回灌进 Redis。

    为什么默认回灌：
        用户点开历史会话通常是想「接着问」。Redis 里的上下文 30 分钟就过期了，
        不回灌的话模型不知道上一轮聊了什么，「那第 4 级呢」这类追问会答非所问。
        回灌只影响多轮上下文，不改变问答链路本身。
    """
    user_id = _require_user(user_id)

    owner = mysql.get_session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="会话不存在或已被删除")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    messages = mysql.get_session_messages(session_id=session_id, user_id=user_id,
                                          limit=MESSAGE_LIMIT_MAX)
    restored = 0
    if restore and messages:
        restored = redis_client.restore_history(session_id, messages)

    sessions = [s for s in mysql.list_user_sessions(user_id, limit=SESSION_LIMIT_MAX)
                if s["session_id"] == session_id]
    meta = sessions[0] if sessions else {"session_id": session_id, "title": "", "role": "student"}

    logger.info(
        "打开历史会话 | 会话=%s | 用户=%s | 消息数=%d | 回灌轮数=%d",
        session_id, user_id, len(messages), restored,
    )
    return {
        "session_id": session_id,
        "title": meta.get("title", ""),
        "role": meta.get("role", "student"),
        "message_count": len(messages),
        "restored_turns": restored,
        "messages": messages,
    }


@router.post("/sessions/{session_id}/restore", summary="把历史上下文回灌到会话")
def restore_session(session_id: str, user_id: str) -> Dict[str, Any]:
    """单独触发上下文回灌（前端如需显式调用，或演示时手动验证）"""
    user_id = _require_user(user_id)
    owner = mysql.get_session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="会话不存在或已被删除")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    messages = mysql.get_session_messages(session_id=session_id, user_id=user_id,
                                          limit=MESSAGE_LIMIT_MAX)
    restored = redis_client.restore_history(session_id, messages)
    return {"session_id": session_id, "restored_turns": restored}


@router.delete("/sessions/{session_id}", summary="删除整个会话")
def delete_session(session_id: str, user_id: str) -> Dict[str, Any]:
    """
    删除会话及其全部问答（外键级联）。

    同时清掉 Redis 里的多轮上下文，避免「已删除的会话」还残留记忆。
    已收藏的条目不受影响 —— 收藏存的是快照，删历史不删收藏。
    """
    user_id = _require_user(user_id)
    deleted = mysql.delete_chat_session(session_id=session_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或已被删除")

    redis_client.clear_session(session_id)
    logger.info("删除会话 | 会话=%s | 用户=%s", session_id, user_id)
    return {"session_id": session_id, "deleted": True}


@router.delete("/messages/{message_id}", summary="删除单条问答")
def delete_message(message_id: int, user_id: str) -> Dict[str, Any]:
    """删除历史记录里的单条问答（只允许删自己的）"""
    user_id = _require_user(user_id)
    deleted = mysql.delete_chat_message(message_id=message_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="该问答不存在或已被删除")

    logger.info("删除问答 | 消息ID=%s | 用户=%s", message_id, user_id)
    return {"message_id": message_id, "deleted": True}