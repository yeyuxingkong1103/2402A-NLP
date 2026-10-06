# -*- coding: utf-8 -*-
"""roleplay/sessions.py —— 用户、会话与消息。

在链路中的位置：
    chat.py 每轮对话都会用到本文件：确保用户存在、找/建会话、
    存两条消息（用户 + 助手）、读最近若干轮作为短期记忆。

短期记忆与长期记忆的分工：
    短期（本文件）按**时间**取最近 N 轮 —— 解决"刚才说过什么"
    长期（memory.py）按**语义相似度**跨会话召回 —— 解决"他之前提到过什么"
"""
from __future__ import annotations

import uuid
from typing import Any

from .config import SHORT_MEMORY_LIMIT
from .db import _db, _now
from .roles import get_role

def ensure_user(user_id: str, name: str | None = None) -> dict[str, str]:
    """确保用户存在，并更新其显示名。

    参数：
        user_id: 用户 id，空则归入 "anonymous"
        name: 显示名，空则用 user_id
    返回：
        {"id": ..., "name": ...}

    为什么先 INSERT OR IGNORE 再 UPDATE：
        两条语句合起来等价于 upsert，但能同时兼容"新用户"和"老用户改了名字"两种情况。
        如果只 INSERT OR IGNORE，老用户改名不会生效。
    """
    user_id = (user_id or "anonymous").strip()[:128] or "anonymous"
    display_name = (name or user_id).strip()[:100] or user_id
    db = _db()
    db.execute("INSERT OR IGNORE INTO users(id,name,created_at) VALUES(?,?,?)", (user_id, display_name, _now()))
    db.execute("UPDATE users SET name=? WHERE id=?", (display_name, user_id))
    db.commit()
    return {"id": user_id, "name": display_name}

def create_session(user_id: str, role_id: str, title: str = "") -> dict[str, Any]:
    """为用户在某角色下新建会话。

    参数：
        user_id: 用户 id
        role_id: 角色 id
        title: 会话标题，空着也行（首条消息会自动补上，见 append_message）
    返回：
        新建的会话字典。

    异常：
        角色不存在 -> ValueError。
        先校验角色再建会话：否则会留下一堆指向不存在角色的孤儿会话。
    """
    ensure_user(user_id)
    if not get_role(role_id):
        raise ValueError(f"角色不存在: {role_id}")
    now = _now()
    session = {"id": uuid.uuid4().hex, "user_id": user_id, "role_id": role_id, "title": title[:120], "created_at": now, "updated_at": now}
    _db().execute(
        "INSERT INTO sessions(id,user_id,role_id,title,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        tuple(session.values()),  # 依赖 dict 的插入顺序与上面的列顺序一致（Python 3.7+ 保证有序）
    )
    _db().commit()
    return session

def get_session(session_id: str) -> dict[str, Any] | None:
    """按 id 取会话；不存在返回 None。"""
    row = _db().execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None

def list_sessions(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """列出某用户的会话，最近的在前。

    参数：
        user_id: 用户 id
        limit: 条数上限，被 clamp 到 [1, 200] —— 防止调用方传个大数把整库拖出来
    """
    limit = min(max(int(limit), 1), 200)
    rows = _db().execute("SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(row) for row in rows]

def append_message(session: dict[str, Any], speaker: str, content: str) -> dict[str, Any]:
    """追加一条消息，并顺带更新会话的活跃时间与标题。

    参数：
        session: 会话字典
        speaker: "user" 或 "assistant"
        content: 消息正文
    返回：
        写入的消息字典。

    这里每写一条消息都要做三件事（同一个事务里完成）：
        1. 插入消息
        2. 更新会话的 updated_at —— 会话列表按它倒序，不更新的话刚聊过的会话不会浮到最前
        3. 如果会话还没有标题，用首条消息的前 40 字当标题 ——
           这样会话列表里显示的是"用户问了什么"，而不是一串 uuid。
           CASE WHEN title='' 保证标题只被设置一次，后续消息不会覆盖它
    """
    message = {
        "id": uuid.uuid4().hex,
        "session_id": session["id"],
        "user_id": session["user_id"],
        "role_id": session["role_id"],
        "speaker": speaker,
        "content": content,
        "created_at": _now(),
    }
    db = _db()
    db.execute("INSERT INTO messages(id,session_id,user_id,role_id,speaker,content,created_at) VALUES(?,?,?,?,?,?,?)", tuple(message.values()))
    db.execute("UPDATE sessions SET updated_at=?, title=CASE WHEN title='' THEN ? ELSE title END WHERE id=?", (_now(), content[:40], session["id"]))
    db.commit()
    return message

def recent_messages(session_id: str, limit: int = SHORT_MEMORY_LIMIT) -> list[dict[str, Any]]:
    """取最近若干条消息，用作短期记忆。

    参数：
        session_id: 会话 id
        limit: 条数，被 clamp 到 [1, 50]
    返回：
        按时间正序的消息列表（最早的在前）。

    为什么 SQL 用 DESC 查、返回前又 reversed：
        SQL 用 ORDER BY created_at DESC + LIMIT 才能"从尾部取最近 N 条"；
        但喂给模型时必须按时间正序（先发生的在前），否则模型读到的对话顺序是倒的。
        所以取完要反转一次。
    """
    rows = _db().execute(
        "SELECT speaker,content,created_at FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, min(max(int(limit), 1), 50)),
    ).fetchall()
    return [dict(row) for row in reversed(rows)]

def history(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """取会话历史消息（给前端展示用）。

    参数：
        session_id: 会话 id
        limit: 条数，被 clamp 到 [1, 500]
    返回：
        按时间正序的完整消息记录。

    与 recent_messages 的区别：
        这里返回全部字段（含 id/created_at）供界面渲染，条数上限也更宽（500 vs 50）——
        因为它是"翻看历史"，而后者是"喂给模型的上下文"，用途不同、预算也不同。
    """
    rows = _db().execute(
        "SELECT id,session_id,user_id,role_id,speaker,content,created_at FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, min(max(int(limit), 1), 500)),
    ).fetchall()
    return [dict(row) for row in reversed(rows)]
