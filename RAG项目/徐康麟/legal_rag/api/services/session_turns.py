# -*- coding: utf-8 -*-
"""会话的登记与"Redis 窗口懒回填关系库" —— 从 ``api/app.py`` 拆出。

这两个函数原先都是 `create_app` 内的模块级助手，**只被会话域使用**（已核实引用点），
因此随会话路由一起搬出来是内聚的。

⚠️ :func:`sync_window_into_db` 的"按轮配对回填"注释是**实测踩坑的产物**
（会在关系库里留下"没有提问的孤儿回答"），改动前务必读完。
"""
from __future__ import annotations

from typing import Any

from ... import metrics as M
from ...logging_setup import get_logger

__all__ = ["register_session", "count_orphan_skip", "sync_window_into_db"]

logger = get_logger("legal_rag.api.services.session_turns")


def register_session(app_state: Any, user_id: str, session_id: str,
                     role_id: str, title: str,
                     user_set_title: bool = False) -> int:
    """登记会话（建会话/登记已有会话）；返回受影响行数（0 = 该会话属于别人）。

    ``user_set_title=True`` = 标题是**用户手填的** → 打上"用户设定"标记，
    后续自动标题（首条提问前 30 字）不得覆盖它（F-F / `AC-TI-*`）。
    """
    assert app_state.business is not None
    app_state.business.upsert_user(user_id)
    return app_state.business.upsert_session(session_id, user_id, role_id, title,
                                             user_set_title=user_set_title)


def count_orphan_skip(session_id: str) -> None:
    """记一次"为避免孤儿行而跳过助手消息"（指标失败不影响主链路）。"""
    try:
        M.counter("orphan_assistant_message_total").inc(session=str(session_id)[:64])
    except Exception:  # noqa: BLE001
        logger.debug("孤儿助手行计数失败", exc_info=True)


def sync_window_into_db(app_state: Any, user_id: str, role_id: str,
                        session_id: str) -> int:
    """把 Redis 窗口里、关系库**还没有**的消息懒回填进关系库（幂等）。

    两条用途（`AC-ST-12` 存量不丢）：

    1. 改造前只写在 Redis 的老会话：首次读取历史时把窗口内的消息原样搬进关系库，
       条数与搬之前**逐值一致**（不截断、不改写内容）；
    2. 旁路写入（测试/运维直接 ``sessions.append``）的消息同样会被补齐，
       保证"接口读的是关系库"这一口径不需要调用方特殊照顾。
    """
    business, sessions = app_state.business, app_state.sessions
    if business is None or sessions is None:
        return 0
    window = sessions.raw_messages(user_id, role_id, session_id)
    if not window:
        return 0
    known = {str(item.get("message_id") or "")
             for item in business.list_messages(session_id, user_id)}
    # ⚠️ **按"轮"配对着回填，绝不单独补一条"已记录轮次"的助手消息**：
    #    只补它会在关系库里留下"没有提问的孤儿回答"（会话历史里表现为凭空的回答）。
    #    判定按**相邻配对**：一条助手消息若**紧跟在一条用户消息之后**，它俩就是同一轮；
    #    此时若该用户消息**已经在库里**（同一问题重问），这一轮按已记录处理 ⇒ 助手不再补写。
    #    其它情况照旧补齐（`AC-ST-12`：旁路写入 / 改造前只写 Redis 的存量不能丢）。
    pending_rows: list[dict] = []
    prev_role = ""
    prev_user_recorded = False
    for item in window:
        message_id = str(item.get("message_id") or "")
        role = str(item.get("role") or "")
        already = message_id in known
        if role == "user":
            prev_user_recorded = already
            if not already:
                pending_rows.append(item)
        elif role == "assistant":
            if already:
                pass
            elif prev_role == "user" and prev_user_recorded:
                logger.warning("跳过一条助手消息（避免孤儿行）：session=%s message_id=%s"
                               " —— 它那一轮的**提问已经在库里**（同一问题重问），"
                               "整轮按已记录处理", session_id, message_id[:12])
                count_orphan_skip(session_id)
            else:
                pending_rows.append(item)
        elif not already:
            # 其它角色（理论上不该有）：按普通消息处理
            pending_rows.append(item)
        prev_role = role

    if not pending_rows:
        return 0
    seq = business.next_seq(session_id, user_id)
    rows = []
    for offset, item in enumerate(pending_rows):
        rows.append({
            "message_id": str(item.get("message_id") or ""),
            "session_id": str(session_id), "user_id": str(user_id),
            "role": str(item.get("role") or ""),
            "content": str(item.get("content") or ""),
            "citations": list(item.get("citations") or []),
            "turn_index": max((seq + offset - 1) // 2, 0), "seq": seq + offset,
            "created_at": float(item.get("created_at") or 0.0),
        })
    inserted = business.append_messages(rows)
    if inserted:
        logger.info("会话窗口懒回填关系库：session=%s user=%s 新增 %d 条（存量兼容，幂等）",
                    session_id, user_id, inserted)
    return inserted
