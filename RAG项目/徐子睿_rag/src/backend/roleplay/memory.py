# -*- coding: utf-8 -*-
"""roleplay/memory.py —— 长期记忆（Milvus 向量检索）。

在链路中的位置：
    chat.py 在生成回答前调 retrieve_long_term 取相关历史，
    在回答完成后调 remember_long_term 把用户这轮发言存下来。

做法上与知识库完全同构：同一个向量化器、同一个 MilvusStore、同样的双路召回，
只是集合名不同（MEMORY_COLLECTION）、过滤条件不同（按用户 + 角色隔离）。

全程 fail-open：记忆是增强能力，Milvus 不可用时只应"记不住/想不起"，
绝不能让用户没法聊天。
"""
from __future__ import annotations

import time
import uuid
from typing import Any

try:
    from ..pipeline import embed
    from ..vector_store import and_filter, ensure_collection, equal_filter, search_vectors, upsert_vectors
except ImportError:
    from pipeline import embed
    from vector_store import and_filter, ensure_collection, equal_filter, search_vectors, upsert_vectors

from .config import MEMORY_COLLECTION
from .db import _now

def _ensure_memory_collection() -> bool:
    """确保长期记忆集合可用。

    返回：
        可用返回 True，Milvus 不可达等异常返回 False（不抛出）。

    为什么吞掉异常：
        长期记忆是"锦上添花"的能力。Milvus 挂了的时候，用户至少应该还能聊天，
        只是这一轮记不住/想不起来历史。让记忆功能的失败阻断主对话流程是本末倒置，
        所以这里统一 fail-open 返回布尔值，由调用方决定怎么降级。
    """
    try:
        ensure_collection(MEMORY_COLLECTION)
        return True
    except Exception:
        return False

def remember_long_term(user_id: str, role_id: str, session_id: str, text: str) -> bool:
    """把一条用户发言写入长期记忆。

    参数：
        user_id / role_id / session_id: 归属信息，用于后续按人按角色隔离检索
        text: 用户发言原文
    返回：
        写入成功 True；内容过短、Milvus 不可用或写入异常返回 False。

    两个设计决定：
        1. 太短的内容不入库（< 8 字符）：像"嗯""好的"这类短语没有记忆价值，
           入库只会污染长期记忆的检索结果
        2. 主键用 uuid5 且掺入 time.time_ns()：
           与 pipeline 里"同内容必须幂等覆盖"相反，这里恰恰要允许重复 ——
           用户在不同时间说同一句"我最近很焦虑"，是两个不同时刻的记忆事件，
           都应该被记下来。所以用纳秒时间戳保证主键唯一。
    """
    text = (text or "").strip()
    if len(text) < 8 or not _ensure_memory_collection():
        return False
    try:
        point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{role_id}:{session_id}:{text}:{time.time_ns()}")
        upsert_vectors(
            MEMORY_COLLECTION,
            [{"id": point_id, "vector": embed([text])[0], "payload": {"user_id": user_id, "role_id": role_id, "session_id": session_id, "text": text, "created_at": _now()}}],
        )
        return True
    except Exception:
        return False

def retrieve_long_term(user_id: str, role_id: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
    """按当前问题检索该用户在该角色下的历史记忆。

    参数：
        user_id / role_id: 用于过滤，只在该用户该角色的记忆里搜
        query: 当前用户消息
        limit: 返回条数，被 clamp 到 [1, 10]
    返回：
        [{"text": 记忆内容, "score": 相似度, "created_at": 时间}, ...]；
        Milvus 不可用时返回空列表。

    按 user_id + role_id 过滤是隐私和效果的双重要求：
        不过滤的话，A 用户的记忆会被 B 用户检索到；跨角色的记忆（在"医生"角色下说的
        健康问题和在"虚拟朋友"角色下说的也混在一起）也会互相干扰，让角色人格不一致。
    """
    try:
        hits = search_vectors(
            MEMORY_COLLECTION,
            embed([query])[0],
            limit=max(1, min(limit, 10)),
            filter_expression=and_filter(equal_filter("user_id", user_id), equal_filter("role_id", role_id)),
        )
        return [{"text": hit["payload"].get("text", ""), "score": round(float(hit["score"]), 4), "created_at": hit["payload"].get("created_at", "")} for hit in hits]
    except Exception:
        return []
