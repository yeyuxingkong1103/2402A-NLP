# -*- coding: utf-8 -*-
"""
Redis 缓存与会话访问模块

承担两类短生命周期数据：
    1. 会话短期记忆 —— 多轮对话的上下文，按会话有效期自动过期；
    2. 高频查询缓存 —— 重复问题直接返回，避免重走完整检索链路。

缓存失效设计（对应 F8 验收点）：
    缓存 Key 中携带「知识库版本号」。知识库重建后版本号自增，
    历史缓存自然失效，不会返回过期答案。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

import redis

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Key 前缀
# ---------------------------------------------------------------------------
_KEY_SESSION_HISTORY = "rag:session:{sid}:history"
_KEY_SESSION_META = "rag:session:{sid}:meta"
# 缓存 Key 中的 role 段（增量改造）：
# 同一个问题在「学生身份 / 职场身份」下的答案口吻不同，若不区分身份，
# 先以职场身份提问、再切到学生身份问同一句会命中同一缓存，答案口吻串味。
# 旧格式 Key（不含 r{role} 段）自然读不到，到期自动清理，无需手工处理。
_KEY_QUERY_CACHE = "rag:cache:v{kbver}:r{role}:query:{digest}"
_KEY_KB_VERSION = "rag:kb:version"

_client: Optional[redis.Redis] = None


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------

def get_client() -> redis.Redis:
    """获取 Redis 客户端单例"""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
            socket_timeout=5,
        )
        logger.info("Redis 客户端已连接：%s:%s", settings.redis_host, settings.redis_port)
    return _client


def health_check() -> bool:
    """连通性检查，供 /api/health 使用"""
    try:
        get_client().ping()
        return True
    except Exception as exc:
        logger.error("Redis 健康检查失败：%s", exc)
        return False


# ---------------------------------------------------------------------------
# 知识库版本号（缓存失效的钥匙）
# ---------------------------------------------------------------------------

def get_kb_version() -> int:
    """读取知识库版本号，不存在时初始化为 1"""
    client = get_client()
    try:
        value = client.get(_KEY_KB_VERSION)
        if value is None:
            client.set(_KEY_KB_VERSION, 1)
            return 1
        return int(value)
    except Exception as exc:
        logger.warning("读取知识库版本号失败，按 1 处理：%s", exc)
        return 1


def bump_kb_version() -> int:
    """
    知识库重建后调用：版本号自增。
    历史查询缓存因 Key 中带有旧版本号而自然失效。
    """
    client = get_client()
    try:
        version = client.incr(_KEY_KB_VERSION)
        logger.info("知识库版本号已更新为 %s，历史查询缓存失效", version)
        return int(version)
    except Exception as exc:
        logger.error("更新知识库版本号失败：%s", exc)
        return get_kb_version()


# ---------------------------------------------------------------------------
# 会话短期记忆
# ---------------------------------------------------------------------------

def create_session(user_id: str = "") -> str:
    """
    创建一个新会话，返回 session_id。

    user_id（增量改造）：会话元信息里记录归属用户，便于排查与后续扩展。
    为空时行为与改造前完全一致，旧调用方（curl / JMeter 压测）不受影响。
    """
    sid = uuid.uuid4().hex
    client = get_client()
    try:
        meta_key = _KEY_SESSION_META.format(sid=sid)
        mapping = {"session_id": sid}
        if user_id:
            mapping["user_id"] = user_id
        client.hset(meta_key, mapping=mapping)
        client.expire(meta_key, settings.session_ttl_seconds)
    except Exception as exc:
        logger.warning("创建会话元信息失败（不影响使用）：%s", exc)
    return sid


def get_session_meta(session_id: str) -> Dict[str, str]:
    """读取会话元信息（session_id / user_id / last_active 等）"""
    if not session_id:
        return {}
    client = get_client()
    try:
        return dict(client.hgetall(_KEY_SESSION_META.format(sid=session_id)) or {})
    except Exception as exc:
        logger.warning("读取会话元信息失败：%s", exc)
        return {}


def restore_history(session_id: str, turns: List[Dict[str, Any]]) -> int:
    """
    把 MySQL 里的历史问答回灌进 Redis 会话上下文（增量改造）。

    为什么需要：
        Redis 会话记忆有 TTL（SESSION_TTL_SECONDS，默认 30 分钟）会自然过期，
        而 MySQL 历史记录是永久的。用户点开昨天的会话继续追问时，
        Redis 里已经没有上下文了 —— 若不回灌，模型就不知道上一轮问了什么，
        「那第 4 级呢」这类省略主语的追问会答非所问。

    与 append_turn 的区别：
        这里用 delete + rpush 覆盖写，而不是追加，保证重复打开同一会话不会
        把历史叠加成多份；同样只保留最近 session_max_turns 轮（与正常对话一致）。

    返回回灌的轮数；任何异常都不抛出，最坏情况只是上下文没恢复。
    """
    if not session_id or not turns:
        return 0

    payloads: List[str] = []
    for turn in turns[-settings.session_max_turns:]:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if not question:
            continue
        payloads.append(json.dumps(
            {"question": question, "answer": answer, "sources": turn.get("sources") or []},
            ensure_ascii=False,
        ))
    if not payloads:
        return 0

    client = get_client()
    key = _KEY_SESSION_HISTORY.format(sid=session_id)
    meta_key = _KEY_SESSION_META.format(sid=session_id)
    try:
        pipe = client.pipeline()
        pipe.delete(key)
        pipe.rpush(key, *payloads)
        pipe.expire(key, settings.session_ttl_seconds)
        pipe.hset(meta_key, mapping={"last_active": "1", "restored": "1"})
        pipe.expire(meta_key, settings.session_ttl_seconds)
        pipe.execute()
    except Exception as exc:
        logger.warning("回灌会话上下文失败（不影响浏览历史）：%s", exc)
        return 0

    logger.info("已回灌会话上下文 | 会话=%s | 轮数=%d", session_id, len(payloads))
    return len(payloads)


def get_history(session_id: str) -> List[Dict[str, Any]]:
    """
    读取会话历史（按时间正序）。

    返回：[{question, answer, sources}, ...]，最多 session_max_turns 轮。
    """
    if not session_id:
        return []
    client = get_client()
    key = _KEY_SESSION_HISTORY.format(sid=session_id)
    try:
        raw_items = client.lrange(key, 0, -1)
    except Exception as exc:
        logger.warning("读取会话历史失败，按空历史处理：%s", exc)
        return []

    history: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            history.append(json.loads(raw))
        except (ValueError, TypeError):
            continue
    return history


def append_turn(
    session_id: str,
    *,
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """追加一轮问答到会话历史，并刷新会话有效期"""
    if not session_id:
        return

    client = get_client()
    key = _KEY_SESSION_HISTORY.format(sid=session_id)
    payload = json.dumps(
        {"question": question, "answer": answer, "sources": sources or []},
        ensure_ascii=False,
    )
    try:
        pipe = client.pipeline()
        pipe.rpush(key, payload)
        # 只保留最近 N 轮，防止上下文无限增长
        pipe.ltrim(key, -settings.session_max_turns, -1)
        pipe.expire(key, settings.session_ttl_seconds)
        pipe.hset(_KEY_SESSION_META.format(sid=session_id), mapping={"last_active": "1"})
        pipe.expire(_KEY_SESSION_META.format(sid=session_id), settings.session_ttl_seconds)
        pipe.execute()
    except Exception as exc:
        logger.warning("写入会话历史失败（不影响本次问答）：%s", exc)


def clear_session(session_id: str) -> None:
    """清空指定会话的上下文（对应 F7 新建对话）"""
    if not session_id:
        return
    client = get_client()
    try:
        client.delete(
            _KEY_SESSION_HISTORY.format(sid=session_id),
            _KEY_SESSION_META.format(sid=session_id),
        )
        logger.info("会话上下文已清空：%s", session_id)
    except Exception as exc:
        logger.warning("清空会话失败：%s", exc)


# ---------------------------------------------------------------------------
# 高频查询缓存
# ---------------------------------------------------------------------------

def _digest(question: str) -> str:
    """对问题做归一化后取 MD5，作为缓存 Key 的一部分"""
    normalized = " ".join((question or "").strip().lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def get_cached_answer(question: str, role: str = "student") -> Optional[Dict[str, Any]]:
    """
    读取查询缓存。

    命中时直接返回完整响应体（含溯源信息），无需重走检索与生成。
    """
    client = get_client()
    key = _KEY_QUERY_CACHE.format(
        kbver=get_kb_version(), role=role or "student", digest=_digest(question)
    )
    try:
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("读取查询缓存失败，按未命中处理：%s", exc)
        return None


def set_cached_answer(question: str, payload: Dict[str, Any], role: str = "student") -> None:
    """写入查询缓存（仅缓存正常作答结果，拒答结果不缓存）

    role 参与 Key（增量改造）：不同身份的答案口吻不同，必须分开缓存。
    """
    client = get_client()
    key = _KEY_QUERY_CACHE.format(
        kbver=get_kb_version(), role=role or "student", digest=_digest(question)
    )
    try:
        client.setex(key, settings.cache_ttl_seconds, json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        logger.warning("写入查询缓存失败（不影响本次问答）：%s", exc)


def clear_query_cache() -> int:
    """
    清空全部查询缓存（知识库重建后调用）。

    注：正常情况下版本号自增即可让缓存失效，此函数用于主动清理残留。
    """
    client = get_client()
    try:
        removed = 0
        for key in client.scan_iter(match="rag:cache:*", count=500):
            client.delete(key)
            removed += 1
        logger.info("已清理查询缓存 %d 条", removed)
        return removed
    except Exception as exc:
        logger.warning("清理查询缓存失败：%s", exc)
        return 0


def set_session_user(session_id: str, user_id: str) -> None:
    """
    把登录用户写入会话元信息（增量改造）。

    由 API 层旁路调用：pipeline 内部不感知 user_id，问答主链路保持零改动。
    任何异常都只告警，不影响问答。
    """
    if not session_id or not user_id:
        return
    client = get_client()
    try:
        meta_key = _KEY_SESSION_META.format(sid=session_id)
        client.hset(meta_key, mapping={"user_id": user_id, "last_active": "1"})
        client.expire(meta_key, settings.session_ttl_seconds)
    except Exception as exc:
        logger.warning("写入会话用户失败（不影响问答）：%s", exc)