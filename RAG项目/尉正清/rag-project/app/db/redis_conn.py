# app/db/redis_conn.py
"""Redis 连接：承担短期记忆与热点缓存。

用到多种数据类型，对应不同用途：
  List   —— 最近 N 轮对话（短期记忆），RPUSH + LTRIM；以及待归档缓冲
  Hash   —— 会话状态（当前角色、轮次、最后活跃时间）
  Set    —— 已入库的来源文件名，供 /api/stats 展示与去重
  zSet   —— 角色热度排行（被提问次数）
  String —— 统计结果的短时缓存（/api/stats 要扫 Milvus 全量，缓存 30 秒）
"""
import threading
from typing import List, Optional

import redis

from app.config import settings
import logging

logger = logging.getLogger(__name__)

# key 命名规范，避免不同业务互相覆盖
KEY_CHAT = "chat:{user_id}:{role_id}:{session_id}"      # List  短期记忆
KEY_SESSION_META = "session:meta:{session_id}"          # Hash  会话状态
KEY_ROLE_HOT = "role:hot"                               # zSet  角色热度
KEY_INGESTED = "ingest:done:{role_id}"                  # Set   已入库文件
KEY_MEM_PENDING = "mem:pending:{user_id}:{role_id}:{session_id}"  # List 待归档对话


def _chat_key(user_id: str, role_id: str, session_id: str) -> str:
    return KEY_CHAT.format(user_id=user_id, role_id=role_id, session_id=session_id)


_client: Optional[redis.Redis] = None
_lock = threading.Lock()


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
                _client.ping()
                logger.info("Redis 已连接: %s:%s", settings.REDIS_HOST, settings.REDIS_PORT)
    return _client


# ---------------- 短期记忆（List） ----------------
def push_turn(user_id: str, role_id: str, session_id: str,
              question: str, answer: str, keep: int) -> List[str]:
    """追加一轮对话，裁剪到最近 keep 轮，并返回**被裁掉的旧记录**。

    返回值用于归档进长期记忆。必须在裁剪前取，否则这些内容就永久丢了。
    """
    r = get_redis()
    key = _chat_key(user_id, role_id, session_id)
    limit = keep * 2                        # 一轮 = 2 条

    overflow: List[str] = []
    total = r.llen(key)
    if total + 2 > limit:
        cut = total + 2 - limit
        overflow = r.lrange(key, 0, cut - 1) or []

    pipe = r.pipeline()
    pipe.rpush(key, f"用户：{question}", f"助手：{answer}")
    pipe.ltrim(key, -limit, -1)
    pipe.expire(key, 7 * 24 * 3600)
    pipe.execute()
    return overflow


def get_recent_turns(user_id: str, role_id: str, session_id: str,
                     turns: int) -> List[str]:
    """取最近 turns 轮对话，返回扁平的文本行列表。"""
    if not session_id:
        return []
    try:
        lines = get_redis().lrange(
            _chat_key(user_id, role_id, session_id), -turns * 2, -1)
        return lines or []
    except Exception as e:                                  # pragma: no cover
        logger.warning("读取短期记忆失败: %s", e)
        return []
def clear_history(user_id: str, role_id: str, session_id: str) -> None:
    get_redis().delete(_chat_key(user_id, role_id, session_id),
                       KEY_MEM_PENDING.format(user_id=user_id, role_id=role_id,
                                              session_id=session_id))


# ---------------- 待归档缓冲（List） ----------------
def add_pending(user_id: str, role_id: str, session_id: str,
                lines: List[str], ttl: int = 7 * 24 * 3600) -> int:
    """把滑出短期窗口的记录暂存起来，返回缓冲区当前行数。"""
    if not lines:
        return 0
    r = get_redis()
    key = KEY_MEM_PENDING.format(user_id=user_id, role_id=role_id,
                                 session_id=session_id)
    pipe = r.pipeline()
    pipe.rpush(key, *lines)
    pipe.expire(key, ttl)
    pipe.execute()
    return r.llen(key)


def take_pending(user_id: str, role_id: str, session_id: str) -> List[str]:
    """取出全部待归档内容并清空缓冲。"""
    r = get_redis()
    key = KEY_MEM_PENDING.format(user_id=user_id, role_id=role_id,
                                 session_id=session_id)
    pipe = r.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    res = pipe.execute()
    return res[0] or []


# ---------------- 会话状态（Hash） ----------------
def set_session_meta(session_id: str, **fields) -> None:
    if not fields:
        return
    r = get_redis()
    r.hset(KEY_SESSION_META.format(session_id=session_id), mapping=fields)
    r.expire(KEY_SESSION_META.format(session_id=session_id), 7 * 24 * 3600)


def get_session_meta(session_id: str) -> dict:
    try:
        return get_redis().hgetall(KEY_SESSION_META.format(session_id=session_id)) or {}
    except Exception:                                       # pragma: no cover
        return {}


# ---------------- 角色热度（zSet） ----------------
def incr_role_hot(role_id: str, n: int = 1) -> None:
    try:
        get_redis().zincrby(KEY_ROLE_HOT, n, role_id)
    except Exception as e:                                  # pragma: no cover
        logger.warning("角色热度写入失败: %s", e)


def top_roles(k: int = 10) -> List[tuple]:
    try:
        return get_redis().zrevrange(KEY_ROLE_HOT, 0, k - 1, withscores=True)
    except Exception:                                       # pragma: no cover
        return []


# ---------------- 已入库来源（Set） ----------------
def mark_ingested(role_id: str, sources: List[str]) -> None:
    """记录某角色下已入库的来源文件，用于去重与追溯。"""
    if not sources:
        return
    try:
        get_redis().sadd(KEY_INGESTED.format(role_id=role_id), *sources)
    except Exception as e:                                  # pragma: no cover
        logger.warning("记录入库来源失败: %s", e)


def ingested_sources(role_id: str) -> List[str]:
    try:
        return sorted(get_redis().smembers(KEY_INGESTED.format(role_id=role_id)))
    except Exception:                                       # pragma: no cover
        return []


def forget_ingested(role_id: str, source: str) -> None:
    try:
        get_redis().srem(KEY_INGESTED.format(role_id=role_id), source)
    except Exception:                                       # pragma: no cover
        pass


# ---------------- 缓存（String，带 TTL） ----------------
def cache_get(key: str) -> Optional[str]:
    try:
        return get_redis().get(key)
    except Exception:                                       # pragma: no cover
        return None


def cache_set(key: str, value: str, ttl: int = 300) -> None:
    try:
        get_redis().setex(key, ttl, value)
    except Exception as e:                                  # pragma: no cover
        logger.warning("缓存写入失败: %s", e)
