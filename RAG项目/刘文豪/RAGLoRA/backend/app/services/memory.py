# -*- coding: utf-8 -*-
"""短期记忆：Redis LIST 保存每个会话最近 N 轮对话。

结构：
    Key   conv:{user_id}:{conversation_id}
    Type  LIST（最新在右侧）
    TTL   3600s（每次写入续期）

选 LIST 而非 String 的理由：LRANGE 天然按时间序取最近 N 条，
LTRIM 天然做容量封顶，避免每次都要读出整个 JSON 数组再序列化。

Redis 不可用时全部接口降级为空/静默失败，由调用方回源 MySQL，不影响主流程。
"""
import json

from ..core import config
from ..core.logging import get_logger

log = get_logger("memory")

_client = None
_unavailable_logged = False


def get_client():
    """惰性创建 Redis 客户端。

    ⚠️ encoding_errors="replace" 必须加：Windows 版 Redis 的 INFO 响应
       含 GBK 编码的中文路径字节，redis-py 默认严格 UTF-8 解码会抛
       UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd7。
    """
    global _client
    if _client is None:
        import redis
        _client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True,
            encoding_errors="replace",
            socket_connect_timeout=3,
            socket_timeout=3,
        )
    return _client


def _key(user_id: int, conversation_id: int) -> str:
    return f"conv:{user_id}:{conversation_id}"


def _warn_once(e: Exception) -> None:
    global _unavailable_logged
    if not _unavailable_logged:
        log.warning("Redis 不可用，短期记忆降级（将由 MySQL 回源）: %s", str(e)[:120])
        _unavailable_logged = True


def is_available() -> bool:
    try:
        get_client().ping()
        return True
    except Exception as e:
        _warn_once(e)
        return False


def get_recent(user_id: int, conversation_id: int,
               turns: int | None = None) -> list[dict]:
    """取最近 N 轮（1 轮 = 问 + 答，即 2*N 条消息），按时间正序返回。"""
    turns = turns or config.MEMORY_TURNS
    try:
        raw = get_client().lrange(_key(user_id, conversation_id), -turns * 2, -1)
        out = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out
    except Exception as e:
        _warn_once(e)
        return []


def append(user_id: int, conversation_id: int, role: str, content: str) -> bool:
    """追加一条消息并裁剪到 N 轮。"""
    import time
    key = _key(user_id, conversation_id)
    try:
        client = get_client()
        client.rpush(key, json.dumps(
            {"role": role, "content": content, "ts": int(time.time())},
            ensure_ascii=False,
        ))
        client.ltrim(key, -config.MEMORY_TURNS * 2, -1)
        client.expire(key, config.MEMORY_TTL)
        return True
    except Exception as e:
        _warn_once(e)
        return False


def append_turn(user_id: int, conversation_id: int,
                question: str, answer: str) -> None:
    """写入一轮完整问答。"""
    append(user_id, conversation_id, "user", question)
    append(user_id, conversation_id, "assistant", answer)


def clear(user_id: int, conversation_id: int) -> bool:
    try:
        get_client().delete(_key(user_id, conversation_id))
        return True
    except Exception as e:
        _warn_once(e)
        return False


def rebuild_from_db(user_id: int, conversation_id: int, db) -> list[dict]:
    """缓存未命中时从 MySQL 重建（懒加载）。"""
    from ..models import Message

    rows = (db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.id.desc())
            .limit(config.MEMORY_TURNS * 2).all())
    rows = list(reversed(rows))
    if not rows:
        return []

    try:
        client = get_client()
        key = _key(user_id, conversation_id)
        client.delete(key)
        for m in rows:
            client.rpush(key, json.dumps(
                {"role": m.role, "content": m.content, "ts": 0}, ensure_ascii=False))
        client.expire(key, config.MEMORY_TTL)
    except Exception as e:
        _warn_once(e)

    return [{"role": m.role, "content": m.content} for m in rows]


def get_context(user_id: int, conversation_id: int, db) -> list[dict]:
    """取对话上下文：先查 Redis，未命中回源 MySQL。"""
    cached = get_recent(user_id, conversation_id)
    if cached:
        return cached
    return rebuild_from_db(user_id, conversation_id, db)
