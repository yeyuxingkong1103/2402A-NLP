"""
session_memory.py — 短期记忆

Redis List，key = chat:{user_id}，只保留最近 SHORT_TERM_ROUNDS 轮
（一轮 = 用户提问 + 模型回答，即 2 条消息）。

LOCAL_MODE=True 时落到 storage/sessions.json，无需 Redis。
另外维护一个每用户累计轮次计数器，供长期记忆判断"是否该提取关系记忆"。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import config

_logger = None


def _log():
    """延迟初始化日志，避免模块导入时就写日志文件。"""
    global _logger
    if _logger is None:
        from loguru import logger

        _logger = logger
    return _logger


_lock = threading.RLock()
_SESSION_FILE = config.STORAGE_DIR / "sessions.json"
_ROUNDS_FILE = config.STORAGE_DIR / "rounds.json"

_redis_client = None
_redis_probed = False


def _get_redis():
    """返回 Redis 客户端；生产模式下连不上会打一次告警并降级到本地文件。"""
    global _redis_client, _redis_probed

    if _redis_probed:
        return _redis_client

    _redis_probed = True
    if config.LOCAL_MODE:
        return None

    try:
        import redis

        client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD or None,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        client.ping()
        _redis_client = client
    except Exception as exc:
        _log().warning(f"Redis 不可用（{exc}），短期记忆已降级为本地文件存储。")
        _redis_client = None
    return _redis_client


def _session_key(user_id: str) -> str:
    return f"chat:{user_id}"


def _rounds_key(user_id: str) -> str:
    return f"chat_rounds:{user_id}"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """原子写入：先写临时文件再替换，避免进程中断留下半个文件。"""
    with _lock:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)


def get_short_term(user_id: str) -> list[dict]:
    """取最近若干轮对话，返回 [{"role": "user"/"assistant", "content": ...}]。"""
    limit = config.SHORT_TERM_ROUNDS * 2

    client = _get_redis()
    if client is not None:
        try:
            raw = client.lrange(_session_key(user_id), -limit, -1)
            return [json.loads(item) for item in raw]
        except Exception as exc:
            _log().warning(f"读取 Redis 短期记忆失败：{exc}")

    record = _read_json(_SESSION_FILE).get(user_id, [])
    return record[-limit:]


def save_short_term(user_id: str, user_input: str, reply: str) -> None:
    """追加一轮对话，并裁剪到配置的轮数上限。"""
    limit = config.SHORT_TERM_ROUNDS * 2
    now = int(time.time())
    turns = [
        {"role": "user", "content": user_input, "ts": now},
        {"role": "assistant", "content": reply, "ts": now},
    ]

    client = _get_redis()
    if client is not None:
        try:
            key = _session_key(user_id)
            client.rpush(key, *[json.dumps(t, ensure_ascii=False) for t in turns])
            client.ltrim(key, -limit, -1)
            # 一周无对话自动过期，避免冷用户长期占用内存
            client.expire(key, 7 * 24 * 3600)
            return
        except Exception as exc:
            _log().warning(f"写入 Redis 短期记忆失败：{exc}")

    with _lock:
        sessions = _read_json(_SESSION_FILE)
        record = sessions.get(user_id, [])
        record.extend(turns)
        sessions[user_id] = record[-limit:]
        _write_json(_SESSION_FILE, sessions)


def clear_short_term(user_id: str) -> None:
    """清空某个用户的短期记忆（"开启新对话"）。长期记忆不受影响。"""
    client = _get_redis()
    if client is not None:
        try:
            client.delete(_session_key(user_id))
            return
        except Exception as exc:
            _log().warning(f"清空 Redis 短期记忆失败：{exc}")

    with _lock:
        sessions = _read_json(_SESSION_FILE)
        sessions.pop(user_id, None)
        _write_json(_SESSION_FILE, sessions)


def bump_round(user_id: str) -> int:
    """把该用户的累计轮数加一并返回新值。

    累计而非按会话重置：否则用户每 4 轮就开一次新对话时，关系记忆永远不会被提取。
    """
    client = _get_redis()
    if client is not None:
        try:
            key = _rounds_key(user_id)
            value = int(client.incr(key))
            client.expire(key, 30 * 24 * 3600)
            return value
        except Exception as exc:
            _log().warning(f"更新 Redis 轮次计数失败：{exc}")

    with _lock:
        rounds = _read_json(_ROUNDS_FILE)
        value = int(rounds.get(user_id, 0)) + 1
        rounds[user_id] = value
        _write_json(_ROUNDS_FILE, rounds)
    return value


def get_rounds(user_id: str) -> int:
    """读取累计轮数（不清零），主要用于排查问题。"""
    client = _get_redis()
    if client is not None:
        try:
            return int(client.get(_rounds_key(user_id)) or 0)
        except Exception:
            pass
    return int(_read_json(_ROUNDS_FILE).get(user_id, 0))


def clear_rounds(user_id: str) -> None:
    """重置累计轮数，随 clear_all 一起调用。"""
    client = _get_redis()
    if client is not None:
        try:
            client.delete(_rounds_key(user_id))
            return
        except Exception:
            pass

    with _lock:
        rounds = _read_json(_ROUNDS_FILE)
        rounds.pop(user_id, None)
        _write_json(_ROUNDS_FILE, rounds)


if __name__ == "__main__":
    uid = "_selftest_user"
    clear_short_term(uid)
    clear_rounds(uid)

    for i in range(1, 7):
        save_short_term(uid, f"第{i}轮提问", f"第{i}轮回答")
        bump_round(uid)

    history = get_short_term(uid)
    print(f"累计轮数 = {get_rounds(uid)}，保留消息 = {len(history)} 条")
    assert get_rounds(uid) == 6
    assert len(history) == config.SHORT_TERM_ROUNDS * 2, "没有裁剪到配置的轮数上限"
    assert history[0]["content"] == "第2轮提问", "保留的不是最近的若干轮"

    clear_short_term(uid)
    clear_rounds(uid)
    assert get_short_term(uid) == [] and get_rounds(uid) == 0
    print("session_memory 自检通过。")
