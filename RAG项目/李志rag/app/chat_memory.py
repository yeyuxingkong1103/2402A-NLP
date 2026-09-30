import json
from functools import lru_cache

import redis

from app.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def redis_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _key(user_id: int, role_id: int, session_id: str) -> str:
    safe_session = "".join(char for char in session_id if char.isalnum() or char in "_-.")
    return f"chat:{user_id}:{role_id}:{safe_session or 'default'}"


def recent_messages(user_id: int, role_id: int, session_id: str) -> list[dict[str, str]]:
    return [
        json.loads(value)
        for value in redis_client().lrange(_key(user_id, role_id, session_id), 0, -1)
    ]


def append_turn(user_id: int, role_id: int, session_id: str, message: str, answer: str) -> None:
    key = _key(user_id, role_id, session_id)
    pipe = redis_client().pipeline()
    pipe.rpush(key, json.dumps({"role": "user", "content": message}, ensure_ascii=False))
    pipe.rpush(key, json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False))
    pipe.ltrim(key, -20, -1)
    pipe.execute()


def clear_memory(user_id: int, role_id: int, session_id: str) -> None:
    redis_client().delete(_key(user_id, role_id, session_id))
