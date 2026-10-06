import json
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4


class RedisStore:
    def __init__(self, url: str):
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)

    def set_json(self, key: str, value: dict, ttl: int) -> None:
        self.client.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    def get_json(self, key: str) -> dict | None:
        value = self.client.get(key)
        return json.loads(value) if value else None

    def increment(self, key: str, ttl: int) -> int:
        pipe = self.client.process()
        pipe.incr(key)
        pipe.expire(key, ttl)
        count, _ = pipe.execute()
        return int(count)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def append_history(self, user_id: str, session_id: str, message: dict, ttl: int) -> None:
        key = f"rag:history:{user_id}:{session_id}"
        self.client.rpush(key, json.dumps(message, ensure_ascii=False))
        self.client.expire(key, ttl)

    def get_history(self, user_id: str, session_id: str) -> list[dict]:
        key = f"rag:history:{user_id}:{session_id}"
        return [json.loads(item) for item in self.client.lrange(key, 0, -1)]

    def delete_history(self, user_id: str, session_id: str) -> None:
        self.client.delete(f"rag:history:{user_id}:{session_id}")

    def set_json_value(self, key: str, value: dict, ttl: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl and ttl > 0:
            self.client.setex(key, ttl, payload)
        else:
            self.client.set(key, payload)

    def get_json_value(self, key: str) -> dict | None:
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set_if_absent_json_value(self, key: str, value: dict, ttl: int | None = None) -> bool:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl and ttl > 0:
            return bool(self.client.set(key, payload, nx=True, ex=ttl))
        return bool(self.client.set(key, payload, nx=True))

    def acquire_lock(self, key: str, ttl: int = 30) -> str | None:
        token = uuid4().hex
        return token if self.client.set(key, token, nx=True, ex=ttl) else None

    def release_lock(self, key: str, token: str) -> None:
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """
        self.client.eval(script, 1, key, token)

    @contextmanager
    def lock(self, key: str, ttl: int = 30) -> Iterator[bool]:
        token = self.acquire_lock(key, ttl)
        try:
            yield bool(token)
        finally:
            if token:
                self.release_lock(key, token)

    def health(self) -> dict:
        return {"connected": bool(self.client.ping()), "backend": "redis"}
