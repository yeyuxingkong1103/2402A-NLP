# -*- coding: utf-8 -*-
"""Redis short-term conversation memory."""

from app.config import enabled, setting


class ConversationMemory:
    def __init__(self, max_turns=10):
        self.max_items = max_turns * 2
        if enabled("RAG_TEST_MODE"):
            import fakeredis

            self.client = fakeredis.FakeRedis(decode_responses=True)
            self.backend = "fakeredis"
        else:
            import redis

            self.client = redis.Redis.from_url(
                setting("RAG_REDIS_URL", "redis://127.0.0.1:6379/0"),
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self.client.ping()
            self.backend = "redis"

    @staticmethod
    def key(user_id, role_id):
        return f"chat:{user_id}:{role_id}"

    def add(self, user_id, role_id, role, content):
        key = self.key(user_id, role_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, f"{role}:{content}")
        pipe.ltrim(key, -self.max_items, -1)
        pipe.expire(key, int(setting("RAG_MEMORY_TTL_SECONDS", "86400")))
        pipe.execute()

    def add_turn(self, user_id, role_id, question, answer):
        key = self.key(user_id, role_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, f"user:{question}", f"assistant:{answer}")
        pipe.ltrim(key, -self.max_items, -1)
        pipe.expire(key, int(setting("RAG_MEMORY_TTL_SECONDS", "86400")))
        pipe.execute()

    def history(self, user_id, role_id, budget=1600):
        lines = []
        for item in self.client.lrange(self.key(user_id, role_id), 0, -1):
            role, _, content = item.partition(":")
            lines.append(f"{'用户' if role == 'user' else '助手'}：{content}")
        selected = []
        for line in reversed(lines):
            if len(line) + 1 > budget:
                if not selected:
                    selected.append(line[:budget])
                break
            selected.append(line)
            budget -= len(line) + 1
        return "\n".join(reversed(selected)) if selected else "（无历史对话）"

    def clear(self, user_id, role_id):
        self.client.delete(self.key(user_id, role_id))

    def ping(self):
        return bool(self.client.ping())
