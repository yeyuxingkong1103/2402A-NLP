import json


class ShortTermMemory:
    def __init__(self, redis_client):
        self.redis = redis_client

    def _key(self, conversation_id: int) -> str:
        return f"chat:{conversation_id}:messages"

    async def append(self, conversation_id: int, role: str, content: str) -> None:
        await self.redis.rpush(
            self._key(conversation_id), json.dumps({"role": role, "content": content})
        )

    async def get(self, conversation_id: int) -> list[dict]:
        raw = await self.redis.lrange(self._key(conversation_id), 0, -1)
        return [json.loads(x) for x in raw]

    async def clear(self, conversation_id: int) -> None:
        await self.redis.delete(self._key(conversation_id))

    async def len(self, conversation_id: int) -> int:
        return await self.redis.llen(self._key(conversation_id))
