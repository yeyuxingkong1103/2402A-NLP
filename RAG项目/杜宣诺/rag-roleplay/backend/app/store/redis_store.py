# 导入 json：把消息（字典）转成字符串存进 Redis，取出来再转回字典
import json

# 导入 redis 的异步客户端（async 版，配合 FastAPI 的异步）
import redis.asyncio as redis

# 读配置（Redis 地址）
from ..config import get_settings


class RedisStore:
    """Redis 存储：负责存"短期记忆"——最近的对话。"""

    def __init__(self, url: str | None = None):
        # 连接 Redis。decode_responses=True 表示：取出来的值是字符串，
        # 而不是 bytes（省得每次手动解码）
        self.redis = redis.from_url(url or get_settings().redis_url, decode_responses=True)

    async def push_message(self, session_id: int, role: str, content: str) -> None:
        """【存消息】把一条消息追加到某个会话的历史里。"""
        # rpush = 从列表右边插入。
        # key 是 session:{会话id}:history，value 是 JSON 字符串（角色 + 内容）
        await self.redis.rpush(
            f"session:{session_id}:history",
            json.dumps({"role": role, "content": content})
        )

    async def get_recent(self, session_id: int, rounds: int) -> list[dict]:
        """【取最近对话】拿最近 N 轮消息。"""
        # lrange = 取列表的一段。(-2*rounds, -1) 表示：
        # 从倒数第 2*rounds 个，取到最后一个（因为一轮有 user + assistant 两条）
        items = await self.redis.lrange(f"session:{session_id}:history", -2 * rounds, -1)
        # 把每个 JSON 字符串还原成字典，返回列表
        return [json.loads(i) for i in items]

    async def get_summary(self, session_id: int) -> str | None:
        """【取摘要】拿这个会话的历史摘要（超出 N 轮后被压缩的）。"""
        return await self.redis.get(f"session:{session_id}:summary")

    async def set_summary(self, session_id: int, text: str) -> None:
        """【存摘要】保存历史摘要。"""
        await self.redis.set(f"session:{session_id}:summary", text)