import json
from uuid import uuid4

import redis

from backend.app.core.config import get_settings
from embedding.bge_m3_embedder import get_embedding_service
from vectorstores.milvus_store import MilvusStore


settings = get_settings()


class MemoryService:
    """负责 Redis 短期记忆和 Milvus 长期记忆。"""

    def __init__(self) -> None:
        # 创建 Redis 客户端，用来保存最近几轮对话。
        self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        # 创建向量模型服务，用来给长期记忆生成向量。
        self.embedding_service = get_embedding_service()
        # 创建 Milvus 服务，用来保存和检索长期记忆。
        self.milvus = MilvusStore()

    def append_short_message(self, session_id: int, role: str, content: str) -> None:
        # Redis key 按会话隔离，避免不同会话互相污染。
        key = f"short_memory:session:{session_id}"
        try:
            # 把消息序列化成 JSON 后追加到列表右侧。
            self.redis.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            # 只保留最近 N 条消息，避免上下文无限膨胀。
            self.redis.ltrim(key, -settings.short_memory_window, -1)
            # 设置过期时间，形成短期记忆。
            self.redis.expire(key, settings.short_memory_ttl_seconds)
        except redis.RedisError:
            return

    def get_short_messages(self, session_id: int) -> list[dict]:
        # Redis key 按会话读取。
        key = f"short_memory:session:{session_id}"
        try:
            # 读取最近的短期记忆消息。
            raw_items = self.redis.lrange(key, 0, -1)
        except redis.RedisError:
            return []
        # 反序列化成 Python 字典列表。
        return [json.loads(item) for item in raw_items]

    def save_long_memory(self, user_id: int, content: str) -> None:
        # 内容太短时不保存长期记忆，避免噪声。
        if len(content.strip()) < 10:
            return
        # 给长期记忆生成唯一编号。
        memory_id = f"mem-{user_id}-{uuid4().hex}"
        # 使用 BGE-M3 给记忆文本生成向量。
        vector = self.embedding_service.encode([content])[0]
        try:
            # 写入 Milvus 的用户级长期记忆集合。
            self.milvus.insert_memory(memory_id, user_id, content, vector)
        except Exception:
            return

    def search_long_memory(self, user_id: int, question: str) -> list[str]:
        # 使用问题生成向量，用来检索相关历史偏好和摘要。
        vector = self.embedding_service.encode([question])[0]
        try:
            # 返回当前用户的长期记忆。
            return self.milvus.search_memory(vector, user_id)
        except Exception:
            return []
