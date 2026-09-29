# -*- coding: utf-8 -*-
"""短期记忆：用 Redis List 保存每个 用户×角色 会话的最近 N 轮对话。"""
import json
# 解析：消息以 JSON 字符串存入 Redis List


class RedisMemoryStore:
    """key = chat:history:{user_id}:{role_id}，List 尾部追加，超过 max_rounds 轮裁掉最旧的。"""

    def __init__(self, redis_client, max_rounds: int = 10, ttl_seconds: int = 604800):
        # 解析：构造——传入 Redis 客户端与记忆参数
        self.client = redis_client
        # 解析：保存 Redis 客户端
        self.max_rounds = max_rounds
        # 解析：最多保留轮数（默认 10 轮）
        self.ttl_seconds = ttl_seconds
        # 解析：记忆过期时间（默认 7 天）

    def _key(self, user_id: int, role_id: int) -> str:
        # 解析：生成该会话的记忆 key
        return f"chat:history:{user_id}:{role_id}"
        # 解析：key 含用户ID×角色ID——会话两维隔离

    def get_history(self, user_id: int, role_id: int) -> list[dict]:
        # 解析：读记忆
        raw = self.client.lrange(self._key(user_id, role_id), 0, -1)
        # 解析：取整个 List（0 到 -1 = 全部元素）
        return [json.loads(item) for item in raw]
        # 解析：逐条 JSON 反序列化成 {role, content} 字典

    def append(self, user_id: int, role_id: int, sender: str, content: str) -> None:
        # 解析：写一条消息进记忆
        key = self._key(user_id, role_id)
        # 解析：取该会话 key
        pipe = self.client.pipeline()
        # 解析：用 pipeline 打包多条命令（一次网络往返，提升性能）
        pipe.rpush(key, json.dumps({"role": sender, "content": content}, ensure_ascii=False))
        # 解析：尾部追加消息（JSON 序列化，中文不转义）
        pipe.ltrim(key, -self.max_rounds * 2, -1)  # 1 轮 = 2 条消息
        # 解析：只保留最近 max_rounds*2 条（O(1) 裁剪，超出部分丢弃）
        pipe.expire(key, self.ttl_seconds)
        # 解析：续期 TTL（每次写入刷新过期时间）
        pipe.execute()
        # 解析：执行整组命令

    def clear(self, user_id: int, role_id: int) -> None:
        # 解析：清空某会话记忆
        self.client.delete(self._key(user_id, role_id))
        # 解析：直接删除 key（测试与隐私场景用）
