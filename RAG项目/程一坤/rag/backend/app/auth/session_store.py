"""会话令牌的创建、读取、撤销：只负责 Redis 读写，不涉及 HTTP。"""

import json
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionUser:
    """会话用户信息。"""

    user_id: str
    is_admin: bool


class SessionStore:
    """会话存储：管理 Redis 中的会话令牌与用户信息。

    Redis 键约定（必须遵守，便于排查与清理）：
    - session:<token>            → JSON {"user_id": "...", "is_admin": true}，TTL = session_ttl_seconds
    - user_sessions:<user_id>    → SET，成员为该用户当前所有 token，用于 revoke_all_sessions 反查
    """

    def __init__(self, redis_client, session_ttl_seconds: int) -> None:
        """初始化会话存储。

        参数：
        - redis_client: Redis 客户端实例（或测试替身）
        - session_ttl_seconds: 会话令牌有效期（秒）
        """
        self.redis = redis_client
        self.session_ttl_seconds = session_ttl_seconds

    def create_session_token(self, user_id: str, is_admin: bool) -> str:
        """创建会话并返回令牌。

        令牌用 secrets.token_urlsafe(32) 生成，不包含任何可解码的用户信息。
        """
        # 生成随机令牌（URL 安全，约 43 字符）
        token = secrets.token_urlsafe(32)

        # 会话数据
        session_data = json.dumps({"user_id": user_id, "is_admin": is_admin})

        # 存储会话：session:<token> → JSON
        session_key = f"session:{token}"
        self.redis.set(session_key, session_data, ex=self.session_ttl_seconds)

        # 记录用户会话集合：user_sessions:<user_id> → SET，设置 TTL
        user_sessions_key = f"user_sessions:{user_id}"
        self.redis.sadd(user_sessions_key, token)
        # 顺手清掉悬空成员：集合的 TTL 每次登录都会被刷新，但成员对应的
        # session:<token> 会各自到期——只增不减的话，长期活跃用户的集合会
        # 无限堆积（批次 24 在真实 Redis 观察到 user_sessions:pending 残留 19 个
        # 全悬空 token 且 TTL=-1）。放在 expire 之前，清理完再统一续期。
        self._prune_dangling_tokens(user_sessions_key)
        self.redis.expire(user_sessions_key, self.session_ttl_seconds)

        return token

    def _prune_dangling_tokens(self, user_sessions_key: str) -> int:
        """移除集合内 session:<token> 已不存在的悬空 token，返回清理个数。

        用 get 而不是 exists 判存在：测试替身（tests/conftest.py、
        tests/test_session_store.py 的 FakeRedis）只实现了 get，
        这样生产 Redis 与测试替身语义一致，不必为测试新增方法。
        """
        dangling = [
            token
            for token in self.redis.smembers(user_sessions_key)
            if self.redis.get(f"session:{token}") is None
        ]
        if dangling:
            self.redis.srem(user_sessions_key, *dangling)
        return len(dangling)

    def read_session(self, token: str) -> SessionUser | None:
        """读取会话；不存在或已过期返回 None。"""
        session_key = f"session:{token}"
        session_data = self.redis.get(session_key)

        if session_data is None:
            return None

        # 解析 JSON
        data = json.loads(session_data)
        return SessionUser(user_id=data["user_id"], is_admin=data["is_admin"])

    def revoke_session(self, token: str) -> None:
        """注销单个会话；令牌不存在时按幂等处理。"""
        session_key = f"session:{token}"

        # 先读取会话数据以获取 user_id
        session_data = self.redis.get(session_key)
        if session_data is not None:
            data = json.loads(session_data)
            user_id = data["user_id"]

            # 删除会话
            self.redis.delete(session_key)

            # 从用户会话集合中移除
            user_sessions_key = f"user_sessions:{user_id}"
            self.redis.srem(user_sessions_key, token)

            # 如果用户会话集合为空，删除该 key
            if len(self.redis.smembers(user_sessions_key)) == 0:
                self.redis.delete(user_sessions_key)

    def revoke_all_sessions(self, user_id: str) -> None:
        """撤销某用户的全部会话（改密码、禁用用户时使用）。"""
        user_sessions_key = f"user_sessions:{user_id}"

        # 获取该用户的所有令牌
        tokens = self.redis.smembers(user_sessions_key)

        # 删除所有会话
        for token in tokens:
            session_key = f"session:{token}"
            self.redis.delete(session_key)

        # 删除用户会话集合
        self.redis.delete(user_sessions_key)
