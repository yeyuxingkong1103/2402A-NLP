"""Redis 短期记忆：保存最近消息、会话摘要与摘要节流状态。

设计动机：LLM 无状态，多轮问答的"上文"必须在请求之间落到外部存储。
选 Redis 而不是 MySQL 的原因：
  · 短期记忆本质是"可丢弃"的缓存——TTL 到期自然消失，不需要持久化保障
  · 消息窗口用 List 的 rpush + ltrim 就能 O(1) 完成追加与截断
key 按 user_id/session_id 双层隔离，保证跨用户/跨会话互不可见。

每个会话三个 key（同 TTL，一起生一起灭）：
  · messages       List，滑动窗口；**超窗口的更早消息被 ltrim 永久丢弃**
  · summary        String，压缩后的前情（批次 21 起由 summary_service 写入）
  · summary_state  String(JSON)，摘要节流计数（窗口长度恒定，只能靠它判断"又过几轮"）
"""

from __future__ import annotations

import json
from typing import Any


def message_key(user_id: str, session_id: str) -> str:
    """返回指定用户和会话的消息列表 key。

    Args:
        user_id: 用户标识
        session_id: 会话标识

    Returns:
        形如 short_memory:{user_id}:{session_id}:messages 的 Redis key
    """
    # 统一前缀便于 SCAN 排查；双层 id 保证隔离性，杜绝串会话
    return f"short_memory:{user_id}:{session_id}:messages"


def summary_key(user_id: str, session_id: str) -> str:
    """返回指定用户和会话的摘要 key（参数与 Returns 同 message_key 约定）。"""
    return f"short_memory:{user_id}:{session_id}:summary"


def summary_state_key(user_id: str, session_id: str) -> str:
    """返回会话摘要的节流状态 key（参数与 Returns 同 message_key 约定）。

    为什么需要它：消息列表是固定长度的滑动窗口，窗口满了之后**长度不再变化**，
    无法据此判断"距上次摘要又过了几轮"。摘要服务因此需要一个会话级计数器，
    记录 {"turns": 累计轮次, "summarized_turns": 上次摘要时的轮次}。
    与消息/摘要同 TTL，会话沉睡后一起过期，不留脏数据。
    """
    return f"short_memory:{user_id}:{session_id}:summary_state"


class ShortTermMemoryStore:
    """按用户和会话隔离的 Redis 短期记忆存储。

    两个窗口参数（max_messages 限制条数、ttl_seconds 限制时长）缺一不可：
    只限条数不限时长 → 旧会话消息永不过期占内存；
    只限时长不限条数 → 超长会话会把 prompt 撑爆。
    """

    def __init__(self, redis_client: Any, ttl_seconds: int, max_messages: int) -> None:
        """初始化存储，并校验记忆窗口和过期时间。

        Args:
            redis_client: 已连接的 Redis 客户端（不在此处建连，便于测试注入 fake）
            ttl_seconds: key 的过期秒数
            max_messages: 消息列表最多保留条数

        Raises:
            ValueError: ttl_seconds 或 max_messages 非正数（配置错误尽早暴露）
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")
        if max_messages <= 0:
            raise ValueError("max_messages 必须大于 0")
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def append_message(self, user_id: str, session_id: str, message: dict[str, Any]) -> None:
        """追加一条消息，并限制列表最多保留指定数量。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            message: 消息字典（至少含 role/content），JSON 序列化后存入
        """
        key = message_key(user_id, session_id)
        # ensure_ascii=False：中文按原样存储，便于人工直接用 redis-cli 排查
        self.redis.rpush(key, json.dumps(message, ensure_ascii=False))
        # ltrim 只保留最后 max_messages 条——滑动窗口；紧跟追加执行，列表长度恒定
        self.redis.ltrim(key, -self.max_messages, -1)
        # 每次写入都续期：活跃会话不过期，沉睡 ttl 秒后整条链路自然消亡
        self.redis.expire(key, self.ttl_seconds)

    def read_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """读取指定用户会话的全部短期消息。

        Args:
            user_id: 用户标识
            session_id: 会话标识

        Returns:
            按时间正序（最老在前）的消息列表；会话不存在时返回空列表
        """
        key = message_key(user_id, session_id)
        # lrange(0, -1) 取全量且天然保持写入顺序，正序正是对话上下文需要的顺序
        return [json.loads(value) for value in self.redis.lrange(key, 0, -1)]

    def write_summary(self, user_id: str, session_id: str, summary: str) -> None:
        """写入会话摘要并设置 TTL。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            summary: 摘要文本（整键覆盖写——摘要只有"最新一份"有意义）
        """
        # 用 SET + ex 一步完成写入与续期，避免 SET/EXPIRE 两步之间出现无 TTL 的裸 key
        self.redis.set(summary_key(user_id, session_id), summary, ex=self.ttl_seconds)

    def read_summary(self, user_id: str, session_id: str) -> str | None:
        """读取指定用户会话的摘要。

        Args:
            user_id: 用户标识
            session_id: 会话标识

        Returns:
            摘要文本；不存在或已过期时返回 None
        """
        return self.redis.get(summary_key(user_id, session_id))

    def read_summary_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        """读取会话摘要的节流状态。

        Args:
            user_id: 用户标识
            session_id: 会话标识

        Returns:
            状态字典（如 {"turns": 13, "summarized_turns": 10}）；
            key 不存在、内容非法、JSON 解析失败时统一返回空字典——
            调用方按"从第 0 轮开始"处理即可，不需要区分这些情形。
        """
        raw = self.redis.get(summary_state_key(user_id, session_id))
        if not raw:
            return {}
        try:
            state = json.loads(raw)
        except (TypeError, ValueError):
            # 被外部写坏的值当作不存在：宁可多摘要一次，也不能让摘要链路报错
            return {}
        return state if isinstance(state, dict) else {}

    def write_summary_state(self, user_id: str, session_id: str, state: dict[str, Any]) -> None:
        """写入会话摘要的节流状态并设置 TTL（整键覆盖写）。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            state: 状态字典，随摘要一起按 ttl_seconds 过期
        """
        # 与 write_summary 同款：SET + ex 一步完成，避免出现无 TTL 的裸 key
        self.redis.set(
            summary_state_key(user_id, session_id),
            json.dumps(state, ensure_ascii=False),
            ex=self.ttl_seconds,
        )

    def delete_session_memory(self, user_id: str, session_id: str) -> None:
        """删除指定用户会话的消息、摘要与摘要节流状态。

        Args:
            user_id: 用户标识
            session_id: 会话标识

        说明：DEL 一个不存在的 key 是 no-op，因此无须先判存在性，天然幂等。
        摘要状态必须一并删除：否则同一 session_id 被复用时会带着旧计数，
        导致新会话的首次摘要被错误节流掉。
        """
        self.redis.delete(
            message_key(user_id, session_id),
            summary_key(user_id, session_id),
            summary_state_key(user_id, session_id),
        )
