# -*- coding: utf-8 -*-
"""
Redis 短期记忆层：保存每个"用户+角色"会话的最近 N 条消息
大模型本身无记忆，用 Redis List 做滑动窗口

数据结构：
    Key   = f"chat:{user_id}:{role_id}"   （每个用户每个角色一个独立会话）
    Value = Redis List，每条元素是一个 JSON 字符串：{"role":"user","content":"..."}
    过期   = 2 小时（无活动自动清理，避免内存浪费）

Redis 5 种数据类型在本项目中的用途：
    String : 计数器、缓存单值
    List   : 对话历史（本项目核心用法，LPUSH + LTRIM 做滑动窗口）
    Set    : 在线用户集合
    Dict(Hash) : 角色配置缓存
    ZSet   : 按时间排序的会话列表（可做"最近会话"功能）
"""

import json  # 消息序列化
from typing import List, Dict, Optional  # 类型标注

import redis  # Redis 客户端库

from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, MEMORY_WINDOW  # Redis 配置
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# Redis 连接池：decode_responses=True 自动把 bytes 转成 str
_pool = redis.ConnectionPool(  # 连接池（全局复用，不每次新建连接）
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD or None,
    db=REDIS_DB, decode_responses=True,
)
_redis_client = redis.Redis(connection_pool=_pool)  # 客户端实例

SESSION_TTL = 7200  # 会话过期时间：2 小时（秒），无活动自动清理


def _session_key(user_id: int, role_id: int) -> str:
    """构造会话 Key：每个用户每个角色一个独立的对话历史"""
    return f"chat:{user_id}:{role_id}"  # 如 "chat:1:3"


def add_message(user_id: int, role_id: int, role: str, content: str):
    """
    向会话追加一条消息（LPUSH + LTRIM 做固定长度滑动窗口）

    Args:
        user_id: 用户 ID
        role_id: 角色 ID
        role: 消息角色，"user"（用户说的）或 "assistant"（角色回的）
        content: 消息内容
    """
    key = _session_key(user_id, role_id)  # 会话 Key
    msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)  # 序列化为 JSON
    _redis_client.lpush(key, msg)  # LPUSH：从左（头部）插入，最新消息在最前
    _redis_client.ltrim(key, 0, MEMORY_WINDOW - 1)  # LTRIM：只保留前 N 条，旧的自动删除（O(1)）
    _redis_client.expire(key, SESSION_TTL)  # 设置/刷新过期时间（2 小时无活动自动清理）


def get_history(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """
    获取会话历史消息列表

    Returns:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        顺序：从新到旧（LPUSH 插入，lrange 0 到 -1 就是新到旧）
        喂给 LLM 时需要反转成旧到新
    """
    key = _session_key(user_id, role_id)  # 会话 Key
    raw = _redis_client.lrange(key, 0, -1)  # LRANGE：取全部（已 LTRIM 限制在 N 条内）
    messages = [json.loads(item) for item in raw]  # 逐条反序列化
    return messages  # 新到旧


def get_history_for_llm(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """
    获取会话历史，格式化为 LLM 能直接用的消息列表（旧到新排序）

    返回格式（与 OpenAI/DeepSeek messages 格式一致）：
        [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            ...
        ]
    """
    messages = get_history(user_id, role_id)  # 新到旧
    messages.reverse()  # 反转成旧到新（LLM 要求时间顺序）
    return messages


def clear_session(user_id: int, role_id: int):
    """清空指定会话的记忆（用户切换角色或主动重置时调用）"""
    key = _session_key(user_id, role_id)
    _redis_client.delete(key)  # DEL：删除整个 Key
    logger.info(f"会话已清空：user={user_id} role={role_id}")


def set_user_active(user_id: int):
    """标记用户在线（Set 用法示例）"""
    _redis_client.sadd("online_users", str(user_id))  # SADD：加入在线集合


def get_online_users() -> int:
    """获取在线用户数（Set 用法示例）"""
    return _redis_client.scard("online_users")  # SCARD：返回集合元素数
