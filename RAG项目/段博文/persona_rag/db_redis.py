# -*- coding: utf-8 -*-
"""
Redis 短期记忆层：保存每个"用户+角色"会话的最近 N 条消息。

在系统中的位置：
    上游是 main.py（FastAPI 路由）：每轮问答前调 get_history_for_llm() 取上下文，
    生成回答后再调 add_message() 把「问」和「答」依次写回；下游只有 Redis 一个
    外部依赖。本模块只管「会话态」，与 MySQL 的「审计日志」职责分离。

大模型本身无记忆，用 Redis List 做滑动窗口：
    Key   = f"chat:{user_id}:{role_id}"   （每个用户每个角色一个独立会话）
    Value = Redis List，每条元素是一个 JSON 字符串：{"role":"user","content":"..."}
    过期   = 2 小时（无活动自动清理，避免内存浪费）

关键设计取舍：
    1. 用 List + LPUSH/LTRIM，而不是把整段历史存成一个 JSON 大字符串：LTRIM 裁剪
       是 O(1)，LRANGE 一次读完，天然贴合「只保留最近 N 条」的滑动窗口语义；
    2. 写入时新消息在头部（LPUSH），只有喂给 LLM 前才 reverse，读写两端都是 O(1)；
    3. 依赖 decode_responses=True 自动把 bytes 解码成 str，业务层不必到处 .decode()；
    4. 本模块不做 try/except 兜底：Redis 故障时异常向上抛，交由上层决定是否降级，
       避免在这里静默吞掉基础设施故障。

Redis 5 种数据类型在本项目中的用途：
    String : 计数器、缓存单值
    List   : 对话历史（本项目核心用法，LPUSH + LTRIM 做滑动窗口）
    Set    : 在线用户集合
    Dict(Hash) : 角色配置缓存
    ZSet   : 按时间排序的会话列表（可做"最近会话"功能）
"""

import json  # 消息序列化（dict 与 JSON 字符串互转）
from typing import List, Dict, Optional  # 类型标注

import redis  # Redis 客户端库

from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, MEMORY_WINDOW  # Redis 配置
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# Redis 连接池：decode_responses=True 自动把 bytes 转成 str
# 连接池是进程级全局对象：多个请求/线程共用同一批 TCP 连接，省掉每次请求的握手开销；
# password 用 `or None` 兜底——传空串时 redis-py 会真的发一条空密码 AUTH 命令从而报错。
_pool = redis.ConnectionPool(  # 连接池（全局复用，不每次新建连接）
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD or None,
    db=REDIS_DB, decode_responses=True,
)
_redis_client = redis.Redis(connection_pool=_pool)  # 客户端实例（线程安全，可全局复用）

# 注意：redis-py 在 import / 构造时并不会真正建连，第一条命令才去连接；
# 所以 Redis 没启动时 import 本模块不会炸，只会在首次读写时报连接错误。
SESSION_TTL = 7200  # 会话过期时间：2 小时（秒），无活动自动清理


def _session_key(user_id: int, role_id: int) -> str:
    """
    构造会话 Key：每个用户每个角色一个独立的对话历史。

    参数：
        user_id：用户 ID（MySQL users.id，正整数）。
        role_id：角色 ID（MySQL roles.id，正整数）。
    返回：
        str，形如 "chat:1:3"；同一用户切换角色会得到不同的 Key，历史不会串台。
    """
    return f"chat:{user_id}:{role_id}"  # 如 "chat:1:3"


def add_message(user_id: int, role_id: int, role: str, content: str):
    """
    向会话追加一条消息（LPUSH + LTRIM 做固定长度滑动窗口）

    Args:
        user_id: 用户 ID
        role_id: 角色 ID
        role: 消息角色，"user"（用户说的）或 "assistant"（角色回的）；
              该值会被原样透传给 LLM，取值写错会导致 messages 格式非法
        content: 消息内容

    返回：
        None（三条 Redis 命令都成功即返回）。
    异常与降级：
        不做捕获，Redis 不可用时直接抛 redis.RedisError，由上层决定是否降级。
    """
    key = _session_key(user_id, role_id)  # 会话 Key
    # ensure_ascii=False：中文按原样存，可读且更省空间
    msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)  # 序列化为 JSON
    _redis_client.lpush(key, msg)  # LPUSH：从左（头部）插入，最新消息在最前
    _redis_client.ltrim(key, 0, MEMORY_WINDOW - 1)  # LTRIM：只保留前 N 条，旧的自动删除（O(1)）
    _redis_client.expire(key, SESSION_TTL)  # 设置/刷新过期时间（2 小时无活动自动清理）


def get_history(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """
    获取会话历史消息列表

    参数：
        user_id / role_id：用于定位会话 Key（语义见 _session_key）。
    返回：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        顺序：从新到旧（LPUSH 插入，lrange 0 到 -1 就是新到旧）
        喂给 LLM 时需要反转成旧到新
    异常与降级：
        Key 不存在时 Redis 返回空列表，这里就返回 []，不抛异常（新会话的正常情况）；
        若某条元素不是合法 JSON（人为写坏 Key 的情况），json.loads 会抛 ValueError，
        这里刻意不吞掉，便于尽早暴露脏数据。
    """
    key = _session_key(user_id, role_id)  # 会话 Key
    raw = _redis_client.lrange(key, 0, -1)  # LRANGE：取全部（已 LTRIM 限制在 N 条内）
    messages = [json.loads(item) for item in raw]  # 逐条反序列化
    return messages  # 新到旧


def get_history_for_llm(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """
    获取会话历史，格式化为 LLM 能直接用的消息列表（旧到新排序）

    参数：
        user_id / role_id：定位会话 Key，语义同上。

    返回格式（与 OpenAI/DeepSeek messages 格式一致）：
        [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            ...
        ]
    说明：
        这是主链路真正调用的函数；返回空列表表示新会话，此时 LLM 只能看到本轮
        问题和检索到的知识，不会凭空「记得」之前的对话。
    """
    messages = get_history(user_id, role_id)  # 新到旧
    messages.reverse()  # 反转成旧到新（LLM 要求时间顺序，顺序反了会答非所问）
    return messages


def clear_session(user_id: int, role_id: int):
    """
    清空指定会话的记忆（用户切换角色或主动重置时调用）

    参数：
        user_id / role_id：定位会话 Key。
    返回：
        None。
    说明：
        用 DEL 整体删除，而不是逐条 LPOP：只需一次网络往返，还能顺带清掉 TTL。
    """
    key = _session_key(user_id, role_id)
    _redis_client.delete(key)  # DEL：删除整个 Key
    logger.info(f"会话已清空：user={user_id} role={role_id}")


def set_user_active(user_id: int):
    """
    标记用户在线（Set 用法示例）

    参数：
        user_id：用户 ID；写入前统一 str() 转换，保证集合内元素类型一致。
    返回：
        None。SADD 是幂等的，重复标记同一个用户不会产生重复元素。
    """
    _redis_client.sadd("online_users", str(user_id))  # SADD：加入在线集合


def get_online_users() -> int:
    """
    获取在线用户数（Set 用法示例）

    返回：
        int，集合基数（元素个数）；集合不存在时 SCARD 返回 0，不报错。
    """
    return _redis_client.scard("online_users")  # SCARD：返回集合元素数