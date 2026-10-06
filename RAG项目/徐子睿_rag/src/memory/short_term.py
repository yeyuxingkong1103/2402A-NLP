"""src/memory/short_term.py —— 短期记忆（Redis，带进程内降级）。

在链路中的位置：
    src/online/chain.py → 【本文件】 → Redis（或进程内字典）
    src/online/retriever.py 也用本文件的检索缓存（retrieval_cache_get/set）

短期记忆 = 当前会话最近若干轮对话。它和长期记忆的分工：
    短期  原文保留最近 N 轮，保证对话连贯（"刚才你说的那个"能接上）
    长期  把历史发言向量化，按语义相关性跨会话召回（很久以前提过的偏好也能想起来）

核心设计：Redis 可用就用 Redis，不可用就退回进程内字典。
    **这是本项目能在没装 Redis 的机器上跑通的第三个降级开关**
    （前两个是 EMBEDDING_BACKEND=hash 和 LLM_BACKEND=mock）。
    降级后会话记忆只存在于当前进程 —— 重启即丢、多进程不共享，
    但单机演示完全够用，流程能完整走通。

Redis 数据结构的选择：
    zset（有序集合）存消息  —— 用时间戳做 score，天然按时间排序，
                              且能 O(logN) 地裁掉旧消息
    hash 存会话元信息       —— 记录 role_id 和最近活跃时间
    zset "hot:roles"        —— 各角色的使用热度统计
    set  "online:users"     —— 活跃用户去重统计
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from configs.settings import get_settings

# 降级用的进程内存储：{会话key: [消息...]}。defaultdict 省去"键不存在先建空列表"的样板代码
_memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
_redis = None  # 三态：None=还没试过；False=试过且失败（不再重试）；对象=可用连接


def _client():
    """获取 Redis 客户端（惰性连接 + 失败缓存）。

    返回：
        可用的 Redis 客户端；不可用时返回 None。

    三态设计的意义（_redis 取 None / False / 连接对象）：
        如果用 None 同时表示"没试过"和"试过失败"，
        那么 Redis 没启动时，每次读写都要重新尝试连接并等一次超时 ——
        本来只是"降级到内存"，结果变成每次操作都卡几百毫秒。
        用 False 记住失败结果后，后续调用直接返回 None，降级路径是零成本的。

    一次 ping 是必要的：
        from_url 只是构造客户端对象、不会真正连接，
        不做 ping 探测的话，"Redis 没启动"这个事实要等到第一次真正读写才暴露，
        而那时代码已经走进 Redis 分支了。

    降级是捕获**所有**异常（except Exception）：
        连接被拒、超时、认证失败、DNS 解析失败……对使用者来说都等价于"Redis 不可用"，
        区分它们没有意义，统一降级即可。
    """
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis

        # decode_responses=True 让取回的是 str 而不是 bytes，省去到处 decode
        _redis = redis.from_url(get_settings().redis_url, decode_responses=True)
        _redis.ping()
        return _redis
    except Exception:
        _redis = False
        return None


def _key(user_id: int, session_id: int) -> str:
    """构造会话的 Redis 键名。

    参数：
        user_id: 用户 id
        session_id: 会话 id
    返回：
        形如 "chat:{用户}:{会话}:messages" 的键名。

    键名里带 user_id 是有意为之：
        会话 id 在多租户环境下不保证全局唯一，
        只用 session_id 做键可能让两个用户的会话撞在一起。
    """
    return f"chat:{user_id}:{session_id}:messages"


def append_message(user_id: int, session_id: int, role_id: str, speaker: str, content: str) -> None:
    """追加一条消息到短期记忆，并裁剪掉过旧的记录。

    参数：
        user_id / session_id: 会话定位
        role_id: 角色 id
        speaker: "user" 或 "assistant"
        content: 消息正文

    Redis 分支做的四件事：
        1. zadd 写入消息，score 用时间戳 -> 天然按时间有序
        2. zremrangebyrank 裁掉最早的记录，只保留最近 short_memory_rounds 轮
        3. hset 更新会话元信息（当前角色、最后活跃时间）
        4. 更新"角色热度"和"在线用户"两个统计集合

    `* 2` 的含义（保留条数 = rounds * 2）：
        一轮对话 = 一句用户提问 + 一句助手回答，所以"保留 N 轮"对应 2N 条消息。
        这个换算在 append 和 recent 两处必须一致，否则读到的条数与预期不符。

    zremrangebyrank(key, 0, -N-1) 的区间语义：
        从第 0 位（最早）删到倒数第 N-1 位，即"只保留最后 N 条"。
        负号是 Redis 的下标语法（-1 表示最后一个元素）。

    内存分支用 `del messages[:-N]`：
        切片删除保留最后 N 条，语义与上面完全对应 ——
        两条路径的裁剪行为一致，才能在切换后端时不影响对话表现。
    """
    item = {"speaker": speaker, "role_id": role_id, "content": content, "ts": time.time()}
    client = _client()
    if client:
        key = _key(user_id, session_id)
        client.zadd(key, {json.dumps(item, ensure_ascii=False): item["ts"]})
        client.zremrangebyrank(key, 0, -get_settings().short_memory_rounds * 2 - 1)
        client.hset(f"chat:{user_id}:{session_id}:meta", mapping={"role_id": role_id, "updated_at": str(item["ts"])})
        client.zincrby("hot:roles", 1, role_id)
        client.sadd("online:users", str(user_id))
        return
    messages = _memory[_key(user_id, session_id)]
    messages.append(item)
    del messages[:-get_settings().short_memory_rounds * 2]


def recent_messages(user_id: int, session_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """读取最近的消息（用于拼提示词的短期记忆层）。

    参数：
        user_id / session_id: 会话定位
        limit: 条数上限，不传则按配置的轮数换算（rounds * 2）
    返回：
        按时间**正序**排列的消息列表。

    Redis 的 zrange(key, -limit, -1)：
        负下标表示"从尾部数"，-limit 到 -1 即"最后 limit 条"，
        且 zrange 天然按 score 升序返回，正好是时间正序 —— 不需要额外反转。

    内存分支用切片 [-limit:]：
        同样是取最后 limit 条，且保持正序。

    两条路径都返回正序，是为了让调用方拿到的是"能直接读懂"的对话流；
    如果返回倒序，拼进提示词后模型读到的对话是反的，会严重影响回答质量。
    """
    limit = limit or get_settings().short_memory_rounds * 2
    client = _client()
    if client:
        values = client.zrange(_key(user_id, session_id), -limit, -1)
        return [json.loads(item) for item in values]
    return list(_memory[_key(user_id, session_id)][-limit:])


def retrieval_cache_get(key: str) -> str | None:
    """读检索结果缓存。

    参数：
        key: 缓存键（retriever 用 md5 生成的哈希）
    返回：
        缓存的 JSON 字符串；无缓存或 Redis 不可用时返回 None。

    注意本函数在 Redis 不可用时**直接返回 None**（不做内存缓存）：
        检索缓存是纯性能优化，进程内再缓存一层收益有限、
        还会让"同一个查询在重启前后行为不一致"，不如不做。
    """
    client = _client()
    return client.get(f"cache:retrieval:{key}") if client else None


def retrieval_cache_set(key: str, value: str, ttl: int = 300) -> None:
    """写检索结果缓存。

    参数：
        key: 缓存键
        value: 要缓存的 JSON 字符串
        ttl: 过期秒数，默认 300（5 分钟）

    为什么要 TTL：
        知识库会更新（上传/删除文档）。永久缓存会让用户一直检索到旧结果，
        表现为"我明明删了这份文档，怎么还能搜到"。
        5 分钟是一个折中：既能挡住重复提问的重复计算，
        又不至于让数据变更长期不生效。

    Redis 不可用时静默跳过（不抛异常）：
        缓存写失败绝不该影响检索本身的可用性。
    """
    client = _client()
    if client:
        client.setex(f"cache:retrieval:{key}", ttl, value)
