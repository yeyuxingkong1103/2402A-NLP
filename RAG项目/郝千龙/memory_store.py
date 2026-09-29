# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，保证中文字符正常解析
"""短期记忆：Redis List；不可用时退回进程内字典。"""  # 模块文档字符串：概述本模块职责与降级策略
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，让 dict[str, list[dict]] 等类型注解在低版本也能用

import json  # 导入 json 模块，用于消息字典与字符串之间的序列化/反序列化
from collections import defaultdict  # 导入 defaultdict，缺失键自动创建默认值，简化降级字典的写入
from typing import Any  # 导入 Any 类型，标注返回值中可承载任意类型的元素

from config import REDIS_ENABLED, REDIS_URL, SHORT_MEMORY_TURNS  # 从配置模块导入开关、连接串、记忆轮数三个参数
from logger import log  # 导入统一日志器，记录连接成功/失败等关键事件

_local: dict[str, list[dict]] = defaultdict(list)  # 进程内降级记忆：键为 _key() 生成串，值为消息列表；缺失键自动初始化为空 list
_redis = None  # Redis 客户端单例：None=未初始化，对象=已连接，False=连接失败过（用于短路避免反复重连）


def _client():
    """获取 Redis 客户端；未启用/连接失败返回 None（调用方走降级路径）。"""  # 函数文档字符串：说明降级返回约定
    global _redis  # 声明使用模块级全局变量 _redis，以便在函数内复用单例并写入失败标记
    if not REDIS_ENABLED:
        return None  # 配置层未启用 Redis，直接返回 None，调用方据此走进程内降级
    if _redis is not None:
        return _redis  # 已有可用客户端或已判定失败（False），复用结果避免每次请求重新建连/探活
    try:
        import redis  # 延迟导入 redis 库：未启用或不可用时不会触发模块加载，降低冷启动依赖

        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)  # 用 URL 形式构造客户端，decode_responses=True 让返回值自动为 str 而非 bytes
        _redis.ping()  # 主动探活：发送 PING 确认连接真正可用，避免懒加载在首条命令才暴露连接问题
        log.info("redis connected: %s", REDIS_URL)  # 记录连接成功日志，便于运维定位记忆后端状态
        return _redis  # 返回已建立并验证过的客户端单例
    except Exception as exc:
        log.warning("redis unavailable, fallback to memory: %s", exc)  # 降级告警：记录异常原因，便于排查 Redis 不可用根因
        _redis = False  # 关键降级标记：置为 False 让后续 _client() 直接走短路分支，避免每请求都重连拖垮系统
        return None  # 返回 None，调用方据此切换到 _local 进程内字典路径


def _key(user_id: int, role_code: str, session_id: int) -> str:
    """记忆 Key：按 用户:角色:会话 三级隔离，实现多用户多角色互不干扰。"""  # 文档字符串：点明三级隔离设计意图，避免不同维度数据串线
    return f"chat:{user_id}:{role_code}:{session_id}"  # 用冒号分层拼接三级 ID 作为 Redis Key，符合 Redis Key 命名惯例且天然按维度隔离


def push_messages(user_id: int, role_code: str, session_id: int, messages: list[dict]) -> None:
    """追加消息到短期记忆：RPush 写入 + LTrim 截断为最近 N*2 条 + 过期 7 天。"""  # 文档字符串：概括"三连"操作（RPush+LTrim+Expire）保证定长滑动窗口
    key = _key(user_id, role_code, session_id)  # 先算出本次消息要写入的隔离 Key
    client = _client()  # 取 Redis 客户端，None 表示需要走进程内降级
    if client:
        pipe = client.pipeline()  # 开启 pipeline 事务式批量打包，把多条命令一次发出，减少网络往返开销
        for msg in messages:
            pipe.rpush(key, json.dumps(msg, ensure_ascii=False))  # RPush 追加到 List 尾部；ensure_ascii=False 保留中文可读性，减小体积
        pipe.ltrim(key, -SHORT_MEMORY_TURNS * 2, -1)  # LTrim 从尾部截断：只保留最近 N*2 条（每轮=用户+助手两条），丢弃更早的历史，O(1) 操作
        pipe.expire(key, 7 * 24 * 3600)  # 设置 7 天 TTL：会话长期不活跃时让 Key 自动清理，控制存储成本与隐私合规
        pipe.execute()  # 一次性提交 pipeline 中所有命令，原子地完成"写入+截断+过期"三连
        return  # Redis 路径完成，直接返回，不再走降级
    # 降级路径：进程内字典，同样只保留最近 N 轮，保证两种后端语义一致
    bucket = _local[key]  # defaultdict 自动初始化空 list，避免 KeyError
    bucket.extend(messages)  # 把新消息追加到本地尾部，对应 Redis 的 RPush
    _local[key] = bucket[-SHORT_MEMORY_TURNS * 2 :]  # 切片截断为最近 N*2 条，对应 Redis 的 LTrim，保持降级与主路径一致


def get_messages(user_id: int, role_code: str, session_id: int) -> list[dict[str, Any]]:
    """读取短期记忆（最近 N 轮对话），供拼装多轮提示词使用。"""  # 文档字符串：说明用途是把历史塞回提示词，是"多轮对话"的载体
    key = _key(user_id, role_code, session_id)  # 用同一套三级隔离规则还原 Key，确保写读一致
    client = _client()  # 取客户端，决定走 Redis 还是降级路径
    if client:
        items = client.lrange(key, 0, -1)  # lrange 0 -1 取整个 List；由于写入侧已 LTrim，长度自然被限制在 N*2 条内
        return [json.loads(x) for x in items]  # 逐条反序列化为 dict，恢复成可拼装进提示词的消息对象
    return list(_local.get(key, []))  # 降级：返回进程内列表的拷贝，避免外部修改污染内部存储


# =====================================================================
# 知识点说明（RAG：记忆模块 / Redis）
# ---------------------------------------------------------------------
# 1. 大模型无记忆：HTTP 调用是无状态的，"多轮对话"必须由应用层把历史
#    塞回提示词。本模块即"短期记忆"层：Redis List 存最近
#    SHORT_MEMORY_TURNS*2 条消息，RPush 追加 + LTrim 截断（O(1)）。
# 2. Redis 五种常用数据类型选型：List 适合"定长滑动窗口对话历史"；
#    String（KV 缓存）、Hash（对象属性）、
#    Set（去重）、ZSet（按分数/时间排序，可做长期记忆索引或排行榜）。
# 3. 过期策略：Expire 7 天自动清理热数据，兼顾存储成本与隐私。
# 4. 长期记忆：跨会话的用户画像/历史摘要可向量化后存入 Milvus
#    （"记忆的 RAG 化"），检索时与知识库一起召回——本项目预留此扩展。
# 5. 降级设计：Redis 不可用时回退进程内字典，开发环境零依赖可跑。
# =====================================================================
