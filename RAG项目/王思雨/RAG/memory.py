# -*- coding: utf-8 -*-
"""记忆模块：Redis 短期记忆（最近几轮对话）+ Milvus 长期记忆（跨会话问答沉淀）。"""

import json                                   # 导入 json，用于序列化每条记忆
import time                                   # 导入 time，用于生成时间戳
from pathlib import Path                      # 导入 Path，用于路径处理

from pymilvus import DataType, MilvusClient   # 导入 Milvus 客户端与字段类型枚举
import config                                 # 导入配置模块，记忆集合与条数从这里读
import db                                     # 导入数据库模块，复用 Redis 连接封装
import vector_store                           # 导入向量库模块，复用 Milvus 客户端与向量化
from logger import get_logger                 # 导入日志工具，用于记录记忆读写

logger = get_logger("memory")                 # 创建本模块的 logger 实例

SHORT_KEY_PREFIX = "mem:short:"               # 短期记忆的 Redis 键前缀
SHORT_TTL = 60 * 60 * 24                      # 短期记忆过期时间：1 天，单位秒
MAX_LEN_QUESTION = 2048                       # question 字段最大字符数
MAX_LEN_ANSWER = 8192                         # answer 字段最大字符数
MAX_LEN_SOURCE = 128                          # source 字段最大字符数
SEARCH_FETCH_MULTIPLIER = 4                   # 检索时多取几倍候选，过滤用户后再截断


# ===================== 一、短期记忆（Redis） =====================

def _short_key(user_id: int) -> str:          # 内部函数：拼短期记忆键名
    """拼出某个用户的短期记忆键名，用户编号作为前缀实现多用户隔离。"""
    return f"{SHORT_KEY_PREFIX}{user_id}"     # 形如 mem:short:1


def push_short_memory(user_id: int, role: str, content: str) -> int:
    """把一轮对话追加进短期记忆，只保留最近 N 条；Redis 连不上时抛 RuntimeError。"""
    try:                                      # Redis 不可用时统一转成 RuntimeError
        client = db.get_redis_client()        # 复用 db 模块的 Redis 客户端
        client.ping()                         # 主动探测，提前暴露连接问题
    except Exception as exc:                  # 连不上
        raise RuntimeError(f"Redis 连接失败：{exc}") from exc   # 抛出便于上层捕获
    key = _short_key(user_id)                 # 该用户的记忆键
    payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)   # 序列化成 JSON
    client.lpush(key, payload)                # 从左端插入，最新的在最前面
    limit = config.LLM_HISTORY_LIMIT          # 保留条数从配置读
    client.ltrim(key, 0, limit - 1)           # 只保留最近 limit 条，旧的丢弃
    client.expire(key, SHORT_TTL)             # 设置 1 天过期，避免长期占用内存
    count = client.llen(key)                  # 当前条数
    logger.info("短期记忆已更新：用户 %s，当前 %d 条", user_id, count)   # 记录日志
    return count                              # 返回当前条数


def get_short_memory(user_id: int) -> list:
    """读取短期记忆并按时间正序返回（旧到新），Redis 连不上时返回空列表。"""
    try:                                      # 读操作降级为返回空列表
        client = db.get_redis_client()        # 获取 Redis 客户端
        raw_items = client.lrange(_short_key(user_id), 0, -1)   # 取出全部（最新的在最前）
    except Exception as exc:                  # 连不上
        logger.warning("短期记忆读取降级为空列表（Redis 不可用）：%s", exc)   # 记录降级原因
        return []                             # 返回空列表
    history = []                              # 保存解析后的记忆
    for item in reversed(raw_items):          # lpush 存的是倒序，反转成正序
        try:                                  # 单条解析失败不影响整体
            record = json.loads(item)         # 反序列化 JSON
        except Exception:                     # 解析失败
            continue                          # 跳过这条
        if isinstance(record, dict) and record.get("content"):   # 结构正确且有内容
            history.append({"role": record.get("role", "user"),
                            "content": record["content"]})   # 收进结果
    logger.info("短期记忆读取：用户 %s，%d 条", user_id, len(history))   # 记录日志
    return history                            # 返回正序历史


def clear_short_memory(user_id: int) -> int:
    """清空某个用户的短期记忆，返回删除的键数量；Redis 连不上时抛 RuntimeError。"""
    try:                                      # 写操作不可静默失败
        client = db.get_redis_client()        # 获取 Redis 客户端
        removed = client.delete(_short_key(user_id))   # 删除该用户的记忆键
    except Exception as exc:                  # 连不上
        raise RuntimeError(f"Redis 连接失败：{exc}") from exc   # 抛出异常
    logger.info("短期记忆已清空：用户 %s", user_id)   # 记录日志
    return removed                            # 返回删除数量


# ===================== 二、长期记忆（Milvus） =====================

def get_memory_client() -> MilvusClient:      # 获取 Milvus 客户端
    """获取 Milvus 客户端，地址来自配置，用户编号字段用于隔离不同用户。"""
    if not config.MILVUS_HOST:                # 主机没配置
        raise RuntimeError("MILVUS_HOST 未配置，请复制 .env.example 为 .env 并填写")   # 报错
    uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"   # 拼出服务地址
    return MilvusClient(uri=uri, timeout=config.MILVUS_TIMEOUT)   # 创建并返回客户端


def build_memory_schema():                    # 构建长期记忆字段定义
    """构建长期记忆 Collection 的字段定义：id、vector、question、answer、user_id、source、created_at。"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)   # 主键自增
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)   # 主键
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)   # 稠密向量
    schema.add_field(field_name="question", datatype=DataType.VARCHAR,
                     max_length=MAX_LEN_QUESTION)   # 问题原文
    schema.add_field(field_name="answer", datatype=DataType.VARCHAR,
                     max_length=MAX_LEN_ANSWER)     # 答案原文
    schema.add_field(field_name="user_id", datatype=DataType.INT64)   # 所属用户，用于隔离
    schema.add_field(field_name="source", datatype=DataType.VARCHAR,
                     max_length=MAX_LEN_SOURCE)     # 来源，存会话 ID 或 manual
    schema.add_field(field_name="created_at", datatype=DataType.INT64)   # 创建时间戳
    return schema                             # 返回字段定义


def create_memory_collection(recreate: bool = False) -> None:
    """创建长期记忆集合；recreate=True 时先删除再重建，集合不存在时自动创建。"""
    client = get_memory_client()              # 获取 Milvus 客户端
    name = config.MEMORY_COLLECTION           # 集合名从配置读
    if client.has_collection(name):           # 集合已存在
        if not recreate:                      # 且不要求重建
            logger.info("记忆集合 %s 已存在，跳过创建", name)   # 记录日志
            return                            # 直接返回
        client.drop_collection(name)          # 要求重建则先删除
    index_params = client.prepare_index_params()   # 创建索引参数对象
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")   # 向量索引
    client.create_collection(collection_name=name, schema=build_memory_schema(),
                             index_params=index_params)   # 创建集合并挂索引
    logger.info("记忆集合 %s 创建完成", name)   # 记录日志


def save_long_memory(user_id: int, question: str, answer: str, source: str = "") -> None:
    """把一轮问答写入长期记忆；Milvus 不可用时抛 RuntimeError。"""
    if not question or not answer:            # 问答缺一不可
        logger.warning("长期记忆写入跳过：问题或答案为空")   # 记录告警
        return                                # 直接返回
    try:                                      # Milvus 不可用时统一转成 RuntimeError
        client = get_memory_client()          # 获取客户端
        name = config.MEMORY_COLLECTION       # 集合名
        if not client.has_collection(name):   # 集合还没建
            create_memory_collection()        # 自动创建
        dense_vec, _ = vector_store.embed_query(question)   # 用问题文本生成稠密向量
        client.insert(collection_name=name, data=[{        # 插入一行
            "vector": dense_vec,              # 问题向量
            "question": question[:MAX_LEN_QUESTION],   # 问题原文，超长截断
            "answer": answer[:MAX_LEN_ANSWER],         # 答案原文，超长截断
            "user_id": int(user_id),          # 所属用户，检索时用于过滤
            "source": (source or "")[:MAX_LEN_SOURCE],   # 来源标记
            "created_at": int(time.time()),   # 创建时间戳
        }])                                   # 插入结束
        logger.info("长期记忆已写入：用户 %s，来源 %s", user_id, source or "未标注")   # 记录日志
    except Exception as exc:                  # 写入失败
        raise RuntimeError(f"长期记忆写入失败：{exc}") from exc   # 抛出异常


def search_long_memory(user_id: int, query: str, top_k: int = None) -> list:
    """按用户过滤后做向量检索，只返回该用户自己的记忆；不可用时返回空列表。"""
    limit = top_k or config.MEMORY_TOP_K      # 未指定条数时从配置读
    try:                                      # 检索降级为返回空列表
        client = get_memory_client()          # 获取客户端
        name = config.MEMORY_COLLECTION       # 集合名
        if not client.has_collection(name):   # 集合不存在说明还没有任何记忆
            logger.info("长期记忆检索：集合 %s 不存在，返回空列表", name)   # 记录说明
            return []                         # 返回空列表
        client.load_collection(name)          # 加载集合到内存
        client.refresh_load(name)             # 刷新已加载快照，否则刚写入的记忆搜不到
        dense_vec, _ = vector_store.embed_query(query)   # 查询文本向量化
        fetch = limit * SEARCH_FETCH_MULTIPLIER   # 多取候选，过滤后仍有足够结果
        outcome = client.search(              # 执行向量检索
            collection_name=name,             # 集合名
            data=[dense_vec],                 # 查询向量
            anns_field="vector",              # 检索向量字段
            search_params={"metric_type": "COSINE", "params": {}},   # 余弦相似度
            filter=f"user_id == {int(user_id)}",   # 只查该用户自己的记忆，防止串用户
            limit=fetch,                      # 取候选条数
            output_fields=["question", "answer", "user_id"],   # 需要取回的字段
        )                                     # 检索结束
        hits = outcome[0] if outcome else []  # 取出第一条查询的结果
    except Exception as exc:                  # 连不上或检索失败
        logger.warning("长期记忆检索降级为空列表：%s", exc)   # 记录降级原因
        return []                             # 返回空列表
    results = []                              # 保存整理后的记忆
    for hit in hits:                          # 逐条整理
        entity = hit.get("entity", {})        # 取出实体字段
        if int(entity.get("user_id", -1)) != int(user_id):   # 双保险：再核对一次归属
            continue                          # 不是本人的记忆就跳过
        results.append({                      # 组装结果
            "question": entity.get("question", ""),   # 历史问题
            "answer": entity.get("answer", ""),       # 历史答案
            "score": round(float(hit.get("distance", 0.0)), 4),   # 相似度分数
        })                                    # 组装结束
        if len(results) >= limit:             # 取够条数就停
            break                             # 跳出循环
    logger.info("长期记忆检索：用户 %s，返回 %d 条", user_id, len(results))   # 记录日志
    return results                            # 返回结果
