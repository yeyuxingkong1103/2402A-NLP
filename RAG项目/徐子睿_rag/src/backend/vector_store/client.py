# -*- coding: utf-8 -*-
"""vector_store/client.py —— Milvus 客户端与集合生命周期管理。

在链路中的位置：
    backend/pipeline.py（写入） / backend/retrieval.py（读取）
        → vector_store 包 → 【本文件】 → Milvus

本文件负责"连得上、建得对、装得进内存"三件事：
    连接复用（进程级单例）、集合与索引的幂等创建、load 状态管理、健康探测。

包内分工：
    filters.py  过滤表达式构造与记录整形（纯字符串/字典加工，不碰网络）
    ops.py      记录级查/搜/写/删（依赖本文件的 ensure_collection）
    __init__.py 把三者的公开名再导出，使本包对外的用法与拆分前完全一致
"""
from __future__ import annotations

import os
import threading
from typing import Any

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")  # Milvus 服务地址
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")                    # 需要鉴权的 Milvus 部署才填，本地留空
VECTOR_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1024"))     # 必须与 bge-m3 的输出维度一致，
                                                                 # 改成别的 embedding 模型时这里要同步改
COLLECTION = os.getenv("RAG_COLLECTION", "rag_docs_v6")          # 知识库集合名。v6 是版本号：
                                                                 # 切分策略或字段结构大改时换新集合，
                                                                 # 避免与旧数据结构混在一起

# 进程级单例。用 _client_lock 保护，多线程并发请求时只会建一个连接
_client: Any = None
_loaded_collections: set[str] = set()  # 已 load 的集合。load 是重操作，同一个集合只做一次
_client_lock = threading.RLock()       # 用 RLock 而非 Lock：ensure_collection 内层会再取一次锁
def milvus_client() -> Any:
    """创建并复用 Milvus 客户端。

    返回：
        进程级共享的 MilvusClient。
    说明：
        延迟 import pymilvus：没装 pymilvus 时也能 import 本模块（例如只跑单元测试）。
        token 为空时不传该参数，避免本地无鉴权部署被一个空 token 干扰。
    """
    global _client
    with _client_lock:
        if _client is None:
            from pymilvus import MilvusClient

            kwargs = {"uri": MILVUS_URI}
            if MILVUS_TOKEN:
                kwargs["token"] = MILVUS_TOKEN
            _client = MilvusClient(**kwargs)
        return _client


def close_milvus_client() -> None:
    """关闭客户端并清空"已加载"记录（进程退出或重建连接时用）。

    说明：
        用 getattr 探测 close 是否存在，兼容不同版本 pymilvus 的接口差异。
        同时清空 _loaded_collections，否则新连接会以为集合已经 load 过而跳过加载。
    """
    global _client
    with _client_lock:
        if _client is not None:
            close = getattr(_client, "close", None)
            if close:
                close()
            _client = None
            _loaded_collections.clear()


def ensure_collection(collection_name: str = COLLECTION) -> Any:
    """确保集合、向量索引和加载状态都已就绪。

    参数：
        collection_name: 集合名，默认知识库集合
    返回：
        可用的 Milvus 客户端。

    做三件事（都做幂等处理，可反复调用）：
        1. 集合不存在就建，并当场建好向量索引
        2. 索引建好后 load 进内存，否则检索接口会报 "collection not loaded"
        3. 用 _loaded_collections 记住已加载的集合，不重复 load

    表结构：
        id      VARCHAR(256) 主键 —— 由 pipeline 用 uuid5 生成，保证同名文档重复构建时幂等
        vector  FLOAT_VECTOR(1024) —— 语义检索用
        其余字段走 enable_dynamic_field（动态字段），所以 pipeline 里写的
        payload（source/page/section/text…）不需要预先在 schema 里声明就能存取。
        这也是为什么要开 dynamic field：加一个 metadata 字段不必迁移集合。

    索引参数：
        AUTOINDEX —— 让 Milvus 按数据规模自选索引类型，本地部署省心，不必手工调 HNSW 参数
        COSINE    —— 余弦相似度。文本向量更该看方向而不是长度，所以不用 L2 距离
        consistency_level="Strong" —— 强一致。上传后立刻检索必须能搜到，
        用默认的 Bounded 可能出现"刚入库却搜不到"的困惑
    """
    client = milvus_client()
    with _client_lock:
        if not client.has_collection(collection_name=collection_name):
            from pymilvus import DataType

            # auto_id=False：主键由我们提供（uuid5），不用 Milvus 自增，这样才能实现幂等覆盖
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
        if collection_name not in _loaded_collections:
            client.load_collection(collection_name=collection_name)
            _loaded_collections.add(collection_name)
    return client


def milvus_health() -> tuple[bool, str]:
    """探测 Milvus 是否可用。

    返回：
        (是否连通, 说明文本)。连通时说明为 "connected"，
        否则是 "异常类型: 异常信息" —— 直接把原因带回接口，排障时不用翻日志。
    """
    try:
        ensure_collection(COLLECTION)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "connected"


def collection_count(collection_name: str = COLLECTION) -> int:
    """取集合中的记录数。

    参数：
        collection_name: 集合名
    返回：
        记录条数（即知识库里的 chunk 总数）。

    用途：
        上传前/检索前判断"知识库是不是空的"，好给出"请先上传 PDF"的友好提示，
        而不是走一轮完整检索再返回空结果。
    """
    client = ensure_collection(collection_name)
    stats = client.get_collection_stats(collection_name=collection_name)
    return int(stats.get("row_count", 0))
