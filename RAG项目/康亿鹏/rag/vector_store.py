"""Milvus 向量库：连接、建集合（不存在时自动创建）、写入与检索。"""  # 模块说明
from functools import lru_cache  # 缓存装饰器，用于全局单例
from langchain_community.vectorstores import Milvus
from langchain_milvus import Milvus  # LangChain 对 Milvus 的封装

import config  # 全局配置
from embeddings import get_bge_m3_embeddings  # BGE-M3 Embedding 工厂

INDEX_PARAMS = {  # 向量索引参数（建集合时使用）
    "index_type": "HNSW",  # HNSW 图索引：高召回、低延迟
    "metric_type": "COSINE",  # 余弦相似度度量
    "params": {"M": 16, "efConstruction": 200},  # 建图参数：每节点连接数、构建时搜索宽度
}

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}  # 检索参数：搜索宽度 ef（越大越准越慢）


@lru_cache(maxsize=1)
def get_embeddings():  # 工厂函数：全局共享 Embedding 模型
    """全局共享 Embedding 模型（避免重复加载）。"""
    return get_bge_m3_embeddings()  # 首次调用时才加载模型


@lru_cache(maxsize=8)  # 按 domain 缓存多个集合实例（medical/education）
def get_vector_store(domain: str) -> Milvus:  # 工厂函数：按领域获取对应集合的向量库实例（domain 必传）
    """全局共享 Milvus 向量库实例；domain 必须为 domain_collections 的合法键。"""
    # 兜底：langchain-milvus 的 Milvus 构造时会创建 AsyncMilvusClient（基于 grpc.aio），
    # grpc.aio 要求当前线程有 event loop。FastAPI 把同步 retrieve 丢到线程池时线程无 loop，
    # 会报 RuntimeError: There is no current event loop in thread。
    # 这里确保当前线程有 loop 再构造；构造后 lru_cache 缓存，后续 similarity_search 走
    # 同步 pymilvus ORM，不再需要 loop。
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    collection_name = config.domain_collections[domain]  # 领域 -> 集合名（domain 必传，无默认回退）
    connection_args = {"uri": config.milvus_uri}  # Milvus 连接地址
    return Milvus(  # 构造 LangChain Milvus 客户端
        embedding_function=get_embeddings(),  # 注入 Embedding（写入/查询时自动向量化）
        collection_name=collection_name,  # 集合名称（按领域切换）
        connection_args=connection_args,  # 连接参数
        index_params=INDEX_PARAMS,  # 集合不存在时按此参数自动创建索引
        search_params=SEARCH_PARAMS,  # 检索时的搜索参数
        auto_id=True,  # 主键由 Milvus 自动生成
        drop_old=False,  # 不删除已有集合（重复启动时幂等复用）
    )
