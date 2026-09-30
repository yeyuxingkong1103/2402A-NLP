# -*- coding: utf-8 -*-
"""
Milvus 数据层：多集合知识库入库 + 混合检索 + 长期记忆

架构：
    rag_legal         ← 法律顾问知识库
    rag_psychology    ← 心理专家知识库
    rag_companion     ← 虚拟朋友知识库
    rag_long_term_memory ← 长期记忆（共用）

每个角色检索自己对应的集合，互不干扰
"""

import uuid  # 生成主键
from datetime import datetime  # 时间戳
from typing import List, Optional, Dict  # 类型标注

from langchain_ollama import OllamaEmbeddings  # Ollama 向量模型封装
from langchain_milvus import Milvus  # LangChain 的 Milvus 封装
from langchain_core.documents import Document  # 统一文档结构
from pymilvus import MilvusClient  # 原生客户端（全量查询用）

from config import (  # 配置
    MILVUS_URI, MILVUS_COLLECTION, MILVUS_COLLECTIONS,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, MILVUS_MEMORY_COLLECTION,
    OLLAMA_BASE_URL, EMBED_MODEL, EMBED_DIM,
    get_collection_for_role,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# ==================== 模型单例（进程内只加载一次） ====================
_embedding: Optional[OllamaEmbeddings] = None  # 向量模型单例
_vectorstores: Dict[str, Milvus] = {}  # 集合名 -> 向量存储（多集合）
_milvus_client: Optional[MilvusClient] = None  # 原生客户端单例


def get_embedding() -> OllamaEmbeddings:
    """获取向量模型单例（首次调用时初始化，之后复用）"""
    global _embedding
    if _embedding is None:  # 未初始化
        _embedding = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)  # 初始化
        logger.info(f"向量模型已加载：{EMBED_MODEL}")  # 日志
    return _embedding


def get_vectorstore(collection_name: str = None) -> Milvus:
    """
    获取向量存储单例（按集合名缓存）
    首次调用时建连接 + 配置索引，之后复用
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    if collection_name not in _vectorstores:  # 该集合未初始化
        index_params = {  # 索引参数（langchain-milvus 0.4 要求字典格式）
            "index_type": MILVUS_INDEX_TYPE,  # AUTOINDEX
            "metric_type": MILVUS_METRIC_TYPE,  # COSINE
            "params": {},  # AUTOINDEX 下留空
        }
        _vectorstores[collection_name] = Milvus(  # 初始化
            embedding_function=get_embedding(),  # 向量模型
            collection_name=collection_name,  # 集合名
            connection_args={"uri": MILVUS_URI},  # 连接参数
            index_params=index_params,  # 索引配置
            auto_id=False,  # 主键我们自己给
        )
        logger.info(f"Milvus 向量存储已连接：{MILVUS_URI}/{collection_name}")  # 日志
    return _vectorstores[collection_name]


def get_milvus_client() -> MilvusClient:
    """获取原生客户端单例（用于全量查询拉文本重建 BM25）"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)  # 连接
        logger.info(f"Milvus 原生客户端已连接：{MILVUS_URI}")  # 日志
    return _milvus_client


# ==================== 入库操作 ====================

def _collect_existing_keys(collection_name: str) -> set:
    """
    拉取集合内全部 (source, chunk_index) 已存在键，用于入库幂等去重。

    为什么用 query_iterator 而不是 offset 分页：
        Milvus 规定单次 query 的 offset + limit ≤ 16384。集合超过 1.6 万条后，
        翻页到 16000 时 16000+500=16500 就会抛 code=1100，
        导致已有键集合为空、重复数据被全部写入。
        query_iterator 由 Milvus 内部游标翻页，不受这个窗口限制。
    """
    client = get_milvus_client()  # 原生客户端
    keys = set()  # 已存在键集合
    iterator = client.query_iterator(  # 游标迭代器（不受 offset 上限约束）
        collection_name=collection_name,
        filter="source != ''",  # 只取有来源的记录
        output_fields=["source", "chunk_index"],  # 只取幂等键字段
        batch_size=2000,  # 每批条数
    )
    try:
        while True:
            rows = iterator.next()  # 取下一批
            if not rows:  # 取完了
                break
            for r in rows:  # 逐个收集键
                keys.add((r.get("source", ""), r.get("chunk_index", -1)))
    finally:
        iterator.close()  # 释放游标
    return keys


def ingest_documents(docs: List[Document], batch_size: int = 10,
                     collection_name: str = None) -> int:
    """
    向量化并写入指定 Milvus 集合，返回当前语料总数

    metadata 必须包含固定键：
        source      : 来源文件名
        section     : 章节路径（Markdown 有，txt 为空）
        chunk_index : 块序号
        summary     : 块摘要（可选，用于展示）
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    if not docs:  # 空列表
        logger.warning("入库文档列表为空，跳过")  # 警告日志
        return 0

    # ---- 幂等去重：用 (source, chunk_index) 唯一键，跳过已存在的记录 ----
    client_m = get_milvus_client()  # 原生客户端
    existing_keys = set()  # 已存在键集合
    if client_m.has_collection(collection_name):  # 集合已存在才需要查
        try:
            existing_keys = _collect_existing_keys(collection_name)  # 游标拉全量键
        except Exception as e:
            # 去重是数据安全的底线：拿不到已有键就宁可中止，也不能放行入库。
            # 之前这里是 warning 后继续，去重静默失效，直接把重复数据写进了集合。
            raise RuntimeError(
                f"无法读取集合 {collection_name} 的已有键，为避免写入重复数据已中止入库：{e}"
            ) from e

    # 过滤掉已存在的文档
    filtered = []  # 真正要入库的
    skipped = 0  # 已存在被跳过的数量
    for d in docs:  # 逐个文档
        key = (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1))  # 幂等键
        if key in existing_keys:  # 已存在
            skipped += 1  # 跳过计数
            continue
        existing_keys.add(key)  # 标记本次批次已计划入库（防止批内重复）
        filtered.append(d)  # 加入待入库

    if not filtered:  # 全部重复
        logger.info(f"集合 {collection_name} 无新文档需要入库（跳过 {skipped} 条重复）")  # 日志
        return 0

    # ---- 写入（带进度条） ----
    vs = get_vectorstore(collection_name)  # 获取该集合的向量存储
    try:
        from tqdm import tqdm  # 进度条
    except ImportError:
        tqdm = None  # 未安装则退化为普通循环
    total_batches = (len(filtered) + batch_size - 1) // batch_size  # 总批次数
    bar = tqdm(total=total_batches, desc=f"[{collection_name}]", unit="batch") if tqdm else None
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        vs.add_documents(documents=batch, ids=batch_ids)
        if bar:
            bar.set_postfix_str(f"{len(batch)}条/批")  # 尾部信息
            bar.update(1)  # 前进一格
        else:
            logger.info(f"  写入批次 {i // batch_size + 1}/{(len(filtered) - 1) // batch_size + 1}（{len(batch)} 条）")
    if bar:
        bar.close()  # 关闭进度条

    client = get_milvus_client()  # 原生客户端
    total = client.get_collection_stats(collection_name).get("row_count", 0) if client.has_collection(collection_name) else len(filtered)
    logger.info(f"入库 {len(filtered)} 条（跳过重复 {skipped} 条）到集合 {collection_name}，当前语料 {total} 条")
    return total


def ingest_jsonl(file_path: str, collection_name: str = None) -> int:
    """
    读取一个已处理好的 JSONL 文件并入库到指定 Milvus 集合。

    JSONL 每行格式（与预处理输出对齐）：
        {"text": "...", "metadata": {"source": "...", "section": "...",
                                     "chunk_index": 0, "summary": "...", ...}}
    也兼容直接形如 {"page_content": "...", "metadata": {...}} 的记录。

    返回：入库条数。文件读取失败或为空时返回 0。
    """
    import json  # JSON 解析
    from langchain_core.documents import Document  # 统一文档结构

    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    docs = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # 逐行读
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    rec = json.loads(line)  # 解析一行
                except json.JSONDecodeError:
                    logger.warning(f"  跳过损坏行：{line[:80]}...")  # 坏行日志
                    continue

                # 兼容两种字段命名：text/page_content
                if "text" in rec:
                    content, meta = rec["text"], rec.get("metadata") or {}
                elif "page_content" in rec:
                    content, meta = rec["page_content"], rec.get("metadata") or {}
                else:
                    continue  # 缺少内容字段，跳过

                if not content or not str(content).strip():
                    continue  # 空内容跳过
                # 补齐 metadata 固定键
                meta.setdefault("source", "jsonl")
                meta.setdefault("section", "")
                meta.setdefault("chunk_index", len(docs))
                docs.append(Document(page_content=str(content), metadata=meta))
    except FileNotFoundError:
        logger.error(f"文件不存在：{file_path}")  # 找不到文件
        return 0

    if not docs:
        logger.warning(f"JSONL 无有效记录：{file_path}")  # 空文件
        return 0

    logger.info(f"读取 {file_path}：{len(docs)} 条有效记录，入库集合 {collection_name}")  # 日志
    return ingest_documents(docs, collection_name=collection_name)


# ==================== 全量拉取（BM25 重建用） ====================

def fetch_all_text(collection_name: str = None) -> List[Document]:
    """分页拉取指定集合内全部文本，返回 Document 列表（用于重建 BM25）"""
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()  # 原生客户端
    if not client.has_collection(collection_name):  # 集合不存在
        return []  # 返回空

    all_docs: List[Document] = []  # 收集文档
    # 用游标迭代器而不是 offset 分页：Milvus 单次 query 的 offset+limit ≤ 16384，
    # 大集合（如 6.6 万条的心理库）翻页必然报错，导致 BM25 索引重建失败、关键词检索整路失效
    iterator = client.query_iterator(  # 游标迭代器
        collection_name=collection_name,
        filter="",  # 空过滤 = 匹配全部
        # 除主键和原文外，还要带上来源类字段：这些 Document 会直接喂给 BM25Retriever，
        # 缺了 source 会导致 BM25 召回的片段在引用卡片里显示为空来源
        output_fields=["pk", "text", "source", "section", "chunk_index"],
        batch_size=2000,  # 每批条数
    )
    try:
        while True:
            rows = iterator.next()  # 取下一批
            if not rows:  # 取完了
                break
            for row in rows:  # 逐行转 Document
                all_docs.append(Document(
                    page_content=row["text"],
                    metadata={  # 完整保留 metadata，BM25 召回时才能显示来源
                        "pk": row["pk"],
                        "source": row.get("source", ""),
                        "section": row.get("section", ""),
                        "chunk_index": row.get("chunk_index", -1),
                    },
                ))
    finally:
        iterator.close()  # 释放游标

    logger.info(f"从 Milvus 集合 {collection_name} 拉取 {len(all_docs)} 条文档")  # 日志
    return all_docs


# ==================== 长期记忆存储 ====================

def save_long_term_memory(user_id: int, role_id: int, content: str, summary: str = ""):
    """
    将重要对话存入 Milvus 作为长期记忆（与知识库分离，用独立集合）

    长期记忆 vs 短期记忆：
        短期（Redis）：最近 20 条对话，滑动窗口，2 小时过期
        长期（Milvus）：重要事实永久存储，用向量检索找回
    """
    embedding = get_embedding()  # 向量模型
    vec = embedding.embed_query(content)  # 向量化
    client = get_milvus_client()  # 原生客户端

    # 使用配置的集合名（修复之前硬编码的 bug）
    memory_collection = MILVUS_MEMORY_COLLECTION

    if not client.has_collection(memory_collection):
        from pymilvus import CollectionSchema, FieldSchema, DataType  # schema 构造器
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="role_id", dtype=DataType.INT64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        client.create_collection(memory_collection, schema=schema)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_index(memory_collection, index_params=index_params)
        logger.info(f"长期记忆集合已创建：{memory_collection}")

    client.insert(memory_collection, {
        "pk": str(uuid.uuid4()), "text": content, "vector": vec,
        "user_id": user_id, "role_id": role_id, "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
    })
    logger.info(f"长期记忆已保存：user={user_id} role={role_id}")


def search_long_term_memory(user_id: int, query: str, top_k: int = 3) -> List[Document]:
    """检索某用户的长期记忆（按角色过滤）"""
    embedding = get_embedding()
    vec = embedding.embed_query(query)
    client = get_milvus_client()

    memory_collection = MILVUS_MEMORY_COLLECTION  # 使用配置的集合名

    if not client.has_collection(memory_collection):
        return []
    results = client.search(
        collection_name=memory_collection,
        data=[vec], anns_field="vector",
        filter=f"user_id == {user_id}",
        limit=top_k, output_fields=["text", "summary", "created_at"],
    )
    if not results or not results[0]:
        return []
    return [Document(page_content=r["entity"]["text"], metadata={"summary": r["entity"].get("summary", "")})
            for r in results[0]]
