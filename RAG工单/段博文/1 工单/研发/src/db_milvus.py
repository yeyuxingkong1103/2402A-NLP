# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
Milvus 数据层（简化版，单集合）：向量模型/向量存储/原生客户端三套单例 +
入库（带 UUID 主键 + 幂等去重）+ 全量拉文本（给 BM25 建索引用）。

在系统中的位置：
    上游是入库脚本（pdf_parser 解析 PDF → text_splitter 切分 → 调本模块
    ingest_documents）和检索链路（retriever 调向量召回、调 fetch_all_text
    拉文本重建 BM25）；本模块是唯一直接持有 Milvus 连接的地方。
    入库完成后由 retriever.refresh_bm25() 重建关键词索引，本模块只负责向量
    这一路。

架构（单集合）：
    rag_pdf_qa  ← 所有 PDF 文档的 chunk 都落到这一个集合
    检索时不做角色路由，所有问题都查同一个集合。

关键设计取舍：
    1. 模型与连接都做进程内单例（_embedding / _vectorstore / _milvus_client）：
       加载 bge-m3 权重要几秒、建连接也有开销，而 FastAPI 是常驻进程，
       复用即可；
    2. 主键由自己生成 UUID（auto_id=False），再配合 (source, chunk_index)
       做幂等去重，同一份文件重复入库不会产生重复向量；
    3. 同时使用 LangChain 封装（Milvus）和原生客户端（MilvusClient）：
       前者用于带向量的写入，后者用于不带向量的全量 query（重建 BM25 时
       不需要向量，走原生客户端更快也更省事）。
"""

import uuid  # 生成主键
from typing import List, Optional  # 类型标注

from langchain_huggingface import HuggingFaceEmbeddings  # 本地向量模型（直接加载 bge-m3 权重）
from langchain_milvus import Milvus  # LangChain 的 Milvus 封装
from langchain_core.documents import Document  # 统一文档结构
from pymilvus import MilvusClient  # 原生客户端（全量查询用）

from config import (  # 配置
    MILVUS_URI, MILVUS_COLLECTION,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE,
    EMBED_MODEL_PATH, EMBED_DEVICE, EMBED_DIM,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# ==================== 模块级单例（进程内只加载一次） ====================
_embedding: Optional[HuggingFaceEmbeddings] = None  # 向量模型单例
_vectorstore: Optional[Milvus] = None  # 向量存储单例（单集合，直接用全局变量而非字典）
_milvus_client: Optional[MilvusClient] = None  # 原生客户端单例


def get_embedding() -> HuggingFaceEmbeddings:
    """
    获取向量模型单例（首次调用时加载本地 bge-m3，之后复用）

    返回：
        HuggingFaceEmbeddings 实例，已绑定本地权重路径与运行设备。
    说明：
        normalize_embeddings=True 是关键：向量先做 L2 归一化，配合集合的
        COSINE 度量，相似度就等价于余弦值，分数天然落在 [-1,1]，方便直接
        和阈值比较；若改成 False，分数分布会变化，SCORE_THRESHOLD 就需要
        重新标定。
        用 global 声明是因为要「写」模块级变量；模型文件缺失时构造阶段就
        会抛异常，这里不做兜底，属于启动期配置错误，应当早失败。
    """
    global _embedding
    if _embedding is None:  # 未初始化
        _embedding = HuggingFaceEmbeddings(  # 加载本地权重，不联网
            model_name=EMBED_MODEL_PATH,  # 模型目录
            model_kwargs={"device": EMBED_DEVICE},  # 运行设备
            encode_kwargs={"normalize_embeddings": True},  # 归一化，配合 COSINE 度量
        )
        logger.info(f"向量模型已加载：{EMBED_MODEL_PATH}")
    return _embedding


def get_vectorstore() -> Milvus:
    """
    获取向量存储单例（首次调用时建连接 + 配置索引，之后复用）

    返回：
        langchain_milvus.Milvus 实例（已绑定向量模型与索引配置）。
    说明：
        单集合架构下只需一个全局变量，不必用字典；集合不存在时由
        langchain-milvus 自动创建，维度取自 EMBED_DIM——这也是换向量模型
        必须重建集合的原因（维度在建模时已固化）。
    """
    global _vectorstore
    if _vectorstore is None:  # 未初始化
        index_params = {  # 索引参数（langchain-milvus 0.4 要求字典格式）
            "index_type": MILVUS_INDEX_TYPE,  # AUTOINDEX（交给 Milvus 按数据量自动挑索引）
            "metric_type": MILVUS_METRIC_TYPE,  # COSINE（与向量归一化配套）
            "params": {},  # AUTOINDEX 下留空；若改成 HNSW 才需要填 M / efConstruction
        }
        _vectorstore = Milvus(  # 初始化
            embedding_function=get_embedding(),  # 向量模型
            collection_name=MILVUS_COLLECTION,  # 集合名
            connection_args={"uri": MILVUS_URI},  # 连接参数
            index_params=index_params,  # 索引配置
            auto_id=False,  # 主键我们自己给（这样才能用固定 UUID 做幂等去重）
        )
        logger.info(f"Milvus 向量存储已连接：{MILVUS_URI}/{MILVUS_COLLECTION}")
    return _vectorstore


def get_milvus_client() -> MilvusClient:
    """
    获取原生客户端单例（用于全量查询拉文本重建 BM25）

    返回：
        pymilvus.MilvusClient 实例（连接 MILVUS_URI，不做向量运算）。
    说明：
        与 LangChain 封装并存的原因：重建 BM25 只需要 text 字段，用原生
        query 可以直接分页拉取、不必触发 embedding，省时省算力。
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)  # 连接
        logger.info(f"Milvus 原生客户端已连接：{MILVUS_URI}")
    return _milvus_client


# ==================== 入库操作 ====================

def ingest_documents(docs: List[Document], batch_size: int = 10) -> int:
    """
    向量化并写入 Milvus 集合，返回当前语料总数

    metadata 必须包含固定键：
        source      : 来源文件名
        chunk_index : 块序号（同源内唯一，配合 source 做幂等键）
        page_number : 页码（可选，便于检索命中后展示出处）

    参数：
        docs：待入库的 Document 列表（来自 text_splitter 切分结果）。
        batch_size：每批写入条数，默认 10；调大能减少网络往返，但单批太大
                    时向量化会一次占用较多内存。
    返回：
        int，入库完成后集合内的语料总数；无新文档入库（空输入或全部重复）
        时返回 0。
    异常与降级：
        拉取已有键失败只在日志里告警并跳过去重（继续入库）；最终统计语料
        数时若集合不存在则退化为本次入库条数。
    """
    if not docs:  # 空列表
        logger.warning("入库文档列表为空，跳过")
        return 0

    # ---- 幂等去重：用 (source, chunk_index) 唯一键，跳过已存在的记录 ----
    client = get_milvus_client()  # 原生客户端
    existing_keys = set()  # 已存在键集合（放内存里做 O(1) 命中判断）
    if client.has_collection(MILVUS_COLLECTION):  # 集合已存在才需要查
        try:
            offset = 0  # 分页偏移
            page_size = 500  # 每页条数（避免一次拉太多撑爆内存）
            while True:  # 分页拉取已存在的 (source, chunk_index)
                rows = client.query(
                    collection_name=MILVUS_COLLECTION,
                    filter="source != ''",  # 只取有来源的记录
                    output_fields=["source", "chunk_index"],  # 只取幂等键字段
                    limit=page_size,
                    offset=offset,
                )
                if not rows:  # 没数据了
                    break
                for r in rows:  # 逐个收集键
                    existing_keys.add((r.get("source", ""), r.get("chunk_index", -1)))
                if len(rows) < page_size:  # 最后一页
                    break
                offset += page_size  # 翻页
        except Exception as e:  # 查询失败不阻断入库，仅警告
            logger.warning(f"查询已有键失败（{e}），本次跳过去重")

    # 过滤掉已存在的文档
    filtered = []  # 真正要入库的
    skipped = 0  # 已存在被跳过的数量
    for d in docs:  # 逐个文档
        key = (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1))  # 幂等键
        if key in existing_keys:  # 已存在
            skipped += 1
            continue
        existing_keys.add(key)  # 标记本次批次已计划入库（防止批内重复）
        filtered.append(d)  # 加入待入库

    if not filtered:  # 全部重复
        logger.info(f"无新文档需要入库（跳过 {skipped} 条重复）")
        return 0

    # ---- 写入（带进度条） ----
    vs = get_vectorstore()  # 获取向量存储
    try:
        from tqdm import tqdm  # 进度条
    except ImportError:
        tqdm = None  # 未安装则退化为普通循环
    total_batches = (len(filtered) + batch_size - 1) // batch_size  # 总批次数（向上取整）
    bar = tqdm(total=total_batches, desc=f"[{MILVUS_COLLECTION}]", unit="batch") if tqdm else None
    for i in range(0, len(filtered), batch_size):  # 按 batch_size 步长切片
        batch = filtered[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]  # 现场生成 UUID 主键（同文档重复入库也是新主键，交由上面的去重拦截）
        vs.add_documents(documents=batch, ids=batch_ids)
        if bar:
            bar.set_postfix_str(f"{len(batch)}条/批")
            bar.update(1)
        else:
            logger.info(f"  写入批次 {i // batch_size + 1}/{total_batches}（{len(batch)} 条）")
    if bar:
        bar.close()

    total = client.get_collection_stats(MILVUS_COLLECTION).get("row_count", 0) if client.has_collection(MILVUS_COLLECTION) else len(filtered)
    logger.info(f"入库 {len(filtered)} 条（跳过重复 {skipped} 条），当前语料 {total} 条")
    return total


# ==================== 全量拉取（BM25 重建用） ====================

def fetch_all_text() -> List[str]:
    """
    分页拉取集合内全部文本，返回纯文本列表（用于重建 BM25）

    返回：
        List[str]，每个元素是一条 chunk 的原文；集合不存在时返回空列表 []
        （不抛异常，调用方按「知识库未初始化」处理）。
    说明：
        用原生 query 而不走向量检索：这里要的是「全量文本」而不是「相似
        结果」，BM25 索引必须覆盖全部语料；分页 offset 逐页推进，直到某页
        不足 page_size 或返回空，即认为已取完。
    """
    client = get_milvus_client()
    if not client.has_collection(MILVUS_COLLECTION):  # 集合不存在
        return []

    all_text: List[str] = []  # 收集文本
    offset = 0
    page_size = 500  # 每页条数（太小网络往返多，太大单次响应占内存）
    while True:
        rows = client.query(
            collection_name=MILVUS_COLLECTION,
            filter="",  # 空过滤 = 匹配全部
            output_fields=["text"],  # 只取原文（不取向量，省带宽）
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:  # 逐行取文本
            all_text.append(row.get("text", ""))
        if len(rows) < page_size:  # 最后一页
            break
        offset += page_size  # 翻页
    logger.info(f"从 Milvus 拉取 {len(all_text)} 条文本（用于 BM25）")
    return all_text


def get_collection_count() -> int:
    """
    获取集合中的文档数（用于健康检查和前端展示）

    返回：
        int，集合内语料总数；集合不存在时返回 0。
    说明：
        get_collection_stats 返回的是 dict，row_count 字段是字符串型数字，
        这里做一次 int 转换兜底；查询失败时返回 0，不抛异常（健康检查不该
        因为统计失败就 500）。
    """
    client = get_milvus_client()
    if not client.has_collection(MILVUS_COLLECTION):
        return 0
    try:
        stats = client.get_collection_stats(MILVUS_COLLECTION)
        return int(stats.get("row_count", 0))
    except Exception as e:
        logger.warning(f"获取集合统计失败：{e}")
        return 0
