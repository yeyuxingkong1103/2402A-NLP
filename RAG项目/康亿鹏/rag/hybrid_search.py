"""稀疏检索与多路融合：BM25（jieba 中文分词）召回 + RRF 融合。

与 Milvus 稠密向量召回互补：BM25 擅长精确关键词/专名匹配，稠密向量擅长语义相近。
BM25 索引在进程内按集合懒加载：首次查询时从 Milvus 拉取该集合全部片段构建，
之后每次查询先比对 count(*)，入库/重置导致片段数变化时自动重建。
"""  # 模块说明
import threading  # 线程锁，保护索引懒加载
from typing import Any, Dict, List, Sequence  # 类型注解

import jieba  # 中文分词
from langchain_core.documents import Document  # LangChain 文档对象
from rank_bm25 import BM25Okapi  # BM25 经典算法实现

import config  # 全局配置

text_field = "text"  # langchain-milvus 默认的文本字段名（项目未自定义）
batch_size = 500  # query_iterator 每批拉取行数
vector_field_types = {100, 101, 102, 103, 104}  # Milvus 各类型向量字段（标量字段之外全部跳过）

_client: Any = None  # 进程内共享的 MilvusClient
_lock = threading.Lock()  # 索引构建锁，避免并发请求重复建同一个索引
_index_cache: Dict[str, Dict[str, Any]] = {}  # collection_name -> {"count", "docs", "bm25"}


def get_milvus_client():  # 懒加载 MilvusClient（与 vector_store 复用同一 Milvus 服务）
    global _client
    if _client is None:
        from pymilvus import MilvusClient  # 懒导入，避免无关命令也依赖 pymilvus

        _client = MilvusClient(uri=config.milvus_uri)
    return _client


def tokenize(text: str) -> List[str]:  # 中文分词：BM25 建索引和查询共用同一套切词口径
    """jieba 分词并小写化（兼容英文），过滤纯空白 token。"""
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


def count_entities(collection_name: str) -> int:  # 轻量统计集合片段总数
    """用 count(*) 查询片段总数；集合不存在（如 reset 后）返回 0。"""
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return 0
    # rows = [{"count(*)": 1234}]
    rows = client.query(collection_name, filter="", output_fields=["count(*)"])
    return int(rows[0]["count(*)"])


def load_corpus(collection_name: str) -> List[Document]:  # 拉取集合全部片段到内存
    """从 Milvus 拉取集合内全部片段（正文 + 全部标量元数据），用于构建 BM25 语料。"""
    client = get_milvus_client()
    description = client.describe_collection(collection_name)  # 拿集合 schema 确定可查字段
    # 只查标量字段（跳过各类向量字段），主键 pk 用于和稠密结果按 pk 对齐去重
    # scalar_fields每个元素是字段名
    scalar_fields = [
        field["name"]
        for field in description["fields"]
        if field["type"] not in vector_field_types
    ]
    iterator = client.query_iterator(  # 游标分页，避免单次 query 的条数上限
        collection_name=collection_name,
        filter="",# 筛选行
        output_fields=scalar_fields,# 筛选列
        batch_size=batch_size,
    )
    docs: List[Document] = []
    try:
        while True:
            batch = iterator.next() # 从游标里取下一批数据
            if not batch:  # 游标耗尽
                break
            for row in batch:# row是每一行数据
                # 取出正文作为 page_content，从row取出text，同时从row删除text
                content = row.pop(text_field, "")
                docs.append(Document(page_content=content, metadata=row))
    finally:
        iterator.close()
    return docs


def get_bm25_index(collection_name: str) -> Dict[str, Any]:  # 获取/重建某集合的 BM25 索引
    """按片段总数做缓存校验：数量没变直接复用，变化了（重新入库/reset）则重建。"""
    count = count_entities(collection_name)
    cached = _index_cache.get(collection_name)
    if cached is not None and cached["count"] == count:
        return cached
    with _lock:  # 加锁后双重检查，避免并发请求重复构建
        cached = _index_cache.get(collection_name)
        if cached is not None and cached["count"] == count:
            return cached
        docs = load_corpus(collection_name) if count else []  # 空集合不拉数据
        bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs]) if docs else None
        entry = {"count": count, "docs": docs, "bm25": bm25}
        _index_cache[collection_name] = entry
        return entry


def bm25_search(collection_name: str, query: str, top_k: int) -> List[Document]:  # BM25 稀疏召回
    """在指定集合上做 BM25 关键词召回，返回分数最高的 top_k 条（无命中返回空列表）。"""
    if top_k <= 0:
        return []
    index = get_bm25_index(collection_name)
    bm25 = index["bm25"]
    if bm25 is None:  # 集合为空或不存在
        return []
    scores = bm25.get_scores(tokenize(query))  # 对语料中每个片段打分
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    results: List[Document] = []
    for idx, score in ranked[:top_k]:
        if score <= 0:  # 已按分数降序排列，遇 0 说明后续均无关键词命中
            break
        source = index["docs"][idx]
        results.append(  # 复制一份文档并写入 BM25 得分，避免污染缓存中的语料对象
            Document(
                page_content=source.page_content,
                metadata={**source.metadata, "bm25_score": float(score)},
            )
        )
    return results


def doc_key(doc: Document) -> str:  # 文档去重标识：优先 Milvus 主键，兜底正文哈希
    pk = doc.metadata.get("pk")
    return f"pk:{pk}" if pk is not None else f"hash:{hash(doc.page_content)}"


def rrf_fuse(  # 多路召回结果 RRF 融合
    ranked_lists: Sequence[Sequence[Document]],  # 每路结果需按相关性降序排列
    k: int = 60,  # RRF 常数：越大，不同名次之间的分差越平滑
    top_k: int = None,  # 融合后保留条数；None = 保留全部
) -> List[Document]:
    """RRF（Reciprocal Rank Fusion）：fused_score = Σ 1/(k + rank + 1)。

    不依赖各路分数的量纲（余弦相似度与 BM25 分不可直接比较），只看排名，简单稳健。
    """
    fused_scores: Dict[str, float] = {}
    fused_docs: Dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):  # rank 从 0 开始
            key = doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in fused_docs:  # 同一文档多路命中时保留首个对象，累加排名分
                fused_docs[key] = doc
    ordered = sorted(
        fused_docs.values(),
        key=lambda doc: fused_scores[doc_key(doc)],
        reverse=True,
    )
    for doc in ordered:  # 把融合分写入元数据，便于排查检索效果
        doc.metadata = {**doc.metadata, "rrf_score": fused_scores[doc_key(doc)]}
    return ordered[:top_k] if top_k else ordered
