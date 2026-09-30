"""
retrieval_lc.py — 多路召回的 LangChain 包装层

三条召回路都包装成 LangChain 的 BaseRetriever 子类，再用 EnsembleRetriever
按加权 RRF 融合；retrieval.py 因此只剩索引构建和混合检索主流程：

    VectorRetriever  向量语义召回（vector_store）
    BM25Retriever    关键词召回（retrieval.BM25Index）
    CacheRetriever   Redis 缓存召回（同一问题直接复用上次结果）

三路都用 Document 传递，两路的分数写在 metadata 里：vector_score / bm25_score，
取不到的那一路记 None，融合后据此判断候选来自哪条路。
"""

from __future__ import annotations

import json
import threading
from typing import Any

import config
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from vector_store import get_store

# 三路权重：两路检索沿用 config 里的权重，缓存路固定为补充分量，避免缓存主导排序
VECTOR_WEIGHT = config.HYBRID_VECTOR_WEIGHT
BM25_WEIGHT = config.HYBRID_BM25_WEIGHT
CACHE_WEIGHT = 0.2

# 「缓存最近 100 条查询」：每个集合一个 List，超出的从尾部淘汰
CACHE_SIZE = 100
_CACHE_TTL = 3600

_redis_client = None
_redis_probed = False
_redis_lock = threading.Lock()


# ---------------------------------------------------------------- Redis 缓存
def _redis():
    """返回 Redis 客户端；LOCAL_MODE 或连不上时返回 None，缓存路自动失效。"""
    global _redis_client, _redis_probed

    with _redis_lock:
        if _redis_probed:
            return _redis_client
        _redis_probed = True
        if config.LOCAL_MODE:
            return None
        try:
            import redis

            client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            client.ping()
            _redis_client = client
        except Exception:
            _redis_client = None
        return _redis_client


def _cache_key(collection: str) -> str:
    return f"retr_cache:{collection}"


def cache_get(collection: str, query: str) -> list[dict] | None:
    """取同一集合下问题完全相同的缓存结果，没有命中返回 None。"""
    client = _redis()
    if client is None:
        return None
    try:
        for raw in client.lrange(_cache_key(collection), 0, CACHE_SIZE - 1):
            entry = json.loads(raw)
            if entry.get("q") == query:
                return entry.get("docs") or []
    except Exception:
        return None
    return None


def cache_put(collection: str, query: str, docs: list[dict]) -> None:
    """把本轮的最终结果存进缓存，只保留最近 CACHE_SIZE 条查询。"""
    client = _redis()
    if client is None or not docs:
        return
    try:
        key = _cache_key(collection)
        client.lpush(key, json.dumps({"q": query, "docs": docs}, ensure_ascii=False))
        client.ltrim(key, 0, CACHE_SIZE - 1)
        client.expire(key, _CACHE_TTL)
    except Exception:
        return


# ---------------------------------------------------------------- Document
def to_document(doc: dict) -> Document:
    """dict 片段 → Document：正文进 page_content，其余字段进 metadata。"""
    return Document(page_content=doc.get("text") or "", metadata={k: v for k, v in doc.items() if k != "text"})


def from_document(document: Document) -> dict:
    """Document → dict：正文放回 text 字段。"""
    return {**document.metadata, "text": document.page_content}


# ---------------------------------------------------------------- 三条召回
class VectorRetriever(BaseRetriever):
    """向量语义召回，包装 vector_store.search。"""

    collection: str
    filters: dict[str, Any] | None = None
    top_k: int = config.VECTOR_TOP_K

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        import embeddings

        hits = get_store().search(
            self.collection,
            embeddings.encode_query(query),
            top_k=self.top_k,
            filters=self.filters,
        )
        documents: list[Document] = []
        for hit in hits:
            item = {k: v for k, v in hit.items() if k != "embedding"}
            item["vector_score"] = float(item.pop("score", 0.0) or 0.0)
            item["bm25_score"] = None
            documents.append(to_document(item))
        return documents


class BM25Retriever(BaseRetriever):
    """关键词召回，包装 retrieval.BM25Index。"""

    collection: str
    top_k: int = config.BM25_TOP_K

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        # 延迟导入：BM25 索引建在 retrieval 里，模块级导入会形成循环依赖
        import retrieval

        try:
            hits = retrieval.build_bm25_index(self.collection).search(query, top_k=self.top_k)
        except Exception:
            return []
        for hit in hits:
            hit["vector_score"] = None
        return [to_document(hit) for hit in hits]


class CacheRetriever(BaseRetriever):
    """Redis 缓存召回，命中同一问题时直接给出上次的融合结果。"""

    collection: str

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return [to_document(doc) for doc in (cache_get(self.collection, query) or [])]


def build_ensemble(collection: str, filters: dict[str, Any] | None = None) -> EnsembleRetriever:
    """组合三路召回，交给 EnsembleRetriever 做加权 RRF 融合。"""
    return EnsembleRetriever(
        retrievers=[
            VectorRetriever(collection=collection, filters=filters),
            BM25Retriever(collection=collection),
            CacheRetriever(collection=collection),
        ],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT, CACHE_WEIGHT],
    )


if __name__ == "__main__":
    import embeddings
    from vector_store import reset_store

    reset_store()
    store = get_store()
    store.drop("_lc_test")
    for text in ("劳动合同解除时，用人单位应当支付经济补偿。", "今天天气晴朗，适合外出散步。"):
        store.upsert("_lc_test", [{"text": text, "embedding": embeddings.encode_query(text)}])

    docs = build_ensemble("_lc_test").invoke("劳动合同解除的赔偿")
    print(f"融合后端类型：{type(build_ensemble('_lc_test')).__name__}，召回 {len(docs)} 条")
    for doc in docs:
        print(f"  vector={doc.metadata.get('vector_score')} | {doc.page_content[:20]}")

    assert docs and "劳动" in docs[0].page_content, "融合结果不合理"
    store.drop("_lc_test")
    print("retrieval_lc 自检通过。")
