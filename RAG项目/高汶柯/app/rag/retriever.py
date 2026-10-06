"""混合检索 + 多路召回 + 相似度过滤 + 去重。

召回路径：
- Milvus dense 向量检索
- Milvus 原生 BM25（2.5+，若启用）
- 本地 BM25（rank_bm25）兜底
- Redis 缓存 / MongoDB 长期记忆 / Neo4j 图谱 / MySQL 文档元数据
多路结果经 RRF 融合后去重。
"""
from __future__ import annotations

import time

import jieba

from app.config import settings
from app.core.registry import get_embedder
from app.db.milvus_store import milvus_store
from app.db.mongo_store import mongo_store
from app.db.mysql_store import list_kb_docs
from app.db.neo4j_store import neo4j_store
from app.db.redis_store import redis_store
from app.logging_conf import log
from app.rag.query_rewrite import expand_queries

_CORPUS_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 120


def clear_cache() -> None:
    _CORPUS_CACHE.clear()


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def _corpus(domain: str | None) -> list[dict]:
    key = domain or "*"
    cached = _CORPUS_CACHE.get(key)
    now = time.time()
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    try:
        docs = milvus_store.all_entities(domain)
    except Exception as exc:  # noqa: BLE001
        log.warning("读取语料失败: %s", exc)
        docs = []
    _CORPUS_CACHE[key] = (now, docs)
    return docs


def _bm25_local(query: str, domain: str | None, top_k: int) -> list[dict]:
    docs = _corpus(domain)
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:  # noqa: BLE001
        log.warning("rank_bm25 不可用: %s", exc)
        return []
    bm25 = BM25Okapi([_tokenize(d.get("text", "")) for d in docs])
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    out: list[dict] = []
    for i in order:
        if scores[i] > 0:
            d = dict(docs[i])
            d["score"] = float(scores[i])
            out.append(d)
    return out


def _rrf(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion 融合多路召回结果。"""
    agg: dict[str, dict] = {}
    for lst in rank_lists:
        for rank, d in enumerate(lst):
            key = (d.get("text") or "").strip()
            if not key:
                continue
            if key not in agg:
                agg[key] = dict(d)
                agg[key]["rrf"] = 0.0
            agg[key]["rrf"] += 1.0 / (k + rank + 1)
    return sorted(agg.values(), key=lambda x: -x["rrf"])



def _dedup(docs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        key = (d.get("text") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def retrieve(query: str, domain: str | None = None, top_k: int | None = None,
             use_rag: bool = True) -> dict:
    top_k = top_k or settings.retrieve_top_k
    if not use_rag:
        return {"docs": [], "queries": [query], "route": []}

    queries = expand_queries(query)
    embedder = get_embedder()
    rank_lists: list[list[dict]] = []
    route: list[str] = []

    # 路1：多查询 dense 向量检索（带相似度阈值过滤）
    for q in queries:
        try:
            vector = embedder.encode_one(q)
        except Exception as exc:  # noqa: BLE001
            log.error("向量化失败: %s", exc)
            break
        dense = milvus_store.search_dense(vector, top_k, domain)
        dense = [d for d in dense if d.get("score", 1.0) >= settings.score_threshold]
        rank_lists.append(dense)
    route.append("milvus_dense")

    # 路2：Milvus 原生 BM25
    if milvus_store.native_bm25 and queries:
        try:
            vec0 = embedder.encode_one(queries[0])
            rank_lists.append(milvus_store.search_hybrid(queries[0], vec0, top_k, domain))
            route.append("milvus_native_bm25")
        except Exception as exc:  # noqa: BLE001
            log.warning("原生混合检索失败: %s", exc)

    # 路3：本地 BM25
    try:
        rank_lists.append(_bm25_local(queries[0], domain, top_k))
        route.append("bm25_local")
    except Exception as exc:  # noqa: BLE001
        log.warning("本地 BM25 失败: %s", exc)

    fused = _rrf(rank_lists)

    # 路4~7：其他数据源
    extras: list[dict] = []
    cached = redis_store.cache_get(query)
    if cached:
        extras.append({"text": cached, "source": "redis_cache", "domain": domain or "general",
                       "chunk_type": "cache", "score": 0.0})
        route.append("redis_cache")
    for mem in mongo_store.search_memory(query, 3):
        if mem.get("summary"):
            extras.append({"text": mem["summary"], "source": "mongo_memory",
                           "domain": domain or "general", "chunk_type": "memory", "score": 0.0})
    graph = neo4j_store.search(query, domain, 3)
    if graph:
        extras.extend(graph)
        route.append("neo4j")
    if domain:
        for doc in list_kb_docs(domain)[:2]:
            extras.append({"text": f"知识库文档：《{doc['filename']}》（域：{domain}）",
                           "source": "mysql_kb", "domain": domain, "chunk_type": "meta", "score": 0.0})

    docs = _dedup(fused + extras)
    return {"docs": docs[: top_k * 2], "queries": queries, "route": route}
