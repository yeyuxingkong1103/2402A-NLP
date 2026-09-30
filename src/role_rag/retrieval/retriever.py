"""三路混合检索：稠密（Milvus）⊕ 稀疏（BGE-M3 lexical，Milvus）⊕ 关键词（BM25，内存）。

三路召回后在应用侧做加权 RRF 融合，并支持：

* ``mode``：dense / sparse / bm25 / hybrid，便于做消融对比；
* ``fusion``：app（应用侧加权 RRF，可含 BM25）/ milvus（Milvus 原生 hybrid_search）；
* 同文档限流（``per_doc_limit``）避免单个文档霸榜；
* Redis 结果缓存（命中即返回，并把 cache_hit / cache_miss 写入统计）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Config, get_config
from ..errors import ValidationError
from ..logging_conf import get_logger
from ..models.embedder import get_embedder
from ..store.milvus_store import MilvusStore, get_milvus
from ..store.redis_store import RedisStore, get_redis
from .bm25 import BM25Cache, get_bm25_cache
from .fusion import weighted_rrf

logger = get_logger(__name__)

VALID_MODES = {"dense", "sparse", "bm25", "hybrid"}
VALID_FUSION = {"app", "milvus"}


@dataclass(slots=True)
class RetrievedChunk:
    """一条召回结果（含各路得分与排名，便于前端调试面板展示）。"""

    chunk_id: str
    text: str
    score: float
    pk: int = 0
    doc_id: str = ""
    doc_title: str = ""
    section: str = ""
    source: str = ""
    scope: str = ""
    chunk_index: int = 0
    routes: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self, with_text: bool = True, text_limit: int = 600) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "source": self.source,
            "scope": self.scope,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 6),
            "routes": self.routes,
        }
        if with_text:
            payload["text"] = self.text if len(self.text) <= text_limit else self.text[:text_limit] + "…"
        return payload


@dataclass(slots=True)
class RetrievalResult:
    query: str
    role_id: str
    mode: str
    fusion: str
    results: list[RetrievedChunk]
    candidates: dict[str, int]
    timings: dict[str, float]
    cached: bool = False
    sparse_terms: list[tuple[str, float]] = field(default_factory=list)
    expr: str = ""

    def to_dict(self, with_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "role_id": self.role_id,
            "mode": self.mode,
            "fusion": self.fusion,
            "cached": self.cached,
            "candidates": self.candidates,
            "timings": {key: round(value, 4) for key, value in self.timings.items()},
            "sparse_terms": self.sparse_terms,
            "expr": self.expr,
            "results": [item.to_dict(with_text=with_text) for item in self.results],
        }


class HybridRetriever:
    """角色感知的混合检索器。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.embedder = get_embedder(self.config)
        self.milvus: MilvusStore = get_milvus(self.config)
        self.redis: RedisStore = get_redis(self.config)
        self.bm25: BM25Cache = get_bm25_cache(self.config)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 参数
    def _params(self) -> dict[str, Any]:
        section = self.config.retrieval
        return {
            "top_k_dense": int(section.get("top_k_dense", 20)),
            "top_k_sparse": int(section.get("top_k_sparse", 20)),
            "top_k_bm25": int(section.get("top_k_bm25", 20)),
            "final_k": int(section.get("final_k", 6)),
            "rrf_k": int(section.get("rrf_k", 60)),
            "weights": {key: float(value) for key, value in (section.get("weights") or {}).items()},
            "per_doc_limit": int(section.get("per_doc_limit", 3)),
            "cache_enabled": bool(section.get("cache_enabled", True)),
        }

    def scope_expr(self, role_id: str) -> str:
        return self.milvus.scope_expr(role_id, include_shared=bool(self.config.get("ingest.include_shared", True)))

    # ------------------------------------------------------------------ 主入口
    def search(
        self,
        query: str,
        role_id: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        fusion: str = "app",
        use_cache: bool = True,
        debug: bool = False,
    ) -> RetrievalResult:
        query = (query or "").strip()
        if not query:
            raise ValidationError("检索 query 不能为空")
        if mode not in VALID_MODES:
            raise ValidationError(f"不支持的检索模式：{mode}（可选 {sorted(VALID_MODES)}）")
        if fusion not in VALID_FUSION:
            raise ValidationError(f"不支持的融合方式：{fusion}（可选 {sorted(VALID_FUSION)}）")

        params = self._params()
        final_k = int(top_k or params["final_k"])
        expr = self.scope_expr(role_id)
        timings: dict[str, float] = {}
        started = time.perf_counter()

        cache_name = RedisStore.retrieval_cache_key(role_id, mode, final_k, query, scope=expr)
        if use_cache and params["cache_enabled"]:
            cached = self.redis.cache_get(cache_name)
            if cached is not None:
                self.redis.mark_cache(True)
                timings["total"] = time.perf_counter() - started
                return RetrievalResult(
                    query=query,
                    role_id=role_id,
                    mode=mode,
                    fusion=fusion,
                    results=[self._from_dict(item) for item in cached.get("results", [])],
                    candidates=cached.get("candidates", {}),
                    timings=timings,
                    cached=True,
                    sparse_terms=[tuple(item) for item in cached.get("sparse_terms", [])],
                    expr=expr,
                )
            self.redis.mark_cache(False)

        mark = time.perf_counter()
        dense_vector, sparse_query = self.embedder.encode_query(query)
        timings["embed"] = time.perf_counter() - mark

        route_payloads: dict[str, dict[str, dict[str, Any]]] = {}
        route_rankings: dict[str, list[str]] = {}
        route_scores: dict[str, dict[str, float]] = {}

        if mode in {"dense", "hybrid"} and fusion == "app":
            mark = time.perf_counter()
            hits = self.milvus.search_dense(dense_vector.tolist(), expr, params["top_k_dense"])
            timings["dense"] = time.perf_counter() - mark
            self._collect("dense", hits, route_payloads, route_rankings, route_scores)

        if mode in {"sparse", "hybrid"} and fusion == "app":
            mark = time.perf_counter()
            hits = self.milvus.search_sparse(sparse_query, expr, params["top_k_sparse"])
            timings["sparse"] = time.perf_counter() - mark
            self._collect("sparse", hits, route_payloads, route_rankings, route_scores)

        if mode in {"bm25", "hybrid"}:
            mark = time.perf_counter()
            index = self.bm25.get(role_id)
            bm25_hits = index.search(query, params["top_k_bm25"])
            timings["bm25"] = time.perf_counter() - mark
            hits = [{"score": hit.score, **hit.payload} for hit in bm25_hits]
            self._collect("bm25", hits, route_payloads, route_rankings, route_scores)

        if fusion == "milvus" and mode == "hybrid":
            mark = time.perf_counter()
            hits = self.milvus.hybrid_search(
                dense_vector.tolist(), sparse_query, expr, params["top_k_dense"], rrf_k=params["rrf_k"]
            )
            timings["milvus_hybrid"] = time.perf_counter() - mark
            self._collect("milvus", hits, route_payloads, route_rankings, route_scores)

        weights = dict(params["weights"])
        if mode == "dense":
            weights = {"dense": 1.0}
        elif mode == "sparse":
            weights = {"sparse": 1.0}
        elif mode == "bm25":
            weights = {"bm25": 1.0}
        elif fusion == "milvus":
            weights = {"milvus": 1.0}

        mark = time.perf_counter()
        fused = weighted_rrf(route_rankings, weights=weights, k=params["rrf_k"], scores=route_scores)
        timings["fuse"] = time.perf_counter() - mark

        results = self._materialise(fused, route_payloads, final_k, params["per_doc_limit"])
        timings["total"] = time.perf_counter() - started
        sparse_terms = self.embedder.sparse_to_terms(sparse_query, 10) if debug else []

        payload = {
            "candidates": {route: len(keys) for route, keys in route_rankings.items()},
            "results": [
                {
                    "chunk_id": item.chunk_id, "text": item.text, "pk": item.pk, "doc_id": item.doc_id,
                    "doc_title": item.doc_title, "section": item.section, "source": item.source,
                    "scope": item.scope, "chunk_index": item.chunk_index, "score": item.score,
                    "routes": item.routes,
                }
                for item in results
            ],
            "sparse_terms": sparse_terms,
        }
        if use_cache and params["cache_enabled"]:
            self.redis.cache_set(cache_name, payload, ttl=self.redis.cache_ttl)

        return RetrievalResult(
            query=query,
            role_id=role_id,
            mode=mode,
            fusion=fusion,
            results=results,
            candidates=payload["candidates"],
            timings=timings,
            cached=False,
            sparse_terms=sparse_terms,
            expr=expr,
        )

    # ------------------------------------------------------------------ 内部
    def _collect(
        self,
        route: str,
        hits: Sequence[dict[str, Any]],
        payloads: dict[str, dict[str, dict[str, Any]]],
        rankings: dict[str, list[str]],
        scores: dict[str, dict[str, float]],
    ) -> None:
        order: list[str] = []
        score_map: dict[str, float] = {}
        bucket = payloads.setdefault(route, {})
        for hit in hits:
            chunk_id = self._chunk_id(hit)
            if chunk_id in score_map:
                continue
            bucket[chunk_id] = hit
            order.append(chunk_id)
            score_map[chunk_id] = float(hit.get("score", 0.0))
        rankings[route] = order
        scores[route] = score_map

    @staticmethod
    def _chunk_id(hit: dict[str, Any]) -> str:
        explicit = hit.get("chunk_id")
        if explicit:
            return str(explicit)
        return f"{hit.get('doc_id', '')}#{hit.get('chunk_index', 0)}"

    def _materialise(
        self,
        fused: Sequence[Any],
        payloads: dict[str, dict[str, dict[str, Any]]],
        final_k: int,
        per_doc_limit: int,
    ) -> list[RetrievedChunk]:
        merged: dict[str, dict[str, Any]] = {}
        for bucket in payloads.values():
            merged.update(bucket)

        results: list[RetrievedChunk] = []
        doc_counter: dict[str, int] = {}
        for item in fused:
            payload = merged.get(item.key)
            if payload is None:
                continue
            doc_id = str(payload.get("doc_id", ""))
            if per_doc_limit > 0 and doc_counter.get(doc_id, 0) >= per_doc_limit:
                continue
            doc_counter[doc_id] = doc_counter.get(doc_id, 0) + 1
            results.append(
                RetrievedChunk(
                    chunk_id=item.key,
                    text=str(payload.get("text", "")),
                    score=float(item.score),
                    pk=int(payload.get("pk", 0) or 0),
                    doc_id=doc_id,
                    doc_title=str(payload.get("doc_title", "")),
                    section=str(payload.get("section", "")),
                    source=str(payload.get("source", "")),
                    scope=str(payload.get("scope", "")),
                    chunk_index=int(payload.get("chunk_index", 0) or 0),
                    routes={route: dict(values) for route, values in item.routes.items()},
                )
            )
            if len(results) >= final_k:
                break
        return results

    @staticmethod
    def _from_dict(item: dict[str, Any]) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(item.get("chunk_id", "")),
            text=str(item.get("text", "")),
            score=float(item.get("score", 0.0)),
            pk=int(item.get("pk", 0) or 0),
            doc_id=str(item.get("doc_id", "")),
            doc_title=str(item.get("doc_title", "")),
            section=str(item.get("section", "")),
            source=str(item.get("source", "")),
            scope=str(item.get("scope", "")),
            chunk_index=int(item.get("chunk_index", 0) or 0),
            routes=item.get("routes", {}) or {},
        )

    # ------------------------------------------------------------------ 调试
    def compare(self, query: str, role_id: str, top_k: int = 5) -> dict[str, Any]:
        """并列比较 dense / sparse / bm25 / hybrid 四种模式（检索调试面板）。"""

        report: dict[str, Any] = {"query": query, "role_id": role_id, "modes": {}}
        for mode in ("dense", "sparse", "bm25", "hybrid"):
            result = self.search(query, role_id, top_k=top_k, mode=mode, use_cache=False, debug=True)
            report["modes"][mode] = {
                "candidates": result.candidates,
                "timings": {key: round(value, 4) for key, value in result.timings.items()},
                "results": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_title": item.doc_title,
                        "section": item.section,
                        "score": round(item.score, 6),
                        "routes": item.routes,
                        "preview": item.text[:120],
                    }
                    for item in result.results
                ],
            }
        native = self.search(query, role_id, top_k=top_k, mode="hybrid", fusion="milvus", use_cache=False)
        report["modes"]["hybrid_milvus"] = {
            "candidates": native.candidates,
            "timings": {key: round(value, 4) for key, value in native.timings.items()},
            "results": [
                {"chunk_id": item.chunk_id, "doc_title": item.doc_title,
                 "score": round(item.score, 6), "preview": item.text[:120]}
                for item in native.results
            ],
        }
        report["sparse_terms"] = self.embedder.sparse_to_terms(
            self.embedder.encode_query(query)[1], 12
        )
        return report


_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever(config: Config | None = None) -> HybridRetriever:
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = HybridRetriever(config)
        return _retriever
