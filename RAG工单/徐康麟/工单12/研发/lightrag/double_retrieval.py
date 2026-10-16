# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""双层检索：局部（向量）+ 全局（图谱扩展）+ RRF 融合（查询响应流程第 ②③ 步）。

两条后端路径，**唯一变量是检索通道**（生成与引用逻辑完全共用，见 :mod:`lightrag.qa`）：

  ① ``backend="rag"``（基线，扁平 RAG）
     ``向量检索(问题)`` + ``BM25(问题)`` → RRF（权重 ``rag_vector_weight`` / ``rag_bm25_weight``）

  ② ``backend="lightrag"``（双层）
     局部通道：``向量检索(问题 + 局部关键词)`` → ``local_top_k``
     全局通道：图谱实体命中 → 沿关系扩展 ``graph_hops`` 跳 → 收集关联块 → ``global_top_k``
     融合：RRF（权重 ``local_weight`` / ``global_weight``）

复用：
  * :mod:`lightrag.vector_index` 的 ``VectorIndex``（FAISS/NumPy）与 ``BM25Index``（jieba+BM25）
  * :mod:`lightrag.routing` 的分流结果
  * 图谱产物 ``nodes.jsonl`` / ``edges.jsonl``（含 ``chunk_ids`` / ``source_chunks``）
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Cfg
from .routing import Route, norm_ws

# 文档过滤：两份招股书去水印重复件后只保留这两个 doc_key（与建图口径一致）
DEFAULT_DOC_FILTER = ("招股说明书1无水", "招股说明书2pd")


@dataclass
class Hit:
    """一条检索命中（带页码与来源，供引用与逐题复核）。"""

    chunk_id: str
    score: float
    source: str                     # rag_vector | rag_bm25 | lightrag_local | lightrag_global
    rank: int = 0
    page: int = 0
    page_end: int = 0
    doc_key: str = ""
    type: str = ""
    char_count: int = 0
    content_head: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "score": round(float(self.score), 6), "source": self.source,
            "rank": int(self.rank), "page": int(self.page), "doc_key": self.doc_key,
            "type": self.type, "char_count": int(self.char_count),
            "content_head": self.content_head,
        }


@dataclass
class RetrievalResult:
    backend: str
    hits: list[Hit] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "hits": [h.to_dict() for h in self.hits], "trace": self.trace}


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def rrf_fuse(rankings: Sequence[Sequence[tuple[str, float]]], weights: Sequence[float], *,
             k: int = 60) -> dict[str, float]:
    """加权 RRF：``score(d) = Σ_c w_c / (k + rank_c(d))``。"""
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (cid, _raw) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + float(weight) / (float(k) + rank)
    return fused


def _over_fetch(search_fn: Any, query: Any, top_k: int, allowed: set[str] | None) -> list[tuple[str, float]]:
    """超取 + 文档过滤：先取 ``top_k * 6`` 个候选，再按 ``allowed``（doc_key 过滤）截回 top_k。

    过滤放在检索侧而不是索引侧，是为了**不重建索引**（索引 3013 块含带水印重复件）。
    """
    want = int(top_k) if not allowed else max(int(top_k) * 6, 60)
    raw = list(search_fn(query, want))
    if not allowed:
        return raw[: int(top_k)]
    kept = [item for item in raw if item[0] in allowed]
    return kept[: int(top_k)]


def _allowed_ids(chunks_by_id: dict[str, dict[str, Any]], doc_filter: Sequence[str] | None) -> set[str] | None:
    if not doc_filter:
        return None
    keep = set(str(x) for x in doc_filter)
    return {cid for cid, meta in chunks_by_id.items() if str(meta.get("doc_key")) in keep}


def _to_hits(ranking: Sequence[tuple[str, float]], *, source: str,
             chunks_by_id: dict[str, dict[str, Any]], limit: int) -> list[Hit]:
    out: list[Hit] = []
    for rank, (cid, score) in enumerate(ranking[:limit], start=1):
        meta = chunks_by_id.get(cid) or {}
        out.append(Hit(
            chunk_id=cid, score=float(score), source=source, rank=rank,
            page=int(meta.get("page") or 0), page_end=int(meta.get("page_end") or meta.get("page") or 0),
            doc_key=str(meta.get("doc_key") or ""), type=str(meta.get("type") or ""),
            char_count=int(meta.get("char_count") or 0),
            content_head=norm_ws(str(meta.get("content") or ""))[:160],
        ))
    return out


# ---------------------------------------------------------------------------
# 全局通道：图谱扩展
# ---------------------------------------------------------------------------
def expand_graph(route: Route, *, nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                 cfg: Cfg, edge_mode: str = "all") -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """从命中实体出发沿关系扩展 ``graph_hops`` 跳，收集关联块。

    块来源：① 命中/邻居节点自带的 ``chunk_ids``（该实体出现过的块）；
            ② 关系边的 ``source_chunks``（该关系被抽出的块）。
    边的贡献按 ``weight`` 加权，``constraint_ok=false`` 的边再乘
    ``retrieval.global_flagged_edge_weight``（默认 0.5）——**保留但不与合规边同权**。

    ``edge_mode``（供「两组口径」对比，见 t2 报告）：
      * ``all``           = 用全量 167 条边（含 121 条 ``constraint_ok=false``）
      * ``constraint_ok`` = 只用 46 条头尾类型相容的边
    """
    r = cfg.retrieval
    max_hops = max(int(r.get("graph_hops") or 1), 1)
    max_hops = min(max_hops, int(r.get("graph_max_hops") or 3))
    max_entities = int(r.get("graph_expand_top_entities") or 8)
    max_relations = int(r.get("graph_max_relations") or 60)
    flagged_w = float(r.get("global_flagged_edge_weight") or 0.5)
    strict = str(edge_mode) == "constraint_ok"

    edges_used = [e for e in edges if (bool(e.get("constraint_ok")) or not strict)]
    node_by_name = {norm_ws(str(n.get("name"))): n for n in nodes}
    adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in edges_used:
        adj[norm_ws(str(e.get("head")))].append(e)
        adj[norm_ws(str(e.get("tail")))].append(e)

    # 种子实体：先按分流命中的实体，再退回「全局关键词是某节点名的子串」
    seeds: list[str] = [norm_ws(x) for x in route.matched_entities if norm_ws(x) in node_by_name]
    if not seeds:
        for kw in route.global_keywords:
            k = norm_ws(str(kw))
            for name in node_by_name:
                if k and (k in name or name in k):
                    seeds.append(name)
    seeds = list(dict.fromkeys(seeds))[:max_entities]

    chunk_score: Counter[str] = Counter()
    used_edges = 0
    visited: set[str] = set()
    frontier = list(seeds)
    for _hop in range(max_hops):
        nxt: list[str] = []
        for name in frontier:
            if name in visited:
                continue
            visited.add(name)
            node = node_by_name.get(name)
            if not node:
                continue
            for cid in (node.get("chunk_ids") or []):
                chunk_score[str(cid)] += 1.0
            for e in adj.get(name, [])[:max_relations]:
                used_edges += 1
                w = float(e.get("weight") or 1)
                if not e.get("constraint_ok"):
                    w *= flagged_w
                for cid in (e.get("source_chunks") or []):
                    chunk_score[str(cid)] += w
                other = norm_ws(str(e.get("tail"))) if norm_ws(str(e.get("head"))) == name else norm_ws(str(e.get("head")))
                if other and other not in visited:
                    nxt.append(other)
        frontier = list(dict.fromkeys(nxt))
        if not frontier:
            break

    ranking = sorted(chunk_score.items(), key=lambda kv: (-kv[1], kv[0]))
    trace = {
        "seeds": seeds[:max_entities],
        "seed_count": len(seeds),
        "hops": max_hops,
        "edges_touched": used_edges,
        "candidate_chunks": len(ranking),
        "flagged_edge_weight": flagged_w,
    }
    return ranking, trace


# ---------------------------------------------------------------------------
# 两条后端路径
# ---------------------------------------------------------------------------
def retrieve_rag(question: str, *, cfg: Cfg, vindex: Any, bm25: Any, embed_query: Any,
                 chunks_by_id: dict[str, dict[str, Any]], doc_filter: Sequence[str] | None = None) -> RetrievalResult:
    """RAG 基线：向量 + BM25 → RRF（权重 0.5/0.5）。"""
    r = cfg.retrieval
    allowed = _allowed_ids(chunks_by_id, doc_filter)
    top_k = int(r.top_k)
    vec_rank = _over_fetch(vindex.search, embed_query(question), int(r.local_top_k), allowed)
    bm_rank = _over_fetch(bm25.search, question, int(r.bm25_top_k), allowed)

    fused = rrf_fuse([vec_rank, bm_rank],
                     [float(r.rag_vector_weight), float(r.rag_bm25_weight)],
                     k=int(r.rrf_k))
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    hits = _to_hits(order, source="rag_fused", chunks_by_id=chunks_by_id, limit=top_k)
    # 标注每条命中的主通道（便于复核"这条块从哪来"）
    primary = {cid: src for src, ranking in (("rag_vector", vec_rank), ("rag_bm25", bm_rank))
               for cid, _ in ranking}
    for h in hits:
        h.source = primary.get(h.chunk_id, "rag_fused")
    trace = {
        "vector_top": [c for c, _ in vec_rank][:10],
        "bm25_top": [c for c, _ in bm_rank][:10],
        "fused_top": [c for c, _ in order],
        "weights": {"vector": float(r.rag_vector_weight), "bm25": float(r.rag_bm25_weight)},
        "rrf_k": int(r.rrf_k),
        "doc_filter": list(doc_filter or []),
    }
    return RetrievalResult(backend="rag", hits=hits, trace=trace)


def retrieve_lightrag(question: str, route: Route, *, cfg: Cfg, vindex: Any, bm25: Any, embed_query: Any,
                      chunks_by_id: dict[str, dict[str, Any]], nodes: Sequence[dict[str, Any]],
                      edges: Sequence[dict[str, Any]], edge_mode: str = "all",
                      doc_filter: Sequence[str] | None = None) -> RetrievalResult:
    """LightRAG 双层：局部（向量）+ 全局（图谱） → 加权 RRF。

    ``edge_mode`` 见 :func:`expand_graph`（``all`` / ``constraint_ok``）。
    """
    r = cfg.retrieval
    allowed = _allowed_ids(chunks_by_id, doc_filter)
    top_k = int(r.top_k)

    # 局部通道：问题 + 局部关键词一起嵌入（关键词提供了"具体实体"的锚点）
    local_query = question
    if route.local_keywords:
        local_query = question + " " + " ".join(route.local_keywords[:8])
    local_rank: list[tuple[str, float]] = []
    if route.route in ("local", "both"):
        local_rank = _over_fetch(vindex.search, embed_query(local_query), int(r.local_top_k), allowed)

    # 全局通道：图谱扩展
    global_rank: list[tuple[str, float]] = []
    graph_trace: dict[str, Any] = {}
    if route.route in ("global", "both"):
        global_rank, graph_trace = expand_graph(route, nodes=nodes, edges=edges, cfg=cfg, edge_mode=edge_mode)
        if allowed is not None:
            global_rank = [item for item in global_rank if item[0] in allowed]

    rankings: list[Sequence[tuple[str, float]]] = []
    weights: list[float] = []
    if local_rank:
        rankings.append(local_rank)
        weights.append(float(r.local_weight))
    if global_rank:
        rankings.append(global_rank)
        weights.append(float(r.global_weight))
    if not rankings:                        # 两路皆空：退回问题向量检索，避免返回空结果
        fallback = _over_fetch(vindex.search, embed_query(question), top_k, allowed)
        rankings = [fallback]
        weights = [1.0]
        graph_trace["fallback"] = "两路皆空，退回向量检索"

    fused = rrf_fuse(rankings, weights, k=int(r.rrf_k))
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    hits = _to_hits(order, source="lightrag_fused", chunks_by_id=chunks_by_id, limit=top_k)
    primary = {cid: src for src, ranking in (("lightrag_local", local_rank), ("lightrag_global", global_rank))
               for cid, _ in ranking}
    for h in hits:
        h.source = primary.get(h.chunk_id, "lightrag_fused")

    trace = {
        "route": route.route,
        "edge_mode": edge_mode,
        "routing": route.to_dict(),
        "local_query_head": norm_ws(local_query)[:160],
        "local_top": [c for c, _ in local_rank][:10],
        "global_top": [c for c, _ in global_rank][:10],
        "fused_top": [c for c, _ in order],
        "weights": {"local": float(r.local_weight), "global": float(r.global_weight)},
        "graph": graph_trace,
        "rrf_k": int(r.rrf_k),
        "doc_filter": list(doc_filter or []),
    }
    return RetrievalResult(backend="lightrag", hits=hits, trace=trace)
