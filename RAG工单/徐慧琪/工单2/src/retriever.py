# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：混合检索（向量 + BM25，RRF 融合）

为什么不是纯向量（沿用并强化工单1的结论）：
  招股书问题多为精确事实（注册资本、法定代表人、某年收入、奖项名）。
  稠密向量擅长语义，但对"7,360.00 万元"这类数字/专名的精确匹配不如关键词；
  BM25 又答不了换种说法的问法。两路召回用 RRF（名次融合，k=60）取长补短，
  无需对余弦分与 BM25 分做量纲归一化。

工单02 增强：
  - BM25 语料直接 scroll Qdrant payload（单一数据源），随建库自动更新；
  - 命中的是**子块**（300~450 字），父块文本随 payload 一并带出，
    由 context 模块做"小块检索 + 大块上下文"的展开与压缩。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
import threading
import time
from typing import Sequence

from src import config, embedder, vector_store

# 中英混排分词：数字(含小数/千分位) | 英文单词 | 连续汉字
_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z\-]*|[一-鿿]+")
# 作为整体出现的领域专名（jieba 可能切错）
_PROTECTED_TERMS = (
    "注册资本", "法定代表人", "主营业务收入", "军用领域", "国家科技进步一等奖",
    "募集资金", "补充流动资金", "重要供应商", "实际控制人", "发行股数",
    "招股意向书", "电子信息行业", "技术标准", "视频指挥系统",
)

_bm25 = None
_bm25_payloads: list[dict] = []
_bm25_lock = threading.Lock()
_bm25_built_at: float = 0.0


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
def _split_by_protected(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(text):
        hit = None
        for term in _PROTECTED_TERMS:
            if text.startswith(term, i):
                hit = term
                break
        if hit:
            if buf:
                out.append(buf)
                buf = ""
            out.append(hit)
            i += len(hit)
        else:
            buf += text[i]
            i += 1
    if buf:
        out.append(buf)
    return out


def tokenize(text: str) -> list[str]:
    """中英混合分词：汉字走 jieba（保护词优先），英文/数字整体保留并小写。"""
    import jieba

    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        if tok[0].isascii():
            tokens.append(tok.lower())
            continue
        for piece in _split_by_protected(tok):
            if piece in _PROTECTED_TERMS:
                tokens.append(piece)
            else:
                tokens.extend(t for t in jieba.cut(piece) if t.strip())
    return tokens


# ---------------------------------------------------------------------------
# BM25 索引（懒加载 + 缓存）
# ---------------------------------------------------------------------------
def _build_bm25(force: bool = False, name: str | None = None) -> None:
    global _bm25, _bm25_payloads, _bm25_built_at
    if _bm25 is not None and not force:
        return
    with _bm25_lock:
        if _bm25 is not None and not force:
            return

        from rank_bm25 import BM25Okapi

        _bm25 = None
        _bm25_payloads = []

        t0 = time.time()
        try:
            payloads = vector_store.scroll_all(name=name)
            if not payloads:
                raise RuntimeError("向量库为空，无法构建 BM25 索引。请先执行：python build_index.py")
            corpus = [tokenize(p.get("text", "")) for p in payloads]
            # 发布顺序：先 payloads 后索引（读者以 _bm25 非空为就绪标志）
            _bm25_payloads = payloads
            _bm25 = BM25Okapi(corpus)
            _bm25_built_at = time.time()
        except Exception:
            _bm25 = None
            _bm25_payloads = []
            raise
        print(f"[retriever] BM25 索引就绪：{len(payloads)} 篇，"
              f"用时 {time.time() - t0:.1f}s", flush=True)


def refresh(name: str | None = None) -> None:
    """向量库变更后重建 BM25 索引。"""
    _build_bm25(force=True, name=name)


def bm25_stats() -> dict:
    _build_bm25()
    return {"docs": len(_bm25_payloads), "built_at": _bm25_built_at}


# ---------------------------------------------------------------------------
# 单路召回
# ---------------------------------------------------------------------------
def _dense_search(query: str, top_k: int, name: str | None) -> list[dict]:
    vec = embedder.embed_query(query)
    return vector_store.search(vec, top_k=top_k, name=name)


def _sparse_search(query: str, top_k: int) -> list[dict]:
    _build_bm25()
    tokens = tokenize(query)
    if not tokens or _bm25 is None:
        return []
    scores = _bm25.get_scores(tokens)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    out = []
    for i in idx:
        if scores[i] <= 0:
            break
        p = dict(_bm25_payloads[i])
        p["bm25_score"] = float(scores[i])
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists: Sequence[tuple[str, Sequence[dict]]],
             k: int | None = None) -> list[dict]:
    """多路结果按 RRF 融合：score(d) = Σ 1/(k + rank_r(d))。

    只依赖名次，避免余弦分与 BM25 分量纲不同带来的归一化负担。
    """
    k = k or config.RRF_K
    merged: dict[str, dict] = {}
    for source_name, hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.get("chunk_id") or f"p{hit.get('page_idx')}-{hash(hit.get('text', '')) % 10**8}"
            item = merged.setdefault(key, {"payload": dict(hit), "rrf": 0.0, "ranks": {}})
            item["rrf"] += 1.0 / (k + rank)
            item["ranks"][source_name] = rank
            if source_name == "dense" and "score" in hit:
                item["payload"]["vector_score"] = float(hit["score"])
            if source_name == "sparse" and "bm25_score" in hit:
                item["payload"]["bm25_score"] = float(hit["bm25_score"])
            for kk, vv in hit.items():
                item["payload"].setdefault(kk, vv)

    out: list[dict] = []
    for item in merged.values():
        payload = item["payload"]
        payload["rrf_score"] = item["rrf"]
        payload["ranks"] = item["ranks"]
        out.append(payload)
    out.sort(key=lambda p: p["rrf_score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def search(query: str, top_k: int | None = None, hybrid: bool | None = None,
           name: str | None = None) -> list[dict]:
    """混合检索入口。返回候选**子块**（含 parent_text），按 RRF 降序。"""
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or config.VECTOR_TOP_K
    hybrid = config.HYBRID_ENABLED_DEFAULT if hybrid is None else hybrid

    dense = _dense_search(query, top_k, name)
    if not hybrid:
        for i, h in enumerate(dense, start=1):
            h["rrf_score"] = 1.0 / (config.RRF_K + i)
            h["ranks"] = {"dense": i}
        return dense

    sparse = _sparse_search(query, config.BM25_TOP_K)
    fused = rrf_fuse([("dense", dense), ("sparse", sparse)])
    return fused[:top_k]


def search_multi(queries: Sequence[str], top_k: int | None = None,
                 name: str | None = None) -> list[dict]:
    """多查询（子问题分解）检索：各子查询独立混合召回后整体 RRF 融合。"""
    lists: list[tuple[str, Sequence[dict]]] = []
    for i, q in enumerate(queries):
        if not q.strip():
            continue
        dense = _dense_search(q, config.VECTOR_TOP_K, name)
        sparse = _sparse_search(q, config.BM25_TOP_K)
        lists.append((f"dense{i}", dense))
        lists.append((f"sparse{i}", sparse))
    if not lists:
        return []
    fused = rrf_fuse(lists)
    return fused[: (top_k or config.VECTOR_TOP_K)]


def stats() -> dict:
    _build_bm25()
    return {
        "vector_top_k": config.VECTOR_TOP_K,
        "bm25_top_k": config.BM25_TOP_K,
        "bm25_docs": len(_bm25_payloads),
        "rrf_k": config.RRF_K,
    }


def _cli() -> None:
    import sys

    q = " ".join(sys.argv[1:]) or "武汉兴图新科电子股份有限公司注册资本是多少？"
    t0 = time.time()
    for i, hit in enumerate(search(q, top_k=5), start=1):
        print(f"\n[{i}] rrf={hit['rrf_score']:.4f} "
              f"vec={hit.get('vector_score', 0):.3f} "
              f"bm25={hit.get('bm25_score', 0):.2f} "
              f"第{int(hit.get('page_idx', 0)) + 1}页 {hit.get('heading_path')}")
        print("   ", (hit.get("text", ""))[:160].replace("\n", " "))
    print(f"\n用时 {time.time() - t0:.2f}s")


if __name__ == "__main__":  # 冒烟：python -m src.retriever "注册资本是多少"
    from src import bootstrap
    bootstrap.run_with_large_stack(_cli)
