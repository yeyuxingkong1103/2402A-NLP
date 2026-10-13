# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：召回层（三策略调度 + BM25 稀疏召回）

本工单把召回升级为**三种可选策略**（详见 src/strategies.py）：
  · vector   —— 仅稠密向量召回
  · fulltext —— 仅自研倒排索引全文召回（布尔/短语/模糊/多字段）
  · hybrid   —— 两路召回 + 融合算法（RRF/加权平均/投票）★推荐

本文件保留两类职责：
  1. 兼容上游调用点：`search()/search_multi()` 现在按**运行时设置**（src/runtime.py）
     选择策略完成召回合流，返回候选（精排仍在 rag.py 统一调用，避免重复精排）；
  2. BM25 稀疏召回臂：`rank_bm25` 版稀疏检索保留为**可选对照臂**
     （用于消融实验与上游可比性验证；生产 hybrid 默认走自研倒排索引）。

分词统一来自 src/tokenizer.py（与倒排索引完全同口径）。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
import threading
import time
from typing import Sequence

from src import config, doc_router, embedder, vector_store
from src.tokenizer import tokenize  # noqa: F401 —— 兼容 re-export（上游测试引用）

_bm25 = None
_bm25_payloads: list[dict] = []
_bm25_lock = threading.Lock()
_bm25_built_at: float = 0.0


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
def _dense_search(query: str, top_k: int, name: str | None,
                  sources: Sequence[str] | None = None,
                  exclude_block_types: Sequence[str] | None = None) -> list[dict]:
    vec = embedder.embed_query(query)
    return vector_store.search(vec, top_k=top_k, name=name,
                               query_filter=doc_router.build_filter(
                                   list(sources) if sources else None,
                                   list(exclude_block_types) if exclude_block_types else None))


def _sparse_search(query: str, top_k: int,
                   sources: Sequence[str] | None = None,
                   exclude_block_types: Sequence[str] | None = None) -> list[dict]:
    """BM25 召回；sources / exclude_block_types 非空时按同口径过滤。"""
    _build_bm25()
    tokens = tokenize(query)
    if not tokens or _bm25 is None:
        return []
    allowed = set(sources) if sources else None
    banned = set(exclude_block_types) if exclude_block_types else None
    scores = _bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0 or len(out) >= top_k:
            break
        payload = _bm25_payloads[i]
        if allowed is not None and payload.get("source") not in allowed:
            continue
        if banned is not None and payload.get("block_type") in banned:
            continue
        p = dict(payload)
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
            # 融合键必须带 source：两个文档的 chunk_id 各自从 c000001 开始，
            # 只用 chunk_id 会把"招股1 的第 N 块"和"招股2 的第 N 块"错误合并成一条
            # （工单3 多文档修正）。
            cid = hit.get("chunk_id")
            key = (f"{hit.get('source', '')}#{cid}" if cid
                   else f"{hit.get('source', '')}#p{hit.get('page_idx')}"
                        f"-{hit.get('seq', 0)}")
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
# 对外接口（召回层：按运行时设置的三策略调度，精排由调用方统一执行）
# ---------------------------------------------------------------------------
def _cfg_from_runtime(top_k: int | None, sources, exclude_block_types,
                      hybrid: bool | None):
    """把「显式参数 + 运行时设置」合成一次检索配置。

    · 界面/SDK 通过 src/runtime.py 设置策略与参数 → 这里读取为默认值；
    · 上游调用点传入的 sources / exclude_block_types / hybrid 显式参数优先；
    · hybrid=False 等价于「只用向量召回」（兼容旧行为与消融实验）。
    """
    from src import runtime, strategies

    cfg = runtime.get()
    if hybrid is False:
        cfg = strategies.with_overrides(cfg, strategy="vector")
    if top_k:
        cfg = strategies.with_overrides(cfg, top_k_recall=top_k)
    if sources is not None:
        cfg = strategies.with_overrides(cfg, sources=list(sources))
    if exclude_block_types is not None:
        cfg = strategies.with_overrides(cfg,
                                        exclude_block_types=list(exclude_block_types))
    return cfg


def search(query: str, top_k: int | None = None, hybrid: bool | None = None,
           name: str | None = None,
           sources: Sequence[str] | None = None,
           exclude_block_types: Sequence[str] | None = None) -> list[dict]:
    """单查询召回入口。返回候选**子块**（含 parent_text），按融合分降序。

    实际使用的策略由运行时设置决定（vector / fulltext / hybrid）。
    name 参数仅用于兼容旧签名（collection 现由嵌入模型派生，见 config.collection_name）。
    """
    from src import strategies

    query = (query or "").strip()
    if not query:
        return []
    cfg = _cfg_from_runtime(top_k, sources, exclude_block_types, hybrid)
    return strategies.retrieve(query, cfg, do_rerank=False)["hits"]


def search_multi(queries: Sequence[str], top_k: int | None = None,
                 name: str | None = None,
                 sources: Sequence[str] | None = None,
                 exclude_block_types: Sequence[str] | None = None) -> list[dict]:
    """多查询（子问题分解）检索：各子查询独立召回后整体融合。"""
    from src import strategies

    qs = [q for q in (queries or []) if q and q.strip()]
    if not qs:
        return []
    cfg = _cfg_from_runtime(top_k, sources, exclude_block_types, None)
    return strategies.retrieve_multi(qs, cfg, do_rerank=False)["hits"]


def search_with_info(query: str, cfg=None) -> dict:
    """带完整观测信息的检索（供界面/评估展示召回与精排过程）。"""
    from src import runtime, strategies

    cfg = cfg or runtime.get()
    return strategies.retrieve(query, cfg)


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
