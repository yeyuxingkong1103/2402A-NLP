# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：三种检索策略的配置与应用（工单硬性要求1，本工单核心）

┌────────────┬──────────────────────────────────────────────────────────────┐
│ vector     │ 向量检索（召回 + Reranker 精排）                              │
│            │ bge 稠密向量余弦召回 TOP_K_RECALL → Reranker 模型精排         │
│            │ → TOP_K_RERANK 条送 LLM                                      │
├────────────┼──────────────────────────────────────────────────────────────┤
│ fulltext   │ 全文检索（自研倒排索引）                                      │
│            │ 位置倒排表 + BM25 打分，支持布尔/短语/模糊/多字段；            │
│            │ 命中后可选择再过 Reranker 精排（默认开）                       │
├────────────┼──────────────────────────────────────────────────────────────┤
│ hybrid     │ 混合检索（推荐）                                              │
│            │ 向量 + 全文 双路召回 → RRF/加权平均/投票 融合（权重可调）      │
│            │ → **必须**经过 Reranker 模型重新排序 → TOP_K_RERANK 条送 LLM   │
└────────────┴──────────────────────────────────────────────────────────────┘

可调配置（工单要求「允许用户根据需求调整检索策略」）：
  strategy / top_k_recall / top_k_rerank / score_threshold /
  vector_weight / fulltext_weight / fusion / reranker_model /
  rerank_algorithm / embedding_model / sources / exclude_block_types

观测（工单硬性要求2）：
  每次检索返回 info，含 召回候选数、各阶段耗时(ms)、精排前后 Top-K 名次变化。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import time
from dataclasses import dataclass, field, replace
from typing import Sequence

from src import config, fusion as fusion_mod


@dataclass
class RetrievalConfig:
    """一次检索的全部可调参数（界面/SDK 都通过它传参）。"""

    strategy: str = field(default_factory=lambda: config.RETRIEVAL_STRATEGY_DEFAULT)
    top_k_recall: int = field(default_factory=lambda: config.TOP_K_RECALL)
    top_k_rerank: int = field(default_factory=lambda: config.TOP_K_RERANK)
    score_threshold: float = field(default_factory=lambda: config.SCORE_THRESHOLD)
    # 两路权重（自动归一化，不必保证和为 1）
    vector_weight: float = field(default_factory=lambda: config.VECTOR_WEIGHT)
    fulltext_weight: float = field(default_factory=lambda: config.FULLTEXT_WEIGHT)
    fusion: str = field(default_factory=lambda: config.FUSION_METHOD_DEFAULT)
    # 精排
    rerank_enabled: bool = field(default_factory=lambda: config.RERANKER_ENABLED)
    rerank_algorithm: str = field(default_factory=lambda: config.RERANK_ALGORITHM_DEFAULT)
    reranker_model: str = field(default_factory=lambda: config.RERANKER_MODEL_NAME)
    fulltext_rerank: bool = True          # 全文检索是否也过精排
    # 嵌入模型（决定用哪个 collection）
    embedding_model: str = field(default_factory=lambda: config.EMBEDDING_MODEL_NAME)
    # 过滤
    sources: Sequence[str] | None = None
    exclude_block_types: Sequence[str] | None = None

    def normalized_weights(self, method: str | None = None) -> dict[str, float]:
        """把两路权重换算成融合算法需要的口径。

        两种口径（实测教训）：
        · weighted（加权平均）：分数是"凸组合"，按**和为 1** 归一化；
        · rrf / vote（名次类融合）：若也按和为 1 归一化，会**系统性压制少数臂**——
          实测 id1「发行股数」：正确答案是全文臂第 1 名（权重 0.4 → 0.4/61=0.0066），
          却被向量臂第 15 名（0.6/75=0.0080）压过，正确答案进不了候选。
          故按**最大值为 1** 归一化：让被信任的那一路保持满权重，另一路按比例缩放，
          默认等权（0.5/0.5）时即退化为上游验证过的"无权重 RRF"（k=60）。
        """
        vw = max(0.0, float(self.vector_weight))
        fw = max(0.0, float(self.fulltext_weight))
        method = method or self.fusion
        if vw <= 0 and fw <= 0:
            return {"dense": 1.0, "fulltext": 1.0}
        if method == "weighted":
            total = vw + fw
            return {"dense": vw / total, "fulltext": fw / total}
        top = max(vw, fw) or 1.0
        return {"dense": vw / top, "fulltext": fw / top}


# ---------------------------------------------------------------------------
# 单路召回
# ---------------------------------------------------------------------------
def _dense_recall(query: str, top_k: int, cfg: RetrievalConfig) -> list[dict]:
    """向量召回：Query → bge 向量 → Qdrant 余弦检索。"""
    from src import doc_router, embedder, vector_store

    vec = embedder.embed_query(query, model=cfg.embedding_model)
    qfilter = doc_router.build_filter(
        list(cfg.sources) if cfg.sources else None,
        list(cfg.exclude_block_types) if cfg.exclude_block_types else None)
    return vector_store.search(
        vec, top_k=top_k, name=config.collection_name(cfg.embedding_model),
        query_filter=qfilter)


def _fulltext_recall(query: str, top_k: int, cfg: RetrievalConfig) -> list[dict]:
    """全文召回：自研倒排索引（布尔/短语/模糊/多字段）。"""
    from src import fulltext

    hits = fulltext.search(query, top_k=top_k * 2)
    allowed = set(cfg.sources) if cfg.sources else None
    banned = set(cfg.exclude_block_types) if cfg.exclude_block_types else None
    out: list[dict] = []
    for h in hits:
        if allowed is not None and h.get("source") not in allowed:
            continue
        if banned is not None and h.get("block_type") in banned:
            continue
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def _take_with_arm_quota(fused: list[dict], ranked_lists, top_k: int,
                         max_in: int | None = None) -> list[dict]:
    """融合结果截断时给**每一路前 N 名保底**，避免单路强命中被融合挤出候选。

    为什么需要（工单6 实测，两轮迭代）：
      id260「军用领域收入分别是多少」——正确块只被**全文臂**命中（RRF=1/(60+rank)），
      而该语料里大量块被两路同时命中，融合分轻易超过"单路命中"，
      于是正确块被挤出融合 Top-20，**精排根本没机会看到它**（精排看到它时给 0.998）。
      迭代一：每路保底 5 名 + 填充融合序 → 仍不够，因为该块在全文臂只排第 12；
      迭代二（现行）：保底额度提到 `ARM_QUOTA=20`，并把精排输入上限放宽到
      `RERANK_MAX_IN`（默认 60）—— 候选集合变成
          「融合 Top-K」 ∪ 「dense 前 20」 ∪ 「fulltext 前 20」，去重后精排。
      代价是精排候选变多（GPU 上 30~40 条约 0.4~0.8s，仍在 3s 预算内），
      收益是**任何一路的头部命中都不再丢失**。
    """
    from src import config as _cfg

    quota = max(0, int(getattr(_cfg, "ARM_QUOTA", 0) or 0))
    cap = max_in or top_k
    if quota <= 0 or not ranked_lists:
        return fused[:top_k]

    keep: list[dict] = []
    seen: set[str] = set()

    def _add(hit: dict) -> None:
        key = fusion_mod.doc_key(hit)
        if key in seen:
            return
        seen.add(key)
        keep.append(hit)

    for hit in fused[:top_k]:                       # 融合序打底
        _add(hit)
    for _arm, hits in ranked_lists:                 # 每路前 quota 名保底
        for hit in hits[:quota]:
            _add(hit)
    return keep[:cap]


# ---------------------------------------------------------------------------
# 精排（统一走 rerankers 注册表）
# ---------------------------------------------------------------------------
def _apply_rerank(query: str, cands: list[dict], cfg: RetrievalConfig) -> dict:
    from src import rerankers

    if not cfg.rerank_enabled:
        return {"hits": cands[:cfg.top_k_rerank],
                "info": {"algorithm": "off", "n_in": len(cands),
                         "n_out": min(len(cands), cfg.top_k_rerank),
                         "elapsed_ms": 0.0, "rank_changes": [],
                         "n_changed": 0, "degraded": True,
                         "degrade_reason": "精排已关闭（配置）"}}
    return rerankers.rerank_detailed(
        query, cands,
        algorithm=cfg.rerank_algorithm,
        model=cfg.reranker_model,
        top_k=cfg.top_k_rerank,
        score_threshold=cfg.score_threshold)


# ---------------------------------------------------------------------------
# 三策略主体
# ---------------------------------------------------------------------------
def retrieve(query: str, cfg: RetrievalConfig | None = None,
             analysis: dict | None = None, do_rerank: bool = True) -> dict:
    """检索统一入口。

    返回：
      {
        "hits": [...],           # 最终结果（精排后，含引用元数据）
        "info": {                # 工单要求的可观测信息
            strategy, arms, recall_counts, fusion, weights,
            recall_ms, fusion_ms, rerank_ms, total_ms,
            rerank: {n_in, elapsed_ms, rank_changes, degraded, ...}
        }
      }

    do_rerank=False：只做召回（rag.py 的分阶段耗时口径要求精排单独计时，
    由调用方接着调用 rerankers.rerank_detailed，避免重复精排）。
    """
    cfg = cfg or RetrievalConfig()
    query = (query or "").strip()
    if not query:
        return {"hits": [], "info": {"strategy": cfg.strategy, "error": "空查询"}}

    t_all = time.time()
    info: dict = {
        "strategy": cfg.strategy,
        "embedding_model": cfg.embedding_model,
        "fusion": cfg.fusion,
        "weights": cfg.normalized_weights() if cfg.strategy == "hybrid" else {},
        "top_k_recall": cfg.top_k_recall,
        "top_k_rerank": cfg.top_k_rerank,
        "score_threshold": cfg.score_threshold,
        "arms": [],
        "recall_counts": {},
    }

    # ---- 1) 召回阶段 -----------------------------------------------------
    t0 = time.time()
    wide = max(cfg.top_k_recall * 2, cfg.top_k_recall + 10)   # 宽召回再截断

    if cfg.strategy == "vector":
        dense = _dense_recall(query, wide, cfg)
        for i, h in enumerate(dense, start=1):
            h["vector_rank"] = i
        info["arms"] = ["dense"]
        info["recall_counts"] = {"dense": len(dense)}
        cands = dense[:cfg.top_k_recall]

    elif cfg.strategy == "fulltext":
        ft = _fulltext_recall(query, wide, cfg)
        for i, h in enumerate(ft, start=1):
            h["fulltext_rank"] = i
        info["arms"] = ["fulltext"]
        info["recall_counts"] = {"fulltext": len(ft)}
        # 全文检索的"名次分"：直接按 fulltext_score 排序（已是倒排索引输出序）
        for i, h in enumerate(ft, start=1):
            h["rrf_score"] = 1.0 / (config.RRF_K + i)
        cands = ft[:cfg.top_k_recall]

    elif cfg.strategy == "hybrid":
        dense = _dense_recall(query, wide, cfg)
        ft = _fulltext_recall(query, wide, cfg)
        for i, h in enumerate(dense, start=1):
            h["vector_rank"] = i
        for i, h in enumerate(ft, start=1):
            h["fulltext_rank"] = i
        info["arms"] = ["dense", "fulltext"]
        info["recall_counts"] = {"dense": len(dense), "fulltext": len(ft)}

        ranked_lists = [("dense", dense), ("fulltext", ft)]
        weights = cfg.normalized_weights(cfg.fusion)
        fused = fusion_mod.fuse(cfg.fusion, ranked_lists, weights=weights,
                                rrf_k=config.RRF_K)
        info["fusion_ms"] = round((time.time() - t0) * 1000, 2)
        cands = _take_with_arm_quota(fused, ranked_lists, cfg.top_k_recall,
                                     max_in=config.RERANK_MAX_IN)
        info["arm_quota"] = config.ARM_QUOTA
        info["rerank_in"] = len(cands)

    else:
        raise ValueError(f"未知检索策略 {cfg.strategy!r}；"
                         f"可选：{list(config.RETRIEVAL_STRATEGIES)}")

    info["recall_ms"] = round((time.time() - t0) * 1000, 2)
    info["n_candidates"] = len(cands)

    # ---- 2) 精排阶段 -----------------------------------------------------
    if not do_rerank:
        info["rerank_ms"] = 0.0
        info["rerank"] = {"algorithm": "deferred", "n_in": len(cands)}
        info["total_ms"] = round((time.time() - t_all) * 1000, 2)
        info["n_hits"] = len(cands)
        return {"hits": cands, "info": info}

    t1 = time.time()
    if cfg.strategy == "fulltext" and not cfg.fulltext_rerank:
        hits = cands[:cfg.top_k_rerank]
        rerank_info = {"algorithm": "off", "n_in": len(cands),
                       "n_out": len(hits), "elapsed_ms": 0.0,
                       "rank_changes": [], "n_changed": 0, "degraded": True,
                       "degrade_reason": "全文检索未启用精排（配置）"}
    else:
        result = _apply_rerank(query, cands, cfg)
        hits, rerank_info = result["hits"], result["info"]
    info["rerank_ms"] = round((time.time() - t1) * 1000, 2)
    info["rerank"] = rerank_info
    info["rerank_algorithm"] = rerank_info.get("algorithm")
    info["reranker_model"] = (cfg.reranker_model
                              if rerank_info.get("algorithm") == "model"
                              else rerank_info.get("algorithm"))

    info["total_ms"] = round((time.time() - t_all) * 1000, 2)
    info["n_hits"] = len(hits)
    return {"hits": hits, "info": info}


def retrieve_multi(queries: Sequence[str], cfg: RetrievalConfig | None = None,
                   do_rerank: bool = True) -> dict:
    """多查询检索（子问题分解）：各子查询独立召回，多路整体融合后再精排。

    与 retrieve 的区别只在召回阶段：把 N 个子查询视为 N×2 条召回通路一起融合，
    使"多问法覆盖到的块"获得更高名次。
    """
    cfg = cfg or RetrievalConfig()
    qs = [q for q in (queries or []) if q and q.strip()]
    if not qs:
        return {"hits": [], "info": {"strategy": cfg.strategy, "error": "空查询"}}
    if len(qs) == 1:
        return retrieve(qs[0], cfg)

    t_all = time.time()
    wide = max(cfg.top_k_recall * 2, cfg.top_k_recall + 10)
    ranked_lists: list[tuple[str, list[dict]]] = []
    counts: dict[str, int] = {}

    for i, q in enumerate(qs):
        if cfg.strategy in ("vector", "hybrid"):
            dense = _dense_recall(q, wide, cfg)
            ranked_lists.append((f"dense{i}", dense))
            counts[f"dense{i}"] = len(dense)
        if cfg.strategy in ("fulltext", "hybrid"):
            ft = _fulltext_recall(q, wide, cfg)
            ranked_lists.append((f"fulltext{i}", ft))
            counts[f"fulltext{i}"] = len(ft)

    if not ranked_lists:
        return {"hits": [], "info": {"strategy": cfg.strategy, "error": "无召回通路"}}

    method = cfg.fusion if cfg.strategy == "hybrid" else "rrf"
    weights = cfg.normalized_weights(method)
    fused = fusion_mod.fuse(method, ranked_lists, weights=weights,
                            rrf_k=config.RRF_K)
    cands = _take_with_arm_quota(fused, ranked_lists, cfg.top_k_recall,
                                 max_in=config.RERANK_MAX_IN)
    if not do_rerank:
        return {"hits": cands, "info": {
            "strategy": cfg.strategy, "multi_query": True, "queries": qs,
            "arms": [n for n, _ in ranked_lists], "recall_counts": counts,
            "fusion": method, "weights": weights, "n_candidates": len(cands),
            "rerank": {"algorithm": "deferred", "n_in": len(cands)},
            "total_ms": round((time.time() - t_all) * 1000, 2),
            "n_hits": len(cands)}}
    result = _apply_rerank(qs[0], cands, cfg)

    return {"hits": result["hits"], "info": {
        "strategy": cfg.strategy, "multi_query": True, "queries": qs,
        "arms": [n for n, _ in ranked_lists], "recall_counts": counts,
        "fusion": method, "weights": weights,
        "n_candidates": len(cands),
        "rerank": result["info"],
        "total_ms": round((time.time() - t_all) * 1000, 2),
        "n_hits": len(result["hits"]),
    }}


def describe() -> dict:
    """当前默认配置 + 可选项清单（供界面渲染）。"""
    cfg = RetrievalConfig()
    return {
        "strategies": config.RETRIEVAL_STRATEGIES,
        "default_strategy": cfg.strategy,
        "fusion_methods": list(config.FUSION_METHODS),
        "default_fusion": cfg.fusion,
        "rerank_algorithms": list(config.RERANK_ALGORITHMS),
        "default_rerank_algorithm": cfg.rerank_algorithm,
        "reranker_models": {k: v["desc"] for k, v in config.RERANKER_MODELS.items()},
        "default_reranker_model": cfg.reranker_model,
        "embedding_models": {k: v["desc"] for k, v in config.EMBEDDING_MODELS.items()},
        "default_embedding_model": cfg.embedding_model,
        "defaults": {
            "top_k_recall": cfg.top_k_recall,
            "top_k_rerank": cfg.top_k_rerank,
            "score_threshold": cfg.score_threshold,
            "vector_weight": cfg.vector_weight,
            "fulltext_weight": cfg.fulltext_weight,
        },
    }


def with_overrides(cfg: RetrievalConfig, **kwargs) -> RetrievalConfig:
    """便捷覆盖（界面/评估脚本用）。"""
    valid = {k: v for k, v in kwargs.items() if v is not None}
    return replace(cfg, **valid)
