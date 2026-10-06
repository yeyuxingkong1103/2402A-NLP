# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：检索（向量 + BM25 混合召回，RRF 融合）

为什么不是"纯向量检索"：
  招股说明书里大量问题问的是**精确事实**（注册资本、法定代表人、某年收入、
  某奖项名称）。稠密向量擅长语义相似，但对"注册资本 7,360.00 万元"这类
  数字/专名的精确匹配反而不如关键词检索；反之 BM25 又答不了
  "公司在哪个领域是重要供应商"这种换个说法的问法。
  两者召回结果用 RRF（Reciprocal Rank Fusion）融合，取长补短。

RRF:
    score(d) = Σ_r 1 / (k + rank_r(d))       k 取 60（业界经验值）
  只依赖**名次**不依赖分数，因此无需对余弦相似度与 BM25 分数做归一化，
  避免了两种分数量纲不同带来的调参负担。

BM25 语料来源：
  直接 scroll Qdrant 里已入库的 payload（单一数据源，避免 chunks.json
  与向量库不一致导致"召回到不存在的点"）。首次调用时构建并缓存。
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
# 中文里作为整体出现的常见财务/法律专名，jieba 有时会切错，这里做保护
_PROTECTED_TERMS = (
    "注册资本", "法定代表人", "主营业务收入", "军用领域", "国家科技进步一等奖",
    "募集资金", "补充流动资金", "重要供应商", "实际控制人", "发行股数",
)

_RRF_K = 60
_bm25 = None
_bm25_payloads: list[dict] = []
_bm25_lock = threading.Lock()
_bm25_built_at: float = 0.0


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """中英混合分词：汉字走 jieba，英文/数字整体保留并小写。"""
    import jieba  # 延迟导入：仅检索时才需要，缩短进程启动时间

    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        if tok[0].isascii():
            tokens.append(tok.lower())
            continue
        # 先用保护词切开，再对剩余汉字做 jieba 分词
        for piece in _split_by_protected(tok):
            if piece in _PROTECTED_TERMS:
                tokens.append(piece)
            else:
                tokens.extend(t for t in jieba.cut(piece) if t.strip())
    return tokens


def _split_by_protected(text: str) -> list[str]:
    """把保护词从汉字串中切出来，其余部分原样返回。"""
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


# ---------------------------------------------------------------------------
# BM25 索引（懒加载 + 缓存）
# ---------------------------------------------------------------------------
def _build_bm25(force: bool = False) -> None:
    """从 Qdrant 拉取全部 payload 构建 BM25 索引。"""
    global _bm25, _bm25_payloads, _bm25_built_at
    if _bm25 is not None and not force:
        return
    with _bm25_lock:
        if _bm25 is not None and not force:
            return

        from rank_bm25 import BM25Okapi

        # 重建前先清空：否则重建失败（例如知识库刚被删除、库为空）时，
        # 旧索引会原封不动留着，此后 sparse 一路仍能检出**已删除**的片段，
        # 与"知识库已删除"的状态自相矛盾。
        _bm25 = None
        _bm25_payloads = []

        t0 = time.time()
        try:
            payloads = vector_store.scroll_all()
            if not payloads:
                raise RuntimeError(
                    "向量库为空，无法构建 BM25 索引。请先执行：python build_index.py")

            corpus = [tokenize(p.get("text", "")) for p in payloads]
            # 发布顺序：先 payloads 后索引。读者以 `_bm25 is not None` 作为就绪标志，
            # 若反过来，两条赋值之间被线程切换就会拿到"新索引 + 旧 payload"，
            # 轻则文本错位，重则 IndexError。
            _bm25_payloads = payloads
            _bm25 = BM25Okapi(corpus)
            _bm25_built_at = time.time()
        except Exception:
            _bm25 = None
            _bm25_payloads = []
            raise
        print(f"[retriever] BM25 索引就绪：{len(payloads)} 篇，"
              f"用时 {time.time() - t0:.1f}s", flush=True)


def refresh() -> None:
    """向量库变更后重建 BM25 索引（界面"重建知识库"后调用）。"""
    _build_bm25(force=True)


# ---------------------------------------------------------------------------
# 单路召回
# ---------------------------------------------------------------------------
def _dense_search(query: str, top_k: int) -> list[dict]:
    """向量召回。"""
    vec = embedder.embed_query(query)
    hits = vector_store.search(vec, top_k=top_k)
    return hits


def _sparse_search(query: str, top_k: int) -> list[dict]:
    """BM25 稀疏召回。"""
    _build_bm25()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = _bm25.get_scores(tokens)
    # 取分数最高的 top_k（分数为 0 的不要，说明一个词都没命中）
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
def _rrf_fuse(ranked_lists: Sequence[tuple[str, Sequence[dict]]]) -> list[dict]:
    """把多路召回结果按 RRF 融合。

    ranked_lists: [(来源名, 有序结果列表), ...]
    返回按 RRF 分数降序的候选块，保留各路原始分数便于展示与调试。
    """
    merged: dict[str, dict] = {}
    for source_name, hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.get("chunk_id") or f"p{hit.get('page_idx')}-{hash(hit.get('text', '')) % 10**8}"
            item = merged.setdefault(key, {"payload": dict(hit), "rrf": 0.0, "ranks": {}})
            item["rrf"] += 1.0 / (_RRF_K + rank)
            item["ranks"][source_name] = rank
            # 记录各路原始分数
            if source_name == "dense" and "score" in hit:
                item["payload"]["vector_score"] = float(hit["score"])
            if source_name == "sparse" and "bm25_score" in hit:
                item["payload"]["bm25_score"] = float(hit["bm25_score"])
            # payload 里以首次出现的为准，但补齐后出现的元数据
            for k, v in hit.items():
                item["payload"].setdefault(k, v)

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
def search(query: str, top_k: int | None = None,
           hybrid: bool | None = None) -> list[dict]:
    """混合检索入口。

    返回候选块列表（按 RRF 降序），每条含：
        text / raw_text / page_idx / heading_path / source / chunk_id / type
        vector_score / bm25_score（命中时）/ rrf_score / ranks
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or config.VECTOR_TOP_K
    hybrid = config.HYBRID_ENABLED if hybrid is None else hybrid

    dense = _dense_search(query, top_k)
    if not hybrid:
        for i, h in enumerate(dense, start=1):
            h["rrf_score"] = 1.0 / (_RRF_K + i)
            h["ranks"] = {"dense": i}
        return dense

    sparse = _sparse_search(query, config.BM25_TOP_K)
    fused = _rrf_fuse([("dense", dense), ("sparse", sparse)])
    # 尊重调用方给的 top_k（早期版本写 max(top_k, BM25_TOP_K)，会让 top_k=3
    # 这种小值被静默放大到 20，接口语义两路不一致）
    return fused[:top_k]


def stats() -> dict:
    """检索侧统计信息，供界面展示。"""
    _build_bm25()
    return {
        "hybrid": config.HYBRID_ENABLED,
        "vector_top_k": config.VECTOR_TOP_K,
        "bm25_top_k": config.BM25_TOP_K,
        "bm25_docs": len(_bm25_payloads),
        "bm25_built_at": _bm25_built_at,
        "rrf_k": _RRF_K,
    }


def _cli() -> None:
    import sys

    q = " ".join(sys.argv[1:]) or "武汉兴图新科电子股份有限公司注册资本是多少？"
    t0 = time.time()
    for i, hit in enumerate(search(q, top_k=5), start=1):
        print(f"\n[{i}] rrf={hit['rrf_score']:.4f} "
              f"vec={hit.get('vector_score', 0):.3f} "
              f"bm25={hit.get('bm25_score', 0):.2f} "
              # page_idx 是 0 基存储，展示时统一 +1，与界面/引用里的页码一致
              f"第{int(hit.get('page_idx', 0)) + 1}页 {hit.get('heading_path')}")
        print("   ", (hit.get("raw_text") or hit.get("text", ""))[:160].replace("\n", " "))
    print(f"\n用时 {time.time() - t0:.2f}s")


if __name__ == "__main__":  # 手动冒烟：python -m src.retriever "注册资本是多少"
    from src import bootstrap
    bootstrap.run_with_large_stack(_cli)
