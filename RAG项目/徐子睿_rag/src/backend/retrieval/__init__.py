# -*- coding: utf-8 -*-
"""retrieval —— 混合检索（同名包）。

在链路中的位置（在线侧的核心）：
    backend/server.py 的 /api/search、/api/ask、/api/ask/stream → 【本包】
        → 返回带出处(page/section/source)的片段 → 交给大模型生成答案

本包是这个项目的技术主体，回答了"为什么答案可信"：
    1. 双路召回 —— 向量路管语义相近，BM25 路管字面精确，两者互补
    2. 加权 RRF 融合 —— 两路分数量纲不同不能相加，改用排名倒数融合
    3. LLM 精排 —— 让模型对候选打相关性分，且失败自动回退 RRF 顺序
    4. 全过程 trace —— 每一步的状态/耗时/明细都记录，前端"检索链路"页签据此可视化

为什么改成了包：
    原 retrieval.py 有 652 行（其中注释 251 行），超出"单文件 300 行"的上限。
    按链路阶段拆成 config / bm25 / query / corpus / recall / fusion / rerank / context
    八个模块后，每个文件都在 300 行以内，而且目录结构本身就对应检索链路的执行顺序。

关键设计：本文件把各子模块的公开名全部再导出，
    所以 `from retrieval import rrf_merge, rewrite_query, retrieve_with_trace` 之类用法
    与拆分前完全一致 —— server / roleplay / tests 一行都不用改。

包内分工（按检索执行顺序）：
    config.py   调参常量与规则表
    query.py    查询规范化与规则改写
    bm25.py     分词 + BM25 打分
    corpus.py   语料加载与索引缓存
    recall.py   两路召回
    fusion.py   加权 RRF 融合
    rerank.py   LLM 精排
    context.py  上下文组织
    本文件      retrieve_with_trace / hybrid_search 总编排 + 再导出
"""
from __future__ import annotations

import time
from typing import Any, Callable

try:
    from ..pipeline import COLLECTION
    from ..vector_store import collection_count
except ImportError:
    from pipeline import COLLECTION
    from vector_store import collection_count

from .bm25 import BM25, tokenize
from .config import (
    CHAT_URL,
    DEFAULT_CANDIDATES,
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_RESULTS,
    DEFAULT_SNIPPET_CHARS,
    KEYWORD_K,
    KEYWORD_WEIGHT,
    RERANK_MODEL,
    REWRITE_RULES,
    RRF_K,
    STOP_WORDS,
    TOKEN_RE,
    VECTOR_K,
    VECTOR_THRESHOLD,
)
from .context import organize_context
from .corpus import bm25_index, invalidate_cache, load_corpus
from .fusion import fuse_hits, rrf_merge
from .query import normalize_query, rewrite_query
from .recall import dense_hits, keyword_hits
from .rerank import rerank

def retrieve_with_trace(
    query: str,
    *,
    candidate_k: int = DEFAULT_CANDIDATES,
    final_k: int = DEFAULT_RESULTS,
    char_budget: int = DEFAULT_CONTEXT_CHARS,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """检索链路总入口，逐步执行并记录 trace。

    参数：
        query: 用户问题
        candidate_k: 融合后进入精排的候选数
        final_k: 最终返回的结果数
        char_budget: 上下文字符预算
        trace_callback: 每产生一条 trace 事件立即回调。SSE 流式接口用它把
                        "查询改写完成""语义召回完成"实时推给前端，做出逐步点亮的动效。
    返回：
        完整检索结果，含各阶段中间产物：
        query / rewritten_query / rewrite / dense_hits / sparse_hits /
        candidates / results / context_docs / trace / note / rerank / context

    返回值里同时给中间产物和最终结果，是为了让前端"检索链路"页签能展示
    "两路各召回什么、融合成什么、精排砍掉了谁" —— 这是 V2 迭代"链路可观测"的核心，
    也是答辩时最有说服力的部分：能解释每一次命中是怎么来的。

    trace 每条含 step(步骤名) / status(done|skipped) / detail(人类可读说明) / ms(耗时毫秒)。
    """
    trace: list[dict[str, Any]] = []
    query = (query or "").strip()
    started = time.time()

    def add_trace(step: str, status: str, detail: str, begin: float) -> None:
        """记一条 trace：算耗时、入列表、并即时回调出去。"""
        event = {"step": step, "status": status, "detail": detail, "ms": round((time.time() - begin) * 1000)}
        trace.append(event)
        if trace_callback:
            trace_callback(event)

    # 提前备好"空结果"骨架，所有早退分支都基于它，保证返回结构的 key 集合始终一致
    empty = {"query": query, "rewritten_query": query, "dense_hits": [], "sparse_hits": [], "candidates": [], "results": [], "context_docs": [], "trace": trace}
    if not query:
        return empty | {"note": "请输入问题"}
    # 分词后没有有效词（例如用户只输入了标点）→ 直接拒答，不浪费一次向量化和 BM25 全量打分
    if not tokenize(query):
        add_trace("前置拒答", "skipped", "输入无有效检索词", started)
        return empty | {"note": "查询无有效检索词，请输入有意义的问题"}

    begin = time.time()
    rewrite = rewrite_query(query)
    add_trace("查询改写", "done", rewrite["rewritten_query"], begin)
    search_query = rewrite["rewritten_query"] or query  # 兜底：改写结果为空就用原问题

    begin = time.time()
    vectors = dense_hits(search_query)
    add_trace("语义召回", "done" if vectors else "skipped", f"命中 {len(vectors)} 条", begin)
    begin = time.time()
    keywords, keyword_scores = keyword_hits(search_query)
    # 这里统计的是"分数 >= 1"的条数，而不是返回条数：trace 里展示"有多少条是有效命中"更有信息量
    kept = sum(score >= 1 for _, score in keyword_scores[:KEYWORD_K])
    add_trace("关键词召回", "done" if keywords else "skipped", f"保留 {kept} 条", begin)
    begin = time.time()
    candidates = fuse_hits(vectors, keywords, candidate_k)
    add_trace("RRF 融合", "done" if candidates else "skipped", f"融合后 {len(candidates)} 条", begin)
    begin = time.time()
    results, rerank_meta = rerank(candidates, search_query, final_k)
    # 精排这一步的 status 取决于是否真的应用了模型分数：回退 RRF 时显示 skipped，让用户知道"这轮没精排"
    add_trace("候选精排", "done" if rerank_meta["applied"] else "skipped", rerank_meta["note"], begin)
    begin = time.time()
    context, context_meta = organize_context(results, char_budget)
    add_trace("上下文组织", "done" if context else "skipped", f"{context_meta['count']} 条 / {context_meta['chars']} 字符", begin)

    return {
        "query": query,
        "rewritten_query": search_query,
        "rewrite": rewrite,
        "dense_hits": vectors,
        "sparse_hits": keywords,
        "candidates": candidates,
        "results": results,
        "context_docs": context,
        "trace": trace,
        "note": f"改写完成；语义 {len(vectors)} 条；关键词 {kept} 条；融合 {len(candidates)} 条；结果 {len(results)} 条；上下文 {len(context)} 条",
        "rerank": rerank_meta,
        "context": context_meta,
    }

def hybrid_search(query: str, k: int = DEFAULT_RESULTS) -> dict[str, Any]:
    """混合检索的对外简化接口（给不需要 trace 明细的调用方用）。

    参数：
        query: 用户问题
        k: 返回结果数
    返回：
        与 retrieve_with_trace 类似，但精简为 results/vec/kw/note/trace 等字段。

    与 retrieve_with_trace 的区别：
        先探一次 collection_count，知识库为空时直接给出"请先上传 PDF"的提示，
        而不是让用户等一轮完整的检索流程再返回空结果 —— 冷启动时的体验差别很大。
    """
    if collection_count(COLLECTION) == 0:
        return {"results": [], "vec": 0, "kw": 0, "note": "知识库为空，请先上传 PDF 构建", "trace": [], "rewritten_query": query}
    result = retrieve_with_trace(query, candidate_k=max(DEFAULT_CANDIDATES, k), final_k=k)
    return {
        "results": result["results"],
        "vec": len(result["dense_hits"]),
        "kw": len(result["sparse_hits"]),
        "note": result["note"],
        "trace": result["trace"],
        "rewritten_query": result["rewritten_query"],
        "candidates": result["candidates"],
        "context_docs": result["context_docs"],
        "rerank": result["rerank"],
    }


# 显式声明对外接口：这就是"拆包不改调用方"的契约清单
__all__ = [
    "BM25",
    "CHAT_URL",
    "DEFAULT_CANDIDATES",
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_RESULTS",
    "DEFAULT_SNIPPET_CHARS",
    "KEYWORD_K",
    "KEYWORD_WEIGHT",
    "RERANK_MODEL",
    "REWRITE_RULES",
    "RRF_K",
    "STOP_WORDS",
    "TOKEN_RE",
    "VECTOR_K",
    "VECTOR_THRESHOLD",
    "bm25_index",
    "dense_hits",
    "fuse_hits",
    "hybrid_search",
    "invalidate_cache",
    "keyword_hits",
    "load_corpus",
    "normalize_query",
    "organize_context",
    "rerank",
    "retrieve_with_trace",
    "rewrite_query",
    "rrf_merge",
    "tokenize",
]
