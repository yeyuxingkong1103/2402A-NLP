"""T3 离线测试 ④：重排层。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 1/2、工单 6.5）：

- 重排**确实改变顺序**（不是恒等变换）；
- 重排**提升 top-5 命中**（可复现的定量对比：重排前 vs 重排后命中数）；
- 重排耗时在预算内；
- **如实标注模式**：本机无 ``bge-reranker-base`` 权重，生产路径为 ``mode=rule``，
  测试不得声称跑过模型重排（环境事实 2.2 / 红线）。

重排前口径：融合候选池按 ``score`` 降序取 top-5；
重排后口径：``Reranker.rerank(question, pool, top_n=5)`` 的返回顺序。
"""

from __future__ import annotations

import time

import pytest

from test_retriever_v2 import _rank_of


def _pool(retriever, qu, question, size=20):
    """取融合候选池（未经重排的原始顺序由 score 决定）。"""
    analysis = qu.analyze(question)
    return analysis, retriever.retrieve_multi(qu.search_queries(analysis), analysis, top_k=size)


def test_reranker_mode_is_honestly_reported(reranker):
    """重排模式必须如实上报；本机无重排模型权重时应为 rule。"""
    h = reranker.health()
    assert reranker.mode in ("model", "rule", "off"), f"未知重排模式: {reranker.mode}"
    assert "mode" in h, f"health 缺 mode: {h}"
    print(f"\n[重排] mode={reranker.mode}，health={h}")
    assert reranker.mode == "rule", (
        f"本机无 bge-reranker-base 权重，期望 rule，实际 {reranker.mode}；"
        "若已挂载模型权重请同步更新本断言与文档"
    )


def test_rerank_changes_order_on_some_questions(reranker, retriever, query_understanding, golden):
    """重排必须真的改变顺序（否则等同未生效）。"""
    changed = 0
    for item in golden:
        _, pool = _pool(retriever, query_understanding, item.question)
        before = [it.chunk.chunk_id for it in sorted(pool, key=lambda it: it.score, reverse=True)[:5]]
        after = [it.chunk.chunk_id for it in reranker.rerank(item.question, pool, top_n=5)]
        if before != after:
            changed += 1
    print(f"\n[重排] top-5 顺序发生变化的题数 = {changed}/10")
    assert changed >= 1, "重排未改变任何题目的顺序——规则重排未生效"


@pytest.mark.slow
def test_rerank_improves_or_preserves_top5_hits(reranker, retriever, query_understanding,
                                                golden, evidence_hit_chunks):
    """定量对比：重排前 vs 重排后 top-5 证据命中数，并打印逐题位次。"""
    pre_hits = post_hits = 0
    rows = []
    for item in golden:
        hit_ids = evidence_hit_chunks[item.id]
        _, pool = _pool(retriever, query_understanding, item.question)
        pre = sorted(pool, key=lambda it: it.score, reverse=True)[:5]
        post = reranker.rerank(item.question, pool, top_n=5)
        pre_rank = _rank_of(pre, hit_ids)
        post_rank = _rank_of(post, hit_ids)
        pre_hits += pre_rank is not None
        post_hits += post_rank is not None
        rows.append((item.id, pre_rank, post_rank))
    print("\n[重排] 逐题证据位次（重排前 → 重排后）：")
    for qid, pre_rank, post_rank in rows:
        print(f"   Q{qid:>4} {pre_rank} → {post_rank}")
    print(f"[重排] top-5 命中：重排前 {pre_hits}/10 → 重排后 {post_hits}/10")
    assert post_hits >= pre_hits, f"重排使命中下降：{pre_hits} → {post_hits}"
    assert post_hits >= 9, f"重排后 top-5 命中应 ≥9/10，实际 {post_hits}"


def test_rerank_within_time_budget(reranker, retriever, query_understanding, golden):
    """重排耗时必须在预算内（预算取自 reranker.health()）。"""
    budget = float(reranker.health().get("budget_ms", 0) or 0)
    assert budget > 0, f"health 未给出 budget_ms: {reranker.health()}"
    worst = 0.0
    for item in golden[:5]:
        _, pool = _pool(retriever, query_understanding, item.question)
        t0 = time.perf_counter()
        reranker.rerank(item.question, pool, top_n=5)
        worst = max(worst, (time.perf_counter() - t0) * 1000)
    print(f"\n[重排耗时] 最慢 {worst:.2f} ms（预算 {budget} ms）")
    assert worst <= budget, f"重排耗时 {worst:.2f} ms 超预算 {budget} ms"


def test_rerank_preserves_candidate_pool(reranker, retriever, query_understanding, golden):
    """重排只改顺序、不丢候选：返回内容必须是入参池的子集，且 top_n 生效。"""
    item = golden[0]
    _, pool = _pool(retriever, query_understanding, item.question)
    out = reranker.rerank(item.question, pool, top_n=5)
    assert len(out) == 5, f"top_n=5 应返回 5 条，实际 {len(out)}"
    pool_ids = {it.chunk.chunk_id for it in pool}
    assert {it.chunk.chunk_id for it in out} <= pool_ids, "重排返回了候选池之外的块"
    assert all(it.rerank_score is not None for it in out), "重排结果未写入 rerank_score"
