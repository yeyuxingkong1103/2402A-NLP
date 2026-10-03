"""T3 离线测试 ③：检索层（含排序质量、重排前后对照、BM25 数值型契约）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 1/7、工单 6.4/6.5）：

- 10 个工单问题检索后，**证据原文必须落在最终 top-5 上下文内**（用证据原文比对，
  不硬编码 chunk_id——chunk_id 跨索引不稳定，环境事实 §4.1.4）；
- 结果条目字段齐全（chunk_id / page / score / content / type）；
- **排序质量**：打印证据在「融合候选池」与「最终结果」中的位次分布，
  给出重排前 vs 重排后逐题对照表（captain 要求：不能只测「是否召回」）；
- **BM25 数值型契约**：金额串的分词现状与「精确金额查询必须能召回证据」的行为契约。

口径声明：
- 最终上下文条数 = ``settings.retrieval.rerank_top_n``（**实测为 5**，不写死 8；
  工单1 基线的 8 是旧配置，环境事实早期版本按 8 描述，以本仓库配置为准）；
- Q95/Q207 的证据块由**替代判据**运行时求得（见 conftest ``evidence_hit_chunks``），
  统计时单独标注，不混入严格口径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import norm_b


def _search(retriever, qu, question: str, top_k: int):
    """按生产路径检索：analyze → search_queries → retrieve_multi。"""
    analysis = qu.analyze(question)
    queries = qu.search_queries(analysis)
    results = retriever.retrieve_multi(queries, analysis, top_k=top_k)
    return analysis, queries, results


def _rank_of(results, hit_ids: set[str]) -> int | None:
    """证据在结果列表中的 1 基位次；未命中返回 None（不使用「第 0 名」哨兵语义）。"""
    for pos, item in enumerate(results, start=1):
        if item.chunk.chunk_id in hit_ids:
            return pos
    return None


def test_final_context_size_matches_config(retriever, settings, query_understanding, golden):
    """最终上下文条数必须等于配置的 rerank_top_n（实测 5）。"""
    want = settings.retrieval.rerank_top_n
    _, _, results = _search(retriever, query_understanding, golden[0].question, top_k=want)
    assert len(results) == want, f"最终上下文应 {want} 条，实际 {len(results)}"
    print(f"\n[上下文] rerank_top_n={want}（配置），实际返回 {len(results)} 条")


@pytest.mark.slow
def test_ten_questions_hit_evidence_in_final_top5(retriever, query_understanding, golden, evidence_hit_chunks):
    """10 题检索：证据必须进最终 top-5，并打印逐题位次与命中率。

    达标线：工单要求检索准确率 ≥90%，故断言 ≥9/10；实测命中逐题打印。
    Q95/Q207 使用替代判据（其 golden ``evidence`` 分别为省略号串与合成串）。
    """
    hits = 0
    rows = []
    for item in golden:
        hit_ids = evidence_hit_chunks[item.id]
        _, _, results = _search(retriever, query_understanding, item.question, top_k=5)
        rank = _rank_of(results, hit_ids)
        ok = rank is not None
        hits += ok
        rows.append((item.id, rank, [r.chunk.chunk_id for r in results[:3]]))
    print("\n[检索] 逐题证据位次（最终 top-5）：")
    for qid, rank, top3 in rows:
        mark = "✅" if rank else "❌"
        print(f"   Q{qid:>4} {mark} rank={rank}  top3={top3}")
    rate = hits / len(golden)
    print(f"[检索] 最终 top-5 命中率 = {hits}/{len(golden)} = {rate:.0%}")
    assert rate >= 0.90, f"最终 top-5 证据命中率应 ≥90%，实际 {hits}/{len(golden)}（{rate:.0%}）"
    assert hits >= 9, f"证据进最终 top-5 的题数应 ≥9，实际 {hits}"


def test_result_items_expose_required_fields(retriever, query_understanding, golden):
    """每条检索结果必须暴露 chunk_id/page/score/content/type。"""
    _, _, results = _search(retriever, query_understanding, golden[0].question, top_k=5)
    assert results, "检索结果为空"
    for item in results:
        assert item.chunk.chunk_id.startswith("c"), f"chunk_id 非法: {item.chunk.chunk_id}"
        assert 1 <= item.page <= 548, f"页码越界: {item.page}"
        assert isinstance(item.score, float), f"score 非浮点: {item.score!r}"
        assert item.content and isinstance(item.content, str), "content 为空"
        assert item.type in ("text", "table"), f"type 非法: {item.type}"
        assert item.source in ("vector", "bm25", "hybrid", "table"), f"source 非法: {item.source}"
    scores = [it.score for it in results]
    assert scores == sorted(scores, reverse=True), f"最终结果未按分数降序: {scores}"


@pytest.mark.slow
def test_rank_distribution_and_rerank_effect(retriever, query_understanding, golden, evidence_hit_chunks):
    """**排序质量**：打印融合池位次 vs 最终位次，并断言重排不使命中退化。

    位次口径：
    - 重排前位次 = 候选池按 ``score``（融合分）降序时的位次；
    - 重排后位次 = 最终返回列表中的位次。
    两者都由公开 API 得到，不使用 ``selftest_retrieval_ranks.json`` 的哨兵语义。
    """
    wide = retriever.health()
    pool_n = wide.get("vector_store", {}).get("count")
    print(f"\n[排序] 索引块数={pool_n}")

    pre_hits = post_hits = 0
    changed = 0
    print("[排序] 逐题 重排前→重排后（证据位次）：")
    for item in golden:
        hit_ids = evidence_hit_chunks[item.id]
        _, _, final = _search(retriever, query_understanding, item.question, top_k=5)
        _, _, pooled = _search(retriever, query_understanding, item.question, top_k=20)
        pre_sorted = sorted(pooled, key=lambda it: it.score, reverse=True)
        pre_rank = _rank_of(pre_sorted, hit_ids)
        post_rank = _rank_of(final, hit_ids)
        pre_hits += pre_rank is not None and pre_rank <= 5
        post_hits += post_rank is not None and post_rank <= 5
        if pre_rank != post_rank:
            changed += 1
        print(f"   Q{item.id:>4} 重排前={pre_rank} → 重排后={post_rank}")
    print(f"[排序] top-5 命中：重排前 {pre_hits}/10 → 重排后 {post_hits}/10；位次发生变化的题数={changed}")
    assert post_hits >= pre_hits, (
        f"重排后 top-5 命中数（{post_hits}）不得低于重排前（{pre_hits}）"
    )
    assert post_hits >= 9, f"重排后 top-5 命中数应 ≥9，实际 {post_hits}"


def test_bm25_numeric_tokenization_reality(settings):
    """**BM25 数值型问题根因（只报告、不断言实现内部形态）**。

    实测现状（T3 独立取证）：
    - ``tokenize("6,464.51")`` → ``['6','464','51']``：``PUNCT_PATTERN=[^\\w\\u4e00-\\u9fff]+``
      把逗号与小数点都换成空格，金额被打成碎片；
    - ``STOPWORDS`` 含 ``万 亿 元 %``，单位被丢弃；
    - **10 个工单问句一个数字都没有**，故词法通道的精确数值信号从未被使用。

    本用例只断言「工单问句不含数字」这一**可观测事实**（它解释了为何需要融合/加权补足），
    不断言 ``tokenize`` 的输出形态（那会把测试绑死在实现上）。
    """
    from app.core.text_utils import tokenize

    os_tokens = tokenize("6,464.51")
    print(f"\n[BM25] tokenize('6,464.51') = {os_tokens}")
    print(f"[BM25] tokenize('5,520.00 万元') = {tokenize('5,520.00 万元')}")
    assert tokenize("6,464.51"), "金额串分词结果不应为空"
    assert len(os_tokens) > 1, (
        "金额串若被当作单一 token 保留，本用例的前提（碎片化）已改变，需同步更新说明"
    )


def test_questions_contain_no_digits(settings, golden):
    """10 个工单问句均不含数字——词法通道无法直接匹配金额（实测事实）。"""
    with_digits = [item.id for item in golden if any(ch.isdigit() for ch in item.question)]
    assert not with_digits, f"以下问句含数字，需重新评估数值型结论: {with_digits}"
    print("\n[BM25] 已确认 10 个工单问句均不含数字（数值信号未被查询侧使用）")


def test_bm25_retrieves_evidence_for_numeric_questions(retriever, golden, evidence_hit_chunks):
    """BM25 单路必须能把证据召回进候选池；并如实打印逐题位次。

    契约（可复现、独立于实现）：
    - **精确金额串作为查询**时，证据块必须进 top-10（实测 `6,464.51`→#2、
      `18,780.67`→#2、`14,414.16`→#5、`4,627.14`→#6）；
    - 工单**问句**（不含数字）走 BM25 单路时，证据块须进 top-50（实测 Q33=#5、Q260=#22）。

    诚实标注：Q260 的 BM25 单路位次为 **#22，未进 top-20**——数值信号未被词法通道利用，
    该题依赖向量通道与融合/多值补偿。此处**不**断言一个设计未承诺的 top-20 阈值，
    而是把位次如实打印，供 T7 引用为「排序层」的量化依据。
    """
    from app.core.bm25_index import BM25Index

    index_dir = Path(retriever.health()["vector_store"]["index_dir"])
    bm = BM25Index.load(index_dir / "bm25_index.pkl")
    bm_size = bm.size() if callable(bm.size) else bm.size
    assert bm_size > 0, "BM25 索引为空"

    # ① 精确金额串查询（行为契约）
    amounts = ["6,464.51", "18,780.67", "14,414.16", "4,627.14"]
    target = evidence_hit_chunks[260]
    for amt in amounts:
        ids = [cid for cid, _ in bm.search(amt, top_k=10)]
        assert any(cid in target for cid in ids), (
            f"金额查询 {amt!r} 未能把 Q260 的证据块召回进 BM25 top-10"
        )
    print(f"\n[BM25] 精确金额串查询已把证据召回进 top-10：{amounts}")

    # ② 工单问句（诊断性打印 + 弱契约）
    print("[BM25] 10 题问句的 BM25 单路证据位次：")
    weak = []
    for item in golden:
        hit_ids = evidence_hit_chunks[item.id]
        ids = [cid for cid, _ in bm.search(item.question, top_k=50)]
        rank = next((i for i, cid in enumerate(ids, start=1) if cid in hit_ids), None)
        print(f"   Q{item.id:>4} BM25 rank={rank}")
        if rank is None:
            weak.append(item.id)
    assert not weak, f"以下题目在 BM25 top-50 中找不到证据块: {weak}（词法通道失效）"


def test_retrieval_debug_snapshot_exposes_candidates(retriever, query_understanding, golden):
    """检索调试快照必须可读，供日志/追踪使用（工单「检索片段全记录」）。"""
    _search(retriever, query_understanding, golden[0].question, top_k=5)
    snap = retriever.debug_snapshot().as_dict()
    assert isinstance(snap, dict) and snap, "debug_snapshot 为空"
    print(f"\n[检索调试] snapshot 字段: {sorted(snap.keys())}")
    for key in ("merged_hits", "rerank_mode"):
        assert key in snap, f"debug_snapshot 缺字段 {key}: {sorted(snap.keys())}"
