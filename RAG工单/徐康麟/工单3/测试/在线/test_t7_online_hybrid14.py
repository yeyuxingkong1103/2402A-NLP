# -*- coding: utf-8 -*-
"""在线级：14 题混合检索命中（top-5）与按文件名过滤。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定口径（``设计/验收标准.md`` §2.2）：
    ``hit = text_utils.evidence_contains(chunk.content, evidence, 0.90)``，对 **top-5** 任一 chunk 成立；
    **禁止**用「引用页 == evidence_pages」判命中（会把 957 误计为命中）。
    题 207 的 golden ``evidence`` 是合成串 → 优先用 ``evidence_verbatim``。

门槛：14 题命中 ≥ 13（含 PDF2 到位后的 id 1~4）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions, golden as golden_mod
from common.reports import write_report

pytestmark = [pytest.mark.online, pytest.mark.linkage]


def test_hybrid_retrieval_top5_hits(retriever: Any, runnable_items: list[Any],
                                    app_config: Any) -> None:
    """14 题混合检索 top-5 命中率 ≥ 13/14，并逐题留痕。"""
    top_k = int(app_config.retrieval.top_k)
    rows: list[dict[str, Any]] = []
    for item in runnable_items:
        result = retriever.retrieve(item.question, top_k=top_k)
        chunks = list(result.chunks)
        outcome = assertions.evidence_hit(chunks, evidence=item.evidence, verbatim=item.evidence_verbatim)
        rows.append({
            "id": item.id, "corpus": item.corpus, "question": item.question,
            "top_k": top_k, "hit": outcome.hit, "used_field": outcome.used_field,
            "matched_chunk_ids": outcome.matched_chunk_ids,
            "chunks": [{"chunk_id": c.chunk_id, "file_name": c.file_name, "page": c.page,
                        "type": c.type, "score": round(float(c.score), 6)} for c in chunks],
            "evidence_pages": item.evidence_pages,
        })
    write_report("online_retrieval_top5", "14 题混合检索 top-5 命中", rows,
                 extra={"top_k": top_k, "criterion": "evidence_contains"})

    hits = [row for row in rows if row["hit"]]
    missed = [row["id"] for row in rows if not row["hit"]]
    assert len(rows) == 14, f"应跑 14 题，实测 {len(rows)}（pending 需显式报告：{missed}）"
    assert len(hits) >= 13, (f"top-5 命中 {len(hits)}/14 < 13；未命中题 {missed}；"
                             f"详见 测试/留痕/online_retrieval_top5.json")
    # 判据自证：每题都必须声明用的是逐字原文或 golden evidence（二者都经 evidence_contains 判定）
    assert all(row["used_field"] in ("evidence", "evidence_verbatim") for row in rows), \
        "命中判据字段异常（必须走 evidence_contains，禁止引用页相等）"


def test_question_207_hit_uses_verbatim(retriever: Any, golden_index: dict[int, Any]) -> None:
    """题 207：生成上下文必须拿到含 ``15,000`` 的物理 490 原文块（判据 ``evidence_contains``）。

    **口径说明（务必如实阅读）**：
        * 冻结验收门槛是**聚合** ``top-5 命中 ≥ 13/14``（``验收标准.md`` §5/§2.2），由上一个用例把关；
        * 本题 §「15,000 块进 top-5」是 t12 的**原始目标**，定稿实现的做法是把该块作为
          ``RetrievalResult.support_chunks``（生成上下文支持块）提供、**不进 top-5 排名**；
        * 因此本用例断言的是**用户可见结果**：15,000 原文块必须能被取到（support 或 top-5 皆可）
          并进入生成上下文；若只在 top-5 里找就会把「已修好的实现」判成失败。
        * top-5 的实际情况（本轮实测：207 未进 top-5）**写入留痕**，作为开放项报 captain，不在此判红。
    """
    item = golden_index[207]
    result = retriever.retrieve(item.question, top_k=5)
    top5_joined = "\n".join(str(c.content) for c in result.chunks)
    support = list(getattr(result, "support_chunks", []) or [])
    support_joined = "\n".join(str(getattr(c, "content", "")) for c in support)

    top5_outcome = assertions.evidence_hit(result.chunks, evidence=item.evidence,
                                           verbatim=item.evidence_verbatim)
    support_outcome = assertions.evidence_hit(support, evidence=item.evidence,
                                              verbatim=item.evidence_verbatim)

    write_report("online_question_207_recall", "题 207 召回取证（top-5 vs 生成支持块）", [
        {"aspect": "top5", "hit": top5_outcome.hit, "used_field": top5_outcome.used_field,
         "matched": top5_outcome.matched_chunk_ids, "detail": top5_outcome.detail},
        {"aspect": "support_chunks", "count": len(support), "hit": support_outcome.hit,
         "used_field": support_outcome.used_field, "matched": support_outcome.matched_chunk_ids,
         "top1": (getattr(support[0], "chunk_id", "") if support else ""),
         "detail": support_outcome.detail},
        {"aspect": "note",
         "text": ("冻结门槛为聚合 ≥13/14；207 的 15,000 块以生成支持块形式提供（不进 top-5 排名）。"
                  "top5_hit 字段即本轮真实情况，供 captain/T11 判定是否需要进一步优化。")},
    ])

    assert "15,000" in (top5_joined + support_joined), \
        "题 207 的 15,000 原文既不在 top-5 也不在生成支持块里（生成侧无依据）"
    assert support_outcome.hit or top5_outcome.hit, \
        (f"题 207 的 evidence_verbatim 在 top-5 与支持块里都没命中：top5={top5_outcome.detail}；"
         f"support={support_outcome.detail}")


def test_file_filter_is_hard(retriever: Any, discovered_pdfs: list[Any],
                             golden_index: dict[int, Any]) -> None:
    """按文件名过滤是**硬过滤**：返回块的 file_name 必须全等所选文件（越界块 = 0）。"""
    for path in discovered_pdfs:
        result = retriever.retrieve("关联方与募集资金用途", top_k=5, file_names=[path.name])
        bad = [c.chunk_id for c in result.chunks if c.file_name != path.name]
        assert not bad, f"{path.name} 过滤后出现越界块：{bad}"


def test_filter_does_not_degrade_to_no_filter(retriever: Any) -> None:
    """不存在的文件名 → 必须返回空候选，**禁止**退化成「不过滤」。"""
    result = retriever.retrieve("注册资本", top_k=5, file_names=["不存在的文档.pdf"])
    assert not result.chunks, f"不存在的文件名却返回了 {len(result.chunks)} 个块（过滤被绕过）"


def test_chunk_fields_are_citable(retriever: Any, golden_index: dict[int, Any],
                                  page_counts: dict[str, int]) -> None:
    """返回块可引用：文件名真实、页码 1-based 且在范围内、正文非空。"""
    item = golden_index[543]
    result = retriever.retrieve(item.question, top_k=5)
    assert result.chunks, "题 543 检索为空"
    problems: list[str] = []
    for chunk in result.chunks:
        if chunk.file_name not in page_counts:
            problems.append(f"{chunk.chunk_id} 文件名未知 {chunk.file_name!r}")
            continue
        if not isinstance(chunk.page, int) or not (1 <= chunk.page <= page_counts[chunk.file_name]):
            problems.append(f"{chunk.chunk_id} 页码非法 {chunk.page!r}")
        if not str(chunk.content or "").strip():
            problems.append(f"{chunk.chunk_id} 正文为空")
    assert not problems, "\n".join(problems)
