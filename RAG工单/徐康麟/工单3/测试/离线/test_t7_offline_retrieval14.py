# -*- coding: utf-8 -*-
"""离线级：14 题检索 top-5 命中（判据 = ``evidence_contains``）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本用例是三级套件里**唯一**需要本机 Ollama 的离线用例（检索要嵌入查询向量）：
本机环境事实（``部署/配置/环境事实.md``）明确 Ollama 0.35.0 常驻 ``127.0.0.1:11434``，
因此**服务不可用时判失败而不是跳过**（跳过会让「14 题命中」失去证据）。

判定口径（``设计/验收标准.md`` §2.2）：
    ``hit = evidence_contains(chunk.content, evidence, 0.90)`` 对 top-5 任一 chunk 成立；
    题 207 的 golden ``evidence`` 是合成串 → 用 ``evidence_verbatim``；
    **禁止**用「引用页 == evidence_pages」判定（会把 957 误计为命中）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions
from common.reports import write_report

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def test_14_questions_top5_hits(retriever: Any, runnable_items: list[Any],
                               ollama_status: dict[str, Any], page_counts: dict[str, int]) -> None:
    """14 题命中 ≥ 13/14（top-5），并逐题留痕（含命中块、页码、分数）。"""
    if not ollama_status.get("available"):
        pytest.fail(f"离线级的 14 题检索用例需要本机 Ollama 嵌入服务（{ollama_status.get('detail')}）："
                    f"服务不可用即判失败，不跳过（本机环境事实：Ollama 常驻 11434）")

    files = set(page_counts)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for item in runnable_items:
        result = retriever.retrieve(item.question, top_k=5)
        chunks = list(result.chunks)
        outcome = assertions.evidence_hit(chunks, evidence=item.evidence, verbatim=item.evidence_verbatim)
        strict = assertions.evidence_hit(chunks, evidence=item.evidence, prefer_verbatim=False)
        rows.append({
            "id": item.id, "corpus": item.corpus, "question": item.question,
            "hit": outcome.hit, "used_field": outcome.used_field,
            "strict_golden_evidence_hit": strict.hit,
            "matched_chunk_ids": outcome.matched_chunk_ids,
            "chunks": [{"chunk_id": c.chunk_id, "file_name": c.file_name, "page": c.page,
                        "type": c.type, "score": round(float(c.score), 6)} for c in chunks],
            "evidence_pages": item.evidence_pages,
            "file_filter_ok": all(c.file_name in files for c in chunks),
        })
        if not outcome.hit:
            problems.append(f"题 {item.id} top-5 未命中（{outcome.detail}）")
        if chunks and not all(c.file_name in files for c in chunks):
            problems.append(f"题 {item.id} 返回了不存在的文件名")

    write_report("offline_retrieval14", "14 题检索 top-5 命中（离线级）", rows,
                 extra={"criterion": "evidence_contains", "top_k": 5, "threshold_hits": 13})

    hits = sum(1 for row in rows if row["hit"])
    strict_hits = sum(1 for row in rows if row["strict_golden_evidence_hit"])
    assert len(rows) == 14, f"必须跑满 14 题（PDF2 已到位，无 skip），实测 {len(rows)}"
    assert hits >= 13, (f"top-5 命中 {hits}/14 < 13：\n" + "\n".join(problems)
                        + "\n逐题见 测试/留痕/offline_retrieval14.json")
    # 严格单证据口径只作记录：题 207 的 golden evidence 是合成串（基准缺陷），不得据此判失败
    assert strict_hits >= 13 or all(
        not row["strict_golden_evidence_hit"] for row in rows if row["id"] == 207), \
        f"严格口径命中 {strict_hits}/14 且差集不止题 207，需人工核查"


def test_hit_criterion_is_not_page_equality(runnable_items: list[Any]) -> None:
    """自证：命中判据只能来自 ``evidence_contains``，不得退化成「引用页 == evidence_pages」。

    用例做法：对**空块列表**做命中判定必须一律为 False（页码相等式判据在空块上会误判为真）。
    """
    for item in runnable_items[:5]:
        outcome = assertions.evidence_hit([], evidence=item.evidence, verbatim=item.evidence_verbatim)
        assert outcome.hit is False, f"题 {item.id} 在无任何候选块时被判命中 → 判据退化"
        assert outcome.criterion == "evidence_contains"
