# -*- coding: utf-8 -*-
"""指标汇总（四个指标 + 分类型 + 最差样本），批次 24 自 run_eval.py 拆出。

为什么单独成文件：见 `legal_matching.py` 顶部说明（守"单文件 ≤300 行"）。
本模块**零逻辑改动**：`summarize` 与 `worst_samples` 逐字搬移，函数签名不变。

为什么汇总与渲染要分开：`summarize` 是"数字口径"的唯一实现，
不回归核验会**复用同一个 summarize** 去重算子集指标（口径同源，避免手写算法引入偏差）；
渲染只负责把结果排成 Markdown（见 `render_report.py`）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from multi_turn_grading import summarize_multi_turn  # noqa: E402


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [d for d in details if d.get("golden") and d.get("hit_at_5") is not None]
    answerable = [d for d in details if not d.get("expect_refusal")]
    refusals = [d for d in details if d.get("expect_refusal")]

    recall_at_5 = (
        sum(1 for d in graded if d.get("hit_at_5")) / len(graded) if graded else 0.0
    )
    mrr_at_10 = (
        sum(d.get("mrr10", 0.0) for d in graded) / len(graded) if graded else 0.0
    )

    answered = [d for d in answerable if d.get("citation_audit")]
    total_citations = sum(d["citation_audit"]["total"] for d in answered)
    invalid_citations = sum(len(d["citation_audit"]["invalid"]) for d in answered)
    citation_accuracy = (
        1 - invalid_citations / total_citations if total_citations else 1.0
    )
    no_citation_count = sum(1 for d in answered if d["citation_audit"]["no_citation"])

    refusal_ok = sum(1 for d in refusals if d.get("refusal_by_text"))
    refusal_accuracy = refusal_ok / len(refusals) if refusals else 0.0

    answered_non_refusal = [d for d in answerable if d.get("citation_audit")]
    false_refusal = sum(1 for d in answered_non_refusal if d.get("refusal_by_text"))
    false_refusal_rate = (
        false_refusal / len(answered_non_refusal) if answered_non_refusal else 0.0
    )

    by_type: dict[str, dict[str, Any]] = {}
    for detail in details:
        bucket = by_type.setdefault(
            detail["type"], {"count": 0, "recall_at_5": [], "mrr10": [], "refusal_ok": 0, "refusal_total": 0}
        )
        bucket["count"] += 1
        if detail.get("hit_at_5") is not None:
            bucket["recall_at_5"].append(1 if detail.get("hit_at_5") else 0)
            bucket["mrr10"].append(detail.get("mrr10", 0.0))
        if detail.get("expect_refusal") and detail.get("citation_audit"):
            bucket["refusal_total"] += 1
            bucket["refusal_ok"] += 1 if detail.get("refusal_by_text") else 0
    for bucket in by_type.values():
        bucket["recall_at_5"] = (
            sum(bucket["recall_at_5"]) / len(bucket["recall_at_5"]) if bucket["recall_at_5"] else None
        )
        bucket["mrr10"] = (
            sum(bucket["mrr10"]) / len(bucket["mrr10"]) if bucket["mrr10"] else None
        )
        bucket["refusal_accuracy"] = (
            bucket["refusal_ok"] / bucket["refusal_total"] if bucket["refusal_total"] else None
        )

    as_of_violations = [
        {"id": d["id"], "violation": d["absent_violation"]}
        for d in details
        if d.get("absent_violation")
    ]

    return {
        "recall_at_5": round(recall_at_5, 4),
        "mrr_at_10": round(mrr_at_10, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "refusal_accuracy": round(refusal_accuracy, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "counts": {
            "total": len(details),
            "graded": len(graded),
            "refusal_questions": len(refusals),
            "answered": len(answered),
            "total_citations": total_citations,
            "invalid_citations": invalid_citations,
            "answers_without_citation": no_citation_count,
            "as_of_violations": len(as_of_violations),
        },
        "as_of_violations": as_of_violations,
        "by_type": by_type,
        # 批次 23：multi_turn 三个指标（口径见 data/evaluation/_schema.md）
        "multi_turn": summarize_multi_turn(details),
    }


def worst_samples(details: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """最差样本：优先"该命中却没命中"，其次"拒答题没拒"，再次"引用出错"。"""

    def penalty(detail: dict[str, Any]) -> tuple[int, int]:
        score = 0
        reason = []
        if detail.get("errors"):
            score += 100
            reason.append("执行异常：" + detail["errors"][0])
        if detail.get("expect_refusal") and not detail.get("refusal_by_text"):
            score += 50
            reason.append("拒答题被作答（拒答失败）")
        if detail.get("hit_at_5") is False:
            rank = detail.get("golden_rank")
            score += 20
            reason.append(f"golden 未进 top5（实际排名 {rank if rank else '未召回'}）")
        if detail.get("absent_violation"):
            score += 15
            reason.append(f"出现了不该出现的法源：{detail['absent_violation']}")
        audit = detail.get("citation_audit") or {}
        if audit.get("invalid"):
            score += 10
            reason.append(f"越界引用 {audit['invalid']}")
        if audit.get("no_citation") and not detail.get("expect_refusal"):
            score += 5
            reason.append("回答无任何引用")
        if not detail.get("expect_refusal") and detail.get("refusal_by_text"):
            score += 8
            reason.append("不应答却拒答（误拒）")
        detail["_penalty"] = score
        detail["_reason"] = reason
        return (-score, detail.get("golden_rank") or 999)

    ranked = sorted(details, key=penalty)
    return [d for d in ranked if d.get("_penalty", 0) > 0][:limit]
