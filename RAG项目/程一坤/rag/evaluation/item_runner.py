# -*- coding: utf-8 -*-
"""单题执行（非多轮题型）：检索 → 问答 → 判定，批次 24 自 run_eval.py 拆出。

为什么单独成文件：见 `legal_matching.py` 顶部说明（守"单文件 ≤300 行"）。
本模块**零逻辑改动**：`with_retry` 与 `run_item` 逐字搬移，函数签名不变。

与 `multi_turn_runner.py` 的分工：本模块跑"单问题"题型（direct_article / cross_law /
refusal / followup_rewrite / as_of_date / confusable），多轮题型走 `multi_turn_runner.py`。
两者对外的 detail 结构同构，由 `aggregate.py` 统一汇总。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# 本目录也要进 sys.path：legal_matching / answer_judging 是同级模块，
# 这样无论以 `python evaluation/run_eval.py` 还是 `import evaluation.item_runner`
# 的方式进入，都导入同一个模块对象（不会出现两份模块状态）。
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from answer_judging import citation_audit, refused_by_text, refusal_failure_reason  # noqa: E402
from legal_matching import absent_violation, first_golden_rank  # noqa: E402


class RerankerFallbackError(RuntimeError):
    """评测专用异常：本题没有使用真实 Reranker，结果不能进入干净基线。"""

    def __init__(self, error_type: str) -> None:
        super().__init__(f"Reranker fallback: {error_type}")
        self.error_type = error_type


def _record_api_failure(counter: dict[str, Any] | None, error_type: str) -> None:
    """记录外部 Reranker 失败次数；不记录普通业务或断言错误。"""
    if counter is None:
        return
    counter["total"] = int(counter.get("total", 0)) + 1
    by_type = counter.setdefault("by_type", {})
    by_type[error_type] = int(by_type.get(error_type, 0)) + 1


def _require_clean_result(result: Any, counter: dict[str, Any] | None) -> Any:
    """拒绝 RRF 降级结果，确保评测指标来自真实精排。"""
    stats = getattr(result, "stats", None) or getattr(result, "retrieval_stats", None) or {}
    if stats.get("rerank_fallback"):
        error_type = str(stats.get("rerank_error_type") or "unknown")
        _record_api_failure(counter, error_type)
        raise RerankerFallbackError(error_type)
    return result


def with_retry(func, *, attempts: int = 5, base_delay: float = 3.0, on_retry=None):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:  # noqa: BLE001 - 上游任意异常都要重试
            last_error = error
            if on_retry:
                on_retry(attempt, error)
            if attempt < attempts:
                time.sleep(base_delay * attempt)
    raise last_error  # type: ignore[misc]


def run_item(
    item: dict[str, Any],
    *,
    chat_service,
    retrieval_service,
    retrieve_top_n: int,
    api_failure_counter: dict[str, Any] | None = None,
    retry_attempts: int = 5,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "context": item.get("context"),
        "as_of_date": item.get("as_of_date"),
        "expect_refusal": bool(item.get("expect_refusal")),
        "golden": item.get("golden") or [],
        "errors": [],
    }
    session_id = f"eval_{item['id']}"
    user_id = "eval_runner"

    # 追问题：先用 context 走一遍同会话，让短期记忆里有一轮上文
    if item.get("context"):
        try:
            with_retry(
                lambda: _require_clean_result(
                    chat_service.chat(
                        item["context"], rerank_top_n=5, jurisdiction="中国大陆",
                        user_id=user_id, session_id=session_id,
                    ),
                    api_failure_counter,
                ),
                attempts=retry_attempts,
            )
        except Exception as error:  # noqa: BLE001
            detail["errors"].append(f"context_turn_failed: {type(error).__name__}")

    # 第一步：检索（top10，供 Recall@5 / MRR@10）
    t0 = time.time()
    try:
        retrieval = with_retry(
            lambda: _require_clean_result(
                retrieval_service.retrieve(
                    item["question"],
                    rerank_top_n=retrieve_top_n,
                    as_of_date=item.get("as_of_date"),
                    jurisdiction="中国大陆",
                    user_id=user_id,
                    session_id=session_id,
                ),
                api_failure_counter,
            ),
            attempts=retry_attempts,
        )
    except Exception as error:  # noqa: BLE001
        detail["errors"].append(f"retrieval_failed: {type(error).__name__}: {error}")
        detail["retrieval_seconds"] = round(time.time() - t0, 2)
        return detail
    detail["retrieval_seconds"] = round(time.time() - t0, 2)

    articles = retrieval.articles
    detail["retrieved"] = [
        {"rank": i, "law": a.document_title, "article": a.article_number, "score": round(a.recall_score, 4)}
        for i, a in enumerate(articles[:10], start=1)
    ]
    detail["retrieval_stats"] = retrieval.stats

    if item.get("golden"):
        rank = first_golden_rank(articles, item["golden"])
        detail["golden_rank"] = rank
        detail["hit_at_5"] = bool(rank and rank <= 5)
        detail["reciprocal_rank"] = round(1.0 / rank, 4) if rank and rank <= 10 else 0.0
        detail["mrr_applicable"] = rank is not None and rank <= 10
        hit_at_10 = rank is not None and rank <= 10
        detail["mrr10"] = round(1.0 / rank, 4) if hit_at_10 else 0.0
    if item.get("expect_absent"):
        detail["absent_violation"] = absent_violation(articles, item["expect_absent"])

    # 第二步：真实问答（生产默认 top5 上下文）
    t1 = time.time()
    try:
        result = with_retry(
            lambda: _require_clean_result(
                chat_service.chat(
                    item["question"], rerank_top_n=5,
                    as_of_date=item.get("as_of_date"),
                    jurisdiction="中国大陆", user_id=user_id, session_id=session_id,
                ),
                api_failure_counter,
            ),
            attempts=retry_attempts,
        )
    except Exception as error:  # noqa: BLE001
        detail["errors"].append(f"chat_failed: {type(error).__name__}: {error}")
        detail["answer_seconds"] = round(time.time() - t1, 2)
        return detail
    detail["answer_seconds"] = round(time.time() - t1, 2)

    answer = result.answer or ""
    detail["answer"] = answer
    detail["refused_flag"] = result.refused
    detail["guardrails"] = list(result.guardrail_applied)
    detail["source_count"] = len(result.sources or [])
    detail["sources"] = [
        {"law": s.get("law_name"), "article": s.get("article_number")} for s in (result.sources or [])
    ]
    # 注入提示词的法条原文（[n] 编号与回答一致），供 Faithfulness 打分
    detail["context_excerpts"] = result.context_excerpts or []
    detail["refusal_by_text"] = refused_by_text(answer)
    detail["refusal_failure_reason"] = refusal_failure_reason(answer)
    detail["citation_audit"] = citation_audit(answer, len(result.sources or []))
    return detail
