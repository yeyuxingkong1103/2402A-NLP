# -*- coding: utf-8 -*-
"""multi_turn 题目的执行器：真实多轮链路 + 短期记忆回写 + 会话清理（批次 23）。

为什么单独成文件：`run_eval.py` 已 600+ 行（超出项目"单文件 ≤300 行"约定，见
`docs/目录与命名约定.md` §3.4），本模块只装"多轮执行"这一件事。
判定原语 / 汇总 / 渲染在 `evaluation/multi_turn_grading.py`（纯函数，可单测）。

与 `followup_rewrite` 的本质区别（这才是新增 multi_turn 的原因）：
`followup_rewrite` 的 `context` 只是一段前缀字符串，且**评测器从不回写短期记忆**
（`short_term_memory.append_message` 的唯一调用方是 SSE 接口的 `persist_turn`），
改写器读到的上文永远是空的 —— 该类题实际只验"追问能不能检索到"。
本模块按真实顺序连跑 `turns`，**每轮结束后回写短期记忆**
（等价 `persist_turn` 的 Redis 写侧：user + assistant 两条，窗口 20 条与生产一致，
**不写 MySQL**），下一轮的改写器才真的拿得到上文。

依赖注入：`retry` / `citation_audit` / `refused_by_text` / `golden_rank` 四个判定原语
由调用方（`run_eval.py`）传入 —— 既不让两个模块互相 import（循环），也不复制判定逻辑
（口径只能有一处实现，改口径必须两处同时改，否则评测结果不可比）。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from multi_turn_grading import (
    EVAL_USER_ID,
    cleanup_eval_session,
    context_carryover,
    eval_session_id,
    merge_citation_audits,
    resolve_session_store,
    rewrite_hit,
)


def run_multi_turn_item(
    item: dict[str, Any],
    *,
    chat_service: Any,
    retrieval_service: Any,
    retrieve_top_n: int,
    retry: Callable[..., Any],
    citation_audit: Callable[[str, int], dict[str, Any]],
    refused_by_text: Callable[[str], bool],
    golden_rank: Callable[[list[Any], list[dict[str, str]]], int | None],
    clean_result: Callable[[Any], Any] | None = None,
    writeback: bool = True,
) -> dict[str, Any]:
    """按真实多轮顺序跑完整题，返回与既有题型同构的 detail（供 summarize 复用）。

    每轮顺序与生产链路一致：
    1. `retrieval_service.retrieve(...)`（含改写，改写记录挂在结果上）→ 拿 golden 排名；
    2. `chat_service.chat(...)`（生产默认 top5 上下文 + 真实 LLM）；
    3. 回写短期记忆（`writeback=True` 时），供下一轮改写读取。

    会话清理：**跑前先清一次**（清掉上次异常中断留下的残留），
    跑完在 `finally` 里再清一次 —— 中途抛异常也必须删，绝不留 `eval_<id>` 会话。

    参数：
    - item: 评测集条目（需含 id / turns，rewrite_expect 可选）
    - chat_service / retrieval_service / retrieve_top_n: 与 run_eval 主链路同款
    - retry: run_eval 的退避重试包装（上游偶发挂起）
    - citation_audit / refused_by_text / golden_rank: run_eval 的判定原语（口径单一来源）
    - writeback: False 时**不回写**短期记忆 —— 反向对照用，验证"改写判定确实依赖上下文"
    """
    turns: list[dict[str, Any]] = item.get("turns") or []
    expect = item.get("rewrite_expect") or None
    user_id = EVAL_USER_ID
    session_id = eval_session_id(item["id"])
    store = resolve_session_store(retrieval_service)

    detail: dict[str, Any] = {
        "id": item["id"],
        "type": item["type"],
        # 多轮题没有单一 question：拼成一行供报告/worst_samples 显示
        "question": " → ".join(str(turn.get("question", "")) for turn in turns),
        "turns_count": len(turns),
        "context_writeback": bool(writeback),
        "session_id": session_id,
        "errors": [],
        "turns": [],
    }
    if not turns:
        detail["errors"].append("multi_turn_has_no_turns")
        return detail

    try:
        # 跑前清残：上一次异常中断（Ctrl+C / 进程被杀）可能留下 eval_<id> 会话
        detail["session_cleanup_before"] = cleanup_eval_session(store, user_id, session_id)
        if writeback and store is None:
            detail["errors"].append("writeback_skipped: short_term_memory_unavailable")

        previous_question: str | None = None
        for index, turn in enumerate(turns, start=1):
            question = str(turn.get("question", ""))
            turn_detail: dict[str, Any] = {"turn": index, "question": question}

            # 第一步：检索（top10，供 Recall@5 / MRR@10；改写记录就在结果上）
            start = time.time()
            try:
                retrieval = retry(
                    lambda: (
                        clean_result(
                            retrieval_service.retrieve(
                                question,
                                rerank_top_n=retrieve_top_n,
                                jurisdiction="中国大陆",
                                user_id=user_id,
                                session_id=session_id,
                            )
                        )
                        if clean_result
                        else retrieval_service.retrieve(
                            question,
                            rerank_top_n=retrieve_top_n,
                            jurisdiction="中国大陆",
                            user_id=user_id,
                            session_id=session_id,
                        )
                    ),
                    attempts=5,
                )
            except Exception as error:  # noqa: BLE001 - 单轮失败不丢整题，记错后继续下一轮
                turn_detail["errors"] = [f"retrieval_failed: {type(error).__name__}: {error}"]
                turn_detail["retrieval_seconds"] = round(time.time() - start, 2)
                detail["turns"].append(turn_detail)
                detail["errors"].append(
                    f"turn{index}_retrieval_failed: {type(error).__name__}"
                )
                continue
            turn_detail["retrieval_seconds"] = round(time.time() - start, 2)

            articles = retrieval.articles
            turn_detail["retrieved"] = [
                {
                    "rank": rank,
                    "law": article.document_title,
                    "article": article.article_number,
                    "score": round(article.recall_score, 4),
                }
                for rank, article in enumerate(articles[:10], start=1)
            ]
            turn_detail["retrieval_stats"] = retrieval.stats

            # 改写记录（这一轮的真实改写结果；含 rewritten_query 原文）
            rewrite = retrieval.query_rewrite
            turn_detail["rewrite"] = {
                "changed": bool(rewrite.changed) if rewrite is not None else None,
                "original_query": rewrite.original_query if rewrite is not None else None,
                "rewritten_query": rewrite.rewritten_query if rewrite is not None else None,
                "reasons": list(rewrite.reasons) if rewrite is not None else [],
            }
            # 上下文是否真的接通（只记录不判定；第 1 轮没有上文，跳过）
            if index > 1:
                turn_detail["context_carryover"] = context_carryover(
                    rewrite.rewritten_query if rewrite is not None else None, previous_question
                )
            # 只对 rewrite_expect 指定的轮次做断言
            turn_detail["rewrite_hit"] = (
                rewrite_hit(rewrite, expect) if expect and expect.get("turn") == index else None
            )

            golden = turn.get("golden") or []
            if golden:
                rank = golden_rank(articles, golden)
                turn_detail["golden"] = golden
                turn_detail["golden_rank"] = rank
                turn_detail["hit_at_5"] = bool(rank and rank <= 5)
                turn_detail["mrr10"] = round(1.0 / rank, 4) if rank and rank <= 10 else 0.0

            # 第二步：真实问答（生产默认 top5 上下文）
            start = time.time()
            try:
                result = retry(
                    lambda: (
                        clean_result(
                            chat_service.chat(
                                question,
                                rerank_top_n=5,
                                jurisdiction="中国大陆",
                                user_id=user_id,
                                session_id=session_id,
                            )
                        )
                        if clean_result
                        else chat_service.chat(
                            question,
                            rerank_top_n=5,
                            jurisdiction="中国大陆",
                            user_id=user_id,
                            session_id=session_id,
                        )
                    ),
                    attempts=4,
                )
            except Exception as error:  # noqa: BLE001
                turn_detail["errors"] = [f"chat_failed: {type(error).__name__}: {error}"]
                turn_detail["answer_seconds"] = round(time.time() - start, 2)
                detail["turns"].append(turn_detail)
                detail["errors"].append(
                    f"turn{index}_chat_failed: {type(error).__name__}"
                )
                continue
            turn_detail["answer_seconds"] = round(time.time() - start, 2)

            answer = result.answer or ""
            sources = list(result.sources or [])
            turn_detail["answer"] = answer
            turn_detail["refused_flag"] = result.refused
            turn_detail["guardrails"] = list(result.guardrail_applied)
            turn_detail["source_count"] = len(sources)
            turn_detail["sources"] = [
                {"law": source.get("law_name"), "article": source.get("article_number")}
                for source in sources
            ]
            turn_detail["context_excerpts"] = result.context_excerpts or []
            turn_detail["citation_audit"] = citation_audit(answer, len(sources))
            turn_detail["refusal_by_text"] = refused_by_text(answer)

            # 第三步：回写短期记忆（等价 persist_turn 的 Redis 写侧；不写 MySQL）
            if writeback and store is not None:
                try:
                    store.append_message(user_id, session_id, {"role": "user", "content": question})
                    store.append_message(
                        user_id, session_id, {"role": "assistant", "content": answer}
                    )
                    # 回写后核对窗口，证明"下一轮能读到的上文"与生产一致
                    turn_detail["memory_messages_after"] = len(
                        store.read_messages(user_id, session_id)
                    )
                except Exception as error:  # noqa: BLE001 - 回写失败只记录，本轮成绩仍有效
                    turn_detail["writeback_error"] = f"{type(error).__name__}: {error}"

            previous_question = question
            detail["turns"].append(turn_detail)
    finally:
        # 无论正常结束、continue 跳过、还是抛异常，都必须清掉评测会话
        detail["session_cleanup_after"] = cleanup_eval_session(store, user_id, session_id)

    _rollup(detail)
    if any("RerankerFallbackError" in error for error in detail.get("errors", [])):
        raise RuntimeError("RerankerFallbackError: multi-turn item requires retry")
    return detail


def _rollup(detail: dict[str, Any]) -> None:
    """把逐轮明细汇总成整题字段，让 summarize / worst_samples 不必区分题型。

    - `golden` / `golden_rank` / `hit_at_5` / `mrr10` 取**最后一个有 golden 的轮次**
      （本项目多轮题都是 2 轮，即第 2 轮 —— 指标口径里的 turn2_hit_at_5）；
    - `citation_audit` / `refusal_by_text` 取整题合并口径（不丢前几轮的引用）；
    - `answer` / `retrieved` / `context_excerpts` 取最后一轮（报告展示用）。
    """
    executed = detail["turns"]
    graded = [turn for turn in executed if turn.get("golden_rank") is not None or turn.get("golden")]
    detail["golden"] = graded[-1].get("golden") if graded else []
    detail["golden_rank"] = graded[-1].get("golden_rank") if graded else None
    detail["hit_at_5"] = graded[-1].get("hit_at_5") if graded else None
    detail["mrr10"] = graded[-1].get("mrr10", 0.0) if graded else 0.0
    detail["turn_hit_at_5"] = graded[-1].get("hit_at_5") if graded else None
    # 断言轮次可能不是最后一轮（turn 参数），按 rewrite_expect.turn 取
    asserts = [turn for turn in executed if turn.get("rewrite_hit") is not None]
    detail["rewrite_hit"] = asserts[-1]["rewrite_hit"] if asserts else None
    detail["rewrite_turn"] = asserts[-1]["turn"] if asserts else None
    detail["rewrite"] = asserts[-1].get("rewrite") if asserts else None
    carryovers = [
        turn["context_carryover"] for turn in executed if turn.get("context_carryover") is not None
    ]
    detail["context_carryover"] = carryovers[-1] if carryovers else None
    detail["expect_refusal"] = False

    audits = [turn["citation_audit"] for turn in executed if turn.get("citation_audit")]
    detail["citation_audit"] = merge_citation_audits(audits) if audits else None
    detail["refusal_by_text"] = bool(executed) and all(
        turn.get("refusal_by_text") for turn in executed if turn.get("citation_audit")
    )
    last = executed[-1] if executed else {}
    detail["answer"] = last.get("answer", "")
    detail["retrieved"] = last.get("retrieved", [])
    detail["context_excerpts"] = last.get("context_excerpts", [])
    detail["source_count"] = last.get("source_count", 0)
    detail["sources"] = last.get("sources", [])
    detail["guardrails"] = last.get("guardrails", [])
    # 整题耗时由 run_eval 主循环填（elapsed_seconds），这里只保证键存在
    detail.setdefault("elapsed_seconds", None)
