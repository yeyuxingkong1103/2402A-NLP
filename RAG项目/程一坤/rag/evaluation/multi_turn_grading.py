# -*- coding: utf-8 -*-
"""multi_turn 的判定原语、汇总与报告渲染（批次 23）。

按项目"单文件 ≤300 行、一个文件一个主职责"（`docs/目录与命名约定.md` §3.4）拆分：
本模块只放**纯函数**（无 I/O、无服务依赖，可直接单测），执行循环在
`evaluation/multi_turn_runner.py`，接线在 `evaluation/run_eval.py`。

判定口径的单一来源：`rewrite_hit` / `context_carryover` / `merge_citation_audits` /
`summarize_multi_turn` 只在这里实现一次，`data/evaluation/_schema.md` 的表述与之一字对应 ——
改口径必须同时改这两处，否则评测结果不可比。
"""

from __future__ import annotations

import re
from typing import Any

# 评测专用 user_id：与生产用户空间天然隔离，便于按前缀排查残留
EVAL_USER_ID = "eval_runner"
# 短期记忆窗口：必须与生产装配处一致 —— `app/api/chat_persistence.SHORT_TERM_MAX_MESSAGES`
# 与 `retrieval/assembly.build_default_retrieval_service` 的 `max_messages`。三处一致性由
# `backend/tests/test_short_term_window_binding.py` 强制。
# 为何不用 import 绑定：本模块由 `python -m evaluation.run_eval` 在被导入时，
# backend 尚未进 sys.path（由 prepare_env 运行时插入），顶层 `from app.*` 会直接
# ModuleNotFoundError，故改为测试强制。
SESSION_WINDOW = 20


def eval_session_id(item_id: str) -> str:
    """返回该题的评测会话 id（与既有 followup 路径同构：`eval_<题号>`）。"""
    return f"eval_{item_id}"


def resolve_session_store(retrieval_service: Any) -> Any | None:
    """取检索服务持有的短期记忆存储（非 Redis 环境下可能是 None）。"""
    return getattr(retrieval_service, "short_term_memory", None)


def cleanup_eval_session(short_term_memory: Any | None, user_id: str, session_id: str) -> str | None:
    """删除评测会话的 Redis 短期记忆，返回错误描述（None = 清理成功）。

    为什么返回而不是抛出：清理属于收尾动作，失败不能覆盖"本题成绩已拿到"这个事实；
    调用方把它记进 `detail["session_cleanup_before/after"]`，便于发现"评测残留会话"这类隐患。
    DEL 不存在的 key 是 no-op，因此本函数天然幂等，可重复调用。
    """
    if short_term_memory is None:
        return "short_term_memory_unavailable"
    try:
        short_term_memory.delete_session_memory(user_id, session_id)
        return None
    except Exception as error:  # noqa: BLE001 - 清理失败只记录，不影响成绩
        return f"{type(error).__name__}: {error}"


def rewrite_hit(rewrite: Any | None, expect: dict[str, Any] | None) -> bool | None:
    """判定"该轮改写是否符合预期"（None = 本题没有对该轮的断言）。

    三条件同时成立才算命中：
    1. `changed` 与 `expect_changed` 一致（默认期望 True）；
    2. 改写后查询包含 `must_contain_any` 里任一锚点词（空数组视为不考察）；
    3. 改写后查询**不含** `must_not_contain` 里任何词。
    """
    if rewrite is None or not expect:
        return None
    rewritten = rewrite.rewritten_query or ""
    changed_ok = bool(rewrite.changed) == bool(expect.get("expect_changed", True))
    anchors = expect.get("must_contain_any") or []
    anchor_ok = (not anchors) or any(anchor in rewritten for anchor in anchors)
    forbidden = expect.get("must_not_contain") or []
    forbidden_ok = not any(word in rewritten for word in forbidden)
    return changed_ok and anchor_ok and forbidden_ok


def context_carryover(rewritten_query: str | None, previous_question: str | None) -> bool | None:
    """上一轮问题正文是否真的被带进了本轮改写后查询（只记录不判定）。

    判定方式：把两边都压掉空白与标点后做子串包含 —— 改写器把"上一轮问题去标点"
    当主题词注入，命中即证明上下文确实接通（与"是否命中主题锚点词"是两个问题：
    前者验链路，后者验质量）。
    """
    if not rewritten_query or not previous_question:
        return None

    def squeeze(text: str) -> str:
        return re.sub(r"[\s，。！？?；;：:、,.!]+", "", text)

    previous = squeeze(previous_question)
    return bool(previous) and previous in squeeze(rewritten_query)


def merge_citation_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """把多轮各自的引用审计合并成整题一份（口径与单轮版逐字一致）。"""
    citations: list[int] = []
    invalid: list[int] = []
    total = 0
    for audit in audits:
        citations.extend(audit.get("citations") or [])
        invalid.extend(audit.get("invalid") or [])
        total += int(audit.get("total") or 0)
    return {
        "citations": citations,
        "invalid": invalid,
        "total": total,
        "correct": total - len(invalid),
        # 整题口径：**每一轮**都没有引用编号才算"无引用回答"
        "no_citation": total == 0,
    }


def summarize_multi_turn(details: list[dict[str, Any]]) -> dict[str, Any]:
    """三个 multi_turn 指标（口径见 data/evaluation/_schema.md，两处必须一致）。

    分母说明：`rewrite_rate` 只对**有 rewrite_expect** 的题计分（没有断言的题不该拉低
    或抬高这个数）；`turn2_hit_at_5` 对**所有 multi_turn 题**计分。
    """
    items = [detail for detail in details if detail.get("type") == "multi_turn"]
    asserted = [detail for detail in items if detail.get("rewrite_hit") is not None]
    graded = [detail for detail in items if detail.get("turn_hit_at_5") is not None]
    carryover_recorded = [detail for detail in items if detail.get("context_carryover") is not None]
    return {
        "count": len(items),
        "asserted_count": len(asserted),
        "graded_count": len(graded),
        "rewrite_rate": (
            round(sum(1 for detail in asserted if detail["rewrite_hit"]) / len(asserted), 4)
            if asserted
            else None
        ),
        "turn2_hit_at_5": (
            round(sum(1 for detail in graded if detail["turn_hit_at_5"]) / len(graded), 4)
            if graded
            else None
        ),
        "rewrite_rate_denominator": len(asserted),
        "turn2_hit_at_5_denominator": len(graded),
        "context_carryover_true": sum(1 for detail in carryover_recorded if detail["context_carryover"]),
        "context_carryover_recorded": len(carryover_recorded),
        "context_writeback": sorted({bool(detail.get("context_writeback")) for detail in items}),
        "per_item": [
            {
                "id": detail["id"],
                "rewrite_hit": detail.get("rewrite_hit"),
                "rewrite_turn": detail.get("rewrite_turn"),
                "changed": (detail.get("rewrite") or {}).get("changed"),
                "rewritten_query": (detail.get("rewrite") or {}).get("rewritten_query"),
                "context_carryover": detail.get("context_carryover"),
                "turn2_golden_rank": detail.get("golden_rank"),
                "turn2_hit_at_5": detail.get("turn_hit_at_5"),
            }
            for detail in items
        ],
    }


def render_multi_turn_section(summary: dict[str, Any]) -> list[str]:
    """把 multi_turn 指标渲染成 Markdown 行（无该类题时返回空列表，老报告格式不变）。

    放在本模块而不是 run_eval：渲染与指标口径同源，改一处不会漏改另一处。
    """
    multi = summary.get("multi_turn") or {}
    if not multi.get("count"):
        return []
    writeback = "开启" if multi.get("context_writeback") == [True] else "关闭/混合"
    rate, turn2 = multi.get("rewrite_rate"), multi.get("turn2_hit_at_5")
    rate_text = "—" if rate is None else f"{rate:.4f}"
    turn2_text = "—" if turn2 is None else f"{turn2:.4f}"
    lines = [
        "",
        "## 二之二、multi_turn 指标（真实多轮链路）",
        "",
        f"- 题数：{multi['count']}（其中 {multi['asserted_count']} 条带改写断言）；"
        f"短期记忆回写：**{writeback}**",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 改写命中率 | **{rate_text}** | 分母 {multi['rewrite_rate_denominator']}："
        "changed 与期望一致 **且** 命中主题锚点词 |",
        f"| 第 2 轮 golden 进 top5 | **{turn2_text}** | 分母 {multi['turn2_hit_at_5_denominator']}："
        "改写到底有没有用的落点 |",
        f"| 上下文接通（只记录） | {multi['context_carryover_true']}/"
        f"{multi['context_carryover_recorded']} | 改写后查询是否带上上一轮问题正文 |",
        "",
        "| 题号 | 改写 | 改写后查询 | 上下文接通 | 第2轮 golden 排名 | 第2轮 top5 |",
        "|---|---|---|---|---|---|",
    ]
    for row in multi["per_item"]:
        rewritten = (row["rewritten_query"] or "").replace("|", "\\|")
        mark = "✔" if row["rewrite_hit"] else ("✗" if row["rewrite_hit"] is False else "—")
        lines.append(
            f"| {row['id']} | {mark} | {rewritten} "
            f"| {'✔' if row['context_carryover'] else '—'} "
            f"| {row['turn2_golden_rank'] if row['turn2_golden_rank'] else '未召回'} "
            f"| {'✔' if row['turn2_hit_at_5'] else '✗'} |"
        )
    return lines
