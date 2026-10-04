# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
多轮轮次的组装与写回：把「会话状态 → 改写 → history」这段逻辑收在一处。

【为什么单独成一个模块】API（`app/api/chat.py`）与评测脚本（`scripts/eval_multiturn.py`）
都要走同一段流程。若各写一份，"脚本绿了、接口没接上"这类静默失败就会漏网 ——
而那正是最该被自动化抓住的一类问题。所以这里做成**纯函数 + 显式传参**，
不持有全局信号量、不依赖 FastAPI，两边共用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.query_understanding import (
    QueryUnderstanding, intent_template_of, next_focus_entity)
from app.core.session import SessionStore, Turn


@dataclass
class TurnCtx:
    """一轮多轮问答的准备结果。"""

    session_id: str = ""
    turn_index: int = 0
    question: str = ""            # 用户原话（展示 + 语言判定用它）
    rewritten: str = ""           # 改写后的独立问题（检索/生成用它）
    method: str = "none"
    evidence: str = ""
    resolved_entity: str = ""
    carried_intent: str = ""
    focus_before: str = ""
    expired: bool = False
    rewrite_ms: int = 0
    history: list = field(default_factory=list)
    history_dropped: bool = False
    history_chars: int = 0


async def prepare_turn(store: SessionStore, qu: QueryUnderstanding, question: str,
                       session_id: str | None, *, budget_chars: int = 0) -> TurnCtx:
    """取会话状态 → 改写问题 → 组 history。

    【调用方必须在自己的并发闸门内调用它】改写若走 LLM 兜底会调 qwen3；
    两条并发推理会在 8GB 显存上互抢，把 `max_concurrency=1` 的意义作废。
    （`budget_chars` 先传 0，因为 context 还没算出来；拿到 context 后调用方再
    调一次 `store.history_messages(..., budget_chars=len(ctx))` 收紧。）
    """
    ctx = TurnCtx(question=question, rewritten=question)
    if not session_id:
        return ctx

    lookup = store.get_or_create(session_id)
    sess = lookup.session
    ctx.session_id = sess.id
    ctx.expired = lookup.expired
    ctx.turn_index = len(sess.turns) + 1

    prev = sess.turns[-1] if sess.turns else None
    ctx.focus_before = store.focus_entity(sess.id)
    template = store.intent_template(sess.id)

    t0 = time.perf_counter()
    rw = await qu.rewrite(question,
                          focus_entity=ctx.focus_before,
                          intent_template=template,
                          prev_question=prev.question if prev else "",
                          prev_rewritten=prev.rewritten if prev else "")
    ctx.rewrite_ms = int((time.perf_counter() - t0) * 1000)
    ctx.rewritten = rw.rewritten
    ctx.method = rw.method
    ctx.evidence = rw.evidence
    ctx.resolved_entity = rw.resolved_entity
    ctx.carried_intent = rw.carried_intent

    ctx.history, ctx.history_dropped = store.history_messages(
        sess.id, budget_chars=budget_chars)
    ctx.history_chars = sum(len(m["content"]) for m in ctx.history)
    return ctx


def refresh_history(store: SessionStore, ctx: TurnCtx, *, budget_chars: int) -> None:
    """拿到 context 之后收紧一次 history —— 两者共享 num_ctx 预算。"""
    if not ctx.session_id:
        return
    ctx.history, ctx.history_dropped = store.history_messages(
        ctx.session_id, budget_chars=budget_chars)
    ctx.history_chars = sum(len(m["content"]) for m in ctx.history)


def commit_turn(store: SessionStore, ctx: TurnCtx, answer: str, doc_key: str) -> str:
    """把本轮写回会话（供下一轮消解用）。返回本轮之后的焦点实体。"""
    if not ctx.session_id:
        return ctx.focus_before
    focus = next_focus_entity(resolved_entity=ctx.resolved_entity, doc_key=doc_key,
                              previous=ctx.focus_before)
    store.append(ctx.session_id, Turn(
        question=ctx.question, rewritten=ctx.rewritten, answer=answer,
        focus_entity=focus,
        intent_template=intent_template_of(ctx.rewritten),
        doc_key=doc_key,
    ))
    return focus


def turn_fields(ctx: TurnCtx) -> dict:
    """转成 `ChatResponse` / SSE meta 里的多轮字段。"""
    return {
        "session_id": ctx.session_id,
        "turn_index": ctx.turn_index,
        "rewritten_question": ctx.rewritten,
        "rewrite_method": ctx.method,
        "rewrite_evidence": ctx.evidence,
        "carried_intent": ctx.carried_intent,
        "session_expired": ctx.expired,
        "history_chars": ctx.history_chars,
        "history_dropped": ctx.history_dropped,
        "rewrite_ms": ctx.rewrite_ms,
    }
