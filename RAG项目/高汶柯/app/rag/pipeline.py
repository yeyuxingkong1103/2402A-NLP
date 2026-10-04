""" 总编排器 在线问答编排：检索 → 重排 → 提示词 → 生成 → 后处理 → 记忆。"""
from __future__ import annotations

from typing import Iterator

from app.config import settings
from app.core.registry import get_llm
from app.db.mongo_store import mongo_store
from app.db.mysql_store import bump_session, get_role
from app.db.redis_store import redis_store
from app.logging_conf import log
from app.rag.generator import generate, generate_stream
from app.rag.postprocess import postprocess
from app.rag.prompt import build_context, build_messages
from app.rag.rerank import rerank
from app.rag.retriever import retrieve


def _resolve_role(role_id: int) -> dict:
    role = get_role(role_id)
    if not role:
        raise ValueError("角色不存在")
    return role


def prepare(user_id: str, role_id: int, message: str, use_rag: bool = True) -> dict:
    """完成检索、重排、提示词拼装，返回生成所需的全部上下文。"""
    role = _resolve_role(role_id)
    history = redis_store.get_messages(user_id, role_id)
    domain = role.get("domain", "general")

    result = retrieve(message, domain=domain, use_rag=use_rag)
    docs = rerank(message, result["docs"], settings.rerank_top_k) if use_rag else []
    context = build_context(docs)

    memory = mongo_store.get_summary(user_id, role_id)
    if memory:
        context = f"【历史记忆】{memory}\n\n{context}"

    messages = build_messages(role["system_prompt"], context, history, message)
    return {
        "role": role, "history": history, "docs": docs, "context": context,
        "messages": messages, "queries": result["queries"], "route": result["route"],
    }


def _after_reply(user_id: str, role_id: int, message: str, reply: str, ctx: dict) -> None:
    """写入短期记忆、热问题、缓存、会话计数与长期记忆。"""
    redis_store.push_message(user_id, role_id, "user", message)
    redis_store.push_message(user_id, role_id, "assistant", reply)
    redis_store.touch_session(user_id, role_id)
    redis_store.incr_hot_query(message)
    redis_store.cache_set(message, ctx.get("context", ""), ttl=600)
    try:
        bump_session(user_id, role_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("会话计数失败: %s", exc)
    _maybe_summarize(user_id, role_id, ctx.get("history", []))


def _maybe_summarize(user_id: str, role_id: int, history: list[dict]) -> None:
    """历史较长时压缩为长期记忆摘要（MongoDB）。"""
    if len(history) < settings.memory_max_turns * 2:
        return
    try:
        convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history[-20:])
        summary = get_llm().chat(
            [{"role": "user", "content": f"请把下面对话压缩为不超过120字的中文摘要，保留关键事实与用户偏好：\n{convo}"}],
            temperature=0.3,
        )
        mongo_store.save_summary(user_id, role_id, summary.strip()[:400])
    except Exception as exc:  # noqa: BLE001
        log.warning("长期记忆摘要失败: %s", exc)


def answer(user_id: str, role_id: int, message: str, use_rag: bool = True) -> dict:
    """同步问答。"""
    if settings.rag_engine == "langchain":
        try:
            from app.frameworks.langchain_chain import answer as lc_answer

            result = lc_answer(user_id, role_id, message, use_rag=use_rag)
            if result:
                return result
        except Exception as exc:  # noqa: BLE001
            log.warning("LangChain 引擎失败，回退 native: %s", exc)

    if settings.rag_engine == "llamaindex":
        try:
            from app.frameworks.llamaindex_pipeline import query as li_query

            result = li_query(user_id, role_id, message, use_rag=use_rag)
            if result:
                return result
        except Exception as exc:  # noqa: BLE001
            log.warning("LlamaIndex 引擎失败，回退 native: %s", exc)

    ctx = prepare(user_id, role_id, message, use_rag=use_rag)
    raw = generate(ctx["messages"])
    reply = postprocess(raw, ctx["context"])
    _after_reply(user_id, role_id, message, reply, ctx)
    return {
        "reply": reply,
        "role_id": role_id,
        "sources": [
            {"source": d.get("source"), "score": d.get("rerank_score") or d.get("score")}
            for d in ctx["docs"]
        ],
        "rewritten_query": ctx["queries"][1] if len(ctx["queries"]) > 1 else None,
    }


def answer_stream(user_id: str, role_id: int, message: str, use_rag: bool = True) -> Iterator[str]:
    """流式问答。"""
    ctx = prepare(user_id, role_id, message, use_rag=use_rag)

    def gen() -> Iterator[str]:
        full = ""
        for delta in generate_stream(ctx["messages"]):
            full += delta
            yield delta
        _after_reply(user_id, role_id, message, full, ctx)

    return gen()
