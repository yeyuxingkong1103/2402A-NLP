# -*- coding: utf-8 -*-
"""roleplay/chat.py —— 一次角色对话的总流程。

在链路中的位置：
    backend/server.py 的 /api/roleplay/chat → 【本文件】
        → SQLite（角色/会话/消息） + retrieval（知识检索） + memory（长期记忆） + Ollama（生成）

主流程（每一步失败都有各自的降级出口，不让单点故障中断整轮对话）：
    校验消息和角色 → 找到或创建会话 → 保存用户消息
    → RAG 检索知识库（失败则只跳过检索，对话继续）
    → 按角色的 knowledge_sources 过滤片段
    → 读短期历史 + 长期记忆 → 拼五层提示词
    → 调模型生成（失败则用 fallback_answer 兜底）
    → 清洗输出、按实际片段补引用 → 保存助手消息 → 把用户消息存入长期记忆
"""
from __future__ import annotations

from typing import Any

try:
    from ..retrieval import retrieve_with_trace
except ImportError:
    from retrieval import retrieve_with_trace

from .config import MAX_MESSAGE_CHARS, SHORT_MEMORY_LIMIT
from .llm import call_llm, fallback_answer
from .memory import remember_long_term, retrieve_long_term
from .prompt import build_messages, postprocess_answer
from .roles import get_role
from .sessions import append_message, create_session, ensure_user, get_session, recent_messages

def chat(user_id: str, role_id: str, message: str, session_id: str | None = None, user_name: str | None = None) -> dict[str, Any]:
    """执行一次完整的角色对话。

    参数：
        user_id: 用户 id（默认 anonymous）
        role_id: 角色 id
        message: 本轮用户消息
        session_id: 会话 id，不传则自动新建
        user_name: 用户显示名
    返回：
        {
          answer:       角色回复（已清洗、已补引用）
          citations:    引用列表 [{source, page, text}]
          session_id / user_id / role: 会话与身份信息
          model_source: "ollama" 表示真模型生成，"fallback" 表示降级兜底
          trace:        处理链路（含被跳过的步骤及原因）
          memory:       记忆使用情况（短期条数、长期是否写入/命中数）
        }

    异常：
        消息为空、超长、角色不存在、会话不属于当前用户或角色 -> ValueError（接口层映射为 400）。

    主流程（每一步失败都有各自的降级出口，不让单点故障中断整轮对话）：
        校验消息和角色
        → 找到或创建会话
        → 保存用户消息
        → RAG 检索知识库（失败则只跳过检索，记一条 skipped trace，对话继续）
        → 按角色的 knowledge_sources 过滤片段（角色限定只看某些文档时）
        → 读短期历史 + 长期记忆
        → 拼五层提示词
        → 调模型生成（失败则用 fallback_answer 兜底）
        → 清洗输出、按实际片段补引用
        → 保存助手消息
        → 把用户消息存入长期记忆

    越权校验（session 归属检查）：
        传入的 session_id 必须是当前用户、当前角色的会话。
        不校验的话，改一个 session_id 就能往别人的会话里发消息、读到别人的历史。
    """
    message = (message or "").strip()
    if not message:
        raise ValueError("消息不能为空")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"消息不能超过 {MAX_MESSAGE_CHARS} 个字符")
    role = get_role(role_id)
    if not role:
        raise ValueError(f"角色不存在: {role_id}")
    user = ensure_user(user_id, user_name)
    session = get_session(session_id) if session_id else None
    if session and (session["user_id"] != user["id"] or session["role_id"] != role_id):
        raise ValueError("会话不属于当前用户或角色")
    if not session:
        session = create_session(user["id"], role_id)

    append_message(session, "user", message)  # 先落库：即使后面检索/生成失败，用户的话也不能丢
    trace: list[dict[str, Any]] = []
    context_docs: list[dict[str, Any]] = []
    try:
        bundle = retrieve_with_trace(message, candidate_k=12, final_k=5, char_budget=3200)
        context_docs = bundle.get("context_docs") or bundle.get("results") or []
        # 角色可限定只引用某些文档：这是角色卡 knowledge_sources 字段的落地点
        allowed_sources = {str(source) for source in role.get("knowledge_sources", []) if str(source).strip()}
        if allowed_sources:
            context_docs = [doc for doc in context_docs if any(source in str(doc.get("source", "")) for source in allowed_sources)]
            trace.append({"step": "角色知识库过滤", "status": "done", "detail": f"保留 {len(context_docs)} 条片段", "ms": 0})
        trace.extend(bundle.get("trace") or [])
    except Exception as exc:
        # 检索整体不可用（如 Milvus 没起）：不抛异常，只记一条 skipped 让链路透明可见，
        # 角色仍然能凭短期记忆和长期记忆继续对话
        trace.append({"step": "RAG 检索", "status": "skipped", "detail": f"本地检索不可用：{exc}", "ms": 0})

    short = recent_messages(session["id"], SHORT_MEMORY_LIMIT)
    # 剔除本轮刚存进去的那条用户消息：它是"当前问题"，会在提示词里单独以【本轮用户消息】出现。
    # 不去掉的话，同一个问题在上下文里出现两次，模型可能误以为是重复提问
    if short and short[-1].get("speaker") == "user" and short[-1].get("content") == message:
        short = short[:-1]
    long_memories = retrieve_long_term(user["id"], role_id, message)
    messages = build_messages(role, short, long_memories, context_docs, message)
    try:
        raw_answer, model_source = call_llm(messages)
    except Exception as exc:
        raw_answer, model_source = fallback_answer(role, context_docs), "fallback"
        trace.append({"step": "LLM 生成", "status": "skipped", "detail": f"本地模型不可用：{exc}", "ms": 0})

    answer = postprocess_answer(raw_answer, context_docs)
    append_message(session, "assistant", answer)
    # 长期记忆存的是"用户说的话"而不是模型的回答：
    # 需要被记住的是用户的偏好、处境、诉求，模型每轮的回答本身没有记忆价值
    remembered = remember_long_term(user["id"], role_id, session["id"], message)
    citations = [{"source": doc.get("source", ""), "page": doc.get("page", -1), "text": (doc.get("snippet") or doc.get("text", ""))[:120]} for doc in context_docs]
    return {
        "answer": answer,
        "citations": citations,
        "session_id": session["id"],
        "user_id": user["id"],
        "role": {"id": role["id"], "name": role["name"]},
        "model_source": model_source,
        "trace": trace,
        # memory 字段把记忆的实际情况暴露给前端/调试者：
        # short_count 用 len(short)+2 是因为 short 剔掉了本轮那条消息，
        # 算上"本轮用户消息 + 本轮助手回答"才是这个会话真实的短期记忆条数
        "memory": {"short_backend": "sqlite", "short_count": len(short) + 2, "long_term_backend": "milvus", "long_term_saved": remembered, "long_term_hits": len(long_memories)},
    }
