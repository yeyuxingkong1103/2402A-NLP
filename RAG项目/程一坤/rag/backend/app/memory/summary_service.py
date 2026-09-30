"""会话摘要的写入侧：窗口满之后按节流把更早轮次压成一段前情（批次 21）。

方案来源：reports/batch20_session_summary_plan.md（读侧注入在 chat 层，见下）。

要解决的问题（"为什么必须有它"）：
短期记忆是**滑动窗口 + 硬截断**——`append_message` 里 `ltrim(key, -20, -1)`
意味着窗口满（10 轮）之后，每来一轮就有最早 2 条消息被永久丢弃，任何地方都不再保存。
第 11 轮起用户再问"那这个钱谁出？"，上文的实体早已不在 Redis 里。

------ 与滑动窗口的关系（读代码时最容易被绕进去的一点）------
1. 窗口**未满**时不摘要：最近原文全在 Redis 里，压缩没有信息增量。
2. 窗口**满了之后** `len(messages)` 恒等于 `max_messages`，
   **无法从窗口长度推断"又过了几轮"** —— 这正是方案文档没有指定的落点。
   因此本模块自持一个会话级状态（`short_memory:{u}:{s}:summary_state`，
   由 short_term.py 提供读写），记录 `turns`（累计轮次）与 `summarized_turns`
   （上次摘要时的轮次）；节流判断用两者之差，与窗口长度无关。
   状态与消息/摘要共用同一个 TTL，会话沉睡后一起消失（不残留脏数据）。
3. 摘要输入取 `messages[:-SUMMARY_KEEP_RECENT_MESSAGES]` —— 即"已经离开最近 3 轮
   保护区"的那一段。它包含一部分仍在窗口里的消息，这是刻意的：
   每次摘要都把旧摘要与新滑出的内容一起重算，保证跨轮连续、不出现记忆断点。

------ 失败策略（硬要求）------
任何失败都只记 warning 并返回 None，**绝不覆盖已有摘要**（LLM 超时若写空串，
等于把之前积累的前情清空）。摘要只是上下文增强，不能影响问答链路——
与批次 14 长期记忆同一原则。
"""
from __future__ import annotations

import logging
from typing import Any

from app.memory.summary_policy import (
    SUMMARY_INPUT_MESSAGE_CHARS,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_NEW_TURNS,
)

logger = logging.getLogger("app.memory.summary")

# 摘要生成的提示词：固定四要素，第三人称，只输出摘要正文（不要 JSON/标题/寒暄）
SUMMARY_SYSTEM_PROMPT = (
    "你在为一个法律问答助手压缩会话前情。把用户与助手的早期对话压成一段简短摘要，"
    "供后续轮次理解上下文。要求：\n"
    "1. 固定写清四要素：①用户的身份或处境 ②已讨论过的问题 ③已给出的结论要点 "
    "④仍未解决或待用户补充的事实；\n"
    f"2. 不超过 {SUMMARY_MAX_CHARS} 字，只输出摘要正文，不要标题、不要 Markdown、不要寒暄；\n"
    "3. 用第三人称陈述，不要出现'根据上文'这类指代，摘要必须自包含；\n"
    "4. 只写对话里真实出现过的信息，不得补充对话中没有的法律结论。"
)

# 默认 LLM 客户端缓存：仅在未显式传入 llm_client 时懒建（生产路径）。
# 测试一律显式传入假客户端，不触发这里。
_default_llm_client: Any | None = None


def _resolve_llm_client(llm_client: Any | None) -> Any | None:
    """取 LLM 客户端：显式传入优先，否则懒建并缓存（构建失败返回 None）。"""
    global _default_llm_client
    if llm_client is not None:
        return llm_client
    if _default_llm_client is None:
        from app.models.llm import build_chat_client_from_settings

        _default_llm_client = build_chat_client_from_settings()
    return _default_llm_client


def _resolve_enabled(enabled: bool | None) -> bool:
    """取开关值：显式传入优先，否则读 SESSION_SUMMARY_ENABLED（默认关）。"""
    if enabled is not None:
        return enabled
    from app.core.config import settings

    return bool(getattr(settings, "session_summary_enabled", False))


def _render_transcript(messages: list[Any]) -> str:
    """把消息列表渲染成"用户：…/助手：…"的文本，单条按上限截断。"""
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = "助手" if message.get("role") == "assistant" else "用户"
        lines.append(f"{role}：{content.strip()[:SUMMARY_INPUT_MESSAGE_CHARS]}")
    return "\n".join(lines)


def _summarize(llm_client: Any, previous: str | None, messages: list[Any]) -> str | None:
    """调用 LLM 生成新摘要；输入不可用或调用失败返回 None。"""
    transcript = _render_transcript(messages)
    if not transcript:
        return None
    user_prompt = (
        (f"# 已有摘要（更早轮次，需要与新增对话合并重写）\n{previous}\n\n" if previous else "")
        + f"# 新增对话\n{transcript}"
    )
    raw = llm_client.chat(SUMMARY_SYSTEM_PROMPT, user_prompt)
    text = (raw or "").strip()
    if not text:
        return None
    if len(text) > SUMMARY_MAX_CHARS:
        # 超限只截断不报错：模型偶尔不守字数，截断比丢弃更有用
        logger.warning("会话摘要超长已截断：%d -> %d 字", len(text), SUMMARY_MAX_CHARS)
        text = text[:SUMMARY_MAX_CHARS]
    return text


def maybe_update_session_summary(
    *,
    short_term_memory: Any,
    user_id: str | None,
    session_id: str | None,
    llm_client: Any | None = None,
    enabled: bool | None = None,
    request_id: str | None = None,
) -> str | None:
    """按节流规则判断是否需要更新会话摘要，需要则生成并写回。

    Args:
        short_term_memory: 短期记忆存储（提供 read_messages/read_summary/write_summary
            与摘要状态读写）；None 表示不启用
        user_id / session_id: 定位记忆；任一缺失即跳过
        llm_client: 摘要用 LLM 客户端；None = 懒建默认客户端（测试请显式传入）
        enabled: 开关；None = 读配置 SESSION_SUMMARY_ENABLED（默认 false）
        request_id: 日志链路追踪 id

    Returns:
        本次新写入的摘要文本；未触发/失败/被关闭时返回 None

    时序：本函数是**同步**的，调用方（API 持久化层）负责放后台线程。
    任何异常都只记 warning，绝不向调用方抛出。
    """
    if not _resolve_enabled(enabled):
        # 关闭时"不读不写"，与引入摘要之前的行为逐字一致（便于一键回退/对拍）
        return None
    if not short_term_memory or not user_id or not session_id:
        return None

    try:
        state = short_term_memory.read_summary_state(user_id, session_id)
        turns = int(state.get("turns", 0)) + 1
        summarized_turns = int(state.get("summarized_turns", 0))

        messages = short_term_memory.read_messages(user_id, session_id)
        max_messages = int(getattr(short_term_memory, "max_messages", 0) or 0)

        # 窗口未满：最近原文都还在，压缩无信息增量
        if max_messages <= 0 or len(messages) < max_messages:
            short_term_memory.write_summary_state(
                user_id, session_id, {"turns": turns, "summarized_turns": summarized_turns}
            )
            return None

        # 节流：距上次摘要不足 3 轮（窗口满后 len 恒定，只能靠轮次计数判断）
        if turns - summarized_turns < SUMMARY_MIN_NEW_TURNS:
            short_term_memory.write_summary_state(
                user_id, session_id, {"turns": turns, "summarized_turns": summarized_turns}
            )
            return None

        # 待压缩区间：留出最近 6 条原文不压
        foldable = messages[:-SUMMARY_KEEP_RECENT_MESSAGES] if len(messages) > SUMMARY_KEEP_RECENT_MESSAGES else []
        if not foldable:
            short_term_memory.write_summary_state(
                user_id, session_id, {"turns": turns, "summarized_turns": summarized_turns}
            )
            return None

        client = _resolve_llm_client(llm_client)
        if client is None:
            return None
        previous = short_term_memory.read_summary(user_id, session_id)
        new_summary = _summarize(client, previous, foldable)
        if not new_summary:
            # LLM 失败：保留旧摘要不动（summarized_turns 也不推进，下轮继续尝试）
            short_term_memory.write_summary_state(
                user_id, session_id, {"turns": turns, "summarized_turns": summarized_turns}
            )
            logger.warning(
                "会话摘要生成失败，保留旧摘要", extra={"request_id": request_id}
            )
            return None

        short_term_memory.write_summary(user_id, session_id, new_summary)
        short_term_memory.write_summary_state(
            user_id, session_id, {"turns": turns, "summarized_turns": turns}
        )
        logger.info(
            "会话摘要已更新：%d 条消息 -> %d 字", len(foldable), len(new_summary),
            extra={"request_id": request_id},
        )
        return new_summary
    except Exception as error:  # noqa: BLE001
        # 摘要只是上下文增强：任何异常都不得影响主流程
        logger.warning(
            "会话摘要更新失败（已忽略）：%s: %s",
            type(error).__name__,
            error,
            extra={"request_id": request_id},
        )
        return None
