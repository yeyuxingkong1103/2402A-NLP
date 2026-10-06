#!/usr/bin/env python3
"""上下文预算守卫：提示词超出服务窗口时**先裁证据**，而不是把 400 抛给用户。

为什么需要（`docs/DESIGN-TODO.md` §D7，2026-09-28 实测）：
4B 评测服务的 `--max-model-len 8192`，而检索证据很长时提示会到 **7169 token**，
再加 1024 输出 = **8193** ⇒ vLLM 直接 `400 maximum context length`。
接口层把它转成 503（"没有用演示模型顶替"）——**没有静默失败，但用户拿不到答案**。

本模块做两件事（都是纯函数，便于单测）：

1. ``is_context_overflow(...)``：从错误文本里认出"这是超窗"，而不是别的 400；
2. ``trim_context_block(...)``：把提示词里的 **【检索知识】** 块按比例**从尾部**裁短
   （命中按相关性排序，**头部最相关要留住**），【用户长期记忆】/【历史对话】/【用户问题】
   一字不动 —— 问题被裁掉就没法回答了。

调用方（`generate/openai_compat.py`）在遇到超窗 400 时按 ``TRIM_LADDER`` 逐档重试，
并把"裁了多少"记成可见提示（`llm_metrics.mark_context_trimmed`），
用户看到的回答里会带一句说明（COPY-STANDARD：说人话、不甩锅）。
"""
from __future__ import annotations

from .prompt import CONTEXT_HEADER, HISTORY_HEADER, MEMORY_HEADER, QUESTION_HEADER

#: 逐档重试的保留比例（第二档更狠）；命中按相关性排序，所以留头部
TRIM_LADDER: tuple[float, ...] = (0.6, 0.3)

#: 证据块之后的其它分区标题（裁剪**不得**越过它们）
_OTHER_HEADERS = (MEMORY_HEADER, HISTORY_HEADER, QUESTION_HEADER)

#: 认出"超窗"的标记（vLLM / OpenAI / 各家措辞）
_OVERFLOW_MARKERS = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "reduce the length",
    "prompt is too long",
    "too many tokens",
)


def is_context_overflow(message: str) -> bool:
    """这段错误文本是不是"提示词超窗"？（而不是别的 400）"""
    text = str(message or "").lower()
    return any(marker in text for marker in _OVERFLOW_MARKERS)


def context_block_span(content: str) -> tuple[int, int] | None:
    """定位 `【检索知识】` 分区的 ``[start, end)``；没有就返回 ``None``。

    ``end`` = 该分区之后**第一个**其它分区标题的位置（记忆/历史/问题），
    这样裁剪绝不会吃掉后面的分区。
    """
    text = str(content or "")
    start = text.find(CONTEXT_HEADER)
    if start < 0:
        return None
    end = len(text)
    for header in _OTHER_HEADERS:
        position = text.find(header, start + len(CONTEXT_HEADER))
        if position != -1:
            end = min(end, position)
    return start, end


def trim_context_block(messages: list[dict], keep_ratio: float) -> tuple[list[dict], dict]:
    """把**最后一条 user 消息**里的证据块按 ``keep_ratio`` 裁短，返回 (新 messages, 说明)。

    * 只裁``【检索知识】``块；记忆/历史/问题原样保留；
    * **不改原对象**（返回副本，便于重试时保留原始 messages）；
    * 找不到证据块 ⇒ 原样返回并标 ``trimmed=False``（调用方据此决定要不要重试）。
    """
    ratio = max(0.0, min(1.0, float(keep_ratio)))
    result = [dict(message) for message in messages]
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") != "user":
            continue
        content = str(result[index].get("content") or "")
        span = context_block_span(content)
        if span is None:
            return result, {"trimmed": False, "original_chars": len(content)}
        start, end = span
        block = content[start:end]
        keep_chars = int(len(block) * ratio)
        if keep_chars >= len(block):
            return result, {"trimmed": False, "original_chars": len(content)}
        # 从**尾部**裁（命中按相关性排序，头部最相关）
        result[index] = {**result[index],
                         "content": content[:start] + block[:keep_chars] + content[end:]}
        return result, {
            "trimmed": True,
            "keep_ratio": ratio,
            "original_chars": len(content),
            "block_chars_before": len(block),
            "block_chars_after": keep_chars,
            "dropped_chars": len(block) - keep_chars,
        }
    return result, {"trimmed": False, "original_chars": 0}


def trim_ladder(messages: list[dict]) -> list[tuple[float, list[dict], dict]]:
    """按 ``TRIM_LADDER`` 生成一串"越来越短"的候选，供调用方逐档重试。

    只在真的裁动了东西时才产出候选项 —— 没有证据块可裁就别浪费一次往返。
    """
    out: list[tuple[float, list[dict], dict]] = []
    for ratio in TRIM_LADDER:
        candidate, info = trim_context_block(messages, ratio)
        if info.get("trimmed"):
            out.append((ratio, candidate, info))
    return out
