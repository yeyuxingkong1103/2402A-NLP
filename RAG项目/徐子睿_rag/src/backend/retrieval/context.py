# -*- coding: utf-8 -*-
"""retrieval/context.py —— 上下文组织。

在链路中的位置：
    retrieve_with_trace 的最后一步：把排好序的片段整理成能直接拼进提示词的上下文。

三件事：去重、丢低分、控字符预算。做完之后每个片段还带上 snippet 字段，
供 server.py 拼提示词与生成答案时使用。
"""
from __future__ import annotations

from typing import Any

from .config import DEFAULT_SNIPPET_CHARS

def organize_context(items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把排好序的片段整理成能直接拼进提示词的上下文。

    参数：
        items: rerank 的输出（已按相关性排序）
        budget: 上下文总字符预算（默认 3600）
    返回：
        (选中的片段列表, {"count": 条数, "chars": 实际用掉的字符数, "budget": 预算})
        选中的每条额外带 snippet（截断后的正文）与 snippet_chars。

    做三件事：
        1. 去重 —— 同一 (来源,页码,章节,正文) 只留第一条，避免重复内容占满预算
        2. 丢低分 —— 精排分 < 0.5 的丢弃（第 0 条豁免，保证至少有东西给模型看）
        3. 控预算 —— 累加超过 budget 就停，防止提示词超出模型上下文窗口

    注意是 break 不是 continue：
        片段已按相关性排序，一旦装不下就直接停止尝试后面的，
        而不是跳过长片段去塞短的低分片段 —— 宁可少给，不给无关内容。
    """
    selected, seen, used = [], set(), 0
    for index, item in enumerate(items):
        key = (item["source"], item["page"], item["section"], item["text"])
        if key in seen or (index > 0 and item.get("rerank_score", 0) < 0.5):
            continue
        snippet = item["text"].strip()[:DEFAULT_SNIPPET_CHARS]
        if selected and used + len(snippet) > budget:
            break
        selected.append(item | {"snippet": snippet, "snippet_chars": len(snippet)})
        seen.add(key)
        used += len(snippet)
    return selected, {"count": len(selected), "chars": used, "budget": budget}
