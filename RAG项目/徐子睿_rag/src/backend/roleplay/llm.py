# -*- coding: utf-8 -*-
"""roleplay/llm.py —— 模型调用与降级兜底。

在链路中的位置：
    chat.py 调 call_llm 生成角色回复；失败时用 fallback_answer 兜底。

两条路径体现同一个原则："无依据不编造"。
    call_llm       正常情况下走 Ollama
    fallback_answer 模型不可用时 —— 有检索结果就只做"原文搬运 + 标注页码"（不做再加工），
                    没有检索结果就老实承认"本地模型不可用，不能编造答案"
"""
from __future__ import annotations

from typing import Any

import requests

from .config import OLLAMA_CHAT_URL, ROLEPLAY_MODEL

def call_llm(messages: list[dict[str, str]], temperature: float = 0.4) -> tuple[str, str]:
    """调 Ollama 生成角色回复。

    参数：
        messages: build_messages 的产物
        temperature: 默认 0.4 —— 比知识问答的 0.2 高一点，让角色扮演有表达变化，
                     但又不到 0.7+ 那种会偏离设定的程度
    返回：
        (模型输出原文, "ollama")。第二个值是"来源标记"，让调用方能区分
        这次回答是真模型生成还是降级兜底的。

    异常：
        网络/HTTP 错误直接抛出，由 chat() 捕获并降级到 fallback_answer。
    """
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": ROLEPLAY_MODEL, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"], "ollama"

def fallback_answer(role: dict[str, Any], context_docs: list[dict[str, Any]]) -> str:
    """本地模型不可用时的兜底回答。

    参数：
        role: 角色卡
        context_docs: 本轮检索到的片段
    返回：
        有检索结果时，直接返回第一条片段的原文和页码；
        没有检索结果时，明确说明"模型不可用、不能编造答案"。

    这是本项目"无依据不编造"原则的最后一道体现：
        与其让程序随便拼一句话敷衍用户，不如老实承认能力受限。
        有检索结果时只做"原文搬运 + 标注页码"，不做任何再加工 ——
        因为此时没有任何模型在把关，任何加工都可能引入错误信息。
    """
    if context_docs:
        first = context_docs[0]
        text = (first.get("snippet") or first.get("text") or "").strip()
        return f"根据知识库中第{first.get('page', '?')}页的相关内容：\n{text}"
    return f"我是{role['name']}。当前本地大模型不可用，我不能可靠地编造答案；请稍后重试，或补充更多背景信息。"
