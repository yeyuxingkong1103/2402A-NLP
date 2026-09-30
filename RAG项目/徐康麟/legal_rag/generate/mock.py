# -*- coding: utf-8 -*-
"""离线 Mock 大模型。

用途：
  * 在没有 API key、没有 GPU 时把「检索 -> 提示词 -> 生成 -> 后处理」整条链路跑通；
  * 作为单元测试的确定性后端。

它并不"假装"自己是真模型：输出会明确标注这是离线模拟，
并把检索到的知识摘出来，便于验证检索与引用是否正确。
"""
from __future__ import annotations

import re

from .llm_base import LLMClient
from .prompt import CONTEXT_HEADER, QUESTION_HEADER

_SOURCE_RE = re.compile(r"^\[\d+\]\s*来源：(.+?)（相关度", re.MULTILINE)


class MockLLMClient(LLMClient):
    name = "mock"

    def _complete(self, messages: list[dict]) -> str:
        user_content = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_content = str(message.get("content", ""))
                break

        question = ""
        if QUESTION_HEADER in user_content:
            question = user_content.split(QUESTION_HEADER, 1)[1].strip()

        context = ""
        if CONTEXT_HEADER in user_content:
            context = user_content.split(CONTEXT_HEADER, 1)[1]
            if QUESTION_HEADER in context:
                context = context.split(QUESTION_HEADER, 1)[0]
            context = context.strip()

        sources = _SOURCE_RE.findall(context)
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
        # 跳过 "[n] 来源：..." 这类标注行，取正文
        body = [p for p in paragraphs if not p.startswith("[") or "来源：" not in p]
        excerpt = (body[0] if body else context)[:300]

        lines = [
            "（离线模拟回答：当前 LLM_PROVIDER=mock，未调用真实大模型；"
            "设置 LLM_PROVIDER=deepseek 并提供 DEEPSEEK_API_KEY 即可获得真实回答。）",
            "",
            f"针对问题「{question}」，依据检索到的资料，可归纳如下：",
            excerpt,
        ]
        if sources:
            lines.append("")
            lines.append("参考：" + "、".join(dict.fromkeys(sources)))
        return "\n".join(lines)
