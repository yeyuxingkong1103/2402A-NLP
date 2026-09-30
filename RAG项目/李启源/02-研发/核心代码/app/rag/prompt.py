"""Prompt construction facade preserving the historical public API."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from app.rag.prompt_templates import USER_PROMPT_TEMPLATE, XIAOYUN_SYSTEM_PROMPT


def _fill_contacts(template: str) -> str:
    """Inject configured contacts without exposing fake fallback numbers."""
    hotline = os.getenv("SUPPORT_HOTLINE", "").strip() or "官网公布的客服热线"
    email = os.getenv("SUPPORT_EMAIL", "").strip() or "官网公布的客服邮箱"
    return template.replace("__SUPPORT_HOTLINE__", hotline).replace("__SUPPORT_EMAIL__", email)


@dataclass(slots=True, frozen=True)
class PromptTemplate:
    """Rendered system and user prompt pair."""

    system_prompt: str
    user_prompt_template: str
    variables: dict[str, str]


class PromptBuilder:
    """Render retrieved context, user query, and optional memory context."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        user_prompt_template: str = USER_PROMPT_TEMPLATE,
        enable_citation: bool = True,
        max_context_length: int = 4000,
    ) -> None:
        self.system_prompt = _fill_contacts(system_prompt or XIAOYUN_SYSTEM_PROMPT)
        self.user_prompt_template = user_prompt_template
        self.enable_citation = enable_citation
        self.max_context_length = max_context_length

    def build(
        self,
        query: str,
        context_chunks: list[Any],
        *,
        max_length: int = 300,
        additional_context: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        context = self._build_context(context_chunks)
        if len(context) > self.max_context_length * 4:
            context = context[: self.max_context_length * 4] + "\n...(内容过长已截断)"
        user_prompt = self.user_prompt_template.format(
            context=context, query=query, max_length=max_length
        )
        if additional_context:
            user_prompt = self._format_additional_context(additional_context) + "\n\n" + user_prompt
        return PromptTemplate(
            system_prompt=self.system_prompt,
            user_prompt_template=user_prompt,
            variables={"query": query, "context": context, "max_length": str(max_length)},
        )

    def _build_context(self, chunks: list[Any]) -> str:
        if not chunks:
            return "（知识库中未找到相关内容）"
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            text = getattr(chunk, "text", str(chunk))
            source = getattr(chunk, "source", "未知来源")
            score = getattr(chunk, "score", 0.0)
            parts.append(
                f"[文档{index}] 来源: {source} (相关度: {score:.2f})\n{text}"
                if self.enable_citation
                else f"[文档{index}]\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_additional_context(context: dict[str, Any]) -> str:
        lines = ["## 补充信息"]
        for key, label in (("user_id", "用户ID"), ("order_id", "订单号"), ("history", "最近咨询")):
            if key in context:
                lines.append(f"{label}: {context[key]}")
        if context.get("memory"):
            lines.extend([
                "以下长期记忆仅用于理解用户上下文；业务政策和事实必须以知识库为准：",
                str(context["memory"]),
            ])
        return "\n".join(lines)


class PromptOptimizer:
    """Decide when retrieval context is insufficient or out of scope."""

    def should_refuse(self, query: str, context_chunks: list[Any]) -> bool:
        if not context_chunks or all(getattr(chunk, "score", 1.0) < 0.3 for chunk in context_chunks):
            return True
        return any(
            re.search(pattern, query)
            for pattern in (r"股票|投资|理财", r"法律|律师|起诉", r"医疗|诊断|药物", r"代码|编程|bug")
        )

    @staticmethod
    def optimize_for_streaming(prompt: str) -> str:
        return prompt + "\n\n请注意：您的回答将被流式输出，请确保逻辑连贯，避免突然中断。"


def build_xiaoyun_prompt(query: str, context_chunks: list[Any], **kwargs: Any) -> PromptTemplate:
    """Convenience helper for the default 小云 prompt."""
    return PromptBuilder().build(query, context_chunks, **kwargs)


__all__ = [
    "PromptBuilder",
    "PromptOptimizer",
    "PromptTemplate",
    "USER_PROMPT_TEMPLATE",
    "XIAOYUN_SYSTEM_PROMPT",
    "build_xiaoyun_prompt",
]
