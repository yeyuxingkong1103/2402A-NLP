"""生成：调用大模型（同步 / 流式）。"""
from __future__ import annotations

from typing import Iterator

from app.core.registry import get_llm


def generate(messages: list[dict], temperature: float | None = None) -> str:
    return get_llm().chat(messages, temperature=temperature)


def generate_stream(messages: list[dict], temperature: float | None = None) -> Iterator[str]:
    return get_llm().chat_stream(messages, temperature=temperature)
