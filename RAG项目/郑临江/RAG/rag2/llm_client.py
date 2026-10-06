# -*- coding: utf-8 -*-
"""OpenAI 兼容 LLM 客户端（默认接本地 Ollama）。

同时提供非流式 ``chat`` 与流式 ``stream``（逐 token 产出，供 SSE 使用）。
对 Qwen 等思考模型默认传 ``reasoning_effort="none"`` 关闭思考链，避免
结构化/长输出被冗长的思考过程截断。

    from rag2 import LLMClient

    llm = LLMClient(model="Qwen3.5:4B")
    print(llm.chat([{"role": "user", "content": "你好"}]))
    for piece in llm.stream([{"role": "user", "content": "讲个笑话"}]):
        print(piece, end="")
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .logging_config import get_logger

logger = get_logger("llm_client")


class LLMClient:
    """封装 OpenAI 兼容接口的同步/流式生成。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "Qwen3.5:4B",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = "none",
        timeout: float = 300.0,
    ) -> None:
        from openai import OpenAI  # 懒加载，避免 import rag2 强依赖 openai

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        logger.info("初始化 LLM：%s @ %s", model, base_url)

    # ------------------------------------------------------------------ 组装
    def _build(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort:
            params["reasoning_effort"] = effort
        if extra:
            params.update(extra)
        return params

    # ------------------------------------------------------------------ 生成
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """非流式生成，返回完整文本（空结果返回空串）。"""
        params = self._build(messages, temperature, max_tokens, reasoning_effort, kwargs)
        resp = self.client.chat.completions.create(**params)
        if not resp.choices:
            return ""
        message = resp.choices[0].message
        return (message.content or "") if message else ""

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式生成，逐段产出文本增量。"""
        params = self._build(messages, temperature, max_tokens, reasoning_effort, kwargs)
        params["stream"] = True
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece
