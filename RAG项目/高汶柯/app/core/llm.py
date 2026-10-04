"""统一大模型客户端。

基于 OpenAI 兼容协议，支持多家在线 API 与本地部署（vLLM / SGLang / xInference），
通过环境变量 ``LLM_PROVIDER`` 一键切换。
"""
from __future__ import annotations

import os
from typing import Iterator

from openai import OpenAI

from app.config import settings
from app.logging_conf import log

# Provider 预设：base_url / 默认模型 / API Key 环境变量
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "LLM_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "LLM_API_KEY",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-pro-32k",
        "key_env": "LLM_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "key_env": "LLM_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "LLM_API_KEY",
    },
    "local-vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3-27B",
        "key_env": "LOCAL_API_KEY",
    },
    "local-sglang": {
        "base_url": "http://localhost:30000/v1",
        "model": "Qwen/Qwen3-27B",
        "key_env": "LOCAL_API_KEY",
    },
}


class LLMClient:
    """封装 chat / 流式 chat，屏蔽 Provider 差异。"""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider or "deepseek").lower()
        preset = PROVIDERS.get(self.provider, PROVIDERS["deepseek"])

        self.api_key = api_key or settings.llm_api_key or os.getenv(preset["key_env"], "")
        self.base_url = base_url or settings.llm_base_url or preset["base_url"]
        self.model = model or settings.llm_model or preset["model"]

        if not self.api_key:
            log.warning("LLM API Key 为空，请检查 .env 中的 LLM_API_KEY / %s", preset["key_env"])

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "sk-no-key")
        log.info("LLM 初始化: provider=%s model=%s base_url=%s", self.provider, self.model, self.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
