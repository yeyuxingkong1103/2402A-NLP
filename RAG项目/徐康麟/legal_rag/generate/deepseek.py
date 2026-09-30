# -*- coding: utf-8 -*-
"""DeepSeek 在线 API 客户端。

DeepSeek 走 OpenAI 兼容协议，因此直接复用 OpenAICompatClient，
只把 base_url 与密钥环境变量固定下来。

密钥只从环境变量 DEEPSEEK_API_KEY 读取，绝不硬编码、绝不写入日志。
"""
from __future__ import annotations

import os

from .openai_compat import OpenAICompatClient

DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekClient(OpenAICompatClient):
    name = "deepseek"

    def __init__(self, model: str = "deepseek-chat", base_url: str = DEFAULT_BASE_URL,
                 temperature: float = 0.3, max_tokens: int = 1024,
                 timeout: float = 60.0) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=API_KEY_ENV,
            api_key=os.environ.get(API_KEY_ENV) or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
