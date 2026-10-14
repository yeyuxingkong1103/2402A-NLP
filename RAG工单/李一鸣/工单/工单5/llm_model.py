"""Unified model factory with lazy construction and a small OpenAI-compatible client."""

from __future__ import annotations

from typing import Any

from config import MODELS, ModelConfig


class LLMModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            from openai import OpenAI
            if not self.config.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not configured")
            self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        return self._client

    def invoke(self, prompt: str) -> str:
        response = self._client_or_create().chat.completions.create(
            model=self.config.model_type,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


class ModelFactory:
    _instances: dict[str, LLMModel] = {}

    @classmethod
    def get_model(cls, model_type: str = "deepseek-chat", **overrides: Any) -> LLMModel:
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        if overrides:
            config = ModelConfig(**{**MODELS[model_type].__dict__, **overrides})
            return LLMModel(config)
        if model_type not in cls._instances:
            cls._instances[model_type] = LLMModel(MODELS[model_type])
        return cls._instances[model_type]
