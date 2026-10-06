from __future__ import annotations

import json
import time

import httpx

from ..config import Settings


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4) if text else 0


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(message.get("content", "")) for message in messages)


def require_siliconflow_key(settings: Settings) -> None:
    if not settings.siliconflow_api_key:
        raise RuntimeError("SILICONFLOW_API_KEY 未配置，不能生成真实向量或真实重排结果")


class DeepSeekProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat_payload(self, messages: list[dict], max_tokens: int = 5000, stream: bool = False, thinking_enabled: bool | None = None, temperature: float | None = None) -> dict:
        payload = {"model": self.settings.deepseek_model, "messages": messages, "temperature": self.settings.deepseek_temperature if temperature is None else temperature, "max_tokens": max_tokens, "stream": stream}
        thinking = self.settings.deepseek_thinking if thinking_enabled is None else thinking_enabled
        if thinking is not None:
            payload["enable_thinking"] = bool(thinking)
        return payload

    def chat(self, messages: list[dict], max_tokens: int = 5000, thinking_enabled: bool | None = None, temperature: float | None = None) -> str:
        if not self.settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，不能生成真实模型回答")
        retries = int(getattr(self.settings, "llm_retry_count", 2) or 2)
        delay = float(getattr(self.settings, "llm_retry_delay", 1.0) or 1.0)
        for attempt in range(retries + 1):
            try:
                response = httpx.post(f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"}, json=self.chat_payload(messages, max_tokens, thinking_enabled=thinking_enabled, temperature=temperature), timeout=getattr(self.settings, "deepseek_timeout", 120))
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"] or "")
            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt >= retries:
                    raise
                time.sleep(delay * 2 ** attempt)

    @staticmethod
    def stream_delta(line: str) -> str:
        value = str(line or "")
        if not value.startswith("data:"):
            return ""
        payload = value.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            return ""
        return str(json.loads(payload).get("choices", [{}])[0].get("delta", {}).get("content") or "")

    def stream_chat(self, messages: list[dict], max_tokens: int = 5000, thinking_enabled: bool | None = None, temperature: float | None = None):
        if not self.settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，不能生成真实模型回答")
        retries = int(getattr(self.settings, "llm_retry_count", 2) or 2)
        delay = float(getattr(self.settings, "llm_retry_delay", 1.0) or 1.0)
        emitted = False
        for attempt in range(retries + 1):
            try:
                with httpx.stream("POST", f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"}, json=self.chat_payload(messages, max_tokens, True, thinking_enabled, temperature), timeout=getattr(self.settings, "deepseek_timeout", 120)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        delta = self.stream_delta(str(line or ""))
                        if delta:
                            emitted = True
                            yield delta
                return
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
                if emitted or attempt >= retries:
                    raise
                time.sleep(delay * 2 ** attempt)


class ModelGateway:
    def __init__(self, settings: Settings):
        from .embedding import EmbeddingCache, EmbeddingClient
        from .multimodal import MultimodalAnalyzer
        from .rerank import RerankClient
        self.settings = settings
        self._embedding_cache = EmbeddingCache(settings)
        self.embedding_client = EmbeddingClient(settings, self._embedding_cache)
        self.rerank_client = RerankClient(settings)
        self.chat_client = DeepSeekProvider(settings)
        self.vision = MultimodalAnalyzer(settings)

    last_embedding_error = property(lambda self: self.embedding_client.last_error, lambda self, value: setattr(self.embedding_client, "last_error", value))
    last_rerank_error = property(lambda self: self.rerank_client.last_error, lambda self, value: setattr(self.rerank_client, "last_error", value))
    def embed(self, texts): return self.embedding_client.embed(texts)
    def rank(self, query, docs): return self.rerank(query, docs)
    def rerank(self, query, docs): return self.rerank_client.rerank(query, docs)
    def chat_payload(self, messages, max_tokens=5000, stream=False, thinking_enabled=None, temperature=None): return self.chat_client.chat_payload(messages, max_tokens, stream, thinking_enabled, temperature)
    def chat(self, messages, max_tokens=5000, thinking_enabled=None, temperature=None): return self.chat_client.chat(messages, max_tokens, thinking_enabled, temperature)
    @staticmethod
    def stream_delta(line): return DeepSeekProvider.stream_delta(line)
    def stream_chat(self, messages, max_tokens=5000, thinking_enabled=None, temperature=None): yield from self.chat_client.stream_chat(messages, max_tokens, thinking_enabled, temperature)
    def analyze_media(self, path, media_type, filename): return self.vision.analyze_media(path, media_type, filename)
    def health(self):
        return {"deepseek": bool(self.settings.deepseek_api_key), "siliconflow": bool(self.settings.siliconflow_api_key), "llm_provider": "DeepSeek", "llm_model": self.settings.deepseek_model, "deepseek_thinking": self.settings.deepseek_thinking, "embedding_model": self.settings.embedding_model, "embedding_last_error": self.last_embedding_error, "reranker_model": self.settings.reranker_model, "reranker_last_error": self.last_rerank_error, "web": "online" if self.settings.tavily_api_key else "offline", "multimodal_model": self.settings.multimodal_model}
