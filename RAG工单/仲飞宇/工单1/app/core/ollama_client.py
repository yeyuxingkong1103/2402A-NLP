# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
Ollama 客户端。

【为什么走原生 /api/chat 而不是 OpenAI 兼容的 /v1】
1. `think: false` 是 Ollama 原生参数，OpenAI 兼容端点**不保证透传**；
   一旦思考链没关掉，qwen3 会把思维链吐进正文，既拖慢响应又污染答案。
2. `keep_alive` 同理。
工单01 要求响应 ≤3 秒，这两条都是硬需求，所以统一走原生接口。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

import httpx

from app.config import settings


class OllamaError(RuntimeError):
    """Ollama 不可达或返回异常时抛出，由 API 层转成友好提示。"""


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"无法连接 Ollama ({self.base_url}): {e}") from e

    # ------------------------------------------------------------------
    # 向量化
    # ------------------------------------------------------------------
    async def embed(self, texts: str | list[str],
                    *, num_gpu: int | None = None) -> list[list[float]]:
        """
        bge-m3 向量化，返回 1024 维向量列表。

        num_gpu=0 时强制在 CPU 上跑。这不是性能选择，而是**显存预算**：
        8GB 显存里 qwen3:8b 在 num_ctx=4096 下实测要 6.29GB，而系统空闲
        只有 6.23GB —— bge-m3 一旦加载到 GPU 就会把 qwen3 挤出去，
        下次问答要付 6 秒冷启动（实测 TTFT 从 35ms 变成 6117ms）。
        查询时只嵌入一句短文本，CPU 上开销可忽略，换来 qwen3 常驻。
        """
        payload: dict[str, Any] = {"model": settings.embed_model, "input": texts}
        if num_gpu is not None:
            payload["options"] = {"num_gpu": num_gpu}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.base_url}/api/embed", json=payload)
                r.raise_for_status()
                return r.json()["embeddings"]
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"向量化失败: {e}") from e

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        fmt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": stream,
            # 模型常驻，避免反复冷启动（实测冷启动 7.2s）
            "keep_alive": settings.llm_keep_alive,
            "think": settings.llm_think if think is None else think,
            "options": {
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "num_ctx": settings.llm_num_ctx,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }
        if fmt:
            p["format"] = fmt
        return p

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        fmt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        """非流式生成，返回完整文本。"""
        payload = self._payload(
            messages, stream=False, fmt=fmt,
            temperature=temperature, max_tokens=max_tokens, think=think,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "")
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"生成失败: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[str]:
        """流式生成，逐块 yield 文本增量（NDJSON 协议）。"""
        payload = self._payload(
            messages, stream=True,
            temperature=temperature, max_tokens=max_tokens, think=think,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                async with c.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = obj.get("message", {}).get("content", "")
                        if piece:
                            yield piece
                        if obj.get("done"):
                            break
        except OllamaError:
            raise
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"流式生成失败: {e}") from e

    # ------------------------------------------------------------------
    async def embed_sync_batches(
        self, texts: Iterable[str], batch_size: int = 16
    ) -> list[list[float]]:
        """分批向量化，避免单次请求过大。"""
        out: list[list[float]] = []
        batch: list[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                out.extend(await self.embed(batch))
                batch = []
        if batch:
            out.extend(await self.embed(batch))
        return out


_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
