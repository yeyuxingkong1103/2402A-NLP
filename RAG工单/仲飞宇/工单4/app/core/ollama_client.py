# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单04 - 图像内容解析及检索优化
"""
Ollama 客户端。

【为什么走原生 /api/chat 而不是 OpenAI 兼容的 /v1】
1. `think: false` 是 Ollama 原生参数，OpenAI 兼容端点**不保证透传**；
   一旦思考链没关掉，qwen3 会把思维链吐进正文，既拖慢响应又污染答案。
2. `keep_alive` 同理。
3. **图片（工单04）也只在原生接口上**：Ollama 的 `images` 字段是原生协议，
   `/v1` 的 `image_url` 形态本项目未验证。
工单01 要求响应 ≤3 秒，这两条都是硬需求，所以统一走原生接口。

【工单04 改了什么】原先是**纯文本客户端**：`_payload()` 把模型名写死成
`settings.llm_model`，`messages` 的类型标注是 `list[dict[str, str]]`,
装不下图片。加多模态转写就必须能"换一个模型 + 带图"，所以：
  · `_payload()` 新增 `model` / `keep_alive` 参数（不传 = 沿用 settings，向后兼容）
  · `chat()` / `chat_stream()` 新增 `images`（base64 列表），非空时挂到最后一条
    user 消息的 `images` 字段上
  · 新增 `unload()` —— 8GB 卡装不下 qwen3(5.6GB) + 3B VLM(3.2GB)，
    入库转写时必须**主动赶走**聊天模型，转写完再赶走 VLM（见 pipeline 的显存时序）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

import httpx

from app.config import settings


class OllamaError(RuntimeError):
    """Ollama 不可达或返回异常时抛出，由 API 层转成友好提示。"""


def attach_images(messages: list[dict[str, Any]],
                  images: list[str] | None) -> list[dict[str, Any]]:
    """把 base64 图片挂到最后一条 user 消息上（工单04）。

    【为什么挂在最后一条 user 消息】Ollama 原生协议把 `images` 作为**消息级**
    字段，且只在 user 消息上生效；挂到 system 或 assistant 上会被静默忽略
    （不报错，只是模型看不见图 —— 属于"静默答错"，最难查）。
    没有 user 消息时原样返回，让调用方自己去发现提示词组装的问题。

    不修改入参：`build_messages` 的结果可能被别处复用，就地改会串味。
    """
    if not images:
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["images"] = list(images)
            return out
    return messages


class OllamaClient:
    def __init__(self, base_url: str | None = None,
                 timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        # 超时改为可配置（原先硬编码 120.0）；VLM 转写单张图可达数十秒，
        # 调用方会显式传 settings.vlm_timeout。
        self.timeout = timeout if timeout is not None else settings.ollama_timeout

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
        # 【必须显式钉住 bge-m3】不传 keep_alive 时 Ollama 用默认的 5 分钟，
        # 于是隔几分钟再提问要重载 bge-m3 —— 实测重载后单次嵌入 2355ms
        # （热态只要 35ms），这笔开销直接压在"提问→响应 ≤3 秒"的验收上。
        # 聊天路径本来就钉了 keep_alive，嵌入这条以前漏了。
        payload["keep_alive"] = settings.embed_keep_alive
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
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        fmt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        model: str | None = None,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        think_explicit: bool = True,
        extra_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            # 不传 model = 沿用 settings.llm_model（工单01/02/03 的既有行为）
            "model": model or settings.llm_model,
            "messages": messages,
            "stream": stream,
            # 模型常驻，避免反复冷启动（实测冷启动 7.2s）
            "keep_alive": keep_alive or settings.llm_keep_alive,
            "options": {
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "num_ctx": settings.llm_num_ctx if num_ctx is None else num_ctx,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }
        # 额外采样参数（工单04：视觉模型要压重复，见 vision 的 repeat_penalty）
        if extra_options:
            p["options"].update(extra_options)
        # 【工单04】视觉模型（qwen2.5vl）没有思考链，多传 think 键未必被接受。
        # think_explicit=False 时不下发该键，避免换个模型整个转写就失败。
        if think_explicit:
            p["think"] = settings.llm_think if think is None else think
        if fmt:
            p["format"] = fmt
        return p

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        fmt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        images: list[str] | None = None,
        model: str | None = None,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        """非流式生成，返回完整文本。

        images: base64 编码的图片列表（工单04）。非空时挂到**最后一条 user
        消息**上 —— 这是 Ollama 原生协议的位置，放在别的 role 上会被忽略。
        """
        messages = attach_images(messages, images)
        # 带图 + 指定模型（视觉模型）时不下发 think：qwen2.5vl 无思考链
        payload = self._payload(
            messages, stream=False, fmt=fmt,
            temperature=temperature, max_tokens=max_tokens, think=think,
            model=model, keep_alive=keep_alive, num_ctx=num_ctx,
            think_explicit=not (model and model != settings.llm_model),
            extra_options=extra_options,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as c:
                r = await c.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "")
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"生成失败: {e}") from e

    # ------------------------------------------------------------------
    # 显存管理（工单04）
    # ------------------------------------------------------------------
    async def unload(self, model: str) -> bool:
        """把某个模型从显存里卸掉（keep_alive=0）。

        【为什么必须显式卸载】8GB 卡实测装不下 qwen3:8b(5.58GB) + 3B VLM(约3.2GB)。
        入库转写时如果不主动赶走聊天模型，Ollama 会自己淘汰一个 —— 但那是
        **被动**的：转写完接着做嵌入、或用户马上提一个问，就会付一次数秒的冷启动。
        主动卸载让"什么时候付这笔钱"变成可控的。

        失败不抛：卸载是优化，不是主流程的硬需求（真正的硬需求由 pipeline 的
        阶段顺序保证 —— 转写在 embed 之前，二者天然错峰）。
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post(f"{self.base_url}/api/generate",
                                 json={"model": model, "keep_alive": 0})
                r.raise_for_status()
                return True
        except Exception:  # noqa: BLE001
            return False

    async def resident_models(self) -> list[dict[str, Any]]:
        """当前常驻显存的模型（/api/ps），供 preflight 与入库日志核对。"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{self.base_url}/api/ps")
                r.raise_for_status()
                return r.json().get("models", [])
        except Exception:  # noqa: BLE001
            return []

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
