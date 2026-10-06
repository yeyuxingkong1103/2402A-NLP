"""可插拔 LLM 客户端：ollama > openai 兼容（vLLM/SGLang）> extractive 兜底。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 生成优化（对应 设计/接口设计.md §2.11、环境事实 3.1）

关键设计（直接决定"首字 ≤3 秒"能否达标）：
1. **快速失败**：``probe()`` 超时 0.5 s、**不重试**、结果缓存；探测失败不影响首字延迟；
2. **流式逐块计时**：从请求发出到**首个非空增量**的时间即首字延迟（唯一计时点）；
3. 网络异常**不抛给上层**，统一转成 ``("error", {...})`` 事件，由 generator 走抽取式兜底；
4. 本机可用后端：Ollama ``qwen2.5:3b``（实测首字 0.20 s，环境事实 3）。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging_conf import current_trace_id, log_event, logger, truncate

#: 抽取式兜底后端名（无任何 LLM 服务时使用）
EXTRACTIVE = "extractive"


@dataclass
class BackendInfo:
    """后端信息（探测结果）。"""

    name: str
    model: str
    base_url: str
    available: bool
    probe_ms: float
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """序列化（供 health/日志）。"""
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "available": self.available,
            "probe_ms": self.probe_ms,
            "error": self.error,
        }


@dataclass
class LLMStats:
    """最近一次调用的统计（供 generation 事件与评估使用）。"""

    prompts: int = 0
    stream_chunks: int = 0
    last_first_token_ms: float = 0.0
    last_total_ms: float = 0.0
    last_prompt: str = ""
    last_output: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """LLM 客户端门面（可插拔后端 + 探测缓存 + 流式事件）。"""

    def __init__(self, settings=None) -> None:
        self._settings = settings or get_settings()
        self._info: BackendInfo | None = None
        self._lock = threading.RLock()
        self.stats = LLMStats()

    # ------------------------------------------------------------------
    # 探测
    # ------------------------------------------------------------------
    def probe(self, force: bool = False) -> BackendInfo:
        """探测可用后端（结果缓存；超时 ≤0.5 s 且不重试）。"""
        with self._lock:
            if self._info is not None and not force:
                return self._info
            configured = self._settings.llm.backend
            order: list[str] = []
            if configured == "auto":
                order = ["ollama", "openai", EXTRACTIVE]
            else:
                order = [configured, EXTRACTIVE] if configured != EXTRACTIVE else [EXTRACTIVE]
            for name in order:
                info = self._probe_one(name)
                if info.available:
                    self._info = info
                    logger.info(
                        "app.core.llm_client",
                        "LLM 后端已选定",
                        backend=info.name,
                        model=info.model,
                        base_url=info.base_url,
                        probe_ms=info.probe_ms,
                    )
                    return info
                logger.warning(
                    "app.core.llm_client",
                    "LLM 后端不可用，尝试下一个",
                    backend=name,
                    error=info.error,
                    probe_ms=info.probe_ms,
                )
            self._info = BackendInfo(
                name=EXTRACTIVE,
                model="",
                base_url="",
                available=True,
                probe_ms=0.0,
                error="无可用 LLM 服务，使用抽取式兜底",
            )
            logger.warning("app.core.llm_client", "全部 LLM 后端不可用，降级为抽取式回答")
            return self._info

    def _probe_one(self, name: str) -> BackendInfo:
        """探测单个后端（不抛异常，失败填 available=False）。"""
        settings = self._settings.llm
        started = time.perf_counter()
        if name == EXTRACTIVE:
            return BackendInfo(EXTRACTIVE, "", "", True, 0.0, "")
        if name == "ollama":
            url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=settings.probe_timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                models = [item.get("name", "") for item in payload.get("models", [])]
                available = any(settings.model.split(":")[0] in item for item in models) or bool(models)
                elapsed = (time.perf_counter() - started) * 1000
                return BackendInfo(
                    "ollama",
                    settings.model,
                    settings.ollama_base_url,
                    available,
                    round(elapsed, 2),
                    "" if available else f"模型 {settings.model} 不在 Ollama 模型列表中",
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                return BackendInfo("ollama", settings.model, settings.ollama_base_url, False, round(elapsed, 2), str(exc))
        if name == "openai":
            url = f"{settings.base_url.rstrip('/')}/models"
            try:
                request = urllib.request.Request(url, method="GET", headers={"Authorization": f"Bearer {settings.api_key}"})
                with urllib.request.urlopen(request, timeout=settings.probe_timeout) as response:
                    response.read()
                elapsed = (time.perf_counter() - started) * 1000
                return BackendInfo("openai", settings.model, settings.base_url, True, round(elapsed, 2), "")
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                return BackendInfo("openai", settings.model, settings.base_url, False, round(elapsed, 2), str(exc))
        # 未知后端名：配置错误（不静默）
        logger.error("app.core.llm_client", "未知的 LLM 后端名", backend=name)
        return BackendInfo(name, "", "", False, 0.0, f"未知后端 {name}")

    @property
    def backend(self) -> BackendInfo:
        """当前后端（未探测时先探测）。"""
        return self.probe()

    @property
    def available(self) -> bool:
        """当前是否有真实 LLM 服务（extractive 视为无服务）。"""
        return self.probe().name != EXTRACTIVE

    def reset_probe(self) -> None:
        """清除探测缓存（测试用）。"""
        with self._lock:
            self._info = None

    # ------------------------------------------------------------------
    # 生成（流式）
    # ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[str, Any]]:
        """流式生成，产出事件：

        - ``("first_token", {"first_token_ms": float})``：**仅首个非空增量**（首字延迟唯一计时点）
        - ``("delta", {"text": str})``：增量文本
        - ``("done", {"text": str, "total_ms": float})``：结束
        - ``("error", {"code": str, "message": str})``：任何失败（不抛异常给上层）
        """
        info = self.probe()
        if info.name == EXTRACTIVE:
            yield ("error", {"code": "LLM_UNAVAILABLE", "message": "无可用 LLM 服务（extractive 兜底）"})
            return
        started = time.perf_counter()
        first_token_ms = 0.0
        chunks: list[str] = []
        prompt_text = "\n".join(item.get("content", "") for item in messages)
        try:
            if info.name == "ollama":
                generator = self._stream_ollama(messages, temperature, max_tokens)
            else:
                generator = self._stream_openai(messages, temperature, max_tokens)
            for delta in generator:
                if delta:
                    if not first_token_ms:
                        first_token_ms = (time.perf_counter() - started) * 1000
                        yield ("first_token", {"first_token_ms": round(first_token_ms, 2)})
                    chunks.append(delta)
                    yield ("delta", {"text": delta})
            text = "".join(chunks)
            total_ms = (time.perf_counter() - started) * 1000
            if not first_token_ms:
                first_token_ms = total_ms
            self.stats.prompts += 1
            self.stats.last_first_token_ms = round(first_token_ms, 2)
            self.stats.last_total_ms = round(total_ms, 2)
            self.stats.last_prompt = prompt_text
            self.stats.last_output = text
            log_event(
                "llm_io",
                "app.core.llm_client",
                "chat_stream",
                trace_id=current_trace_id(),
                backend=info.name,
                model=info.model,
                prompt=truncate(prompt_text, 2000),
                output=truncate(text, 2000),
                stream_chunks=len(chunks),
                first_token_ms=round(first_token_ms, 2),
                total_ms=round(total_ms, 2),
            )
            yield ("done", {"text": text, "total_ms": round(total_ms, 2), "first_token_ms": round(first_token_ms, 2)})
        except Exception as exc:
            logger.exception("app.core.llm_client", "LLM 流式生成失败", backend=info.name, model=info.model)
            yield ("error", {"code": "LLM_UNAVAILABLE", "message": f"{type(exc).__name__}: {exc}"})

    def _stream_ollama(
        self, messages: list[dict[str, str]], temperature: float | None, max_tokens: int | None
    ) -> Iterator[str]:
        """Ollama ``/api/generate``（stream=true，逐行 JSON）。"""
        settings = self._settings.llm
        prompt = self._messages_to_prompt(messages)
        payload = {
            "model": settings.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": settings.temperature if temperature is None else temperature,
                "top_p": settings.top_p,
                "num_predict": settings.max_tokens if max_tokens is None else max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=settings.read_timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("app.core.llm_client", "Ollama 流式返回非 JSON 行，已跳过", line=truncate(line, 200))
                    continue
                if item.get("error"):
                    raise RuntimeError(f"Ollama 返回错误: {item['error']}")
                piece = item.get("response", "")
                if piece:
                    yield piece
                if item.get("done"):
                    break

    def _stream_openai(
        self, messages: list[dict[str, str]], temperature: float | None, max_tokens: int | None
    ) -> Iterator[str]:
        """OpenAI 兼容 ``/chat/completions``（SSE ``data: `` 行）。"""
        settings = self._settings.llm
        payload = {
            "model": settings.model,
            "messages": messages,
            "stream": True,
            "temperature": settings.temperature if temperature is None else temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        }
        request = urllib.request.Request(
            f"{settings.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=settings.read_timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    item = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("app.core.llm_client", "OpenAI 流式返回非 JSON 数据块，已跳过", data=truncate(data, 200))
                    continue
                choices = item.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """把 chat 消息拼成 Ollama 的单一 prompt（保留角色语义）。"""
        parts: list[str] = []
        for item in messages:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "system":
                parts.append(f"【系统】{content}")
            elif role == "assistant":
                parts.append(f"【历史回答】{content}")
            else:
                parts.append(f"【用户】{content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """非流式生成（用于查询改写等短任务）；失败返回空串并记日志。"""
        text = ""
        try:
            for event, payload in self.chat_stream(messages, **kwargs):
                if event == "done":
                    text = payload.get("text", "")
                elif event == "error":
                    logger.warning("app.core.llm_client", "非流式生成失败", error=payload.get("message", ""))
        except Exception:
            logger.exception("app.core.llm_client", "非流式生成异常")
        return text

    def health(self) -> dict[str, Any]:
        """健康信息（探测超时 ≤0.5 s）。"""
        info = self.probe()
        return info.as_dict()


_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """获取进程级 LLM 客户端单例。"""
    global _client
    with _client_lock:
        if _client is None:
            _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    """重置单例（测试用）。"""
    global _client
    with _client_lock:
        _client = None
