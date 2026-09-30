"""
llm_client.py — 大模型调用封装

统一走 OpenAI 兼容协议，因此同一套代码既支持测试阶段的 DeepSeek 云端 API，
也支持算力云上用 SGLang / vLLM 部署的 Qwen 服务，
切换只需要改 config.LLM_PROVIDER（或 .env 里的 LLM_PROVIDER）。

提供三种调用方式：
    chat()          一次性返回完整回复
    chat_stream()   流式逐 token 产出，供 SSE 和 Streamlit 使用
    detect_intent() 意图识别，返回领域标签
"""

from __future__ import annotations

import time
from typing import Iterator

import config

_logger = None


def _log():
    global _logger
    if _logger is None:
        from loguru import logger

        _logger = logger
    return _logger


class LLMError(RuntimeError):
    """大模型调用失败。"""


# 缺少密钥、鉴权失败、请求非法等情况下重试不会成功，只按类名判断以避免导入期依赖 openai
_PERMANENT_ERRORS = (
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
)


def _is_permanent(error: Exception) -> bool:
    """判断该错误是否重试也无法恢复。"""
    return isinstance(error, LLMError) or type(error).__name__ in _PERMANENT_ERRORS


class LLMClient:
    """OpenAI 兼容接口的客户端封装。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ):
        cfg = config.get_llm_config()
        self.provider = provider or cfg["provider"]
        self.base_url = base_url or cfg["base_url"]
        self.api_key = api_key if api_key is not None else cfg["api_key"]
        self.model = model or cfg["model"]
        self._client = None

    # ------------------------------------------------------------ 客户端

    @property
    def client(self):
        """懒加载 OpenAI 客户端，避免没有 openai 包时导入即报错。"""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise LLMError(
                f"{self.provider} 的 API Key 为空。请在 config.py 里填写对应的密钥，"
                "或在 .env 中设置相应环境变量。"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("缺少 openai 包，请执行：pip install openai") from exc

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,  # 重试逻辑自己控制，便于记录日志
        )
        return self._client

    @staticmethod
    def _build_messages(system_prompt: str, user_input: str, history: list[dict] | None) -> list[dict]:
        """把系统提示、历史对话、当前问题拼成 messages 列表。"""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _with_retry(self, call, attempts: int = 3):
        """指数退避重试，应对网络抖动和服务重启。

        配置与鉴权类错误重试也不会成功，直接抛出，避免让用户白等数秒。
        """
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if _is_permanent(exc):
                    raise LLMError(str(exc)) from exc
                if attempt < attempts - 1:
                    delay = 2**attempt
                    _log().warning(
                        f"模型调用失败（第 {attempt + 1} 次）：{exc}，{delay}s 后重试"
                    )
                    time.sleep(delay)
        raise LLMError(f"模型调用失败，已重试 {attempts} 次：{last_error}") from last_error

    # ------------------------------------------------------------ 对话

    def chat(
        self,
        system_prompt: str,
        user_input: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """普通对话，返回完整回复文本。

        短任务（意图识别、查询改写、探活）要传 config.LLM_REASONING_EFFORT 关掉思维链：
        推理模型的思考会先占满 max_tokens，导致正文为空。
        """
        messages = self._build_messages(system_prompt, user_input, history)
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

        started = time.time()

        def call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )

        response = self._with_retry(call)
        content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)

        _log().info(
            f"模型调用 provider={self.provider} model={self.model} "
            f"tokens={getattr(usage, 'total_tokens', '?')} "
            f"latency={time.time() - started:.2f}s"
        )
        return content

    def chat_stream(
        self,
        system_prompt: str,
        user_input: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式对话，逐段产出文本增量。

        只在前几次连接失败时重试；一旦开始产出内容就不再重试，
        避免用户已经看到一半的回答被重复输出。
        """
        messages = self._build_messages(system_prompt, user_input, history)
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens

        started = time.time()
        first_token_at: float | None = None
        chars = 0

        def open_stream():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

        stream = self._with_retry(open_stream)

        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if not piece:
                    continue

                if first_token_at is None:
                    first_token_at = time.time()
                chars += len(piece)
                yield piece
        finally:
            elapsed = time.time() - started
            first = (first_token_at - started) if first_token_at else -1
            _log().info(
                f"流式调用 provider={self.provider} model={self.model} "
                f"chars={chars} 首字={first:.2f}s 总耗时={elapsed:.2f}s"
            )

    # ------------------------------------------------------------ 意图识别

    def detect_intent(self, user_input: str, system_prompt: str | None = None) -> str:
        """让模型输出一个领域标签，返回原始文本（由 intent.py 负责解析校验）。"""
        prompt = system_prompt or "判断用户问题所属领域，只输出一个词。"
        return self.chat(
            prompt,
            user_input,
            history=None,
            temperature=0.0,
            max_tokens=16,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        )

    # ------------------------------------------------------------ 探活

    def health(self) -> dict:
        """探活：发一个最小请求，确认服务可达、密钥可用。"""
        started = time.time()
        try:
            reply = self.chat("你是一个测试助手。", "回复 OK 两个字。", temperature=0.0,
                              max_tokens=8, reasoning_effort=config.LLM_REASONING_EFFORT)
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "latency": round(time.time() - started, 2),
                "sample": reply[:20],
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "error": str(exc),
            }


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """返回全局 LLM 客户端单例。"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_client() -> None:
    """丢弃单例，配置变更后（例如切换模型）调用。"""
    global _client
    _client = None


if __name__ == "__main__":
    client = LLMClient()
    print(f"服务商 : {client.provider}")
    print(f"模型   : {client.model}")
    print(f"地址   : {client.base_url}")

    result = client.health()
    if result["ok"]:
        print(f"探活成功，耗时 {result['latency']}s，返回：{result['sample']}")
    else:
        print(f"探活失败：{result['error']}")
