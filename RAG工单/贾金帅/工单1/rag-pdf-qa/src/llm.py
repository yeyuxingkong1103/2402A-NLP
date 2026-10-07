"""
大模型调用封装
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

统一走 OpenAI 兼容协议（DashScope 兼容模式 / Moonshot / 硅基流动 均可）。
所有调用都带**优雅降级**：失败返回空串或 None，由上层决定退路，
绝不因为一次 API 抖动把整个请求打成 500。这是工单「容错机制」的一部分。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Iterator

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_JUDGE_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    WORK_ORDER_NO,  # noqa: F401
)

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """惰性构造 OpenAI 客户端单例。"""
    global _client
    if _client is None:
        from openai import OpenAI

        if not LLM_API_KEY:
            raise RuntimeError("未配置 LLM_API_KEY，请在 .env 中填写。")
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=2,
        )
    return _client


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """非流式对话，返回文本。失败时抛异常，由调用方决定降级策略。"""
    client = get_client()
    kwargs: dict = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def chat_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """
    流式对话，逐段吐正文。

    同时兼容推理型模型的思维链字段（reasoning_content）：
    本工单默认用 qwen-flash（非推理），但换模型时不应改动上层代码。
    思维链内容以 ("__thinking__", text) 形式先吐出，正文以 ("__content__", text)。
    调用方按事件类型分流即可。
    """
    client = get_client()
    stream = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            yield ("__thinking__", reasoning)
        if getattr(delta, "content", None):
            yield ("__content__", delta.content)


def chat_json(messages: list[dict], model: str | None = None, max_tokens: int = 512) -> dict | None:
    """
    要求模型返回 JSON。做三层兜底：
      1) 原生 json_object 模式；
      2) 失败则用普通模式 + 从文本里抠第一个 {...} / [...];
      3) 再失败返回 None，由调用方走规则分支。
    """
    try:
        raw = chat(messages, model=model, temperature=0.0, max_tokens=max_tokens, json_mode=True)
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON 模式调用失败（%s），退回文本抽取", exc)

    try:
        raw = chat(messages, model=model, temperature=0.0, max_tokens=max_tokens)
        return _extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 调用失败：%s", exc)
        return None


def _extract_json(text: str) -> dict | None:
    """从自由文本里抠出第一个 JSON 对象。"""
    text = text.strip()
    if not text:
        return None
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def health_check() -> dict:
    """探针用：真发一次最小请求，而不是只检查 Key 是否存在。"""
    if not LLM_API_KEY:
        return {"status": "unconfigured", "reason": "未配置 LLM_API_KEY"}
    try:
        out = chat([{"role": "user", "content": "ping"}], max_tokens=8)
        return {"status": "ok", "model": LLM_MODEL, "echo": out[:20]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "down", "reason": str(exc)[:200]}
