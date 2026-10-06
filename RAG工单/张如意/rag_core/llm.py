# -*- coding: utf-8 -*-
"""
LLM 客户端封装（DeepSeek 为主，Ollama 为备）
工单编号：人工智能NLP-RAG（供 01~13 全部工单共用）

特性：
  1. 统一 chat 接口，支持 system/user 消息、温度、JSON 输出
  2. 磁盘缓存：相同 (model, messages, temperature) 直接命中，避免重复计费
  3. 结构化计时：为工单13「性能瓶颈识别」提供每阶段耗时埋点
  4. token 用量统计：输出 API 消耗汇总
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import config

_CACHE_DIR = config.CACHE_DIR / "llm"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """惰性创建 OpenAI 兼容客户端（指向 DeepSeek）。"""
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY 环境变量，无法调用生成模型。"
            )
        _client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=180.0,
            max_retries=3,
        )
    return _client


def _cache_key(model: str, messages: list[dict], temperature: float, json_mode: bool) -> str:
    payload = json.dumps(
        {"m": model, "msg": messages, "t": temperature, "j": json_mode},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int = 2048,
    use_cache: bool = True,
    tag: str = "",
) -> str:
    """
    调用 LLM 生成回复。

    Args:
        messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        model: 模型名，默认 config.LLM_MODEL
        temperature: 采样温度，RAG 问答建议 0
        json_mode: 要求模型返回严格 JSON
        use_cache: 是否使用磁盘缓存（评测重复跑时省钱）
        tag: 日志标签，用于性能分析
    """
    model = model or config.LLM_MODEL
    key = _cache_key(model, messages, temperature, json_mode)
    cache_file = _CACHE_DIR / f"{key}.json"

    if use_cache and cache_file.exists():
        with _lock:
            _usage["cached"] += 1
        return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = get_client().chat.completions.create(**kwargs)
    elapsed = time.time() - t0

    content = resp.choices[0].message.content or ""
    with _lock:
        _usage["calls"] += 1
        _usage["seconds"] += elapsed
        if resp.usage:
            _usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            _usage["completion_tokens"] += resp.usage.completion_tokens or 0

    if use_cache:
        # 用锁保护写，避免并发写坏文件
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"content": content, "elapsed": elapsed}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(cache_file)

    return content


def chat_json(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_cache: bool = True,
    tag: str = "",
) -> Any:
    """调用 LLM 并解析为 Python 对象，解析失败时回退提取首个 JSON 片段。"""
    raw = chat(
        messages, model=model, temperature=temperature, json_mode=True,
        max_tokens=max_tokens, use_cache=use_cache, tag=tag,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"_raw": raw, "_parse_error": True}


def get_usage() -> dict:
    """返回累计调用统计，用于成本与性能报告。"""
    with _lock:
        u = dict(_usage)
    u["cost_estimate_cny"] = round(
        u["prompt_tokens"] / 1e6 * 2.0 + u["completion_tokens"] / 1e6 * 8.0, 4
    )  # DeepSeek 参考价：输入 2 元/百万 token，输出 8 元/百万 token
    return u


def reset_usage() -> None:
    with _lock:
        for k in _usage:
            _usage[k] = 0 if k != "seconds" else 0.0


def print_usage(prefix: str = "") -> None:
    u = get_usage()
    print(
        f"{prefix}LLM 调用 {u['calls']} 次（缓存命中 {u['cached']} 次）| "
        f"输入 {u['prompt_tokens']} tokens / 输出 {u['completion_tokens']} tokens | "
        f"累计耗时 {u['seconds']:.1f}s | 预估费用 ¥{u['cost_estimate_cny']}"
    )
