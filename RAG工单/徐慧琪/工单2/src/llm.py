# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：LLM 调用（Ollama）

选型：qwen2.5:3b（本机 Ollama 已 pull）
理由：
  - 工单优先级 "优先 Qwen 系列（qwen2.5-7b/14b）"，qwen2.5 是其首选家族；
  - 本机 Ollama 已离线存有 qwen2.5:3b / 0.5b，调用**不会触发任何下载**；
  - 3B 在 RTX 4060 Laptop 上可满足工单"响应时间不超过 3 秒"的目标；
    若需更高答案质量可改 config.OLLAMA_MODEL 为更大的本地模型。

硬约束遵守：本模块**只调用 /api/chat 与 /api/generate**，
绝不调用 /api/pull，因此不存在拉取模型的代码路径。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401

import json
import time
from typing import Iterator

import requests

from src import config

_session = requests.Session()
_available_models: list[str] | None = None


class LLMUnavailable(RuntimeError):
    """Ollama 不可用或模型缺失。"""


# ---------------------------------------------------------------------------
# 可用性
# ---------------------------------------------------------------------------
def list_models(force: bool = False) -> list[str]:
    """列出 Ollama 本地已有模型（读 /api/tags，不会下载）。"""
    global _available_models
    if _available_models is not None and not force:
        return _available_models
    try:
        r = _session.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        _available_models = [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:
        raise LLMUnavailable(
            f"无法连接 Ollama（{config.OLLAMA_HOST}）：{exc}\n"
            f"请先启动 Ollama 服务：ollama serve") from exc
    return _available_models


def resolve_model() -> str:
    """确认配置的模型存在；缺失时回退到本机其他可用模型，绝不下拉。"""
    models = list_models()
    if config.OLLAMA_MODEL in models:
        return config.OLLAMA_MODEL
    # 尝试带/不带 :latest 后缀的等价名
    for m in models:
        if m.split(":")[0] == config.OLLAMA_MODEL.split(":")[0]:
            return m
    if config.OLLAMA_FALLBACK_MODEL in models:
        print(f"[llm] 未找到 {config.OLLAMA_MODEL}，回退到 "
              f"{config.OLLAMA_FALLBACK_MODEL}（本机已有，未下载）", flush=True)
        return config.OLLAMA_FALLBACK_MODEL
    raise LLMUnavailable(
        f"本机 Ollama 中没有可用生成模型。已安装: {models}\n"
        f"请在 config.py 中把 OLLAMA_MODEL 指向其中之一（禁止下载新模型）。")


def health() -> dict:
    """健康检查，供界面/CLI 展示。"""
    try:
        models = list_models(force=True)
        model = resolve_model()
        return {"ok": True, "host": config.OLLAMA_HOST, "model": model,
                "models": models}
    except Exception as exc:
        return {"ok": False, "host": config.OLLAMA_HOST, "error": str(exc),
                "models": []}


def unload() -> None:
    """请求 Ollama 立即卸载模型、释放显存。

    用途：本机 8G 显存上，Ollama 常驻（keep_alive=30m）会挤占
    embedding/reranker 的加载空间，实测会导致 cross-encoder 加载时段错误
    （进程级崩溃，Python 无法捕获）。warmup 在加载本地模型前调用本函数，
    随后 llm.warmup() 会按需重新加载（冷启动 ~3s，仅发生在启动阶段）。
    """
    try:
        model = resolve_model()
        _session.post(f"{config.OLLAMA_HOST}/api/generate",
                      json={"model": model, "keep_alive": 0,
                            "options": {"num_predict": 1}},
                      timeout=15)
    except Exception:
        pass


def warmup() -> None:
    """预热：让 Ollama 把权重加载进显存，避免首个问题被冷启动拖慢。

    工单要求响应时间 <= 3 秒；冷加载 3B 模型通常要数秒，
    因此服务启动时调用一次本函数，并保持模型常驻。

    【必须与真实调用用同一套 options】实测发现：Ollama 把 num_ctx 当作
    模型实例的一部分，预热若用默认 2048、真实调用用 8192，它会**重新加载一遍
    权重**，首个问题因此多花 6 秒（实测第 260 题总耗时 8.96s，其中分析阶段
    占 6.48s）。这里显式带上 num_ctx，让预热加载的实例与实际调用完全一致。
    """
    model = resolve_model()
    try:
        _session.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 1, "num_ctx": config.OLLAMA_NUM_CTX},
                  "keep_alive": "30m"},
            timeout=config.LLM_TIMEOUT_S,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def chat(messages: list[dict], temperature: float | None = None,
         num_predict: int | None = None, model: str | None = None) -> dict:
    """同步对话。返回 {answer, elapsed, model, eval_count, prompt_eval_count}。"""
    model = model or resolve_model()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": (config.OLLAMA_TEMPERATURE if temperature is None
                            else temperature),
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": num_predict or config.OLLAMA_NUM_PREDICT,
        },
    }
    t0 = time.time()
    try:
        r = _session.post(f"{config.OLLAMA_HOST}/api/chat", json=payload,
                          timeout=config.LLM_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailable(f"LLM 调用失败：{exc}") from exc

    elapsed = time.time() - t0
    answer = (data.get("message") or {}).get("content", "").strip()
    return {
        "answer": answer,
        "elapsed": elapsed,
        "model": data.get("model", model),
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        # Ollama 自报的生成耗时（不含排队/加载），用于区分真实推理速度
        "eval_duration_s": data.get("eval_duration", 0) / 1e9,
    }


def chat_stream(messages: list[dict], temperature: float | None = None,
                model: str | None = None) -> Iterator[str]:
    """流式对话，逐段 yield 文本增量（界面打字机效果用）。"""
    model = model or resolve_model()
    payload = {
        "model": model, "messages": messages, "stream": True, "keep_alive": "30m",
        "options": {"temperature": (config.OLLAMA_TEMPERATURE if temperature is None
                                    else temperature),
                    "num_ctx": config.OLLAMA_NUM_CTX,
                    "num_predict": config.OLLAMA_NUM_PREDICT},
    }
    with _session.post(f"{config.OLLAMA_HOST}/api/chat", json=payload,
                       timeout=config.LLM_TIMEOUT_S, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = (obj.get("message") or {}).get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


def generate(prompt: str, temperature: float | None = None,
             num_predict: int | None = None, model: str | None = None) -> dict:
    """单轮补全（RAGAS 评估等场景使用）。"""
    return chat([{"role": "user", "content": prompt}],
                temperature=temperature, num_predict=num_predict, model=model)


def info() -> dict:
    """返回 LLM 配置摘要。"""
    h = health()
    return {"model": h.get("model", config.OLLAMA_MODEL),
            "host": config.OLLAMA_HOST,
            "available": h["ok"],
            "local_pulled": h.get("models", []),
            "download_disabled": True}
