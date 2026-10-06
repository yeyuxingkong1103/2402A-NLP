# -*- coding: utf-8 -*-
"""在线门控（Ollama 可用性）——在线级与用户级共用。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

纪律：服务不可用时**默认失败**，不允许把「跳过」当成通过；只有显式
``RAG_TEST_ALLOW_ONLINE_SKIP=1`` 才降级为 skip（并在报告里留痕）。
探测口径与产品一致：**0.5 s 超时、不重试**。
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from typing import Any

import pytest

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
PROBE_TIMEOUT_S = 0.5


def probe_ollama() -> dict[str, Any]:
    """同步探测 Ollama（0.5 s 超时、不重试），返回 ``{available, detail, probe_ms}``。"""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_S) as response:  # noqa: S310 —— 固定本机地址
            payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
        names = [row.get("name", "") for row in payload.get("models", [])]
        return {"available": True, "detail": f"模型 {names[:6]}",
                "probe_ms": round((time.perf_counter() - started) * 1000, 2)}
    except (urllib.error.URLError, socket.timeout, OSError, json.JSONDecodeError) as exc:
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}",
                "probe_ms": round((time.perf_counter() - started) * 1000, 2)}


def allow_skip() -> bool:
    """是否允许在线用例在服务不可用时显式跳过（默认否）。"""
    return os.environ.get("RAG_TEST_ALLOW_ONLINE_SKIP", "0") == "1"


def enforce(status: dict[str, Any]) -> None:
    """按探测结果决定：可用则放行；不可用则失败（或按环境变量显式跳过）。"""
    if status.get("available"):
        return
    message = (f"Ollama 不可用（{status.get('detail')}，探测 {status.get('probe_ms')} ms）；"
               f"在线/用户级必须在服务可用时运行 —— 不得以跳过代替通过")
    if allow_skip():
        pytest.skip(message)
    pytest.fail(message)
