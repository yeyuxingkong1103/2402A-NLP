# -*- coding: utf-8 -*-
"""用户级：界面契约（Streamlit 真应用 + 纯标准库备用界面）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定依据（``设计/接口设计.md`` §3.25）：
    * 两套界面**只允许**调用 ``app.core.qa_engine``（不得各自实现检索/生成/引用）；
    * ``streamlit_app.py`` 顶层 ``import streamlit``（算力云可跑）；本机 import 失败时
      **不得**被 ``serve_fallback.py`` 间接导入；
    * ``serve_fallback.py`` 用 ``http.server``（``ThreadingHTTPServer``），
      路由 ``GET /``、``POST /api/ask``、``GET /api/files``、``GET /api/health``，默认 ``127.0.0.1:8501``；
    * 黑盒冒烟：以子进程启动备用界面（端口用 ``RAG_SERVER__PORT`` 覆盖），
      实际请求 ``GET /`` 与 ``GET /api/health`` 必须 200。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from common import paths

pytestmark = [pytest.mark.user]


def _free_port() -> int:
    """取一个空闲端口（避免与占用的 8501 冲突）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_streamlit_app_is_real_streamlit(ui_files: dict[str, Any]) -> None:
    """``streamlit_app.py`` 必须是真 Streamlit 应用：顶层 import + 调用 app/core。"""
    text = ui_files["streamlit_text"]
    assert ui_files["streamlit"].exists(), "缺少 app/ui/streamlit_app.py"
    head = "\n".join(text.splitlines()[:40])
    assert "import streamlit" in head, "顶层必须 import streamlit（算力云可跑）"
    assert "qa_engine" in text, "界面必须复用 app.core.qa_engine，不得自实现检索/生成"


def test_fallback_ui_is_stdlib_only(ui_files: dict[str, Any]) -> None:
    """备用界面只用标准库：不得 import streamlit，也不得间接导入 streamlit_app。"""
    text = ui_files["fallback_text"]
    assert ui_files["fallback"].exists(), "缺少 app/ui/serve_fallback.py"
    assert "ThreadingHTTPServer" in text, "备用界面应使用 http.server.ThreadingHTTPServer"
    assert "import streamlit" not in text, "备用界面不得 import streamlit（本机不可用）"
    assert "streamlit_app" not in text, "备用界面不得间接导入 streamlit_app"
    for route in ("/api/ask", "/api/files", "/api/health"):
        assert route in text, f"备用界面缺少路由 {route}"
    assert "8501" in text, "备用界面默认端口应为 8501"


def test_fallback_ui_smoke_blackbox() -> None:
    """黑盒冒烟：子进程启动备用界面 → ``GET /`` 返回 HTML、``GET /api/health`` 返回 JSON。"""
    script = paths.DEV_DIR / "app" / "ui" / "serve_fallback.py"
    if not script.exists():
        pytest.fail(f"备用界面脚本缺失：{paths.display(script)}")
    if not paths.PYTHON_EXE.exists():
        pytest.fail(f"统一解释器缺失：{paths.PYTHON_EXE}")

    port = _free_port()
    env = dict(os.environ)
    env.update({
        "RAG_SERVER__HOST": "127.0.0.1",
        "RAG_SERVER__PORT": str(port),
        "RAG_PY_CWD": str(paths.REPO_ROOT),
        "PYTHONIOENCODING": "utf-8",
    })
    proc = subprocess.Popen(
        [str(paths.PYTHON_EXE), str(script)], cwd=str(paths.REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 180.0
        last_error = ""
        health: dict[str, Any] | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                output = (proc.stdout.read() if proc.stdout else "")[-2000:]
                pytest.fail(f"备用界面进程提前退出（code={proc.returncode}）：{output}")
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=3.0) as response:
                    health = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
                break
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0)
        assert health is not None, f"备用界面 180 s 未就绪（最后错误 {last_error}）"
        assert health.get("ok") is True, f"/api/health 未返回 ok=true：{health}"

        with urllib.request.urlopen(base + "/", timeout=10.0) as response:
            html = response.read().decode("utf-8", errors="replace")
        assert response.status == 200
        assert "<html" in html.lower(), "首页未返回 HTML"
        assert "api/ask" in html or "fetch(" in html, "首页缺少提问入口（无 JS 调用 /api/ask）"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:                      # pragma: no cover - 兜底强杀
            proc.kill()
