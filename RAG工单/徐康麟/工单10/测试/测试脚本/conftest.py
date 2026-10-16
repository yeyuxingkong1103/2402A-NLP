# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

pytest 公共夹具：``测试/测试脚本/conftest.py``。

约定（阈值全部环境变量可覆盖，测试代码里不写死）：
    * ``RAG_TEST_API_BASE``    被测 API 基址，默认 ``http://127.0.0.1:8200``（= 容器 rag-api 的宿主映射端口）
    * ``RAG_TEST_UI_BASE``     被测 UI  基址，默认 ``http://127.0.0.1:8201``（= 容器 rag-ui 的宿主映射端口）
    * ``RAG_TEST_ASK_TIMEOUT`` 问答请求超时（秒），默认 120
    * ``RAG_TEST_CONCURRENCY`` 并发探活线程数，默认 8
    * ``RAG_TEST_OLLAMA``      宿主机 Ollama 基址，默认 ``http://127.0.0.1:11434``
    * ``RAG_TEST_HOST_ALIAS``  容器内访问宿主机用的域名，默认 ``host.docker.internal``
    * ``RAG_TEST_ROOT``        工单根目录（默认由本文件位置推断）

设计取舍：**不 import 被测应用进程内的对象**。所有断言都走真实 HTTP / 真实文件系统，
这样「本机等价验证」与「容器内运行」用的是同一套测试，换基址即可。
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or str(raw).strip() == "" else str(raw).strip()


API_BASE = _env("RAG_TEST_API_BASE", "http://127.0.0.1:8200").rstrip("/")
UI_BASE = _env("RAG_TEST_UI_BASE", "http://127.0.0.1:8201").rstrip("/")
ASK_TIMEOUT = float(_env("RAG_TEST_ASK_TIMEOUT", "120"))
HEALTH_TIMEOUT = float(_env("RAG_TEST_HEALTH_TIMEOUT", "30"))
CONCURRENCY = int(_env("RAG_TEST_CONCURRENCY", "8"))
OLLAMA_BASE = _env("RAG_TEST_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
HOST_ALIAS = _env("RAG_TEST_HOST_ALIAS", "host.docker.internal")
DATA_DIR = REPO_ROOT / "测试" / "测试数据"
DEPLOY_DIR = REPO_ROOT / "部署"

#: 本次 pytest 会话的起点（unix 秒）：供"陈旧残留 vs 本次新写入"的判定使用
#: （见 test_02_卷挂载与数据可见.py::test_索引卷为只读挂载语义）。
#: 在模块导入期就写入环境变量，保证任何被测进程都不会比它更晚才开始写。
if not os.environ.get("RAG_TEST_SESSION_START"):
    os.environ["RAG_TEST_SESSION_START"] = f"{time.time():.3f}"


def pytest_configure(config) -> None:   # pragma: no cover —— pytest 钩子
    """打印本次会话起点与目标地址，便于测试报告核对时间口径。"""
    print(f"\n[pytest] 会话起点 RAG_TEST_SESSION_START={os.environ['RAG_TEST_SESSION_START']}")
    print(f"[pytest] API={API_BASE}  UI={UI_BASE}  并发={CONCURRENCY}")


def _reachable(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"{type(exc).__name__}: {exc}"


@pytest.fixture(scope="session")
def api_base() -> str:
    return API_BASE


@pytest.fixture(scope="session")
def ui_base() -> str:
    return UI_BASE


@pytest.fixture(scope="session")
def data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


@pytest.fixture(scope="session")
def http() -> requests.Session:
    session = requests.Session()
    session.headers.update({"user-agent": "gongdan10-pytest/1.0"})
    return session


@pytest.fixture(scope="session")
def api_health(http: requests.Session) -> dict[str, Any]:
    """拉一次 ``GET /health`` 的完整 JSON（服务未起则该模块 skip，不算失败）。"""
    ok, detail = _reachable(f"{API_BASE}/healthz", timeout=5)
    if not ok:
        pytest.skip(f"rag-api 未就绪（{API_BASE}）：{detail} —— 请先跑 部署/脚本/本地等价验证.ps1")
    resp = http.get(f"{API_BASE}/health", timeout=HEALTH_TIMEOUT)
    assert resp.status_code == 200, f"/health 返回 {resp.status_code}"
    return resp.json()


@pytest.fixture(scope="session")
def save_evidence(data_dir: Path):
    """把真实响应原样落盘到 ``测试/测试数据/``（证据可追溯，禁止编造）。"""

    def _save(name: str, payload: Any) -> Path:
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, (dict, list)):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path.write_text(str(payload), encoding="utf-8")
        return path

    return _save


def tcp_open(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """TCP 连通性探测（容器网络验收用：能连上即说明网络通路 OK）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"TCP {host}:{port} 可连接"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
