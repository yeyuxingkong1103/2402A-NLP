# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

pytest 公共夹具：``测试/conftest.py``。

纪律：
    * **不启动 2.4 GB 的服务本体** —— 全部用例都是轻量单元/集成测试（秒级）；
    * 需要「真 HTTP」的用例自建 stdlib ``http.server`` 或最小 FastAPI 应用，
      不依赖工单17 的真实 API 进程（那是压测脚本的职责）；
    * 所有阈值从 ``PERF_*`` / ``RAG_*`` 环境变量取，测试里只设**测试专用**的值，
      并在用例结束后还原（避免污染同一进程里的其它用例）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[1]
_RESEARCH = _ROOT / "研发"
for path in (str(_RESEARCH), str(_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _reset_perf_hook() -> None:
    """重置 ``app.core.perf_hook`` 的进程级缓存。

    ⚠️ 实测踩坑：``perf_hook`` 把 ``PerfConfig`` 缓存在模块级 ``_perf``，
    只刷新 ``service.perf_config`` **不足以**让用例之间隔离 ——
    上一个用例留下的 ``enabled=True`` 会让 ``PERF_ENABLED=0`` 的对照组用例
    仍然走优化路径（实测表现为「基座口径应探测 5 次，实际 1 次」）。
    """
    try:
        from app.core import perf_hook

        perf_hook._perf = None          # noqa: SLF001 —— 测试专用重置
        perf_hook._perf_loaded = False  # noqa: SLF001
        perf_hook._pool = None          # noqa: SLF001
        perf_hook._pool_loaded = False  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture()
def perf_env(monkeypatch):
    """把优化开关打开并把阈值调到测试友好值；用完自动还原。"""
    monkeypatch.setenv("PERF_ENABLED", "1")
    monkeypatch.setenv("PERF_ANSWER_CACHE_MAX", "8")
    monkeypatch.setenv("PERF_ANSWER_CACHE_TTL_S", "0.4")
    monkeypatch.setenv("PERF_MAX_INFLIGHT", "2")
    monkeypatch.setenv("PERF_QUEUE_MAX", "2")
    monkeypatch.setenv("PERF_ACQUIRE_TIMEOUT_S", "0.3")
    monkeypatch.setenv("PERF_PROBE_CACHE_TTL_S", "30")
    monkeypatch.setenv("PERF_SQLITE_BATCH", "1")
    monkeypatch.setenv("PERF_ANSWER_CACHE_TABLE_MAX", "50")
    from service import perf_config

    perf_config.get_perf_config(refresh=True)
    _reset_perf_hook()
    yield perf_config.get_perf_config()
    monkeypatch.undo()
    perf_config.get_perf_config(refresh=True)
    _reset_perf_hook()


@pytest.fixture()
def base_env(monkeypatch):
    """优化关闭（基座口径）。"""
    monkeypatch.setenv("PERF_ENABLED", "0")
    from service import perf_config

    perf_config.get_perf_config(refresh=True)
    _reset_perf_hook()
    yield perf_config.get_perf_config()
    monkeypatch.undo()
    perf_config.get_perf_config(refresh=True)
    _reset_perf_hook()


@pytest.fixture()
def tmp_path(request):
    """覆盖 pytest 内建 ``tmp_path``：把临时目录放到**工作区内**。

    ⚠️ 实测踩坑：本机沙箱对系统临时目录（``%TEMP%\\pytest-*``）与 ``--basetemp``
    都返回 ``PermissionError: [WinError 5] 拒绝访问``（连 ``scandir`` 都不行），
    用例会直接 ERROR。改用工作区内的 ``测试/临时数据/<用例名>``，用完立即删除。
    """
    import shutil

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.node.name)[:60]
    base = _ROOT / "测试" / "临时数据" / safe
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def reset_singletons() -> None:
    """重置进程级单例与探测缓存（用例之间必须隔离）。"""
    try:
        from service import singletons
        from service.http_pool import get_http_pool
        from service.limiter import get_gate

        singletons.reset_all()
        get_http_pool(refresh=True).reset()
        get_gate(refresh=True)
    except Exception:  # noqa: BLE001
        pass
