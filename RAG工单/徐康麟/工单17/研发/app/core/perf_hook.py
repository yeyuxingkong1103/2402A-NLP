# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

性能优化钩子（app 层接入点）：``研发/app/core/perf_hook.py``。

## 为什么要这一层

工单17 的修复横跨 ``app/core/*``（业务内核）与 ``service/*``（部署层）。
业务内核被工单7/8/9/10 的 CLI 与脚本复用，**不能强依赖 service 包**。

因此本模块做一件事：**可选地**把 ``service.perf_config`` / ``service.http_pool``
导入进来；导入不到（例如在纯 CLI 环境跑）就返回 ``None``，
所有调用点据此退回基座行为 —— 这就是「可开关 + 不破坏既有调用方」的落地方式。

约定：

* :func:`perf` → ``PerfConfig | None``
* :func:`pool` → ``HttpPool | None``（**只有 ``PERF_ENABLED=1`` 时才非 None**）
* :func:`keep_alive` → Ollama 的 ``keep_alive`` 字符串；优化关闭时为基座的 ``"30m"``
* :func:`probe_cache` → 带 TTL 的探测缓存；优化关闭时返回 ``None``（每次真探）
"""

from __future__ import annotations

import os
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

#: 基座（工单10）的硬编码值 —— 优化关闭时必须原样保留，保证行为逐位一致
BASE_KEEP_ALIVE = "30m"

_perf: Any = None
_perf_loaded = False
_pool: Any = None
_pool_loaded = False


def perf() -> Any:
    """返回 ``PerfConfig``；``service`` 包不可用时返回 ``None``。"""
    global _perf, _perf_loaded
    if not _perf_loaded:
        _perf_loaded = True
        try:
            from service.perf_config import get_perf_config
            _perf = get_perf_config()
        except Exception:  # noqa: BLE001 —— 业务内核不依赖部署层
            _perf = None
    return _perf


def enabled() -> bool:
    """总开关。"""
    cfg = perf()
    return bool(cfg is not None and cfg.enabled)


def pool() -> Any:
    """返回 ``HttpPool``；未开启优化或不可用时返回 ``None``（调用方走 ``urlopen``）。"""
    global _pool, _pool_loaded
    if not enabled():
        return None
    if not _pool_loaded:
        _pool_loaded = True
        try:
            from service.http_pool import get_http_pool
            candidate = get_http_pool()
            _pool = candidate if candidate.enabled else None
        except Exception:  # noqa: BLE001
            _pool = None
    return _pool


def keep_alive() -> str:
    """Ollama ``keep_alive``：优化开启时取配置，否则用基座的 ``"30m"``。"""
    cfg = perf()
    if cfg is not None and cfg.enabled:
        return str(cfg.ollama_keep_alive or BASE_KEEP_ALIVE)
    return BASE_KEEP_ALIVE


def probe_cache() -> Any:
    """嵌入/生成后端探测缓存；未开启优化时返回 ``None``。"""
    cfg = perf()
    if cfg is None or not cfg.enabled or not cfg.probe_cache_enabled:
        return None
    try:
        from service.singletons import get_probe_cache
        return get_probe_cache("backend_probe")
    except Exception:  # noqa: BLE001
        return None


def answer_cache_limit(default: int) -> int:
    """答案缓存条数上限（优化关闭时为基座默认值）。"""
    cfg = perf()
    if cfg is not None and cfg.enabled:
        return int(cfg.answer_cache_max)
    return int(default)


def answer_cache_ttl(default: float = 0.0) -> float:
    """答案缓存 TTL 秒（优化关闭时为 0 = 永不过期，即基座行为）。"""
    cfg = perf()
    if cfg is not None and cfg.enabled:
        return float(cfg.answer_cache_ttl_s)
    return float(default)


def cache_chunk_chars(default: int = 0) -> int:
    """缓存条目瘦身阈值（0 = 不截断，即基座行为）。"""
    cfg = perf()
    if cfg is not None and cfg.enabled:
        return int(cfg.cache_chunk_chars)
    return int(default)


def env_flag(name: str, default: bool = False) -> bool:
    """裸环境变量布尔读取（避免本模块被迫依赖 perf_config）。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


__all__ = ["WORK_ORDER", "BASE_KEEP_ALIVE", "perf", "enabled", "pool", "keep_alive",
           "probe_cache", "answer_cache_limit", "answer_cache_ttl", "cache_chunk_chars",
           "env_flag"]
