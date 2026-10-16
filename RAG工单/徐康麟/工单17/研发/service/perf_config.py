# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

性能优化层配置：``研发/service/perf_config.py``。

设计纪律（共享约定 §4.3「阈值配置化」）：
    * 本模块**只做一件事** —— 把工单17 引入的全部性能阈值收敛到 ``PERF_*`` 环境变量；
    * ``PERF_ENABLED=0``（默认）时所有优化关闭，行为与工单10 基座**逐位一致**；
    * 未识别键不静默忽略：``describe()`` 会列出来，便于排障（不抛错，服务要能起来）。

优化的十项开关与阈值一览见 ``设计/优化方案设计.md`` §1 与 ``部署/配置/.env.example``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

#: 本模块识别的全部环境变量
KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "PERF_ENABLED",
        # F1 HTTP 连接池
        "PERF_HTTP_POOL_ENABLED",
        "PERF_HTTP_POOL_MAX_IDLE",
        "PERF_HTTP_POOL_TIMEOUT_S",
        "PERF_HTTP_POOL_RETRIES",
        # F2 后端探测缓存
        "PERF_PROBE_CACHE_TTL_S",
        "PERF_PROBE_CACHE_ENABLED",
        # F3/F4 限流与异步路由
        "PERF_ASYNC_ROUTES",
        "PERF_MAX_INFLIGHT",
        "PERF_QUEUE_MAX",
        "PERF_ACQUIRE_TIMEOUT_S",
        # F5 答案缓存
        "PERF_ANSWER_CACHE_MAX",
        "PERF_ANSWER_CACHE_TTL_S",
        "PERF_CACHE_CHUNK_CHARS",
        # F6 sqlite
        "PERF_SQLITE_WAL",
        "PERF_SQLITE_SYNCHRONOUS",
        "PERF_SQLITE_BATCH",
        "PERF_ANSWER_CACHE_TABLE_MAX",
        # F7 Ollama 驻留
        "PERF_OLLAMA_KEEP_ALIVE",
        "PERF_OLLAMA_IDLE_UNLOAD_S",
        # F8 会话文件
        "PERF_SESSION_FILE_MAX",
        "PERF_SESSION_SWEEP_S",
        # F9 解析任务队列
        "PERF_PARSE_BACKEND",
        "PERF_PARSE_WORKERS",
        "PERF_PARSE_QUEUE_MAX",
        "PERF_PARSE_MAX_PAGES",
        # F10 健康检查缓存
        "PERF_HEALTH_CACHE_S",
        # F11 日志轮转（磁盘泄漏）
        "PERF_LOG_MAX_MB",
        "PERF_LOG_BACKUP_COUNT",
        # F12 上传内容去重（磁盘泄漏）
        "PERF_UPLOAD_DEDUP",
        "PERF_UPLOAD_DEDUP_ALGO",
    }
)


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    return default if raw is None or str(raw).strip() == "" else str(raw).strip()


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


@dataclass(frozen=True)
class PerfConfig:
    """工单17 性能配置（全部可由环境变量覆盖）。"""

    # —— 总开关 ——
    enabled: bool = False

    # —— F1：HTTP keep-alive 连接池 ——
    http_pool_enabled: bool = True
    http_pool_max_idle: int = 8
    http_pool_timeout_s: float = 60.0
    http_pool_retries: int = 1

    # —— F2：嵌入/生成后端探测结果缓存 ——
    probe_cache_enabled: bool = True
    probe_cache_ttl_s: float = 60.0

    # —— F3/F4：限流与异步路由 ——
    async_routes: bool = True
    max_inflight: int = 8
    queue_max: int = 32
    acquire_timeout_s: float = 30.0

    # —— F5：答案缓存上限与 TTL ——
    answer_cache_max: int = 256
    answer_cache_ttl_s: float = 1800.0
    cache_chunk_chars: int = 400

    # —— F6：sqlite ——
    sqlite_wal: bool = True
    sqlite_synchronous: str = "NORMAL"
    sqlite_batch: int = 1
    answer_cache_table_max: int = 2000

    # —— F7：Ollama 驻留 ——
    ollama_keep_alive: str = "5m"
    ollama_idle_unload_s: float = 0.0      # 0 = 不自动卸载

    # —— F8：会话文件卷 ——
    session_file_max: int = 2000
    session_sweep_s: float = 300.0

    # —— F9：解析任务队列 ——
    parse_backend: str = "memory"          # memory | redis
    parse_workers: int = 2
    parse_queue_max: int = 32
    parse_max_pages: int = 40

    # —— F10：健康检查缓存 ——
    health_cache_s: float = 5.0

    # —— F11：日志轮转（磁盘泄漏：实测单文件 600 MB）——
    log_max_mb: int = 64
    log_backup_count: int = 5

    # —— F12：上传内容去重（磁盘泄漏：实测 2304 份文件只对应 10 组唯一内容，浪费 99.61%）——
    upload_dedup: bool = True
    upload_dedup_algo: str = "sha256"

    unknown_env: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["unknown_env"] = list(self.unknown_env)
        data["work_order"] = WORK_ORDER
        return data

    # -- 生效值（考虑总开关） ------------------------------------------ #
    def effective(self, name: str, default: Any) -> Any:
        """总开关关闭时回退到基座默认值（保证 ``PERF_ENABLED=0`` 逐位一致）。"""
        if not self.enabled:
            return default
        return getattr(self, name, default)


def load_perf_config(env: Mapping[str, str] | None = None) -> PerfConfig:
    """从环境变量构造 :class:`PerfConfig`。"""
    environ: Mapping[str, str] = env if env is not None else os.environ
    unknown = tuple(sorted(k for k in environ if k.startswith("PERF_") and k not in KNOWN_KEYS))
    enabled = _bool(environ, "PERF_ENABLED", False)
    return PerfConfig(
        enabled=enabled,
        http_pool_enabled=_bool(environ, "PERF_HTTP_POOL_ENABLED", True),
        http_pool_max_idle=_int(environ, "PERF_HTTP_POOL_MAX_IDLE", 8),
        http_pool_timeout_s=_float(environ, "PERF_HTTP_POOL_TIMEOUT_S", 60.0),
        http_pool_retries=_int(environ, "PERF_HTTP_POOL_RETRIES", 1),
        probe_cache_enabled=_bool(environ, "PERF_PROBE_CACHE_ENABLED", True),
        probe_cache_ttl_s=_float(environ, "PERF_PROBE_CACHE_TTL_S", 60.0),
        async_routes=_bool(environ, "PERF_ASYNC_ROUTES", True),
        max_inflight=_int(environ, "PERF_MAX_INFLIGHT", 8),
        queue_max=_int(environ, "PERF_QUEUE_MAX", 32),
        acquire_timeout_s=_float(environ, "PERF_ACQUIRE_TIMEOUT_S", 30.0),
        answer_cache_max=_int(environ, "PERF_ANSWER_CACHE_MAX", 256),
        answer_cache_ttl_s=_float(environ, "PERF_ANSWER_CACHE_TTL_S", 1800.0),
        cache_chunk_chars=_int(environ, "PERF_CACHE_CHUNK_CHARS", 400),
        sqlite_wal=_bool(environ, "PERF_SQLITE_WAL", True),
        sqlite_synchronous=_get(environ, "PERF_SQLITE_SYNCHRONOUS", "NORMAL"),
        sqlite_batch=_int(environ, "PERF_SQLITE_BATCH", 1),
        answer_cache_table_max=_int(environ, "PERF_ANSWER_CACHE_TABLE_MAX", 2000),
        ollama_keep_alive=_get(environ, "PERF_OLLAMA_KEEP_ALIVE", "5m"),
        ollama_idle_unload_s=_float(environ, "PERF_OLLAMA_IDLE_UNLOAD_S", 0.0),
        session_file_max=_int(environ, "PERF_SESSION_FILE_MAX", 2000),
        session_sweep_s=_float(environ, "PERF_SESSION_SWEEP_S", 300.0),
        parse_backend=_get(environ, "PERF_PARSE_BACKEND", "memory").lower(),
        parse_workers=_int(environ, "PERF_PARSE_WORKERS", 2),
        parse_queue_max=_int(environ, "PERF_PARSE_QUEUE_MAX", 32),
        parse_max_pages=_int(environ, "PERF_PARSE_MAX_PAGES", 40),
        health_cache_s=_float(environ, "PERF_HEALTH_CACHE_S", 5.0),
        log_max_mb=_int(environ, "PERF_LOG_MAX_MB", 64),
        log_backup_count=_int(environ, "PERF_LOG_BACKUP_COUNT", 5),
        upload_dedup=_bool(environ, "PERF_UPLOAD_DEDUP", True),
        upload_dedup_algo=_get(environ, "PERF_UPLOAD_DEDUP_ALGO", "sha256").lower(),
        unknown_env=unknown,
    )


_CACHE: PerfConfig | None = None


def get_perf_config(*, refresh: bool = False) -> PerfConfig:
    """进程级缓存（``refresh=True`` 强制重载；测试用）。"""
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = load_perf_config()
    return _CACHE


__all__ = ["WORK_ORDER", "KNOWN_KEYS", "PerfConfig", "load_perf_config", "get_perf_config"]
