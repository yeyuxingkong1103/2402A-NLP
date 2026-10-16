# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

容器部署层配置：``研发/service/config.py``。

设计纪律（共享约定 §4.3「阈值配置化」）：
    本模块**只做一件事**——把「部署相关」的阈值全部收敛到环境变量，绝不硬编码在业务代码里。
    业务侧阈值（RAG_RETRIEVAL__* / RAG_ANSWER__* …）由 ``app/core/config.py`` 负责，本模块不重复。

环境变量一览见 ``部署/配置/.env.example``，每个键都在这里给出默认值。

约定：
    * 所有键以 ``DEPLOY_`` 开头，避免与 ``RAG_*``（业务）冲突；
    * 未识别键不静默忽略：``describe()`` 会把它打印出来，便于排障（不抛错，容器要能起来）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

#: 本模块识别的全部环境变量（未列出的 DEPLOY_* 会在 describe() 里以 unknown 列出）
KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "DEPLOY_API_HOST",
        "DEPLOY_API_PORT",
        "DEPLOY_API_TITLE",
        "DEPLOY_LOG_DIR",
        "DEPLOY_LOG_LEVEL",
        "DEPLOY_LOG_JSON",
        "DEPLOY_SESSION_DIR",
        "DEPLOY_SESSION_BACKEND",
        "DEPLOY_SESSION_TTL_S",
        "DEPLOY_SESSION_MAX_TURNS",
        "DEPLOY_REDIS_URL",
        "DEPLOY_REDIS_ENABLED",
        "DEPLOY_REDIS_TIMEOUT_S",
        "DEPLOY_REDIS_KEY_PREFIX",
        "DEPLOY_HEALTH_TIMEOUT_S",
        "DEPLOY_ASK_TIMEOUT_S",
        "DEPLOY_MAX_CONCURRENCY",
        "DEPLOY_WARMUP",
        "DEPLOY_WARMUP_WITH_GENERATE",
        "DEPLOY_UPLOAD_DIR",
        "DEPLOY_RAG_DATA_DIR",
        "DEPLOY_INDEX_DIR",
        "DEPLOY_MODEL_DIR",
        "DEPLOY_RUNTIME_DIR",
        "DEPLOY_UI_PORT",
        "DEPLOY_UI_HOST",
        "DEPLOY_METRICS_ENABLED",
        "DEPLOY_BUILD_REF",
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
class DeployConfig:
    """部署层配置（容器端口 / 卷路径 / 会话后端 / 阈值）。"""

    # —— API 进程 ——
    api_host: str = "0.0.0.0"
    api_port: int = 8200
    api_title: str = "金融问答系统容器 API（工单10）"

    # —— 日志（卷：rag-logs）——
    log_dir: str = "/app/logs"
    log_level: str = "INFO"
    log_json: bool = True

    # —— 会话（卷：rag-sessions，容器间共享）——
    session_dir: str = "/app/sessions"
    session_backend: str = "auto"          # auto | redis | file
    session_ttl_s: int = 86400
    session_max_turns: int = 20

    # —— Redis（可选外部/编排内服务）——
    redis_url: str = "redis://redis:6379/0"
    redis_enabled: bool = True
    redis_timeout_s: float = 0.5
    redis_key_prefix: str = "rag:finqa:"

    # —— 阈值 ——
    health_timeout_s: float = 2.0
    ask_timeout_s: float = 60.0
    max_concurrency: int = 4

    # —— 启动行为 ——
    warmup: bool = True
    warmup_with_generate: bool = False     # 默认只预热检索（冷启动可控），见 优化/部署优化说明.md

    # —— 卷路径（只读/读写）——
    upload_dir: str = "/app/data/uploads"
    rag_data_dir: str = "/app/data"
    index_dir: str = "/app/data/index"
    model_dir: str = "/app/data/models"
    #: 运行期可写目录：qa_runtime.sqlite3（问答历史）与答案缓存落这里。
    #: **必须与 index_dir 分开** —— 索引卷是只读挂载（:ro），写进去会失败。
    runtime_dir: str = "/app/data/runtime"

    # —— UI 容器 ——
    ui_host: str = "0.0.0.0"
    ui_port: int = 8201

    # —— 其它 ——
    metrics_enabled: bool = True
    build_ref: str = "dev"

    unknown_env: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """脱敏后的字典（本模块无密钥，直接返回）。"""
        data = {k: v for k, v in self.__dict__.items()}
        data["unknown_env"] = list(self.unknown_env)
        return data

    @property
    def resolved_session_backend(self) -> str:
        """``auto`` → ``redis``（配置启用时由运行期探测决定真实可用性）。"""
        if self.session_backend in {"redis", "file"}:
            return self.session_backend
        return "redis" if self.redis_enabled else "file"


def load_deploy_config(env: Mapping[str, str] | None = None) -> DeployConfig:
    """从环境变量构造 :class:`DeployConfig`（缺省值即容器镜像内的默认值）。"""
    environ: Mapping[str, str] = env if env is not None else os.environ
    unknown = tuple(sorted(k for k in environ if k.startswith("DEPLOY_") and k not in KNOWN_KEYS))
    return DeployConfig(
        api_host=_get(environ, "DEPLOY_API_HOST", "0.0.0.0"),
        api_port=_int(environ, "DEPLOY_API_PORT", 8200),
        api_title=_get(environ, "DEPLOY_API_TITLE", "金融问答系统容器 API（工单10）"),
        log_dir=_get(environ, "DEPLOY_LOG_DIR", "/app/logs"),
        log_level=_get(environ, "DEPLOY_LOG_LEVEL", "INFO").upper(),
        log_json=_bool(environ, "DEPLOY_LOG_JSON", True),
        session_dir=_get(environ, "DEPLOY_SESSION_DIR", "/app/sessions"),
        session_backend=_get(environ, "DEPLOY_SESSION_BACKEND", "auto").lower(),
        session_ttl_s=_int(environ, "DEPLOY_SESSION_TTL_S", 86400),
        session_max_turns=_int(environ, "DEPLOY_SESSION_MAX_TURNS", 20),
        redis_url=_get(environ, "DEPLOY_REDIS_URL", "redis://redis:6379/0"),
        redis_enabled=_bool(environ, "DEPLOY_REDIS_ENABLED", True),
        redis_timeout_s=_float(environ, "DEPLOY_REDIS_TIMEOUT_S", 0.5),
        redis_key_prefix=_get(environ, "DEPLOY_REDIS_KEY_PREFIX", "rag:finqa:"),
        health_timeout_s=_float(environ, "DEPLOY_HEALTH_TIMEOUT_S", 2.0),
        ask_timeout_s=_float(environ, "DEPLOY_ASK_TIMEOUT_S", 60.0),
        max_concurrency=_int(environ, "DEPLOY_MAX_CONCURRENCY", 4),
        warmup=_bool(environ, "DEPLOY_WARMUP", True),
        warmup_with_generate=_bool(environ, "DEPLOY_WARMUP_WITH_GENERATE", False),
        upload_dir=_get(environ, "DEPLOY_UPLOAD_DIR", "/app/data/uploads"),
        rag_data_dir=_get(environ, "DEPLOY_RAG_DATA_DIR", "/app/data"),
        index_dir=_get(environ, "DEPLOY_INDEX_DIR", "/app/data/index"),
        model_dir=_get(environ, "DEPLOY_MODEL_DIR", "/app/data/models"),
        runtime_dir=_get(environ, "DEPLOY_RUNTIME_DIR", "/app/data/runtime"),
        ui_host=_get(environ, "DEPLOY_UI_HOST", "0.0.0.0"),
        ui_port=_int(environ, "DEPLOY_UI_PORT", 8201),
        metrics_enabled=_bool(environ, "DEPLOY_METRICS_ENABLED", True),
        build_ref=_get(environ, "DEPLOY_BUILD_REF", "dev"),
        unknown_env=unknown,
    )


_CACHE: DeployConfig | None = None


def get_deploy_config(*, refresh: bool = False) -> DeployConfig:
    """进程级缓存（``refresh=True`` 强制重载；测试用）。"""
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = load_deploy_config()
    return _CACHE
