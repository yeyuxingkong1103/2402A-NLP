# -*- coding: utf-8 -*-
"""在线阶段配置加载。

优先级：内置默认值 < ``config.yaml`` < ``RAG2_*`` 环境变量。

    from rag2 import load_config

    cfg = load_config()
    print(cfg.llm.model, cfg.redis.prefix, cfg.milvus.collection)
"""

from __future__ import annotations

import copy
import dataclasses as _dc
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 项目根目录（rag2/ 的上一级，即 RAG_2/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 配置节（dataclass，字段默认值即内置默认配置）
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    name: str = "RAG2 在线问答服务"
    env: str = "dev"
    log_dir: str = "logs"
    log_level: str = "INFO"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "Qwen3.5:4B"
    temperature: float = 0.2
    max_tokens: int = 1024
    reasoning_effort: str | None = "none"  # 关闭 Qwen 思考链，避免结构化输出被截断
    timeout: float = 300.0


@dataclass
class MilvusConfig:
    uri: str = "http://localhost:19530"
    collection: str = "agriculture_knowledge"
    dim: int = 1024
    embed_model: str = "D:/modelscope/bge-m3"
    rerank_model: str = "D:/modelscope/bge-reranker-v2-m3"
    device: str = "cpu"


@dataclass
class RedisConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str = ""
    prefix: str = "rag2:"
    history_ttl: int = 86400   # 会话/消息的短期记忆过期时间（秒）
    token_ttl: int = 43200     # 登录令牌有效期（秒）


@dataclass
class RetrievalConfig:
    top_k: int = 6
    mode: str = "hybrid"       # dense / bm25 / hybrid
    rerank: bool = True
    rerank_top_k: int = 12


@dataclass
class MemoryConfig:
    short_term_turns: int = 5  # 拼进提示词的最近对话轮数


@dataclass
class AdminConfig:
    usernames: list[str] = field(default_factory=lambda: ["admin"])  # 管理员用户名列表


@dataclass
class RAG2Config:
    app: AppConfig = field(default_factory=AppConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    admin: AdminConfig = field(default_factory=AdminConfig)
    roles_path: str = "roles.yaml"

    def roles_file(self) -> Path:
        """解析角色配置文件路径（相对路径以项目根目录为基准）。"""
        p = Path(self.roles_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


# 默认配置（字段默认值同义）
DEFAULTS = RAG2Config()


# ---------------------------------------------------------------------------
# 从字典构建 / 环境变量覆盖
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _coerce(value: Any, default: Any) -> Any:
    """按字段默认值类型做轻量类型转换。"""
    if value is None:
        return None
    if isinstance(default, bool):
        return value if isinstance(value, bool) else _as_bool(str(value))
    if isinstance(default, int) and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _section(cls: type, data: dict[str, Any]) -> Any:
    """用映射构建一个 dataclass（只取已知字段，嵌套 dataclass 递归构建）。"""
    kwargs: dict[str, Any] = {}
    for f in _dc.fields(cls):
        if f.name not in data:
            continue
        if f.default is not _dc.MISSING:
            default = f.default
        elif f.default_factory is not _dc.MISSING:  # pragma: no cover - 防御分支
            default = f.default_factory()
        else:
            default = None
        value = data[f.name]
        if _dc.is_dataclass(default) and isinstance(value, dict):
            kwargs[f.name] = _section(type(default), value)
        else:
            kwargs[f.name] = _coerce(value, default)
    return cls(**kwargs)


# 环境变量 → (配置节, 字段, 转换函数)；配置节为 None 表示顶层字段
_ENV_MAP: dict[str, tuple[str | None, str, Any]] = {
    "RAG2_HOST": ("app", "host", str),
    "RAG2_PORT": ("app", "port", int),
    "RAG2_LOG_DIR": ("app", "log_dir", str),
    "RAG2_LOG_LEVEL": ("app", "log_level", str),
    "RAG2_LLM_BASE_URL": ("llm", "base_url", str),
    "RAG2_LLM_API_KEY": ("llm", "api_key", str),
    "RAG2_LLM_MODEL": ("llm", "model", str),
    "RAG2_LLM_TEMPERATURE": ("llm", "temperature", float),
    "RAG2_LLM_MAX_TOKENS": ("llm", "max_tokens", int),
    "RAG2_MILVUS_URI": ("milvus", "uri", str),
    "RAG2_MILVUS_COLLECTION": ("milvus", "collection", str),
    "RAG2_MILVUS_DIM": ("milvus", "dim", int),
    "RAG2_MILVUS_EMBED_MODEL": ("milvus", "embed_model", str),
    "RAG2_MILVUS_RERANK_MODEL": ("milvus", "rerank_model", str),
    "RAG2_MILVUS_DEVICE": ("milvus", "device", str),
    "RAG2_REDIS_HOST": ("redis", "host", str),
    "RAG2_REDIS_PORT": ("redis", "port", int),
    "RAG2_REDIS_DB": ("redis", "db", int),
    "RAG2_REDIS_PASSWORD": ("redis", "password", str),
    "RAG2_REDIS_PREFIX": ("redis", "prefix", str),
    "RAG2_REDIS_HISTORY_TTL": ("redis", "history_ttl", int),
    "RAG2_REDIS_TOKEN_TTL": ("redis", "token_ttl", int),
    "RAG2_TOP_K": ("retrieval", "top_k", int),
    "RAG2_MODE": ("retrieval", "mode", str),
    "RAG2_RERANK": ("retrieval", "rerank", _as_bool),
    "RAG2_RERANK_TOP_K": ("retrieval", "rerank_top_k", int),
    "RAG2_SHORT_TERM_TURNS": ("memory", "short_term_turns", int),
    "RAG2_ADMIN_USERS": (
        "admin",
        "usernames",
        lambda s: [x.strip() for x in s.split(",") if x.strip()],
    ),
    "RAG2_ROLES_PATH": (None, "roles_path", str),
}


def _apply_env(data: dict[str, Any]) -> None:
    for env_key, (section, field_name, cast) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            value = raw
        if section is None:
            data[field_name] = value
        else:
            data.setdefault(section, {})[field_name] = value


def _build(data: dict[str, Any]) -> RAG2Config:
    return _section(RAG2Config, data)


def load_config(path: str | Path | None = None) -> RAG2Config:
    """加载配置：内置默认值 < ``config.yaml`` < ``RAG2_*`` 环境变量。"""
    data = _dc.asdict(DEFAULTS)

    yaml_path = Path(path) if path else (PROJECT_ROOT / "config.yaml")
    if yaml_path.is_file():
        import yaml  # 懒加载，避免 import rag2 强依赖 PyYAML

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config.yaml 根节点必须是映射：{yaml_path}")
        _deep_merge(data, loaded)

    _apply_env(data)
    return _build(data)


# ---------------------------------------------------------------------------
# 管理员在线改配置（白名单 + 校验 + 就地生效 + 落盘）
# ---------------------------------------------------------------------------
# 可在线修改的字段白名单（其余如 app.port / milvus.uri / redis.host 需重启生效）
EDITABLE_FIELDS: dict[str, set[str]] = {
    "app": {"host", "port"},
    "llm": {"base_url", "api_key", "model", "temperature", "max_tokens"},
    "retrieval": {"top_k", "mode", "rerank", "rerank_top_k"},
    "memory": {"short_term_turns"},
    "redis": {"history_ttl", "token_ttl"},
    "admin": {"usernames"},
}

_RETRIEVAL_MODES = {"dense", "bm25", "hybrid"}
_POSITIVE_INT_FIELDS = {"top_k", "rerank_top_k", "short_term_turns", "history_ttl", "token_ttl"}


def validate_updates(updates: dict[str, Any]) -> list[str]:
    """校验待写入的配置更新，返回错误列表（空表示合法）。"""
    errors: list[str] = []
    if not isinstance(updates, dict):
        return ["请求体必须是对象（映射）"]
    for section, fields in updates.items():
        if section not in EDITABLE_FIELDS:
            errors.append(f"不允许修改配置节：{section}")
            continue
        if not isinstance(fields, dict):
            errors.append(f"配置节 {section} 必须是对象")
            continue
        allowed = EDITABLE_FIELDS[section]
        for key, value in fields.items():
            if key not in allowed:
                errors.append(f"不允许修改字段：{section}.{key}")
                continue
            if key == "mode" and value not in _RETRIEVAL_MODES:
                errors.append(f"retrieval.mode 必须是 {'/'.join(sorted(_RETRIEVAL_MODES))}")
            if key == "port":
                try:
                    if not (1 <= int(value) <= 65535):
                        errors.append("app.port 必须是 1~65535 之间的整数")
                except (TypeError, ValueError):
                    errors.append("app.port 必须是 1~65535 之间的整数")
            if key == "host" and (not isinstance(value, str) or not value.strip()):
                errors.append("app.host 不能为空")
            if key in _POSITIVE_INT_FIELDS:
                try:
                    if int(value) <= 0:
                        errors.append(f"{section}.{key} 必须是正整数")
                except (TypeError, ValueError):
                    errors.append(f"{section}.{key} 必须是正整数")
            if key == "temperature":
                try:
                    if not (0.0 <= float(value) <= 2.0):
                        errors.append("llm.temperature 必须在 0~2 之间")
                except (TypeError, ValueError):
                    errors.append("llm.temperature 必须是数字")
            if key == "usernames":
                if not isinstance(value, list) or not value or not all(
                    isinstance(x, str) and x.strip() for x in value
                ):
                    errors.append("admin.usernames 必须是非空字符串列表")
    return errors


def apply_updates(cfg: RAG2Config, updates: dict[str, Any]) -> RAG2Config:
    """把校验过的 updates 就地写入 cfg 的已知字段（复用 _coerce 做类型转换）。"""
    for section, fields in updates.items():
        if section not in EDITABLE_FIELDS or not isinstance(fields, dict):
            continue
        sub = getattr(cfg, section, None)
        for key, value in fields.items():
            if key not in EDITABLE_FIELDS[section] or not hasattr(sub, key):
                continue
            setattr(sub, key, _coerce(value, getattr(sub, key)))
    return cfg


def _atomic_write(path: Path, text: str) -> Path:
    """临时文件 + os.replace 原子写入，避免写一半损坏。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def save_config(cfg: RAG2Config, path: str | Path | None = None) -> Path:
    """把当前配置落盘到 config.yaml（覆盖，保留优先级头注释）。"""
    yaml_path = Path(path) if path else (PROJECT_ROOT / "config.yaml")
    import yaml  # 懒加载

    header = (
        "# ============================================================\n"
        "# RAG2 在线阶段配置（由管理员后台在线保存，也可手改）\n"
        "# 优先级：内置默认值 < 本文件 < RAG2_* 环境变量\n"
        "# ============================================================\n"
    )
    body = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False, default_flow_style=False)
    return _atomic_write(yaml_path, header + body)


_lock = threading.RLock()
_config: RAG2Config | None = None


def get_config(reload: bool = False) -> RAG2Config:
    """获取全局配置单例。"""
    global _config
    with _lock:
        if _config is None or reload:
            _config = load_config()
        return _config
