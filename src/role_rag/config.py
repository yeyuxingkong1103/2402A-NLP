"""配置加载：内置默认值 < configs/config.yaml < .env / 环境变量。

设计约定
--------
* 配置以字典树保存，用 ``cfg.get("milvus.search.ef", 128)`` 这类点路径读取；
* 所有相对路径都以项目根目录为基准解析（``cfg.path(...)``）；
* 模型路径统一从 ``paths.models_root`` 派生，默认 D:/modelscope。
"""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 内置默认值：即使 configs/config.yaml 被删掉也能启动
DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "role-rag",
        "version": "1.0.0",
        "host": "127.0.0.1",
        "port": 8020,
        "env": "dev",
        "debug": True,
        "warmup": True,
        "log_dir": "./logs",
        "log_level": "INFO",
        "cors_origins": ["*"],
    },
    "paths": {
        "models_root": "D:/modelscope",
        "data_dir": "./data",
        "cache_dir": "./data_cache",
        "web_dir": "./web",
    },
    "models": {
        "embedder": {
            "path": "",  # 留空则用 paths.models_root/bge-m3
            "device": "cuda",
            "dtype": "float16",
            "max_length": 512,
            "batch_size": 8,
            "dense_dim": 1024,
            "sparse_min_weight": 0.02,
            "sparse_top_terms": 192,
            "query_sparse_top_terms": 48,
        },
        "llm": {
            "path": "",  # 留空则用 paths.models_root/Qwen3-0.6B
            "device": "cuda",
            "dtype": "float16",
            "max_new_tokens": 700,
            "temperature": 0.35,
            "top_p": 0.85,
            "repetition_penalty": 1.08,
            "enable_thinking": False,
            "max_input_tokens": 7000,
            "max_context_chars": 7000,
        },
    },
    "milvus": {
        "uri": "http://127.0.0.1:19530",
        "token": "",
        "kb_collection": "role_kb_chunks",
        "memory_collection": "role_memory",
        "dense_index": {"index_type": "HNSW", "metric_type": "COSINE", "M": 16, "efConstruction": 200},
        "sparse_index": {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        "search": {"ef": 128, "timeout": 30},
    },
    "redis": {
        "host": "127.0.0.1",
        "port": 6379,
        "password": "",
        "db": 0,
        "prefix": "rolerag:",
        "socket_timeout": 5,
        "history_ttl": 604800,
        "summary_ttl": 604800,
        "cache_ttl": 900,
        "online_ttl": 300,
        "rate_limit_per_min": 60,
    },
    "memory": {
        "short_term_turns": 5,
        "summary_trigger_turns": 5,
        "long_term_top_k": 4,
        "long_term_min_score": 0.35,
        "write_back": True,
        "extract_rules": True,
    },
    "ingest": {
        "chunk_size": 700,
        "chunk_overlap": 120,
        "min_chunk_chars": 80,
        "max_chunk_chars": 1400,
        "batch_size": 16,
        "include_shared": True,
    },
    "retrieval": {
        "top_k_dense": 20,
        "top_k_sparse": 20,
        "top_k_bm25": 20,
        "final_k": 6,
        "rrf_k": 60,
        "weights": {"dense": 0.55, "sparse": 0.30, "bm25": 0.15},
        "per_doc_limit": 3,
        "min_final_score": 0.0,
        "cache_enabled": True,
        "bm25_refresh_seconds": 60,
    },
    "rag": {
        "rewrite_enabled": False,
        "rewrite_max_tokens": 96,
        "include_references_when_uncited": True,
    },
    "security": {
        "secret": "change-me-in-production",
        "token_ttl": 43200,
        "password_iterations": 120000,
        "users_seed": [],
    },
}


def _parse_env_file(path: Path) -> dict[str, str]:
    """读取 .env（KEY=VALUE，支持 # 注释与引号）。"""

    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


# 环境变量 → 配置点路径（.env 或系统环境变量均可）
ENV_MAP: dict[str, tuple[str, Callable[[str], Any]]] = {
    "ROLE_RAG_HOST": ("app.host", str),
    "ROLE_RAG_PORT": ("app.port", int),
    "ROLE_RAG_ENV": ("app.env", str),
    "ROLE_RAG_DEBUG": ("app.debug", _as_bool),
    "ROLE_RAG_WARMUP": ("app.warmup", _as_bool),
    "ROLE_RAG_LOG_DIR": ("app.log_dir", str),
    "ROLE_RAG_LOG_LEVEL": ("app.log_level", str),
    "ROLE_RAG_MODELS_ROOT": ("paths.models_root", str),
    "ROLE_RAG_DATA_DIR": ("paths.data_dir", str),
    "ROLE_RAG_CACHE_DIR": ("paths.cache_dir", str),
    "ROLE_RAG_EMBED_MODEL": ("models.embedder.path", str),
    "ROLE_RAG_LLM_MODEL": ("models.llm.path", str),
    "ROLE_RAG_DEVICE": ("models.embedder.device", str),
    "ROLE_RAG_LLM_DEVICE": ("models.llm.device", str),
    "ROLE_RAG_LLM_MAX_NEW_TOKENS": ("models.llm.max_new_tokens", int),
    "ROLE_RAG_LLM_TEMPERATURE": ("models.llm.temperature", float),
    "ROLE_RAG_ENABLE_THINKING": ("models.llm.enable_thinking", _as_bool),
    "ROLE_RAG_MILVUS_URI": ("milvus.uri", str),
    "ROLE_RAG_MILVUS_TOKEN": ("milvus.token", str),
    "ROLE_RAG_MILVUS_KB_COLLECTION": ("milvus.kb_collection", str),
    "ROLE_RAG_MILVUS_MEMORY_COLLECTION": ("milvus.memory_collection", str),
    "ROLE_RAG_REDIS_HOST": ("redis.host", str),
    "ROLE_RAG_REDIS_PORT": ("redis.port", int),
    "ROLE_RAG_REDIS_PASSWORD": ("redis.password", str),
    "ROLE_RAG_REDIS_DB": ("redis.db", int),
    "ROLE_RAG_FINAL_K": ("retrieval.final_k", int),
    "ROLE_RAG_CACHE_ENABLED": ("retrieval.cache_enabled", _as_bool),
    "ROLE_RAG_SHORT_TERM_TURNS": ("memory.short_term_turns", int),
    "ROLE_RAG_WRITE_BACK": ("memory.write_back", _as_bool),
    "ROLE_RAG_SECRET": ("security.secret", str),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return
    node[parts[-1]] = value


class Config:
    """点路径访问的配置对象。"""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._data = data
        self.root = root
        self._roles_cache: Any = None

    # ---------------------------------------------------------------- 读取
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # ---------------------------------------------------------------- 子属性
    @property
    def app(self) -> dict[str, Any]:
        return self.section("app")

    @property
    def paths(self) -> dict[str, Any]:
        return self.section("paths")

    @property
    def models(self) -> dict[str, Any]:
        return self.section("models")

    @property
    def milvus(self) -> dict[str, Any]:
        return self.section("milvus")

    @property
    def redis(self) -> dict[str, Any]:
        return self.section("redis")

    @property
    def memory(self) -> dict[str, Any]:
        return self.section("memory")

    @property
    def ingest(self) -> dict[str, Any]:
        return self.section("ingest")

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.section("retrieval")

    @property
    def security(self) -> dict[str, Any]:
        return self.section("security")

    # ---------------------------------------------------------------- 路径
    def path(self, dotted: str, default: str = ".") -> Path:
        """把配置中的相对路径解析为绝对路径。"""

        raw = str(self.get(dotted, default) or default)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    @property
    def models_root(self) -> Path:
        return self.path("paths.models_root", "D:/modelscope")

    @property
    def data_dir(self) -> Path:
        return self.path("paths.data_dir", "./data")

    @property
    def cache_dir(self) -> Path:
        return self.path("paths.cache_dir", "./data_cache")

    @property
    def web_dir(self) -> Path:
        return self.path("paths.web_dir", "./web")

    @property
    def log_dir(self) -> Path:
        return self.path("app.log_dir", "./logs")

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    def embedder_path(self) -> Path:
        raw = str(self.get("models.embedder.path", "") or "").strip()
        if raw:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (self.root / candidate).resolve()
        return self.models_root / "bge-m3"

    def llm_path(self) -> Path:
        raw = str(self.get("models.llm.path", "") or "").strip()
        if raw:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (self.root / candidate).resolve()
        return self.models_root / "Qwen3-0.6B"

    # ---------------------------------------------------------------- 角色表
    def roles_config(self) -> dict[str, Any]:
        if self._roles_cache is None:
            path = self.root / "configs" / "roles.yaml"
            if not path.is_file():
                raise ConfigError(f"角色配置文件不存在：{path}")
            self._roles_cache = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return self._roles_cache


def load_config(root: Path = PROJECT_ROOT, env_file: str | None = ".env") -> Config:
    """加载配置：默认值 → config.yaml → .env → 系统环境变量。"""

    data = copy.deepcopy(DEFAULTS)
    yaml_path = root / "configs" / "config.yaml"
    if yaml_path.is_file():
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"配置根节点必须是映射：{yaml_path}")
        _deep_merge(data, loaded)

    file_env: dict[str, str] = {}
    if env_file:
        file_env = _parse_env_file(root / env_file)

    for env_key, (dotted, cast) in ENV_MAP.items():
        raw = os.environ.get(env_key, file_env.get(env_key))
        if raw is None or raw == "":
            continue
        try:
            _set_path(data, dotted, cast(raw))
        except (TypeError, ValueError) as exc:  # pragma: no cover - 配置错误
            raise ConfigError(f"环境变量 {env_key}={raw!r} 解析失败：{exc}") from exc

    config = Config(data, root)
    if not config.get("security.secret"):
        raise ConfigError("security.secret 未配置")
    return config


_lock = threading.RLock()
_config: Config | None = None


def get_config(reload: bool = False) -> Config:
    """获取全局配置单例。"""

    global _config
    with _lock:
        if _config is None or reload:
            _config = load_config()
        return _config
