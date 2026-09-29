from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load a local .env file without requiring python-dotenv."""
    dotenv_path = ROOT_DIR / ".env"
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# Retrieval feature flags
ENABLE_VECTOR_RETRIEVAL = _env_bool("ENABLE_VECTOR_RETRIEVAL", True)
ENABLE_GRAPH_RETRIEVAL = _env_bool("ENABLE_GRAPH_RETRIEVAL", True)

# Retrieval tuning
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 10)
GRAPH_TOP_K = _env_int("GRAPH_TOP_K", 10)
FUSION_TOP_K = _env_int("FUSION_TOP_K", 10)
FUSION_STRATEGY = _env("FUSION_STRATEGY", "rrf").lower()
FUSION_WEIGHTS = {
    "vector": _env_float("FUSION_WEIGHT_VECTOR", 1.0),
    "graph": _env_float("FUSION_WEIGHT_GRAPH", 1.0),
}
RRF_K = _env_int("RRF_K", 60)
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", False)
RERANKER_MODEL_NAME = _env(
    "RERANKER_MODEL_NAME", r"E:\八维\Model\beg-reranker-v2-m3")
RERANKER_TOP_K = _env_int("RERANKER_TOP_K", 10)

# Milvus
MILVUS_HOST = _env("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = _env("MILVUS_PORT", "19530")
MILVUS_ALIAS = _env("MILVUS_ALIAS", "default")
MILVUS_COLLECTION = _env("MILVUS_COLLECTION", "medical_guideline_chunks_v1")
MILVUS_ID_FIELD = _env("MILVUS_ID_FIELD", "id")
MILVUS_VECTOR_FIELD = _env("MILVUS_VECTOR_FIELD", "vector")
MILVUS_TEXT_FIELD = _env("MILVUS_TEXT_FIELD", "text")
MILVUS_NAME_FIELD = _env("MILVUS_NAME_FIELD", "title")
MILVUS_METRIC_TYPE = _env("MILVUS_METRIC_TYPE", "COSINE").upper()
MILVUS_INDEX_TYPE = _env("MILVUS_INDEX_TYPE", "AUTOINDEX")
MILVUS_NLIST = _env_int("MILVUS_NLIST", 1024)
MILVUS_ALLOW_LOCAL = _env_bool("MILVUS_ALLOW_LOCAL", True)
EMBEDDING_MODEL_NAME = _env(
    "EMBEDDING_MODEL_NAME", r"D:\bge-small-zh-v1.5")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 512)
EMBEDDING_API_URL = _env("EMBEDDING_API_URL", "")
EMBEDDING_API_TIMEOUT = _env_float("EMBEDDING_API_TIMEOUT", 10.0)

# Neo4j
NEO4J_URI = _env("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", "medical-rag-2026")
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")
NEO4J_CONNECTION_TIMEOUT = _env_float("NEO4J_CONNECTION_TIMEOUT", 5.0)

# LLM
LLM_ENABLED = _env_bool("LLM_ENABLED", True)
LLM_API_KEY = _env("LLM_API_KEY", _env("DASHSCOPE_API_KEY", ""))
LLM_BASE_URL = _env("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode")
LLM_MODEL = _env("LLM_MODEL", "qwen3.7-flash")
LLM_PROTOCOL = _env("LLM_PROTOCOL", "openai")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2048)
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 30.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
ANTHROPIC_VERSION = _env("ANTHROPIC_VERSION", "2023-06-01")

# Tavily web search (per-request optional; disabled unless the frontend sends web_search=true).
TAVILY_API_KEY = _env("TAVILY_API_KEY", "")
TAVILY_API_URL = _env("TAVILY_API_URL", "https://api.tavily.com/search")
WEB_SEARCH_TIMEOUT = _env_float("WEB_SEARCH_TIMEOUT", 60.0)
WEB_SEARCH_TOP_K = _env_int("WEB_SEARCH_TOP_K", 3)
WEB_SEARCH_CACHE_TTL = _env_int("WEB_SEARCH_CACHE_TTL", 600)

# Bounded in-process conversation memory. It is used for reference resolution,
# never as a replacement for evidence retrieved in the current request.
MEMORY_MAX_SESSIONS = _env_int("MEMORY_MAX_SESSIONS", 500)
MEMORY_MAX_TURNS = _env_int("MEMORY_MAX_TURNS", 20)
MEMORY_TTL_SECONDS = _env_int("MEMORY_TTL_SECONDS", 3600)
MEMORY_MAX_HISTORY_CHARS = _env_int("MEMORY_MAX_HISTORY_CHARS", 6000)
MEMORY_BACKEND = _env("MEMORY_BACKEND", "hybrid").lower()
MEMORY_REDIS_URL = _env("MEMORY_REDIS_URL", "")
MEMORY_REDIS_KEY_PREFIX = _env("MEMORY_REDIS_KEY_PREFIX", "medrag:memory:")

# MySQL shared by persistent memory.
MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = _env_int("MYSQL_PORT", 3306)
MYSQL_USER = _env("MYSQL_USER", "medrag")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "medrag_local_2026")
MYSQL_DATABASE = _env("MYSQL_DATABASE", "health_db")

# Vector/graph retrieval diagnostics printed to the application terminal.
RETRIEVAL_LOG_ENABLED = _env_bool("RETRIEVAL_LOG_ENABLED", True)
RETRIEVAL_LOG_LEVEL = _env("RETRIEVAL_LOG_LEVEL", "INFO").upper()
RETRIEVAL_LOG_RESULT_LIMIT = _env_int("RETRIEVAL_LOG_RESULT_LIMIT", 20)
RETRIEVAL_LOG_PREVIEW_CHARS = _env_int("RETRIEVAL_LOG_PREVIEW_CHARS", 1000)

# Lightweight user auth (login page + bearer-token sessions, stdlib only).
# When AUTH_ENABLED is true, /api/chat and /api/analyze-image require a valid
# login token.
AUTH_ENABLED = _env_bool("AUTH_ENABLED", False)
AUTH_DB_PATH = Path(
    _env("AUTH_DB_PATH", str(ROOT_DIR / "database" / "users.db")))
AUTH_TOKEN_TTL_SECONDS = _env_int("AUTH_TOKEN_TTL_SECONDS", 7 * 24 * 3600)
AUTH_SEED_DEMO = _env_bool("AUTH_SEED_DEMO", True)

# Per-user chat history retention.
HISTORY_LIMIT = _env_int("HISTORY_LIMIT", 30)
HISTORY_TITLE_CHARS = _env_int("HISTORY_TITLE_CHARS", 24)


def get_llm_enabled() -> bool:
    return LLM_ENABLED


def get_runtime_summary() -> dict[str, object]:
    """Return non-sensitive runtime settings for diagnostics."""
    return {
        "milvus": f"{MILVUS_HOST}:{MILVUS_PORT}",
        "milvus_collection": MILVUS_COLLECTION,
        "neo4j_uri": NEO4J_URI,
        "neo4j_database": NEO4J_DATABASE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_enabled": ENABLE_VECTOR_RETRIEVAL,
        "graph_enabled": ENABLE_GRAPH_RETRIEVAL,
        "llm_enabled": LLM_ENABLED,
        "llm_configured": bool(LLM_API_KEY),
        "memory_backend": MEMORY_BACKEND,
        "memory_redis_configured": bool(MEMORY_REDIS_URL) and MEMORY_BACKEND in {"redis", "hybrid"},
        "memory_ttl_seconds": MEMORY_TTL_SECONDS,
    }
