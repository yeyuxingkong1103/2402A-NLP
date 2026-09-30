"""Structured audit logging for vector and graph retrieval results."""

from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Iterable

from src import config


_LOGGER_NAME = "medical_rag.retrieval.audit"
_logger: logging.Logger | None = None
_logger_lock = threading.Lock()
_OMITTED_KEYS = {
    "vector",
    "embedding",
    "embeddings",
    "password",
    "api_key",
    "raw",
    "milvus_fields",
}


def _is_omitted_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in _OMITTED_KEYS
        or normalized.endswith("_vector")
        or "embedding" in normalized
        or any(marker in normalized for marker in ("password", "api_key", "secret"))
    )


def new_retrieval_id() -> str:
    """Return a short correlation id shared by all engines in one search."""
    return uuid.uuid4().hex[:12]


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    with _logger_lock:
        if _logger is not None:
            return _logger
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(getattr(logging, config.RETRIEVAL_LOG_LEVEL, logging.INFO))
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[RETRIEVAL] %(message)s"))
            logger.addHandler(handler)
        _logger = logger
        return logger


def _safe_value(value: Any, max_chars: int, depth: int = 0) -> Any:
    """Convert driver-specific values into bounded JSON-safe data."""
    if depth >= 4:
        return _clip(str(value), max_chars)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, max_chars)
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item, max_chars, depth + 1)
            for key, item in value.items()
            if not _is_omitted_key(key)
        }
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        bounded = [_safe_value(item, max_chars, depth + 1) for item in values[:30]]
        if len(values) > 30:
            bounded.append(f"…其余 {len(values) - 30} 项已省略")
        return bounded
    if hasattr(value, "items"):
        try:
            return _safe_value(dict(value), max_chars, depth + 1)
        except Exception:
            pass
    return _clip(str(value), max_chars)


def _clip(value: str, max_chars: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"…（截断 {len(text) - max_chars} 字符）"


def _result_payload(item: Any, max_chars: int) -> dict[str, Any]:
    if hasattr(item, "content"):
        content = getattr(item, "content", {}) or {}
        payload: dict[str, Any] = {
            "rank": getattr(item, "rank", None),
            "score": round(float(getattr(item, "score", 0.0)), 6),
            "reason": _clip(str(getattr(item, "reason", "")), max_chars),
            "content": _safe_value(content, max_chars),
        }
        return payload
    return _safe_value(item, max_chars) if isinstance(item, dict) else {
        "value": _safe_value(item, max_chars)
    }


def build_retrieval_payload(
    *,
    engine: str,
    query: str,
    results: Iterable[Any],
    elapsed_ms: float,
    retrieval_id: str = "",
    status: str = "success",
    error: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the bounded JSON object written for one engine invocation."""
    result_list = list(results or [])
    max_results = max(1, config.RETRIEVAL_LOG_RESULT_LIMIT)
    max_chars = max(100, config.RETRIEVAL_LOG_PREVIEW_CHARS)
    payload: dict[str, Any] = {
        "event": "retrieval_results",
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "retrieval_id": retrieval_id or new_retrieval_id(),
        "engine": engine,
        "status": status,
        "query": _clip(query, max_chars),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "result_count": len(result_list),
        "logged_result_count": min(len(result_list), max_results),
        "results": [
            _result_payload(item, max_chars) for item in result_list[:max_results]
        ],
    }
    if error:
        payload["error"] = _clip(error, max_chars)
    if details:
        payload["details"] = _safe_value(details, max_chars)
    return payload


def log_retrieval_results(**kwargs: Any) -> None:
    """Print one UTF-8 JSON event to the terminal without breaking retrieval."""
    if not config.RETRIEVAL_LOG_ENABLED:
        return
    try:
        payload = build_retrieval_payload(**kwargs)
        _get_logger().info(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:  # pragma: no cover - logging must be fail-open
        logging.getLogger(__name__).warning("检索审计日志写入失败: %s", exc)


__all__ = [
    "build_retrieval_payload",
    "log_retrieval_results",
    "new_retrieval_id",
]
