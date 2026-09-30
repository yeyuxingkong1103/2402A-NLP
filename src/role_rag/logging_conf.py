"""日志：控制台 + 轮转文件（app.log / error.log）+ 内存环形缓冲。

所有日志自动带上 request_id，便于按一次请求串起全链路。
环形缓冲供前端「运行日志」面板读取（GET /api/logs）。
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import get_config

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_configured = False
_lock = threading.RLock()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        record.request_id = _request_id.get()
        return True


class RingBufferHandler(logging.Handler):
    """把最近的日志保留在内存里，供 API 读取。"""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "request_id": getattr(record, "request_id", "-"),
            }
            with self._lock:
                self.buffer.append(item)
        except Exception:  # pragma: no cover - 日志自身不能抛错
            pass

    def recent(self, limit: int = 200, level: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self.buffer)
        if level:
            items = [item for item in items if item["level"] == level.upper()]
        return items[-limit:][::-1]


RING = RingBufferHandler()


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id or "-")


def get_request_id() -> str:
    return _request_id.get()


def setup_logging(log_dir: Path | None = None, level: str | None = None) -> None:
    """幂等地初始化日志（多次调用只生效一次）。"""

    global _configured
    with _lock:
        if _configured:
            return
        cfg = get_config().app
        log_dir = Path(log_dir) if log_dir else get_config().log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        level = (level or str(cfg.get("log_level", "INFO"))).upper()

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | rid=%(request_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        rid_filter = RequestIdFilter()

        root = logging.getLogger()
        root.setLevel(level)
        for handler in list(root.handlers):
            root.removeHandler(handler)

        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(fmt)
        console.addFilter(rid_filter)
        root.addHandler(console)

        app_file = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        app_file.setFormatter(fmt)
        app_file.addFilter(rid_filter)
        root.addHandler(app_file)

        err_file = logging.handlers.RotatingFileHandler(
            log_dir / "error.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        err_file.setLevel(logging.ERROR)
        err_file.setFormatter(fmt)
        err_file.addFilter(rid_filter)
        root.addHandler(err_file)

        RING.setFormatter(fmt)
        RING.addFilter(rid_filter)
        root.addHandler(RING)

        # 降噪
        for noisy in ("urllib3", "transformers", "httpx", "asyncio", "multipart", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # pymilvus 的 RPC 失败会打印整段 traceback，降级路径已自行记录，这里直接静音
        logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

        _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def recent_logs(limit: int = 200, level: str | None = None) -> list[dict[str, Any]]:
    return RING.recent(limit=limit, level=level)
