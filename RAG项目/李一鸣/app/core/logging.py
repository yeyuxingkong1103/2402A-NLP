import json
import logging
import logging.handlers
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from app.core.config import Settings, get_settings


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # 统一输出 JSON，便于后续用日志平台按 request_id、角色或耗时字段检索。
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("user_id", "role_id", "conversation_id", "document_id", "latency_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings | None = None) -> None:
    # 同时写控制台和滚动文件；单个日志文件达到 10 MB 后保留 5 个备份。
    settings = settings or get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    if root.handlers:
        return

    formatter = JsonFormatter()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)


def new_request_id() -> str:
    # ContextVar 让同一请求链路中的日志自动携带同一个追踪 ID。
    request_id = uuid.uuid4().hex
    request_id_var.set(request_id)
    return request_id
