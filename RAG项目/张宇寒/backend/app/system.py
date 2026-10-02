from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Full, Queue
import re
import sys
from time import perf_counter
import traceback
from typing import Any
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings


request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="anonymous")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
client_ip_var: ContextVar[str] = ContextVar("client_ip", default="")

_listener: QueueListener | None = None
_previous_hash = ""

SENSITIVE_KEYS = {
    "authorization", "cookie", "password", "token", "api_key", "secret",
    "deepseek_api_key", "siliconflow_api_key", "tavily_api_key",
}
MAX_LOG_RECORD_BYTES = 1024 * 1024
MAX_FIELD_CHARS = 12000
QUEUE_MAX_SIZE = 10000

ID_CARD_PATTERN = re.compile(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b")
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
NAME_WITH_LABEL_PATTERN = re.compile(r"((?:原告|被告|当事人|申请人|被申请人|委托人|债权人|债务人|联系人|姓名)[:：]?)([\u4e00-\u9fff])([\u4e00-\u9fff]{1,3})")
COMMON_NAME_PATTERN = re.compile(r"\b([赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝安常傅卞齐元顾孟平黄和穆萧尹])([\u4e00-\u9fff]{1,2})\b")


class NonBlockingQueueHandler(QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = super().prepare(record)
        prepared.exc_info = record.exc_info
        prepared.exc_text = None
        prepared.request_id = getattr(record, "request_id", None) or request_id_var.get()
        prepared.trace_id = getattr(record, "trace_id", None) or trace_id_var.get()
        prepared.user_id = getattr(record, "user_id", None) or user_id_var.get()
        prepared.session_id = getattr(record, "session_id", None) or session_id_var.get()
        prepared.client_ip = getattr(record, "client_ip", None) or client_ip_var.get()
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Full:
            pass


def bind_request_context(request_id: str, trace_id: str | None = None, user_id: str = "anonymous", session_id: str | None = None, client_ip: str | None = None):
    return [
        (request_id_var, request_id_var.set(request_id)),
        (trace_id_var, trace_id_var.set(trace_id or request_id)),
        (user_id_var, user_id_var.set(user_id or "anonymous")),
        (session_id_var, session_id_var.set(session_id or "")),
        (client_ip_var, client_ip_var.set(client_ip or "")),
    ]


def reset_request_context(tokens) -> None:
    for var, token in reversed(tokens):
        var.reset(token)


def set_user_context(user_id: str | None = None, session_id: str | None = None) -> None:
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def mask_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(secret_key in normalized for secret_key in SENSITIVE_KEYS):
                masked[key] = "***"
            else:
                masked[key] = mask_sensitive(item)
        return masked
    if isinstance(value, list):
        return [mask_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [mask_sensitive(item) for item in value]
    if isinstance(value, str):
        text = ID_CARD_PATTERN.sub(lambda match: match.group(0)[:6] + "********" + match.group(0)[-4:], value)
        text = PHONE_PATTERN.sub(lambda match: match.group(0)[:3] + "****" + match.group(0)[-4:], text)
        text = EMAIL_PATTERN.sub(lambda match: match.group(1) + "***" + match.group(2), text)
        text = NAME_WITH_LABEL_PATTERN.sub(lambda match: match.group(1) + match.group(2) + "*", text)
        text = COMMON_NAME_PATTERN.sub(lambda match: match.group(1) + "*", text)
        if len(text) > MAX_FIELD_CHARS:
            return text[:MAX_FIELD_CHARS] + f"...<truncated {len(text) - MAX_FIELD_CHARS} chars>"
        return text
    return value


def normalize_level(level_name: str) -> str:
    if level_name == "WARNING":
        return "WARN"
    if level_name == "CRITICAL":
        return "FATAL"
    return level_name


def compact_exception(record: logging.LogRecord) -> dict[str, str] | None:
    if not record.exc_info:
        return None
    exc_type, exc_value, exc_traceback = record.exc_info
    stack_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return {
        "exception_type": exc_type.__name__ if exc_type else "Exception",
        "exception_message": mask_sensitive(str(exc_value)),
        "stack_trace": "".join(stack_lines).replace("\r\n", "\\n").replace("\n", "\\n"),
    }


class JsonLogFormatter(logging.Formatter):
    def __init__(self, max_record_bytes: int = MAX_LOG_RECORD_BYTES, with_hash_chain: bool = False):
        super().__init__()
        self.max_record_bytes = max_record_bytes
        self.with_hash_chain = with_hash_chain

    def base_payload(self, record: logging.LogRecord) -> dict:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": normalize_level(record.levelname),
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "process_id": record.process,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "trace_id": getattr(record, "trace_id", None) or trace_id_var.get(),
            "user_id": getattr(record, "user_id", None) or user_id_var.get(),
            "session_id": getattr(record, "session_id", None) or session_id_var.get(),
            "client_ip": getattr(record, "client_ip", None) or client_ip_var.get(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        exception = compact_exception(record)
        if exception:
            payload.update(exception)
        return mask_sensitive(payload)

    def with_hash(self, payload: dict, previous_hash: str) -> dict:
        global _previous_hash
        payload["previous_hash"] = previous_hash
        hash_source = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        record_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()
        payload["record_hash"] = record_hash
        _previous_hash = record_hash
        return payload

    def truncate_payload(self, payload: dict) -> dict:
        truncated = {
            "timestamp": payload.get("timestamp"),
            "level": payload.get("level"),
            "logger": payload.get("logger"),
            "event": payload.get("event"),
            "message": str(payload.get("message", ""))[:1000] + "...<record truncated>",
            "process_id": payload.get("process_id"),
            "request_id": payload.get("request_id"),
            "trace_id": payload.get("trace_id"),
            "user_id": payload.get("user_id"),
            "session_id": payload.get("session_id"),
            "client_ip": payload.get("client_ip"),
            "truncated": True,
        }
        if "previous_hash" in payload:
            truncated["previous_hash"] = payload["previous_hash"]
        return truncated

    def format(self, record: logging.LogRecord) -> str:
        cached_line = getattr(record, "_json_line", None)
        if cached_line:
            return cached_line
        previous_hash = _previous_hash
        payload = self.base_payload(record)
        if self.with_hash_chain:
            payload["previous_hash"] = previous_hash
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(line.encode("utf-8")) > self.max_record_bytes:
            payload = self.truncate_payload(payload)
        if self.with_hash_chain:
            payload = self.with_hash(payload, previous_hash)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if self.with_hash_chain:
            record._json_line = line
        return line


class ConsoleJsonLogFormatter(JsonLogFormatter):
    def __init__(self, max_record_bytes: int = MAX_LOG_RECORD_BYTES):
        super().__init__(max_record_bytes=max_record_bytes, with_hash_chain=False)


class TextLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds")
        return f"{timestamp} {normalize_level(record.levelname)} {record.name} request_id={request_id_var.get()} trace_id={trace_id_var.get()} {record.getMessage()}"


def _level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def _file_formatter(settings: Settings) -> logging.Formatter:
    if settings.log_format.lower() == "text":
        return TextLogFormatter()
    return JsonLogFormatter(settings.log_max_record_bytes, with_hash_chain=True)


def _console_formatter(settings: Settings) -> logging.Formatter:
    if settings.log_format.lower() == "text":
        return TextLogFormatter()
    return ConsoleJsonLogFormatter(settings.log_max_record_bytes)


def _safe_console_stream():
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass
    return stream


def _build_file_handler(settings: Settings) -> logging.Handler:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        Path(settings.log_dir) / settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(_file_formatter(settings))
    handler.setLevel(_level(settings.log_level))
    return handler


def _last_record_hash(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        record_hash = payload.get("record_hash")
        if isinstance(record_hash, str):
            return record_hash
    return ""


def setup_logging(settings: Settings) -> None:
    global _listener, _previous_hash
    level = _level(settings.log_level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    if _listener:
        _listener.stop()
    _previous_hash = _last_record_hash(settings.log_dir / settings.log_file)

    queue: Queue = Queue(maxsize=settings.log_queue_max_size)
    queue_handler = NonBlockingQueueHandler(queue)
    queue_handler.setLevel(level)
    root.addHandler(queue_handler)
    root.setLevel(level)

    console = logging.StreamHandler(_safe_console_stream())
    console.setFormatter(_console_formatter(settings))
    console.setLevel(level)
    file_handler = _build_file_handler(settings)
    _listener = QueueListener(queue, file_handler, console, respect_handler_level=True)
    _listener.start()

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "httpx"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)

    logging.getLogger("law_rag.system").info(
        "日志系统已启动",
        extra={"event": "logging_started", "fields": {"log_dir": str(settings.log_dir), "log_format": settings.log_format}},
    )


def shutdown_logging() -> None:
    global _listener
    if _listener:
        _listener.stop()
        _listener = None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = logging.getLogger("law_rag.request")

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self.settings.log_request_header) or uuid4().hex
        session_id = request.query_params.get("session_id") or ""
        client_ip = request.client.host if request.client else "unknown"
        tokens = bind_request_context(request_id, request_id, session_id=session_id, client_ip=client_ip)
        request.state.request_id = request_id
        start = perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            self.logger.exception(
                "请求处理失败",
                extra={
                    "event": "request_error",
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "query": request.url.query,
                        "status_code": status_code,
                    },
                },
            )
            raise
        finally:
            elapsed_ms = round((perf_counter() - start) * 1000, 2)
            self.logger.info(
                "请求处理完成",
                extra={
                    "event": "request_completed",
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "query": request.url.query,
                        "status_code": status_code,
                        "elapsed_ms": elapsed_ms,
                    },
                },
            )
            if response is not None:
                response.headers[self.settings.log_request_header] = request_id
            reset_request_context(tokens)


def verify_log_integrity(log_path: Path) -> dict:
    if not log_path.exists():
        return {"valid": False, "checked": 0, "message": "日志文件不存在"}
    known_hashes: set[str] = set()
    processes: set[str] = set()
    root_count = 0
    external_anchor_count = 0
    checked = 0
    with log_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                return {"valid": False, "checked": checked, "line": line_number, "message": "日志不是合法 JSON"}
            record_hash = payload.pop("record_hash", "")
            processes.add(str(payload.get("process_id", "default")))
            previous_hash = payload.get("previous_hash", "")
            if previous_hash:
                if previous_hash not in known_hashes:
                    if checked == 0:
                        external_anchor_count += 1
                    else:
                        return {"valid": False, "checked": checked, "line": line_number, "message": "previous_hash 不匹配"}
            else:
                root_count += 1
            hash_source = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            expected_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()
            if record_hash != expected_hash:
                return {"valid": False, "checked": checked, "line": line_number, "message": "record_hash 不匹配"}
            known_hashes.add(record_hash)
            checked += 1
    return {
        "valid": True,
        "checked": checked,
        "processes": len(processes),
        "root_count": root_count,
        "external_anchor_count": external_anchor_count,
        "message": "日志哈希链完整",
    }


from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["system"])


def create_services(settings: Settings) -> dict:
    from .auth import AuthService
    from .models import ModelGateway
    from .memory.api import MemoryOrchestrator
    from .rag.workflow import RagWorkflow
    from .storage.redis import RedisStore
    from .storage.mysql import AuthSessionStore, FileStore, MySQLStore, UserStateStore, UserStore
    from .storage.vector import MilvusStore, WorkspaceVectorStore
    from .workspace import WorkspaceService

    mysql = MySQLStore(settings)
    redis = RedisStore(settings.redis_url)
    milvus = MilvusStore(settings)
    model = ModelGateway(settings)
    auth = AuthService(settings, UserStore(mysql), AuthSessionStore(mysql), redis)
    user_state = UserStateStore(mysql)
    memory = MemoryOrchestrator(settings, redis, mysql, model, user_state)
    workspace = WorkspaceService(settings, FileStore(mysql), WorkspaceVectorStore(milvus), model, user_state)
    rag = RagWorkflow(settings, model, milvus, redis, workspace, memory, user_state)
    return {
        "settings": settings,
        "mysql": mysql,
        "redis": redis,
        "milvus": milvus,
        "model": model,
        "auth": auth,
        "user_state": user_state,
        "memory": memory,
        "workspace": workspace,
        "rag": rag,
    }


def require_log_auditor(request: Request) -> None:
    from .auth import current_user

    settings = request.app.state.services["settings"]
    audit_token = request.headers.get("X-Log-Audit-Token")
    if settings.log_audit_token and audit_token == settings.log_audit_token:
        return
    user = current_user(request)
    allowed_users = {item.strip() for item in settings.log_audit_users.split(",") if item.strip()}
    if allowed_users and (user.get("username") in allowed_users or user.get("user_id") in allowed_users or user.get("email") in allowed_users):
        return
    raise HTTPException(status_code=403, detail="没有日志审计权限")


@router.get("/health")
def health(request: Request):
    services = request.app.state.services
    mysql = services["mysql"].health()
    redis = services["redis"].health()
    milvus = services["milvus"].health()
    model = services["model"].health()
    workspace = services["workspace"].health()
    public_collections = milvus.get("public_collections", [])
    embedding_failed = bool(model.get("embedding_last_error"))
    payload = {
        "status": "degraded" if embedding_failed else "ok",
        "rag": "degraded" if embedding_failed else ("ok" if public_collections else "unknown"),
        "mysql": mysql,
        "redis": redis,
        "milvus": milvus,
        "model": model,
        "workspace": workspace,
        "llm_provider": model.get("llm_provider", "DeepSeek"),
        "llm_model": model.get("llm_model", ""),
        "embedding_model": model.get("embedding_model", ""),
        "reranker_model": model.get("reranker_model", ""),
        "web": model.get("web", "unknown"),
        "indexed_records": milvus.get("indexed_records"),
    }
    return {"success": True, "data": payload, **payload}


@router.get("/api/v1/system/logs/integrity")
def log_integrity(request: Request):
    require_log_auditor(request)
    settings = request.app.state.services["settings"]
    return {"success": True, "data": verify_log_integrity(settings.log_dir / settings.log_file)}
