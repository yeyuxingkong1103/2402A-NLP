from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    from celery import Celery
    from celery.schedules import crontab
except ModuleNotFoundError:  # pragma: no cover - 依赖未安装时保留本地线程兜底
    Celery = None

    def crontab(*args, **kwargs):
        return None

from .config import Settings, get_settings


T = TypeVar("T")
logger = logging.getLogger("law_rag.tasks")
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="law-rag-task")
_workspace_services: dict[str, Any] = {}


class LocalTask:
    def __init__(self, func: Callable[..., T]):
        self.func = func
        self.run = func

    def __call__(self, *args, **kwargs) -> T:
        return self.func(*args, **kwargs)

    def delay(self, *args, **kwargs) -> Future[T]:
        return submit_task(self.func, *args, **kwargs)


class LocalCelery:
    def task(self, **_options):
        def decorator(func: Callable[..., T]) -> LocalTask:
            return LocalTask(func)

        return decorator


def create_celery_app(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    if Celery is None:
        return LocalCelery()
    app = Celery("law_rag", broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
        beat_schedule={
            "daily-memory-job-2am": {
                "task": "law_rag.daily_memory_job",
                "schedule": crontab(hour=2, minute=0),
                "args": (),
            },
        },
    )
    return app


celery_app = create_celery_app()


def submit_task(func: Callable[..., T], *args, **kwargs) -> Future[T]:
    """本地开发兜底：没有启动 Celery worker 时仍可把慢任务放到后台线程。"""
    return _executor.submit(func, *args, **kwargs)


def shutdown_tasks(wait: bool = False) -> None:
    _executor.shutdown(wait=wait, cancel_futures=not wait)


def _resolve_task_path(path: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    upload_root = Path(settings.upload_dir).resolve()
    candidate = Path(path).resolve()
    if upload_root != candidate and upload_root not in candidate.parents:
        raise ValueError("后台任务只能处理上传目录内的文件")
    if not candidate.is_file():
        raise ValueError("后台任务文件不存在")
    return candidate


def _workspace_service(settings: Settings):
    key = str(settings.upload_dir)
    if key not in _workspace_services:
        from .system import create_services

        _workspace_services[key] = create_services(settings)["workspace"]
    return _workspace_services[key]


@celery_app.task(name="law_rag.daily_memory_job")
def daily_memory_job_task(user_limit: int = 500) -> dict[str, Any]:
    from .memory.api import run_daily_memory_job

    return run_daily_memory_job(user_limit=user_limit)


@celery_app.task(name="law_rag.process_user_document")
def process_user_document_task(path: str, file_id: str | None = None, user_id: str = "", session_id: str | None = None) -> dict[str, Any]:
    from data_pipeline import process_user_document

    safe_path = _resolve_task_path(path)
    logger.info("用户文件后台入库开始", extra={"event": "task_process_user_document_started", "fields": {"file_id": file_id or "", "user_id": user_id}})
    result = process_user_document(safe_path, file_id=file_id, user_id=user_id, session_id=session_id)
    logger.info("用户文件后台入库完成", extra={"event": "task_process_user_document_completed", "fields": {"file_id": result["metadata"].get("file_id", ""), "chunk_count": len(result.get("chunks", []))}})
    return result


@celery_app.task(name="law_rag.ocr_document")
def ocr_document_task(path: str) -> dict[str, Any]:
    from data_pipeline import ocr_document

    return ocr_document(str(_resolve_task_path(path)))


@celery_app.task(
    name="law_rag.process_workspace_upload",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_workspace_upload_task(path: str, user: dict, record: dict, document_id: str, safe_name: str, session_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    safe_path = _resolve_task_path(path, settings)
    workspace = _workspace_service(settings)
    workspace._process_saved_upload(user, record, document_id, safe_name, safe_path, session_id)
    return {"document_id": document_id, "status": "processed"}


def enqueue_user_document(path: str, file_id: str | None = None, user_id: str = "", session_id: str | None = None, use_celery: bool = True):
    if use_celery:
        return process_user_document_task.delay(path, file_id=file_id, user_id=user_id, session_id=session_id)
    return submit_task(process_user_document_task.run, path, file_id=file_id, user_id=user_id, session_id=session_id)
