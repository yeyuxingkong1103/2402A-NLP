from backend.app.core.config import settings

try:
    from celery import Celery
except ImportError:  # pragma: no cover - 缺少 Celery 包时仍允许单元测试导入模块。
    Celery = None


def create_celery_app():
    if Celery is None:
        raise RuntimeError("celery is not installed")
    app = Celery(
        "legal_rag",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND or None,
        include=["backend.app.workers.document_tasks"],
    )
    app.conf.update(
        task_track_started=True,
        task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Shanghai",
        enable_utc=False,
    )
    return app


celery_app = create_celery_app() if Celery is not None else None


def check_celery_readiness(timeout_seconds: float = 1.0) -> tuple[bool, str | None]:
    if Celery is None or celery_app is None:
        return False, "celery not installed"
    if not settings.CELERY_BROKER_URL:
        return False, "missing broker"
    try:
        response = celery_app.control.inspect(timeout=timeout_seconds).ping()
    except Exception as exc:
        return False, type(exc).__name__
    return bool(response), None if response else "no workers"
