from __future__ import annotations

import os
import subprocess
import sys

import uvicorn

from backend.app.config import get_settings


def start_celery_worker(settings):
    if not bool(getattr(settings, "workspace_use_celery", False)):
        return None
    try:
        from celery import __version__  # noqa: F401
    except ImportError:
        print("Celery 未安装，后台任务将使用本地线程兜底。", flush=True)
        return None
    workers = max(1, int(getattr(settings, "workspace_background_workers", 4) or 4))
    command = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "backend.app.tasks.celery_app",
        "worker",
        "--loglevel=INFO",
        "--pool=threads",
        f"--concurrency={workers}",
    ]
    try:
        process = subprocess.Popen(command, cwd=str(os.path.dirname(os.path.abspath(__file__))))
    except OSError as exc:
        print(f"Celery worker 启动失败，后台任务将使用本地线程兜底：{exc}", flush=True)
        return None
    print(f"Celery worker 已自动启动，concurrency={workers}，pid={process.pid}", flush=True)
    return process


def stop_celery_worker(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    settings = get_settings()
    celery_process = start_celery_worker(settings)
    try:
        uvicorn.run(
            "backend.app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_config=None,
        )
    finally:
        stop_celery_worker(celery_process)
