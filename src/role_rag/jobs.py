"""后台作业管理：入库、评测等长任务放到线程里执行，前端用 job_id 轮询进度。"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "pending"          # pending / running / succeeded / failed
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": round(self.updated_at - self.created_at, 2),
        }


class JobManager:
    """极简作业管理器（进程内，保留最近 50 个作业）。"""

    def __init__(self, capacity: int = 50) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._capacity = capacity
        self._lock = threading.RLock()

    def submit(self, kind: str, func: Callable[[Callable[[str, float], None]], dict]) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, status="pending", message="排队中")
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self._capacity:
                self._jobs.pop(self._order.pop(0), None)

        def _progress(message: str, progress: float | None = None) -> None:
            job.message = message
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            job.updated_at = time.time()

        def _run() -> None:
            job.status = "running"
            job.updated_at = time.time()
            logger.info("作业开始：%s(%s)", kind, job.job_id)
            try:
                job.result = func(_progress) or {}
                job.status = "succeeded"
                job.progress = 1.0
                job.message = "完成"
            except Exception as exc:  # pragma: no cover - 作业内异常
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "失败"
                logger.error("作业失败 %s：%s\n%s", job.job_id, exc, traceback.format_exc())
            finally:
                job.updated_at = time.time()

        threading.Thread(target=_run, name=f"role-rag-job-{job.job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = [self._jobs[job_id] for job_id in self._order if job_id in self._jobs]
        return list(reversed(jobs))[:limit]


_manager: JobManager | None = None
_lock = threading.Lock()


def get_job_manager() -> JobManager:
    global _manager
    with _lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
