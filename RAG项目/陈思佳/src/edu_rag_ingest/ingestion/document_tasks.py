from __future__ import annotations

"""文档后台任务执行器，用于异步处理上传后的解析和入库任务。"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DocumentTaskRunner:
    def __init__(self, max_workers: int = 1) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="document-worker")

    def submit(self, task: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> Future[dict[str, Any]]:
        future = self.executor.submit(task, *args, **kwargs)
        future.add_done_callback(self._log_failure)
        return future

    @staticmethod
    def _log_failure(future: Future[dict[str, Any]]) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("后台文档任务执行失败")
