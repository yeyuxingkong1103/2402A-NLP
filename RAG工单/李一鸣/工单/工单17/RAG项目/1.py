"""Work order 17: diagnose API concurrency bottlenecks and prevent resource leaks."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable


@dataclass
class ConcurrencyConfig:
    max_in_flight: int = 32
    acquire_timeout: float = 2.0
    request_timeout: float = 60.0
    queue_size: int = 128


class ResourcePool:
    """Bounded reusable client pool with deterministic shutdown."""

    def __init__(self, factory: Callable[[], Any], size: int):
        self.factory, self.size = factory, size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=size)
        self._created = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        if self._closed:
            raise RuntimeError("pool is closed")
        try:
            resource = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._created < self.size:
                    resource = self.factory()
                    if asyncio.iscoroutine(resource):
                        resource = await resource
                    self._created += 1
                else:
                    resource = await self._queue.get()
        try:
            yield resource
        finally:
            if self._closed:
                await self._dispose(resource)
            else:
                await self._queue.put(resource)

    async def _dispose(self, resource: Any) -> None:
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def close(self) -> None:
        self._closed = True
        while not self._queue.empty():
            await self._dispose(self._queue.get_nowait())


class RequestLimiter:
    def __init__(self, config: ConcurrencyConfig | None = None):
        self.config = config or ConcurrencyConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_in_flight)
        self.active = 0
        self.completed = 0
        self.failed = 0

    async def run(self, operation: Callable, *args, **kwargs):
        started = time.perf_counter()
        try:
            await asyncio.wait_for(self.semaphore.acquire(), self.config.acquire_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("server is busy; retry later") from exc
        self.active += 1
        try:
            result = operation(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, self.config.request_timeout)
            self.completed += 1
            return {"result": result, "latency_ms": (time.perf_counter() - started) * 1000}
        except Exception:
            self.failed += 1
            raise
        finally:
            self.active -= 1
            self.semaphore.release()

    def metrics(self) -> dict[str, int]:
        return {"active": self.active, "completed": self.completed, "failed": self.failed}
