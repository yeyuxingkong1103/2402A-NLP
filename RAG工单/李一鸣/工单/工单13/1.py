"""Work order 13: profile an RAG pipeline and apply bounded optimizations."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StageMetric:
    stage: str
    elapsed_ms: float
    item_count: int = 0


class PipelineProfiler:
    def __init__(self):
        self.metrics: list[StageMetric] = []

    def measure(self, stage: str, fn: Callable, *args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        count = len(result) if hasattr(result, "__len__") else 0
        self.metrics.append(StageMetric(stage, (time.perf_counter() - started) * 1000, count))
        return result

    async def measure_async(self, stage: str, fn: Callable, *args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        count = len(result) if hasattr(result, "__len__") else 0
        self.metrics.append(StageMetric(stage, (time.perf_counter() - started) * 1000, count))
        return result

    def report(self) -> list[dict[str, Any]]:
        return [metric.__dict__ for metric in self.metrics]


class TTLCache:
    def __init__(self, max_size: int = 2048, ttl_seconds: float = 300):
        self.max_size, self.ttl_seconds = max_size, ttl_seconds
        self.data: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def key(namespace: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str):
        item = self.data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.monotonic():
            self.data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self.data) >= self.max_size:
            oldest = min(self.data, key=lambda item: self.data[item][0])
            self.data.pop(oldest, None)
        self.data[key] = (time.monotonic() + self.ttl_seconds, value)


class BoundedBatcher:
    """Micro-batch independent embedding calls without unbounded task creation."""

    def __init__(self, batch_size: int = 64, concurrency: int = 4):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run(self, items: list[Any], batch_fn: Callable[[list[Any]], Any]) -> list[Any]:
        batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        async def call(batch):
            async with self.semaphore:
                result = batch_fn(batch)
                return await result if asyncio.iscoroutine(result) else result

        results = await asyncio.gather(*(call(batch) for batch in batches))
        return [item for batch in results for item in batch]
