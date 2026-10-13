from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(float)
        self._histograms = defaultdict(list)

    def inc(self, name: str, value: float = 1, labels=()):
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels=()):
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    @staticmethod
    def _key(name, labels):
        return name + "{" + ",".join(f'{key}="{value}"' for key, value in labels) + "}" if labels else name

    def exposition(self) -> str:
        with self._lock:
            lines = [f"{name} {value}" for name, value in sorted(self._counters.items())]
            for name, values in sorted(self._histograms.items()):
                if values:
                    lines.extend((f"{name}_count {len(values)}", f"{name}_sum {sum(values)}", f"{name}_max {max(values)}"))
            return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()


@contextmanager
def timed(name: str, labels=()):
    start = time.perf_counter()
    try:
        yield
    finally:
        METRICS.observe(name, time.perf_counter() - start, labels)
