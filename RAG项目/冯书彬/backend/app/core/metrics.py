import threading
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RequestMetric:
    count: int = 0
    total_duration_seconds: float = 0.0


class MetricsRegistry:
    """进程内指标注册表；多实例部署时由每个实例分别暴露给采集器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], RequestMetric] = defaultdict(RequestMetric)
        self._health: dict[str, bool] = {}

    def observe_request(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        key = (method, path, str(status_code))
        with self._lock:
            metric = self._requests[key]
            metric.count += 1
            metric.total_duration_seconds += duration_seconds

    def set_health(self, check_name: str, ok: bool) -> None:
        with self._lock:
            self._health[check_name] = ok

    def render_prometheus(self) -> str:
        lines = [
            "# HELP legal_rag_http_requests_total HTTP requests handled by the application.",
            "# TYPE legal_rag_http_requests_total counter",
        ]
        with self._lock:
            request_items = sorted(self._requests.items())
            health_items = sorted(self._health.items())
        for (method, path, status_code), metric in request_items:
            labels = _labels(method=method, path=path, status_code=status_code)
            lines.append(f"legal_rag_http_requests_total{{{labels}}} {metric.count}")
            lines.append(
                f"legal_rag_http_request_duration_seconds_sum{{{labels}}} "
                f"{metric.total_duration_seconds:.6f}"
            )
        lines.extend(
            [
                "# HELP legal_rag_health_check_status Last observed dependency health status.",
                "# TYPE legal_rag_health_check_status gauge",
            ]
        )
        for check_name, ok in health_items:
            lines.append(f'legal_rag_health_check_status{{check="{_escape_label(check_name)}"}} {int(ok)}')
        return "\n".join(lines) + "\n"


registry = MetricsRegistry()


def record_health_statuses(statuses: list[object]) -> None:
    for status in statuses:
        registry.set_health(str(status.name), bool(status.ok))


def _labels(**values: str) -> str:
    return ",".join(f'{key}="{_escape_label(value)}"' for key, value in values.items())


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def monotonic_time() -> float:
    return time.perf_counter()
