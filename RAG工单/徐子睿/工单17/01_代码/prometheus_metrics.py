# -*- coding: utf-8 -*-
"""
工单 17 · Prometheus 监控埋点（草案）
——把 API 响应时间、并发数、队列堆积、内存/显存指标暴露给 /metrics，
   供 Grafana 面板（演示视频要对比优化前后资源曲线）。

接入点：RAGFlow api/apps/__init__.py 注册 Blueprint；或在 service.py 里 wrap 请求。
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _OK = True
except Exception:
    _OK = False

if _OK:
    REQUESTS = Counter("ragflow_requests_total", "total requests", ["endpoint", "status"])
    LATENCY = Histogram("ragflow_request_latency_seconds", "request latency", ["endpoint"],
                        buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30))
    INFLIGHT = Gauge("ragflow_inflight_requests", "in-flight requests")
    QUEUE_PENDING = Gauge("ragflow_queue_pending", "redis stream pending tasks")
    MEM_RSS_MB = Gauge("ragflow_process_rss_mb", "process RSS (MB)")
    GPU_MEM_MB = Gauge("ragflow_gpu_mem_mb", "GPU memory used (MB)")


def observe_request(endpoint: str, status: int, seconds: float):
    if not _OK:
        return
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()
    LATENCY.labels(endpoint=endpoint).observe(seconds)


def update_system_gauges(pending: int | None = None, rss_mb: float | None = None, gpu_mb: float | None = None):
    if not _OK:
        return
    if pending is not None:
        QUEUE_PENDING.set(pending)
    if rss_mb is not None:
        MEM_RSS_MB.set(rss_mb)
    if gpu_mb is not None:
        GPU_MEM_MB.set(gpu_mb)


def metrics_endpoint():
    if not _OK:
        return ("prometheus_client not installed", 500)
    return (generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST})


if __name__ == "__main__":
    observe_request("/api/v1/chats_openai/x/chat/completions", 200, 1.9)
    print("prometheus_client available:", _OK)
