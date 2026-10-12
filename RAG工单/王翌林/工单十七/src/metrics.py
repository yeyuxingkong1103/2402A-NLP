# -*- coding: utf-8 -*-
"""Prometheus 指标采集：请求量、延迟分位、在途请求、缓存、内存。"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_TOTAL = Counter("rag_requests_total", "Total requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds", "Request latency",
    ["endpoint"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10),
)
INFLIGHT = Gauge("rag_inflight_requests", "In-flight requests", ["endpoint"])
QUEUE_DEPTH = Gauge("rag_queue_depth", "Async task queue depth")
CACHE_SIZE = Gauge("rag_cache_size", "Semantic cache entries")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
PROCESS_MEM_RSS_MB = Gauge("rag_process_rss_mb", "Process RSS memory (MB)")


def metrics_text():
    return generate_latest(), CONTENT_TYPE_LATEST
