"""Minimal Prometheus metrics for API request visibility."""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "path", "status"])
ERRORS = Counter("rag_http_errors_total", "HTTP errors", ["method", "path"])
LATENCY = Histogram("rag_http_request_duration_seconds", "HTTP latency", ["method", "path"])


async def metrics_middleware(request, call_next):
    """Record route timing without changing the response path."""
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    status = str(response.status_code)
    REQUESTS.labels(request.method, path, status).inc()
    if response.status_code >= 500:
        ERRORS.labels(request.method, path).inc()
    LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
    return response


def metrics_payload() -> bytes:
    """Return the standard Prometheus exposition payload."""
    return generate_latest()
