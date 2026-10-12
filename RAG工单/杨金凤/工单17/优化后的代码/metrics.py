"""
Prometheus 监控指标
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Response
import time


REQUEST_COUNT = Counter(
    'ragflow_requests_total',
    'Total HTTP requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ragflow_request_duration_seconds',
    'HTTP request latency',
    ['endpoint'],
    buckets=[0.1, 0.5, 1, 2, 3, 5, 10, 30, 60]
)

ACTIVE_REQUESTS = Gauge(
    'ragflow_active_requests',
    'Current number of active requests'
)

MODEL_CACHE_SIZE = Gauge(
    'ragflow_model_cache_size',
    'Number of cached model instances'
)


def setup_metrics(app):
    """在 Flask app 上注册监控中间件和 /metrics 路由"""

    @app.before_request
    def _before_request():
        from flask import request
        request._start_ts = time.time()
        ACTIVE_REQUESTS.inc()

    @app.after_request
    def _after_request(response):
        from flask import request
        ACTIVE_REQUESTS.dec()
        if hasattr(request, '_start_ts'):
            latency = time.time() - request._start_ts
            REQUEST_LATENCY.labels(endpoint=request.path).observe(latency)
            REQUEST_COUNT.labels(
                endpoint=request.path,
                status=response.status_code
            ).inc()
        return response

    @app.route('/metrics')
    def _metrics():
        try:
            from rag.llm.singleton import stats
            MODEL_CACHE_SIZE.set(len(stats()))
        except Exception:
            pass
        return Response(generate_latest(), mimetype='text/plain')

    return app