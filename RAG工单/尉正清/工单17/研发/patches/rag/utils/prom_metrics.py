# 工单17 新增文件：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏
"""Prometheus 指标暴露

在**独立端口**（默认 9390）起一个 metrics HTTP 服务，不占用业务端口，
也不经过 nginx —— 压测把业务端口打满时，监控照样采得到数据。

暴露的指标：

| 指标 | 类型 | 用途 |
|---|---|---|
| `ragflow_http_requests_total` | Counter | 按 方法/路由/状态码 统计请求量，看错误率 |
| `ragflow_http_request_duration_seconds` | Histogram | 按路由统计延迟，P95 从这里出 |
| `ragflow_http_in_flight_requests` | Gauge | 同时在处理的请求数，看堆积 |
| `ragflow_process_resident_memory_bytes` | Gauge | 进程 RSS，**验收里「12 小时内存增长≤20%」看它** |
| `ragflow_process_cpu_percent` | Gauge | 进程 CPU 占用 |
| `ragflow_task_queue_depth` | Gauge | 解析任务队列长度（Redis Stream），看 Task Executor 跟不跟得上 |

用法（已在 api/ragflow_server.py 里接好，容器启动即生效）：

    from rag.utils.prom_metrics import install
    install(app)
"""
import logging
import os
import threading
import time

from prometheus_client import Counter, Gauge, Histogram, start_http_server

HTTP_REQUESTS = Counter(
    "ragflow_http_requests_total", "HTTP 请求总数",
    ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "ragflow_http_request_duration_seconds", "HTTP 请求耗时（秒）",
    ["path"],
    # 分桶贴着实测分布设计：单请求正常 2~5 秒，并发劣化时会落到 10/30 秒的桶里
    buckets=(0.1, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, float("inf")))
IN_FLIGHT = Gauge(
    "ragflow_http_in_flight_requests", "正在处理中的请求数")
PROC_RSS = Gauge(
    "ragflow_process_resident_memory_bytes", "进程常驻内存（字节）")
PROC_CPU = Gauge(
    "ragflow_process_cpu_percent", "进程 CPU 占用（%）")
QUEUE_DEPTH = Gauge(
    "ragflow_task_queue_depth", "解析任务队列长度（Redis Stream）")

_installed = False
QUEUE_KEY = os.environ.get("RAGFLOW_QUEUE_KEY", "rag_flow_svr_queue")


def _queue_depth():
    """RAGFlow 的解析任务队列在 Redis Stream 里，堆积说明 Task Executor 跟不上。"""
    try:
        from common import settings
        return float(settings.redis_conn.connection.xlen(QUEUE_KEY))
    except Exception:
        return 0.0


def _collector():
    """后台线程定期刷新那些「拉取式」的指标（RSS/CPU/队列）。"""
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None
    while True:
        try:
            if proc is not None:
                PROC_RSS.set(proc.memory_info().rss)
                PROC_CPU.set(proc.cpu_percent(interval=None))
            QUEUE_DEPTH.set(_queue_depth())
        except Exception:
            pass
        time.sleep(5)


def install(app, port=None):
    """挂到 Quart app 上：起 metrics 服务 + 注册请求钩子。"""
    global _installed
    if _installed:
        return
    port = int(port or os.environ.get("PROMETHEUS_PORT", "9390"))

    # 独立端口。用 start_http_server 而不是在 Quart 里加路由 ——
    # 这样监控端点在业务被打满时仍然可达。
    start_http_server(port)
    threading.Thread(target=_collector, daemon=True).start()

    @app.before_request
    async def _metrics_before():
        from quart import g
        g._metrics_t0 = time.time()
        IN_FLIGHT.inc()

    @app.after_request
    async def _metrics_after(resp):
        from quart import g, request
        IN_FLIGHT.dec()
        try:
            # 用路由规则而不是原始 path，避免把 chat_id 之类的高基数参数带进标签
            path = request.url_rule.rule if request.url_rule else request.path
            HTTP_REQUESTS.labels(request.method, path, resp.status_code).inc()
            t0 = getattr(g, "_metrics_t0", None)
            if t0:
                HTTP_LATENCY.labels(path).observe(time.time() - t0)
        except Exception:
            pass
        return resp

    _installed = True
    logging.info(f"[metrics] Prometheus 指标已在 :{port}/metrics 暴露")
