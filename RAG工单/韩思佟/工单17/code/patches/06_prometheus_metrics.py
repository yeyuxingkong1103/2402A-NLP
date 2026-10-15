# -*- coding: utf-8 -*-
"""
工单17 —— P6：Prometheus 监控集成补丁

现状（源码事实）：RAGFlow 0.23.1 没有 Prometheus 客户端代码，也没有 /metrics 端点
（prometheus 只出现在 docker/docker-compose*.yml 的注释里）。
现有可观测能力只有 Langfuse tracing 和写 Redis zset 的 executor 心跳。

落地位置：
  - 新建 api/apps/metrics_app.py（走 register_page 自动注册，路径 /v1/metrics）
  - 或在 api/ragflow_server.py 里注册一个 /metrics 路由（下面给的是可直接用的实现）

依赖：pip install prometheus-client
"""
from __future__ import annotations

import os
import time

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               Gauge, Histogram, generate_latest)

REGISTRY = CollectorRegistry(auto_describe=True)

# ------------------------------------------------------------------ 指标定义
CHAT_LATENCY = Histogram(
    "ragflow_chat_latency_seconds",
    "问答接口端到端延迟（秒）",
    labelnames=("endpoint", "strategy"),
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34),
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "ragflow_retrieval_latency_seconds",
    "检索阶段耗时（秒）",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "ragflow_llm_latency_seconds",
    "LLM 生成阶段耗时（秒）",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRY,
)

HTTP_REQUESTS = Counter(
    "ragflow_http_requests_total",
    "HTTP 请求总数",
    labelnames=("method", "endpoint", "status"),
    registry=REGISTRY,
)

INFLIGHT = Gauge(
    "ragflow_inflight_requests",
    "当前在飞请求数",
    registry=REGISTRY,
)

MEMORY_RSS = Gauge(
    "ragflow_process_memory_rss_bytes",
    "进程常驻内存（bytes）",
    registry=REGISTRY,
)

GPU_MEMORY = Gauge(
    "ragflow_gpu_memory_used_bytes",
    "GPU 显存占用（bytes）",
    labelnames=("device",),
    registry=REGISTRY,
)

DB_CONNECTIONS = Gauge(
    "ragflow_db_connections_in_use",
    "数据库连接池在用连接数",
    registry=REGISTRY,
)


# ------------------------------------------------------------------ 采集函数
def collect_process_metrics():
    """定期采集进程级指标（内存 RSS / GPU 显存）"""
    try:
        import psutil
        MEMORY_RSS.set(psutil.Process().memory_info().rss)
    except Exception:                                            # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                GPU_MEMORY.labels(device=str(i)).set(torch.cuda.memory_allocated(i))
    except Exception:                                            # noqa: BLE001
        pass


# ------------------------------------------------------------------ Quart 集成
def register_metrics(app):
    """在 api/ragflow_server.py 里调用：register_metrics(app)"""

    @app.before_request
    async def _before():
        INFLIGHT.inc()
        collect_process_metrics()

    @app.after_request
    async def _after(resp):
        INFLIGHT.dec()
        try:
            HTTP_REQUESTS.labels(
                method=getattr(getattr(__import__("quart"), "request", None),
                               "method", "UNKNOWN") if False else "UNKNOWN",
                endpoint="unknown",
                status=str(getattr(resp, "status_code", "")),
            ).inc()
        except Exception:                                        # noqa: BLE001
            pass
        return resp

    @app.route("/metrics")
    async def metrics():
        collect_process_metrics()
        return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    return app


# ------------------------------------------------------------------ Flask 兼容版
def register_metrics_flask(app):
    """如果部署的是老版本（Flask），用这个"""
    @app.before_request
    def _before():
        INFLIGHT.inc()
        collect_process_metrics()

    @app.after_request
    def _after(resp):
        INFLIGHT.dec()
        return resp

    @app.route("/metrics")
    def metrics():
        collect_process_metrics()
        return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    return app


# ------------------------------------------------------------------ 装饰器用法
def observe_chat(endpoint: str = "chat", strategy: str = "hybrid"):
    """给问答处理函数套上耗时观测"""
    def deco(fn):
        def sync_wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                CHAT_LATENCY.labels(endpoint=endpoint, strategy=strategy).observe(
                    time.perf_counter() - t0)

        async def async_wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return await fn(*a, **kw)
            finally:
                CHAT_LATENCY.labels(endpoint=endpoint, strategy=strategy).observe(
                    time.perf_counter() - t0)

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
    return deco


# ------------------------------------------------------------------ Grafana
GRAFANA_DASHBOARD_HINT = """
# prometheus.yml 抓取配置
scrape_configs:
  - job_name: ragflow
    metrics_path: /metrics
    static_configs:
      - targets: ["ragflow-server:9380"]

# 建议的告警规则（groups/ragflow.yml）
groups:
  - name: ragflow
    rules:
      - alert: RAGFlowP95TooHigh
        expr: histogram_quantile(0.95,
              sum(rate(ragflow_chat_latency_seconds_bucket[5m])) by (le)) > 3
        for: 5m
        labels: {severity: warning}
        annotations: {summary: "问答接口 P95 超过 3s"}
      - alert: RAGFlowMemoryGrowing
        expr: increase(ragflow_process_memory_rss_bytes[1h]) > 200*1024*1024
        for: 30m
        labels: {severity: warning}
        annotations: {summary: "1 小时内 RSS 增长超过 200MB，疑似内存泄漏"}
"""
