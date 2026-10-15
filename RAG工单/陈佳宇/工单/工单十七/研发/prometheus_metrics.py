# -*- coding: utf-8 -*-
# prometheus_metrics.py 工单17 prometheus指标埋点
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# http请求总计数
HTTP_REQUEST_TOTAL = Counter(
    "ragflow_http_request_total",
    "Total http api request count",
    ["api_path","status"]
)

# 接口耗时直方图（P50/P95/P99自动计算）
HTTP_REQUEST_DURATION = Histogram(
    "ragflow_http_duration_seconds",
    "Http api request duration seconds",
    ["api_path"],
    buckets=[0.5,1,2,3,5,8,10]
)

# redis队列长度
REDIS_QUEUE_LEN = Gauge(
    "ragflow_redis_queue_length",
    "Redis stream task queue pending count"
)

# 连接池活跃连接数
POOL_ACTIVE_CONN = Gauge(
    "ragflow_pool_active_conn",
    "DB/Vector pool active connections"
)

def start_metrics_server(port=9090):
    """启动prometheus指标http服务"""
    start_http_server(port)
    print(f"prometheus metrics server start at port:{port}")


# 简易上下文装饰器统计耗时
from functools import wraps
import time

def api_metrics(api_path:str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            t0 = time.perf_counter()
            status = "ok"
            try:
                return func(*args,**kwargs)
            except Exception as e:
                status = "error"
                raise e
            finally:
                cost = time.perf_counter()‑t0
                HTTP_REQUEST_TOTAL.labels(api_path=api_path, status=status).inc()
                HTTP_REQUEST_DURATION.labels(api_path=api_path).observe(cost)
        return wrapper
    return decorator

"""
# 接口示例
@api_metrics(api_path="/v1/chats/completions")
def chat_api():
    pass
"""
