"""压力测试补充工具：保留旧版免费GET基线，答辩主文件不展开。"""
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


def percentile(values, ratio):
    """返回最邻近秩分位数；只服务于旧版免费基线。"""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(ratio * len(ordered)) - 1)
    return round(ordered[index], 3)


def validate_target(url):
    """验证免费GET地址，并阻止误请求会调用DeepSeek的/api/chat。"""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url必须是完整的http或https地址")
    path = unquote(parsed.path).lower().rstrip("/") or "/"
    if path == "/api/chat" or path.startswith("/api/chat/"):
        raise ValueError("免费基线禁止压测/api/chat；完整RAG请使用JMeter")
    return url


def one_request(url, timeout):
    """向免费GET地址发一次请求，并返回状态、耗时和错误类别。"""
    started = time.perf_counter()
    request = Request(url, headers={"User-Agent": "rag-roleplay-pressure-test/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            status, error = int(response.status), None
    except HTTPError as exc:
        status, error = int(exc.code), f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError) as exc:
        status, error = None, type(exc).__name__
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"ok": status is not None and 200 <= status < 400,
            "status": status, "latency_ms": round(elapsed_ms, 3), "error": error}


def run_test(url, requests, concurrency, timeout):
    """兼容旧调用：并发测试免费GET；它不代表完整RAG问答性能。"""
    url = validate_target(url)
    if requests < 1 or concurrency < 1 or timeout <= 0:
        raise ValueError("请求数、并发数和超时时间都必须大于0")
    workers = min(concurrency, requests)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_request, url, timeout) for _ in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    duration = time.perf_counter() - started
    successes = sum(item["ok"] for item in results)
    latencies = [item["latency_ms"] for item in results]
    statuses = Counter(str(item["status"]) for item in results if item["status"] is not None)
    errors = Counter(item["error"] for item in results if item["error"])
    return {
        "kind": "free_http_baseline", "created_at": datetime.now(timezone.utc).isoformat(),
        "target": url, "method": "GET", "requests": requests, "concurrency": workers,
        "timeout_seconds": timeout, "successes": successes, "failures": requests - successes,
        "success_rate": round(successes / requests, 6), "duration_seconds": round(duration, 6),
        "qps": round(requests / duration, 3) if duration else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 3),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "status_counts": dict(sorted(statuses.items())),
        "error_counts": dict(sorted(errors.items())),
        "scope": "FastAPI基础GET接口；不代表完整RAG和DeepSeek问答性能",
    }
