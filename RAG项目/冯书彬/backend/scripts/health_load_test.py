import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class RequestResult:
    duration_seconds: float
    status_code: int | None
    error: str | None = None


def run_load_test(base_url: str, path: str, requests: int, concurrency: int, timeout: float) -> dict[str, object]:
    if requests < 1 or concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    started_at = time.perf_counter()
    results: list[RequestResult] = []
    lock = threading.Lock()

    def request_once() -> RequestResult:
        request_started = time.perf_counter()
        try:
            request = Request(f"{base_url.rstrip('/')}/{path.lstrip('/')}", method="GET")
            with urlopen(request, timeout=timeout) as response:
                response.read()
                result = RequestResult(time.perf_counter() - request_started, response.status)
        except HTTPError as exc:
            result = RequestResult(time.perf_counter() - request_started, exc.code, type(exc).__name__)
        except (URLError, TimeoutError, OSError) as exc:
            result = RequestResult(time.perf_counter() - request_started, None, type(exc).__name__)
        with lock:
            results.append(result)
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_once) for _ in range(requests)]
        for future in as_completed(futures):
            future.result()

    durations = sorted(result.duration_seconds for result in results)
    success_count = sum(result.status_code is not None and 200 <= result.status_code < 400 for result in results)
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for result in results:
        key = str(result.status_code) if result.status_code is not None else "connection_error"
        status_counts[key] = status_counts.get(key, 0) + 1
        if result.error:
            error_counts[result.error] = error_counts.get(result.error, 0) + 1
    elapsed = time.perf_counter() - started_at
    return {
        "base_url": base_url,
        "path": path,
        "requests": len(results),
        "concurrency": concurrency,
        "successes": success_count,
        "failures": len(results) - success_count,
        "elapsed_seconds": round(elapsed, 6),
        "requests_per_second": round(len(results) / elapsed, 3) if elapsed else 0.0,
        "latency_seconds": {
            "min": round(durations[0], 6) if durations else None,
            "median": round(statistics.median(durations), 6) if durations else None,
            "p95": round(_percentile(durations, 0.95), 6) if durations else None,
            "max": round(durations[-1], 6) if durations else None,
        },
        "status_counts": status_counts,
        "error_counts": error_counts,
    }


def _percentile(values: list[float], percentile: float) -> float:
    index = min(len(values) - 1, max(0, int((len(values) - 1) * percentile)))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行健康端点并发验收压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--path", default="/health/live")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()
    report = run_load_test(args.base_url, args.path, args.requests, args.concurrency, args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if int(report["failures"]) <= args.max_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
