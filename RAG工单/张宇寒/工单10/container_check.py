from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def fetch_json(url: str, method: str = "GET", timeout: int = 10) -> dict:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_results(health: dict, evaluation: dict) -> dict:
    if not health.get("ready"):
        raise RuntimeError("GraphRAG索引尚未就绪")
    precision = float(evaluation.get("average_context_precision", 0))
    recall = float(evaluation.get("average_context_recall", 0))
    if precision < 0.8:
        raise RuntimeError(f"Context Precision未达标：{precision:.4f} < 0.8")
    if recall < 0.9:
        raise RuntimeError(f"Context Recall未达标：{recall:.4f} < 0.9")
    if not evaluation.get("under_3_seconds"):
        raise RuntimeError("存在响应时间超过3秒的问题")
    return {
        "status": "passed",
        "question_count": int(evaluation.get("count", 0)),
        "context_precision": precision,
        "context_recall": recall,
        "under_3_seconds": True,
    }


def verify(base_url: str, attempts: int = 20, interval: float = 2.0) -> dict:
    base_url = base_url.rstrip("/")
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            health = fetch_json(f"{base_url}/api/health")
            evaluation = fetch_json(f"{base_url}/api/evaluate-all?top_k=5", method="POST", timeout=30)
            return validate_results(health, evaluation)
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(f"容器服务验收失败：{last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="验证工单10 Docker金融GraphRAG服务")
    parser.add_argument("--base-url", default="http://127.0.0.1:8021")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.base_url), ensure_ascii=False, indent=2))
        return 0
    except RuntimeError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
