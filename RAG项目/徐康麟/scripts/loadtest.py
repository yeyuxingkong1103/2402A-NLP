#!/usr/bin/env python3
"""轻量压测：并发打真实 HTTP 端点，产出可复核的 QPS / 延迟分位 / 限流读数。

**它测的是什么、不是什么**（写在最前面，避免把读数读过头）：

* 测的是**服务层**：HTTP 栈、线程池/并发闸门、鉴权与限流、检索 + 拼提示词 + 后处理；
* **不测** GPU 推理吞吐 —— 本机跑用 `--spawn-server` 时大模型是 **Mock**（离线栈），
  所以 `/chat` 的延迟里**不含**模型生成时间。要看真机生成延迟请用
  `docs/MONITORING.md` 里的 `llm_ttft_seconds` / `llm_itl_seconds` 指标，或 `.pytmp/cloud/bench_vllm.py`。

两种用法::

    # A) 自带一个**隔离**的离线服务（临时目录，绝不碰仓库里的 index/ 与 Redis）
    python scripts/loadtest.py --spawn-server --out eval/results/loadtest-local.json

    # B) 打一个已经在跑的服务（例如云端 27B）
    python scripts/loadtest.py --base-url http://127.0.0.1:18080 --scenario health

场景：
* ``health``  —— GET /health（**读快照**，O(1)、不占线程池：默认路径，量 HTTP + 线程池底噪）
* ``health-deep`` —— GET /health?deep=1（**现采**：依赖探针 + 查库，含探针超时；用来复核
  快照带来的增益，见 D4 / docs/LOAD-TEST.md §3）
* ``chat``    —— POST /chat（走完整链路；离线栈下不含 GPU 生成）
* ``metrics`` —— GET /metrics（大响应体，量序列化成本）
* ``ratelimit`` —— 快速打鉴权端点直到出现 **429**，核对限流是否真的生效、文案是否规范
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 上限保护：别让一次手滑把服务打崩
MAX_REQUESTS = 20000
MAX_CONCURRENCY = 128


# --------------------------------------------------------------------------- 结果统计

def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * ratio), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(samples: list[float], statuses: dict[str, int], wall: float,
              errors: list[str]) -> dict:
    ordered = sorted(samples)
    total = len(samples)
    return {
        "requests": total,
        "wall_seconds": round(wall, 3),
        "qps": round(total / wall, 2) if wall > 0 else 0.0,
        "ok": sum(count for code, count in statuses.items()
                  if str(code).startswith("2")),
        "status_counts": dict(sorted(statuses.items())),
        "errors": errors[:10],
        "error_kinds": len(errors),
        "latency_ms": {
            "min": round(min(ordered) * 1000, 2) if ordered else 0.0,
            "mean": round(statistics.fmean(ordered) * 1000, 2) if ordered else 0.0,
            "p50": round(_percentile(ordered, 0.50) * 1000, 2),
            "p90": round(_percentile(ordered, 0.90) * 1000, 2),
            "p95": round(_percentile(ordered, 0.95) * 1000, 2),
            "p99": round(_percentile(ordered, 0.99) * 1000, 2),
            "max": round(max(ordered) * 1000, 2) if ordered else 0.0,
        },
    }


# --------------------------------------------------------------------------- 场景

class _Client:
    """每线程一条 keep-alive 连接（压测里反复建连接会把 connect 成本算进延迟）。"""

    def __init__(self, host: str, port: int, timeout: float):
        self.host, self.port, self.timeout = host, port, timeout
        self.conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def call(self, method: str, path: str, body: dict | None = None,
             headers: dict | None = None) -> tuple[int, bytes, dict]:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
        head = {"Connection": "keep-alive"}
        if payload:
            head["Content-Type"] = "application/json; charset=utf-8"
        head.update(headers or {})
        for attempt in (1, 2):                       # 连接被服务端关掉时重建一次
            try:
                self.conn.request(method, path, body=payload, headers=head)
                response = self.conn.getresponse()
                data = response.read()
                return response.status, data, dict(response.getheaders())
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = http.client.HTTPConnection(self.host, self.port,
                                                       timeout=self.timeout)
                if attempt == 2:
                    raise
        return 0, b"", {}


def _scenario_requests(scenario: str, user: str, session: str) -> list[tuple[str, str, dict | None]]:
    if scenario == "livez":
        return [("GET", "/livez", None)]
    if scenario == "health":
        return [("GET", "/health", None)]
    if scenario == "health-deep":
        return [("GET", "/health?deep=1", None)]
    if scenario == "metrics":
        return [("GET", "/metrics", None)]
    if scenario == "chat":
        return [("POST", "/chat", {
            "user_id": user, "role_id": "lawyer", "session_id": session,
            "message": "民间借贷的利率司法保护上限是多少？", "stream": False})]
    raise ValueError(f"未知场景：{scenario}（ratelimit 由专门的函数处理）")


def run_load(host: str, port: int, scenario: str, requests: int, concurrency: int,
             timeout: float) -> dict:
    samples: list[float] = []
    statuses: dict[str, int] = {}
    errors: list[str] = []
    lock = threading.Lock()
    plan = _scenario_requests(scenario, "loadtest-user", f"{scenario}-session")

    def worker(worker_id: int) -> None:
        client = _Client(host, port, timeout)
        local_samples: list[float] = []
        local_status: dict[str, int] = {}
        local_errors: list[str] = []
        for index in range(worker_id, requests, concurrency):
            method, path, body = plan[index % len(plan)]
            started = time.perf_counter()
            try:
                status, _data, _headers = client.call(method, path, body)
                local_status[str(status)] = local_status.get(str(status), 0) + 1
            except Exception as exc:  # noqa: BLE001 - 单次失败要计数，不能中断压测
                local_status["EXC"] = local_status.get("EXC", 0) + 1
                local_errors.append(f"{type(exc).__name__}: {exc}")
            local_samples.append(time.perf_counter() - started)
        with lock:
            samples.extend(local_samples)
            for code, count in local_status.items():
                statuses[code] = statuses.get(code, 0) + count
            errors.extend(local_errors)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    return summarize(samples, statuses, wall, errors)


def run_ratelimit(host: str, port: int, probe: int, timeout: float) -> dict:
    """连打鉴权端点直到出现 429：核对"限流真的生效"且**文案规范**。"""
    client = _Client(host, port, timeout)
    statuses: dict[str, int] = {}
    first_429 = 0
    body_429 = ""
    retry_after = ""
    for index in range(1, probe + 1):
        query = urllib.parse.urlencode({"username": f"probe-{index}"})
        status, data, headers = client.call("GET", f"/auth/username-available?{query}")
        statuses[str(status)] = statuses.get(str(status), 0) + 1
        if status == 429 and not first_429:
            first_429 = index
            retry_after = headers.get("Retry-After", "")
            try:
                body_429 = json.dumps(json.loads(data.decode("utf-8", "replace")),
                                      ensure_ascii=False)[:200]
            except ValueError:
                body_429 = data.decode("utf-8", "replace")[:200]
    return {
        "probe_requests": probe,
        "status_counts": dict(sorted(statuses.items())),
        "first_429_at": first_429,
        "body_429": body_429,
        "retry_after": retry_after,
    }


# --------------------------------------------------------------------------- 自带隔离服务

def spawn_server(port: int, corpus_chars: int = 4000) -> tuple[Any, Path]:
    """起一个**完全隔离**的离线服务：临时目录 + 不可用的 Redis（走内存降级）。

    隔离是硬要求：默认配置会连仓库里的 `index/legal_rag.db` 与本机 Redis，
    压测会往里面塞测试账号/会话 —— 那是**污染开发数据**，不能干。
    """
    import tempfile

    import uvicorn

    from legal_rag.api.app import create_app
    from legal_rag.config import ChunkConfig, RagConfig, RetrievalConfig

    tmp = Path(tempfile.mkdtemp(prefix="legalrag-loadtest-"))
    config = RagConfig()
    config.index_dir = tmp / "index"
    config.knowledge_dir = tmp / "knowledge"
    config.knowledge_dir.mkdir(parents=True, exist_ok=True)
    config.vector_store = "memory"
    config.embedding_provider = "offline"
    config.rerank_provider = "cosine"
    config.llm_provider = "mock"
    config.mysql_dsn = ""
    config.sqlite_path = str(tmp / "index" / "loadtest.db")
    config.redis_url = "redis://127.0.0.1:6399/0"      # 故意不可用 -> 内存降级，不碰本机 Redis
    config.upload.upload_dir = str(tmp / "uploads")
    config.logging.sys_metrics_enabled = False
    config.chunk = ChunkConfig(strategy="sentence", chunk_size=120, overlap=10, parent_size=400)
    config.retrieval = RetrievalConfig(vector_top_k=5, keyword_top_k=5, final_top_k=3)
    config.ensure_dirs()
    (config.knowledge_dir / "kb.md").write_text(
        ("民间借贷的利率司法保护上限，是合同成立时一年期贷款市场报价利率的四倍。\n\n"
         "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，"
         "应当每月支付二倍的工资。\n\n") * max(corpus_chars // 120, 1), encoding="utf-8")

    app = create_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(120):
        try:
            status, _data, _headers = _Client("127.0.0.1", port, 5.0).call("GET", "/health")
            if status == 200:
                return server, tmp
        except Exception:  # noqa: BLE001 - 还没起来
            pass
        time.sleep(0.5)
    raise RuntimeError("自带服务 60s 内没起来")


def main() -> int:
    parser = argparse.ArgumentParser(description="轻量压测（服务层）")
    parser.add_argument("--base-url", default="", help="要压的服务地址；不给就用 --spawn-server")
    parser.add_argument("--spawn-server", action="store_true",
                        help="自带一个隔离的离线服务（临时目录 + 内存 Redis 降级）")
    parser.add_argument("--port", type=int, default=18099, help="--spawn-server 时的端口")
    parser.add_argument("--scenario", default="health",
                        choices=("livez", "health", "health-deep", "chat", "metrics",
                                 "ratelimit", "all"))
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--probe", type=int, default=40, help="ratelimit 场景连打次数")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=10.0,
                        help="开测前的静默等待（秒）：等启动期的嵌入冷加载/BM25 建索引跑完，"
                             "否则第一个场景量到的是**启动抖动**而不是稳态性能")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.requests > MAX_REQUESTS or args.concurrency > MAX_CONCURRENCY:
        print(f"!! 请求数/并发超上限（{MAX_REQUESTS}/{MAX_CONCURRENCY}）", file=sys.stderr)
        return 2

    server = None
    tmp: Path | None = None
    if args.spawn_server:
        if not args.base_url:
            print(f"自带隔离服务启动中（端口 {args.port}，临时目录会在结束时删除）...")
            server, tmp = spawn_server(args.port)
            args.base_url = f"http://127.0.0.1:{args.port}"
    if not args.base_url:
        print("!! 请给 --base-url 或加 --spawn-server", file=sys.stderr)
        return 2

    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    if args.settle > 0:
        # 先探一次，确认服务是活的；再静默等待启动期的重活跑完（否则第一个场景量的是启动抖动）
        before_status, _b, _h = _Client(host, port, args.timeout).call("GET", "/health")
        print(f"静默 {args.settle:.0f}s 等服务进入稳态（探活 /health -> {before_status}）...")
        time.sleep(args.settle)

    report: dict[str, Any] = {
        "base_url": args.base_url,
        "spawned_server": bool(server),
        "note": ("大模型=Mock（离线栈）⇒ /chat 延迟**不含** GPU 生成；"
                 "本报告只反映服务层" if server else "打的是外部服务，读数含义以该服务配置为准"),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "scenarios": {},
    }
    scenarios = (["livez", "health", "health-deep", "chat", "metrics", "ratelimit"]
                 if args.scenario == "all" else [args.scenario])
    try:
        for scenario in scenarios:
            if scenario == "ratelimit":
                result = run_ratelimit(host, port, args.probe, args.timeout)
                print(f"\n[ratelimit] 打 {args.probe} 次 -> {result['status_counts']}；"
                      f"首次 429 出现在第 {result['first_429_at']} 次；响应体：{result['body_429']}")
            else:
                result = run_load(host, port, scenario, args.requests, args.concurrency,
                                  args.timeout)
                latency = result["latency_ms"]
                print(f"\n[{scenario}] {result['requests']} 次 / 并发 {args.concurrency}："
                      f"QPS={result['qps']}  成功={result['ok']}  状态={result['status_counts']}")
                print(f"           延迟(ms) min={latency['min']} mean={latency['mean']} "
                      f"p50={latency['p50']} p90={latency['p90']} p95={latency['p95']} "
                      f"p99={latency['p99']} max={latency['max']}")
                if result["error_kinds"]:
                    print(f"           异常 {result['error_kinds']} 种，例：{result['errors'][:2]}")
            report["scenarios"][scenario] = result
    finally:
        if server is not None:
            server.should_exit = True
            time.sleep(0.5)
        if tmp is not None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n证据已写：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
