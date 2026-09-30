"""
load_test.py — 压力测试

对运行中的 /chat 接口施加并发，统计 QPS、平均响应时间、P95 延迟与错误率。

    python scripts/load_test.py --concurrency 50
    python scripts/load_test.py --concurrency 100 --requests 500
    python scripts/load_test.py --concurrency 200 --endpoint /chat/stream

注意：每次请求都会真实调用大模型，产生费用与延迟，
建议先用小并发摸清单次耗时，再逐步加压。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "劳动合同到期不续签，公司需要支付经济补偿吗？",
    "经济补偿金怎么计算？",
    "休息日加班工资怎么算？",
    "试用期最长可以约定多久？",
    "感冒需要吃抗生素吗？",
    "成年人每天建议睡多久？",
    "空腹血糖多少算糖尿病？",
    "现在完成时和一般过去时有什么区别？",
    "被动语态怎么构成？",
    "look forward to 后面接什么形式？",
]


async def run_request(client, url: str, payload: dict, timeout: float) -> tuple[float, bool, str]:
    """发一次请求，返回 (耗时秒, 是否成功, 错误信息)。"""
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            return elapsed, False, f"HTTP {response.status_code}"
        return elapsed, True, ""
    except Exception as exc:
        return time.perf_counter() - started, False, type(exc).__name__


async def worker(client, url, semaphore, total_requests, counter, results, timeout, stream):
    """持续取任务直到总数耗尽。"""
    while True:
        async with semaphore:
            index = counter[0]
            if index >= total_requests:
                return
            counter[0] += 1

        payload = {"user_id": f"load_{index % 20}", "message": random.choice(QUESTIONS)}
        if stream:
            results.append(await run_request(client, url, payload, timeout))
        else:
            results.append(await run_request(client, url, payload, timeout))


def percentile(values: list[float], ratio: float) -> float:
    """简单分位数，避免额外的 numpy 依赖。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[position]


async def main_async(args) -> int:
    try:
        import httpx
    except ImportError:
        print("缺少 httpx，请执行：pip install httpx")
        return 1

    base = args.host.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"
    url = f"{base}:{args.port}{args.endpoint}"

    print("压力测试开始")
    print(f"  目标      : POST {url}")
    print(f"  并发数    : {args.concurrency}")
    print(f"  请求总数  : {args.requests}")
    print(f"  超时      : {args.timeout}s")
    print()

    results: list[tuple[float, bool, str]] = []
    counter = [0]
    semaphore = asyncio.Semaphore(args.concurrency)

    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [
            worker(client, url, semaphore, args.requests, counter, results, args.timeout, args.stream)
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - started

    latencies = [r[0] for r in results]
    succeeded = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    print("压力测试结果")
    print(f"  总耗时      : {wall:.2f}s")
    print(f"  QPS         : {len(results) / wall:.2f}")
    print(f"  成功 / 失败 : {len(succeeded)} / {len(failures)}")
    print(f"  错误率      : {len(failures) / len(results) * 100:.2f}%")

    if latencies:
        print(f"  平均延迟    : {statistics.mean(latencies):.3f}s")
        print(f"  中位数      : {percentile(latencies, 0.5):.3f}s")
        print(f"  P95         : {percentile(latencies, 0.95):.3f}s")
        print(f"  P99         : {percentile(latencies, 0.99):.3f}s")
        print(f"  最快 / 最慢 : {min(latencies):.3f}s / {max(latencies):.3f}s")

    if failures:
        reasons: dict[str, int] = {}
        for _, _, reason in failures:
            reasons[reason] = reasons.get(reason, 0) + 1
        print("\n  失败原因：")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {count}")

    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 服务压力测试")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--endpoint", default="/chat", choices=["/chat", "/chat/stream"])
    parser.add_argument("--concurrency", type=int, default=50, help="并发数（需求要求测 50/100/200）")
    parser.add_argument("--requests", type=int, default=200, help="请求总数")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    args = parser.parse_args()

    args.stream = args.endpoint == "/chat/stream"
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
