# -*- coding: utf-8 -*-
"""压力测试：对 /api/chat 做 QPS 压测（纯 Python 实现，无需 JMeter）。

使用 concurrent.futures.ThreadPoolExecutor 模拟并发用户，
测量：吞吐量 (QPS)、平均延迟、P95/P99 延迟、错误率。

前提：API 服务已启动（python api.py 或 python -m uvicorn api:app）。
若服务未启动，本脚本会提示并退出。

运行方式：
  # 终端1：启动服务
  python api.py
  # 终端2：跑压测
  python tests/stress_test.py
  # 自定义参数
  python tests/stress_test.py --users 20 --rounds 50 --host 127.0.0.1 --port 8000
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
USERNAME = "stress_tester"
PASSWORD = "stress1234"

# 压测用的查询集（中英文混合，模拟真实负载）
QUERIES = [
    "I miss you",
    "我想你",
    "go to sleep",
    "Hurry up",
    "赶快!",
    "I have to go to sleep",
    "今天天气怎么样",
    "education in this world",
    "I never liked biology",
    "I'm at a loss for words",
]


def api_call(method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict | str]:
    """发一个 HTTP 请求，返回 (status_code, json_or_text)。"""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)


def setup_user() -> str:
    """注册并登录压测用户，返回 token。"""
    # 注册（可能已存在，忽略错误）
    api_call("POST", "/api/auth/register", {"username": USERNAME, "password": PASSWORD})
    # 登录
    code, data = api_call("POST", "/api/auth/login", {"username": USERNAME, "password": PASSWORD})
    if code != 200:
        print(f"[ERROR] 登录失败: {code} {data}")
        sys.exit(1)
    token = data["token"]
    print(f"[INFO] 压测用户 {USERNAME} 登录成功，token 已获取")
    return token


def run_stress(users: int, rounds: int, token: str) -> None:
    """并发压测 /api/chat。"""
    total_requests = users * rounds
    latencies: list[float] = []
    errors = 0
    completed = 0

    print(f"\n{'=' * 60}")
    print(f"  压力测试开始")
    print(f"  并发用户数: {users}")
    print(f"  每用户请求数: {rounds}")
    print(f"  总请求数: {total_requests}")
    print(f"  目标接口: POST /api/chat")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    def worker(uid: int) -> tuple[int, float, bool]:
        """单用户发送 rounds 轮请求，返回 (用户ID, 总耗时, 是否有错误)。"""
        local_latencies = []
        local_errors = 0
        for r in range(rounds):
            query = QUERIES[(uid + r) % len(QUERIES)]
            t0 = time.time()
            code, _ = api_call(
                "POST", "/api/chat",
                {"query": query, "role_code": "teacher", "top_k": 3},
                token=token,
            )
            elapsed = time.time() - t0
            local_latencies.append(elapsed)
            if code != 200:
                local_errors += 1
        return uid, local_latencies, local_errors

    with ThreadPoolExecutor(max_workers=users) as pool:
        futures = {pool.submit(worker, uid): uid for uid in range(users)}
        for future in as_completed(futures):
            uid, lats, errs = future.result()
            latencies.extend(lats)
            errors += errs
            completed += 1
            # 进度输出
            if completed % max(1, users // 5) == 0 or completed == users:
                print(f"  进度: {completed}/{users} 用户完成")

    total_time = time.time() - start_time

    # ===== 统计 =====
    qps = len(latencies) / total_time if total_time > 0 else 0
    avg_lat = statistics.mean(latencies) if latencies else 0
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[p95_idx] if latencies_sorted and p95_idx < len(latencies_sorted) else 0
    p99_idx = int(len(latencies_sorted) * 0.99)
    p99 = latencies_sorted[p99_idx] if latencies_sorted and p99_idx < len(latencies_sorted) else 0
    min_lat = min(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0
    error_rate = errors / len(latencies) * 100 if latencies else 0

    print(f"\n{'=' * 60}")
    print(f"  压测结果汇总")
    print(f"{'=' * 60}")
    print(f"  总请求数:     {len(latencies)}")
    print(f"  总耗时:       {total_time:.2f} s")
    print(f"  吞吐量 QPS:   {qps:.2f}")
    print(f"  错误数:       {errors}")
    print(f"  错误率:       {error_rate:.2f}%")
    print(f"  --- 延迟分布（秒）---")
    print(f"  最小延迟:     {min_lat:.4f} s")
    print(f"  平均延迟:     {avg_lat:.4f} s")
    print(f"  P50 延迟:     {p50:.4f} s")
    print(f"  P95 延迟:     {p95:.4f} s")
    print(f"  P99 延迟:     {p99:.4f} s")
    print(f"  最大延迟:     {max_lat:.4f} s")
    print(f"{'=' * 60}\n")

    # 判定是否通过（错误率 < 5% 且 QPS > 0）
    passed = error_rate < 5.0 and qps > 0
    print(f"  结论: {'PASS（通过）' if passed else 'FAIL（不通过）'}")
    print()

    # 写结果到 JTL 格式（兼容 JMeter 结果文件）
    jtl_path = Path(__file__).resolve().parent / "stress_result.jtl"
    with jtl_path.open("w", encoding="utf-8") as f:
        f.write("timeStamp,elapsed,label,responseCode,success,grpThreads,Latency\n")
        for lat in latencies_sorted:
            ts = int(time.time() * 1000)
            f.write(f"{ts},{int(lat * 1000)},/api/chat,200,true,1,{int(lat * 1000)}\n")
    print(f"  结果已写入: {jtl_path}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="RAG 系统 /api/chat 压力测试")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--users", type=int, default=10, help="并发用户数")
    parser.add_argument("--rounds", type=int, default=10, help="每用户请求数")
    args = parser.parse_args()

    BASE_URL = f"http://{args.host}:{args.port}"

    # 检查服务是否可达
    code, data = api_call("GET", "/health")
    if code != 200:
        print(f"[ERROR] 服务不可达 ({BASE_URL}): {code} {data}")
        print("[HINT] 请先启动服务: python api.py")
        sys.exit(1)
    print(f"[INFO] 服务可达: {BASE_URL} (kb_size={data.get('kb_size', '?')})")

    token = setup_user()
    run_stress(args.users, args.rounds, token)


if __name__ == "__main__":
    main()
