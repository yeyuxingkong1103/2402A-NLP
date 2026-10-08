# -*- coding: utf-8 -*-
# 工单17：压测脚本（替代JMeter）
"""
模拟两种压测场景：
- 场景A：20并发用户，持续发送问答请求
- 场景B：10并发用户，5问答+5上传解析
记录响应时间、吞吐量、内存占用。
"""
import json
import statistics
import time
import threading
import urllib.request
from pathlib import Path
from collections import defaultdict

DEV = Path(__file__).resolve().parent
OUT = DEV.parent / "logs"
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(DEV))
from api_server import run_server


def http_post(url, data, timeout=30):
    """简单HTTP POST。"""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def run_scenario(server_info, scenario, duration_sec, concurrency):
    """运行压测场景。"""
    server, instance, port = server_info
    base_url = f"http://127.0.0.1:{port}"
    results = []
    mem_samples = []
    errors = 0
    stop = threading.Event()
    lock = threading.Lock()

    def worker(worker_id, scenario):
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                if scenario == "A":
                    q = f"这是测试问题{worker_id}_{time.time()}"
                    r = http_post(f"{base_url}/api/v1/chat/completions",
                                  {"question": q}, timeout=30)
                else:
                    if worker_id < 5:
                        q = f"混合问答{worker_id}_{time.time()}"
                        r = http_post(f"{base_url}/api/v1/chat/completions",
                                      {"question": q}, timeout=30)
                    else:
                        r = http_post(f"{base_url}/api/v1/upload",
                                      {"file": f"test_{worker_id}.pdf"}, timeout=30)
                elapsed = time.perf_counter() - t0
                with lock:
                    if "error" in r:
                        errors += 1
                    results.append({
                        "worker": worker_id, "elapsed": round(elapsed, 4),
                        "status": "error" if "error" in r else "ok",
                    })
            except Exception as e:
                elapsed = time.perf_counter() - t0
                with lock:
                    errors += 1
                    results.append({"worker": worker_id, "elapsed": round(elapsed, 4),
                                    "status": "error", "error": str(e)})
            # 模拟用户思考时间，避免CPU 100%空转
            time.sleep(0.1)

    # 内存监控线程
    def mem_monitor():
        while not stop.is_set():
            mem_samples.append({
                "time": round(time.perf_counter(), 2),
                "mem_mb": round(instance.mem_usage_mb(), 3),
                "requests": instance.request_count,
            })
            time.sleep(1)

    threads = [threading.Thread(target=worker, args=(i, scenario), daemon=True) for i in range(concurrency)]
    mem_thread = threading.Thread(target=mem_monitor, daemon=True)

    print(f"  启动 {concurrency} 并发, 场景{scenario}, 持续 {duration_sec}s")
    for t in threads: t.start()
    mem_thread.start()

    time.sleep(duration_sec)
    stop.set()
    for t in threads: t.join(timeout=2)
    mem_thread.join(timeout=2)

    # 统计
    latencies = [r["elapsed"] for r in results if r["status"] == "ok"]
    if latencies:
        sl = sorted(latencies)
        p50 = statistics.median(sl)
        p95 = sl[int(len(sl) * 0.95)] if len(sl) >= 20 else sl[-1]
        p99 = sl[int(len(sl) * 0.99)] if len(sl) >= 100 else sl[-1]
        avg = statistics.mean(latencies)
    else:
        p50 = p95 = p99 = avg = 0

    throughput = len(latencies) / duration_sec if duration_sec > 0 else 0
    if mem_samples:
        mem_start = mem_samples[0]["mem_mb"]
        mem_end = mem_samples[-1]["mem_mb"]
        mem_growth_pct = ((mem_end - mem_start) / mem_start * 100) if mem_start > 0.001 else (9999.0 if mem_end > 0.001 else 0.0)
    else:
        mem_start = mem_end = 0
        mem_growth_pct = 0

    summary = {
        "scenario": scenario,
        "concurrency": concurrency,
        "duration_sec": duration_sec,
        "total_requests": len(results),
        "successful": len(latencies),
        "errors": errors,
        "error_rate": round(errors / len(results) * 100, 2) if results else 0,
        "latency_avg": round(avg, 4),
        "latency_p50": round(p50, 4),
        "latency_p95": round(p95, 4),
        "latency_p99": round(p99, 4),
        "throughput_rps": round(throughput, 2),
        "mem_start_mb": round(mem_start, 3),
        "mem_end_mb": round(mem_end, 3),
        "mem_growth_pct": round(mem_growth_pct, 2),
        "mem_samples": mem_samples,
    }
    print(f"  完成: {len(latencies)}成功, {errors}错误, P95={p95:.3f}s, "
          f"吞吐={throughput:.1f}rps, 内存增长={mem_growth_pct:.1f}%")
    return summary


def main():
    results = {}
    for server_type in ["buggy", "optimized"]:
        print(f"\n===== {server_type} 服务器 =====")
        server_info = run_server(server_type, port=0)
        server, instance, port = server_info
        time.sleep(0.5)  # 等待启动

        results[server_type] = {}
        # 场景A: 20并发问答, 15秒
        print("场景A: 20并发问答")
        results[server_type]["scenario_a"] = run_scenario(server_info, "A", duration_sec=15, concurrency=20)
        time.sleep(1)
        # 场景B: 10并发混合负载, 15秒
        print("场景B: 10并发混合负载")
        results[server_type]["scenario_b"] = run_scenario(server_info, "B", duration_sec=15, concurrency=10)

        server.shutdown()
        time.sleep(0.5)

    out_file = OUT / "benchmark_results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已保存: {out_file}")

    # 打印对比
    print("\n===== 基线 vs 优化 对比 =====")
    for scenario in ["scenario_a", "scenario_b"]:
        b = results["buggy"][scenario]
        o = results["optimized"][scenario]
        print(f"\n{scenario}:")
        print(f"  P95:    {b['latency_p95']}s -> {o['latency_p95']}s")
        print(f"  吞吐:   {b['throughput_rps']} -> {o['throughput_rps']} rps")
        print(f"  错误率: {b['error_rate']}% -> {o['error_rate']}%")
        print(f"  内存增长: {b['mem_growth_pct']}% -> {o['mem_growth_pct']}%")


if __name__ == "__main__":
    main()
