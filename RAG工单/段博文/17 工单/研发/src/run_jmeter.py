# -*- coding: utf-8 -*-
# 工单17：JMeter压测运行脚本
"""
本地启动API服务，用JMeter CLI运行4轮压测，收集结果。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEV = Path(__file__).resolve().parent
LOGS = DEV.parent / "logs"
LOGS.mkdir(exist_ok=True)
JMETER = r"D:\测试工具\apache-jmeter-5.6.3\bin\jmeter.bat"
JMX_A = DEV / "scenario_a.jmx"
JMX_B = DEV / "scenario_b.jmx"

sys.path.insert(0, str(DEV))
from api_server import run_server


def wait_health(port, timeout=10):
    for _ in range(timeout * 10):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status == 200:
                return True
        except:
            time.sleep(0.1)
    return False


def run_jmeter(port, scenario, threads, duration, label):
    """运行JMeter CLI压测，返回结果摘要。"""
    csv_file = LOGS / f"jmeter_{label}.csv"
    if csv_file.exists():
        csv_file.unlink()

    cmd = [
        JMETER, "-n", "-t", str(JMX_A if scenario == "A" else JMX_B),
        f"-Jport={port}", f"-Jthreads={threads}", f"-Jduration={duration}",
        f"-Jcsvfile={csv_file}",
        "-Jhost=127.0.0.1",
    ]
    print(f"  JMeter: port={port} scenario={scenario} threads={threads} duration={duration}s")
    result = subprocess.run(cmd, capture_output=True, timeout=duration + 120, encoding="gbk", errors="replace")

    # 解析JMeter输出
    output = (result.stdout or "") + (result.stderr or "")
    # 提取关键指标
    summary = {"label": label, "scenario": scenario, "threads": threads, "duration": duration}

    # 从输出中解析
    for line in output.split("\n"):
        if "Average" in line or "average" in line.lower():
            summary["raw_avg_line"] = line.strip()
        if "90%" in line or "95%" in line or "99%" in line:
            summary["raw_pct_line"] = line.strip()
        if "summary" in line.lower() or "Samples" in line:
            summary.setdefault("raw_lines", []).append(line.strip())

    # 从CSV解析
    if csv_file.exists():
        import csv as csvmod
        rows = list(csvmod.reader(csv_file.open(encoding="utf-8")))
        if len(rows) > 1:
            headers = rows[0]
            data = rows[1:]
            # JMeter CSV字段: timeStamp,elapsed,responseCode,responseMessage,threadId, etc.
            elapsed_idx = headers.index("elapsed") if "elapsed" in headers else 1
            rc_idx = headers.index("responseCode") if "responseCode" in headers else 3
            latencies = [int(r[elapsed_idx]) for r in data if r and r[elapsed_idx].isdigit()]
            errors = sum(1 for r in data if r and r[rc_idx] != "200")
            if latencies:
                sl = sorted(latencies)
                summary["count"] = len(latencies)
                summary["avg_ms"] = round(sum(latencies) / len(latencies), 1)
                summary["p50_ms"] = sl[len(sl) // 2]
                summary["p95_ms"] = sl[int(len(sl) * 0.95)]
                summary["p99_ms"] = sl[int(len(sl) * 0.99)]
                summary["throughput_rps"] = round(len(latencies) / duration, 2)
                summary["error_count"] = errors
                summary["error_rate"] = round(errors / len(latencies) * 100, 2)
    summary["jmeter_output"] = output[-500:]  # 最后500字符
    print(f"    {summary.get('count', 0)} samples, avg={summary.get('avg_ms', '?')}ms, "
          f"p95={summary.get('p95_ms', '?')}ms, errors={summary.get('error_count', 0)}")
    return summary


def main():
    results = {}

    for server_type in ["buggy", "optimized"]:
        print(f"\n===== {server_type} 服务器 =====")
        server_info = run_server(server_type, port=0)
        server, instance, port = server_info
        time.sleep(0.5)
        if not wait_health(port, 10):
            print(f"  服务启动失败!")
            continue

        # 获取初始内存
        try:
            health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
            mem_start = health["mem_mb"]
        except:
            mem_start = 0

        results[server_type] = {}
        # 场景A: 20并发问答 15秒
        print("场景A: 20并发问答")
        sa = run_jmeter(port, "A", 20, 15, f"{server_type}_a")
        # 获取压测后内存
        try:
            health2 = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
            mem_end = health2["mem_mb"]
            mem_growth = ((mem_end - mem_start) / mem_start * 100) if mem_start > 0.001 else 0
        except:
            mem_end = mem_start = 0
            mem_growth = 0
        sa["mem_start_mb"] = round(mem_start, 3)
        sa["mem_end_mb"] = round(mem_end, 3)
        sa["mem_growth_pct"] = round(mem_growth, 2)
        results[server_type]["scenario_a"] = sa
        time.sleep(1)

        # 场景B: 10并发混合 15秒
        print("场景B: 10并发混合负载")
        sb = run_jmeter(port, "B", 10, 15, f"{server_type}_b")
        try:
            health3 = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
            sb_mem_end = health3["mem_mb"]
            sb_mem_start = mem_end
            sb_growth = ((sb_mem_end - sb_mem_start) / sb_mem_start * 100) if sb_mem_start > 0.001 else 0
        except:
            sb_mem_end = sb_mem_start = 0
            sb_growth = 0
        sb["mem_start_mb"] = round(sb_mem_start, 3)
        sb["mem_end_mb"] = round(sb_mem_end, 3)
        sb["mem_growth_pct"] = round(sb_growth, 2)
        results[server_type]["scenario_b"] = sb

        server.shutdown()
        time.sleep(0.5)

    out_file = LOGS / "jmeter_results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已保存: {out_file}")

    # 打印对比
    print("\n===== JMeter 基线 vs 优化 对比 =====")
    for sc in ["scenario_a", "scenario_b"]:
        b = results.get("buggy", {}).get(sc, {})
        o = results.get("optimized", {}).get(sc, {})
        print(f"\n{sc}:")
        print(f"  P95:    {b.get('p95_ms', '?')}ms -> {o.get('p95_ms', '?')}ms")
        print(f"  吞吐:   {b.get('throughput_rps', '?')} -> {o.get('throughput_rps', '?')} rps")
        print(f"  错误率: {b.get('error_rate', '?')}% -> {o.get('error_rate', '?')}%")
        print(f"  内存增长: {b.get('mem_growth_pct', '?')}% -> {o.get('mem_growth_pct', '?')}%")


if __name__ == "__main__":
    main()
