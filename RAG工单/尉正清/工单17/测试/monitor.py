# 工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏
"""压测期间采集容器/GPU/队列的资源曲线

工单任务二要求「用 nvidia-smi 和 docker stats 持续监控 GPU 与容器内存变化」，
并「检查 redis-cli 的 INFO stats 观察任务队列堆积」。这个脚本把三样一起采，
每 N 秒一行落到 JSONL，方便后面画图和算「12 小时 RSS 增长」。

用法：
    python monitor.py --out results/mem_baseline.jsonl --interval 5
    （Ctrl-C 停，或 --duration 600 自动停）
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONTAINERS = ["docker-ragflow-cpu-1", "docker-es01-1", "docker-mysql-1", "docker-redis-1"]


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace").stdout
    except Exception as e:
        return f"__ERR__ {e}"


def docker_mem():
    """docker stats 的容器内存（MiB）。"""
    out = sh("docker stats --no-stream --format \"{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}\"")
    got = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        name, mem, cpu = parts[0], parts[1], parts[2]
        # MemUsage 形如 "512.3MiB / 1GiB"
        try:
            m = mem.split("/")[0].strip()
            val = float(m.replace("MiB", "").replace("GiB", "e3").replace("KiB", "e-3")
                        .replace("B", "").strip())
        except Exception:
            val = None
        got[name] = {"mem_mib": val, "cpu": cpu}
    return got


def gpu():
    out = sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits")
    try:
        used, util = out.strip().splitlines()[0].split(",")
        return {"gpu_mem_mib": float(used), "gpu_util": float(util)}
    except Exception:
        return {}


def redis_queue():
    """RAGFlow 的任务队列长度（Redis Stream）。堆积说明 Task Executor 跟不上。"""
    out = sh("docker exec docker-redis-1 redis-cli -a infini_rag_flow --no-auth-warning "
             "XLEN rag_flow_svr_queue")
    try:
        return {"queue_len": int(out.strip())}
    except Exception:
        return {"queue_len": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--duration", type=float, default=0, help="秒；0 = 跑到手动停")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    print(f"采样中 -> {out}（每 {args.interval}s，Ctrl-C 停）", flush=True)
    with out.open("w", encoding="utf-8") as f:
        try:
            while True:
                rec = {"t": round(time.time() - t0, 1), **gpu(), **redis_queue()}
                rec["containers"] = docker_mem()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 6 == 1:
                    rag = rec["containers"].get("docker-ragflow-cpu-1", {}).get("mem_mib")
                    print(f"  t={rec['t']:6.0f}s  ragflow={rag}MiB  "
                          f"gpu={rec.get('gpu_mem_mib')}MiB  队列={rec.get('queue_len')}", flush=True)
                if args.duration and time.time() - t0 >= args.duration:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"采了 {n} 个点 -> {out}")


if __name__ == "__main__":
    main()
