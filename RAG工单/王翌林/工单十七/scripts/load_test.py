# -*- coding: utf-8 -*-
"""
自包含压测脚本：对指定 RAG API 进行并发压测，记录延迟分位、吞吐、错误率、内存增长。
用法：
  PYTHONPATH=. python scripts/load_test.py --mode optimized --port 8017 --out loadtest/result_opt.json
  PYTHONPATH=. python scripts/load_test.py --mode baseline  --port 8018 --out loadtest/result_base.json
"""
import os, sys, time, json, argparse, threading, subprocess, urllib.request, random

QUESTIONS = [
    "什么是淬火工艺？", "公差与配合的作用？", "链条传动适合什么场合？",
    "带式输送机由哪些部件组成？", "轧制带材时为什么要调节张力？", "水洗洗涤塔的作用？",
]


def http_post(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return time.time() - t0, r.status, None
    except Exception as e:
        return time.time() - t0, 0, str(e)


def worker(url, results, stop_event, errors):
    while not stop_event.is_set():
        # 唯一化问题，避免语义缓存命中，公平测量全链路（检索+生成）性能
        q = random.choice(QUESTIONS) + " #" + str(random.randint(0, 10 ** 9))
        lat, status, err = http_post(url, {"question": q, "chat_id": "c1"})
        results.append(lat)
        if status != 200:
            errors.append(status)


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run(mode, port, out, concurrency=20, duration=60):
    base = f"http://127.0.0.1:{port}"
    # 启动服务
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    app = "src.api:app" if mode == "optimized" else "src.api_baseline:app"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port),
         "--workers", "1", "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # 等待就绪
        for _ in range(40):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read(); break
            except Exception:
                time.sleep(0.5)
        time.sleep(2)
        # 初始内存
        mem0 = _rss(base)
        results, errors = [], []
        stop = threading.Event()
        threads = [threading.Thread(target=worker, args=(base + "/api/v1/chats_openai/c1/chat/completions",
                                                         results, stop, errors)) for _ in range(concurrency)]
        mem_samples = []
        t_start = time.time()
        for t in threads: t.start()
        # 采样内存
        for _ in range(duration):
            time.sleep(1)
            mem_samples.append(_rss(base))
        stop.set()
        for t in threads: t.join()
        elapsed = time.time() - t_start
        mem1 = _rss(base)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    n = len(results)
    summary = {
        "mode": mode, "concurrency": concurrency, "duration_s": round(elapsed, 1),
        "requests": n, "throughput_rps": round(n / elapsed, 2),
        "avg_ms": round(1000 * sum(results) / n, 1) if n else 0,
        "p50_ms": round(1000 * percentile(results, 50), 1),
        "p95_ms": round(1000 * percentile(results, 95), 1),
        "p99_ms": round(1000 * percentile(results, 99), 1),
        "errors": len(errors),
        "rss_start_mb": mem0, "rss_end_mb": mem1,
        "rss_growth_mb": round(mem1 - mem0, 1),
        "rss_growth_pct": round((mem1 - mem0) / max(1, mem0) * 100, 1),
        "mem_samples_mb": mem_samples,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "mem_samples_mb"}, ensure_ascii=False, indent=2))
    return summary


def _rss(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            return json.loads(r.read()).get("rss_mb", 0)
    except Exception:
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["optimized", "baseline"], default="optimized")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--out", default="loadtest/result.json")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--duration", type=int, default=60)
    a = ap.parse_args()
    run(a.mode, a.port, a.out, a.concurrency, a.duration)
