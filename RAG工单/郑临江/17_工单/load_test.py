# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单
Python 版压测脚本：模拟并发请求，输出 P95 延迟与吞吐量。
"""
import concurrent.futures
import statistics
import time
import urllib.request
import json


def fire(query, i):
    req = urllib.request.Request(
        "http://localhost:8000/api/v1/chats_openai/1/chat/completions",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return time.perf_counter() - t0, None
    except Exception as e:
        return time.perf_counter() - t0, e


def run(concurrency, queries, rounds=50):
    latencies = []
    errors = 0
    for r in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(fire, queries[i % len(queries)], i) for i in range(concurrency)]
            for f in concurrent.futures.as_completed(futs):
                lat, err = f.result()
                latencies.append(lat)
                if err:
                    errors += 1
    s = sorted(latencies)
    p95 = s[int(len(s) * 0.95) - 1]
    print(f"并发={concurrency} 请求={len(latencies)} 失败={errors} "
          f"avg={statistics.mean(latencies)*1000:.1f}ms p95={p95*1000:.1f}ms")
    return p95


if __name__ == "__main__":
    queries = ["该静电除尘器的发明人是？", "本发明的分散装置包含哪个组件？"]
    print("场景A：20 并发高频问答")
    run(20, queries)
