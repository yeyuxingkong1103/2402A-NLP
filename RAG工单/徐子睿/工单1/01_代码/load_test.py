# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：load_test —— 并发 / 高可用压测
# 说明：对运行中的服务做分级并发压测，统计吞吐、平均/尾延迟、错误率与稳定性。
#       只依赖标准库，便于在任何机器复现。
# 用法：python evaluation/load_test.py --base http://127.0.0.1:8100 --levels 1,2,4,8 --repeat 5
#       （先启动服务：python -m uvicorn app.server:app --port 8100）
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "武汉兴图新科电子股份有限公司的注册资本是多少？",
    "武汉兴图新科电子股份有限公司的法定代表人是谁？",
    "电子信息行业的上游涉及哪些企业？",
    "募集资金中多少用于补充流动资金？",
    "公司在哪个领域已经成为重要供应商？",
]


def post(base, path, payload, timeout):
    """返回 (耗时秒, 状态码, 响应体文本)；异常记 -1。"""
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
        return time.time() - t0, r.status, body
    except urllib.error.HTTPError as e:
        return time.time() - t0, e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return time.time() - t0, -1, repr(e)


def run_level(base, path, level, tasks, timeout):
    """并发 level 个 worker，跑完 tasks 个请求。"""
    qs = (QUESTIONS * ((tasks // len(QUESTIONS)) + 1))[:tasks]
    lat, codes = [], {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=level) as ex:
        for dt, code, _body in ex.map(lambda q: post(base, path, {"query": q}, timeout), qs):
            lat.append(dt)
            codes[code] = codes.get(code, 0) + 1
    wall = time.time() - t0
    lat_sorted = sorted(lat)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    return {
        "concurrency": level, "requests": len(qs), "wall_s": round(wall, 2),
        "qps": round(len(qs) / wall, 2),
        "avg_ms": round(1000 * statistics.mean(lat)),
        "p95_ms": round(1000 * p95),
        "min_ms": round(1000 * min(lat)), "max_ms": round(1000 * max(lat)),
        "errors": sum(v for k, v in codes.items() if k != 200),
        "codes": codes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--path", default="/api/ask", help="/api/ask 或 /api/search")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--repeat", type=int, default=5, help="每级并发发送的请求数")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    here = os.path.dirname(os.path.abspath(__file__))

    # 健康检查
    try:
        with urllib.request.urlopen(a.base + "/api/health", timeout=10) as r:
            print("health:", r.read().decode("utf-8", "replace")[:200])
    except Exception as e:  # noqa: BLE001
        print("服务不可用：", e)
        sys.exit(2)

    rows = []
    for lv in levels:
        r = run_level(a.base, a.path, lv, a.repeat, a.timeout)
        rows.append(r)
        print("并发=%-3d 请求=%-3d 墙钟=%6.2fs QPS=%5.2f 平均=%6dms p95=%6dms 错误=%d"
              % (r["concurrency"], r["requests"], r["wall_s"], r["qps"],
                 r["avg_ms"], r["p95_ms"], r["errors"]))

    out = a.out or os.path.join(here, "load_test_result.json")
    json.dump({"base": a.base, "path": a.path, "results": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
