# -*- coding: utf-8 -*-
"""
工单 17 · Python 压测脚本（JMeter 的轻量替代，无需 JVM）
支持工单两场景：
  场景 A：N 并发高频问答（已入库文档的简单问题）
  场景 B：混合负载（一半问答、一半上传+解析 PDF）

用法：
  python stress_test.py --self-test
  python stress_test.py --scenario A --users 20 --duration 600 \
      --url http://<host>/api/v1/chats_openai/<chat_id>/chat/completions \
      --api-key <key> [--questions questions.txt] [--out result_A.json]
  python stress_test.py --scenario B --users 10 --duration 600 \
      --url http://<host>/api/v1/chats_openai/<chat_id>/chat/completions \
      --upload-url http://<host>/api/v1/datasets/<ds_id>/documents \
      --pdf-dir D:\_RAG新工单素材\original_problems\documents
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import List

try:
    import aiohttp
except Exception:  # 允许 --self-test 在无 aiohttp 时运行
    aiohttp = None

DEFAULT_QUESTIONS = [
    "根据文本信息，该静电除尘器的发明人是：",
    "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
]


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class Stats:
    def __init__(self):
        self.lat: List[float] = []
        self.ok = 0
        self.err = 0

    def add(self, ms: float, ok: bool):
        self.lat.append(ms)
        self.ok += int(ok)
        self.err += int(not ok)

    def report(self, name: str) -> dict:
        return {
            "scenario": name,
            "total": self.ok + self.err,
            "ok": self.ok,
            "err": self.err,
            "err_rate": round(self.err / max(1, self.ok + self.err), 4),
            "avg_ms": round(statistics.mean(self.lat), 1) if self.lat else 0,
            "p50_ms": round(percentile(self.lat, 0.50), 1),
            "p95_ms": round(percentile(self.lat, 0.95), 1),
            "p99_ms": round(percentile(self.lat, 0.99), 1),
            "max_ms": round(max(self.lat), 1) if self.lat else 0,
        }


async def one_qa(session, url, api_key, question, stats: Stats):
    body = {"question": question, "stream": False}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    ok = False
    try:
        async with session.post(url, json=body, headers=headers, timeout=60) as r:
            await r.read()
            ok = r.status == 200
    except Exception:
        ok = False
    stats.add((time.perf_counter() - t0) * 1000, ok)


async def upload_pdf(session, upload_url, api_key, pdf_path, stats: Stats):
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    ok = False
    try:
        data = aiohttp.FormData()
        with open(pdf_path, "rb") as f:
            data.add_field("file", f, filename=pdf_path.split("\\")[-1])
        async with session.post(upload_url, data=data, headers=headers, timeout=120) as r:
            await r.read()
            ok = r.status in (200, 201)
    except Exception:
        ok = False
    stats.add((time.perf_counter() - t0) * 1000, ok)


async def worker_a(session, args, questions, stats, deadline):
    while time.time() < deadline:
        await one_qa(session, args.url, args.api_key, random.choice(questions), stats)
        await asyncio.sleep(args.think)


async def worker_b_qa(session, args, questions, stats, deadline):
    await worker_a(session, args, questions, stats, deadline)


async def worker_b_up(session, args, pdfs, stats, deadline):
    while time.time() < deadline:
        await upload_pdf(session, args.upload_url, args.api_key, random.choice(pdfs), stats)
        await asyncio.sleep(args.think)


async def run(args, questions, stats_a, stats_up):
    deadline = time.time() + args.duration
    conn = aiohttp.TCPConnector(limit=args.users * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        if args.scenario == "A":
            for _ in range(args.users):
                tasks.append(worker_a(session, args, questions, stats_a, deadline))
        else:  # B 混合负载
            half = max(1, args.users // 2)
            pdfs = args.pdf_list
            for _ in range(half):
                tasks.append(worker_b_qa(session, args, questions, stats_a, deadline))
            for _ in range(args.users - half):
                tasks.append(worker_b_up(session, args, pdfs, stats_up, deadline))
        await asyncio.gather(*tasks)


def self_test():
    import random as _r
    st = Stats()
    for _ in range(200):
        st.add(_r.gauss(2500, 700), True)
    print(json.dumps(st.report("A(self-test)"), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B"], default="A")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--duration", type=int, default=600, help="秒")
    ap.add_argument("--url")
    ap.add_argument("--upload-url")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--questions")
    ap.add_argument("--pdf-dir")
    ap.add_argument("--think", type=float, default=0.0, help="每次请求间隔秒")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if aiohttp is None:
        raise SystemExit("需安装 aiohttp: pip install aiohttp")

    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = [l.strip() for l in open(args.questions, encoding="utf-8") if l.strip()]

    import glob, os
    args.pdf_list = []
    if args.scenario == "B":
        assert args.upload_url and args.pdf_dir, "场景B需 --upload-url 与 --pdf-dir"
        args.pdf_list = glob.glob(os.path.join(args.pdf_dir, "*.pdf"))[:20]
        assert args.pdf_list, "未找到 PDF"

    stats_a, stats_up = Stats(), Stats()
    asyncio.run(run(args, questions, stats_a, stats_up))
    rep = {"qa": stats_a.report("A" if args.scenario == "A" else "B-问答")}
    if args.scenario == "B":
        rep["upload_parse"] = stats_up.report("B-上传解析")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        json.dump(rep, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
