# -*- coding: utf-8 -*-
"""异步压测器：测 QPS / 延迟分位 / 错误率。

为什么自己写而不是用 Jmeter/locust
----------------------------------
本机没有装任何压测工具（`jmeter`/`ab`/`wrk`/`locust`/`hey` 全部不存在），
而按项目的「本地优先、下载需授权」原则，不该为了一次压测去装新东西。
`httpx` + `asyncio` 已足够做并发压测，且能直接复用项目的鉴权流程。

测什么
------
    --scenario health   只打 /api/health（轻，测 Web 层吞吐上限）
    --scenario search   打 /api/search（中，测检索链路）
    --scenario chat     打 /api/chat/completions（重，含 LLM 生成 —— 很慢）

**压测结果必须连同「打了什么」一起看**：`health` 的 QPS 反映 Web 层能力，
`chat` 的 QPS 受限于 LLM 生成，两者差几个数量级，混着报会得出荒谬结论。

用法
----
    python scripts/loadtest.py --scenario health --concurrency 20 --duration 15
    python scripts/loadtest.py --scenario search --concurrency 8  --duration 20
    python scripts/loadtest.py --scenario chat   --concurrency 4  --duration 60
    python scripts/loadtest.py --scenario health --url http://127.0.0.1:8080   # 经 nginx
"""
import argparse
import asyncio
import statistics
import sys
import secrets
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def _setup(base: str) -> tuple[dict, int | None]:
    """注册/登录一个压测用户，返回 (headers, conversation_id)。"""
    async with httpx.AsyncClient(timeout=120) as c:
        u = f"load_{int(time.time())}"
        pw = secrets.token_hex(8)   # 临时用户密码每次随机生成，不写死
        await c.post(f"{base}/api/auth/register",
                     json={"username": u, "password": pw})
        r = await c.post(f"{base}/api/auth/login",
                         json={"username": u, "password": pw})
        h = {"Authorization": f"Bearer {r.json()['access_token']}"}
        chars = (await c.get(f"{base}/api/characters", headers=h)).json()
        doctor = next((x for x in chars if x["slug"] == "doctor"), chars[0])
        conv = await c.post(f"{base}/api/conversations", headers=h,
                            json={"character_id": doctor["id"]})
        return h, conv.json().get("id")


def _request_spec(scenario: str, headers: dict, conv_id: int | None):
    """返回 (method, path, json_body, timeout)。"""
    if scenario == "light":
        # 真正轻量的端点：FastAPI 自动生成的 schema，不碰任何外部依赖。
        # 用途是测**Web 层吞吐**（以及验证 nginx 是否真的在分发）。
        # 注意别拿 /api/health 当轻量端点 —— 它逐个探活 MySQL/Redis/Milvus/Ollama，
        # 实测每次约 2 秒（Ollama 那次调用占大头），QPS 只有个位数。
        return "GET", "/openapi.json", None, 30
    if scenario == "health":
        return "GET", "/api/health", None, 30
    if scenario == "search":
        return "POST", "/api/search", {
            "query": "高血压的诊断标准是什么", "collection": "kb_medical",
            "recall_k": 20, "top_k": 5, "use_rerank": True,
        }, 120
    if scenario == "chat":
        return "POST", "/api/chat/completions", {
            "conversation_id": conv_id, "question": "高血压的诊断标准是什么？",
        }, 300
    raise ValueError(scenario)


async def worker(base: str, headers: dict, conv_id: int | None, scenario: str,
                 deadline: float, results: list) -> None:
    method, path, body, timeout = _request_spec(scenario, headers, conv_id)
    async with httpx.AsyncClient(timeout=timeout) as c:
        while time.time() < deadline:
            t0 = time.time()
            err = None
            try:
                if method == "GET":
                    r = await c.get(f"{base}{path}", headers=headers)
                else:
                    r = await c.post(f"{base}{path}", headers=headers, json=body)
                if r.status_code >= 400:
                    err = f"HTTP {r.status_code}"
                # chat 场景要确认真有回答（否则 200 可能是空响应）
                if scenario == "chat" and not r.json().get("answer"):
                    err = "空回答"
            except Exception as e:
                err = type(e).__name__
            results.append({"lat": time.time() - t0, "err": err})


async def run(base: str, scenario: str, concurrency: int, duration: int,
              warmup: int = 3) -> dict:
    headers, conv_id = await _setup(base)

    # 预热：首请求含模型加载/连接建立，计入会严重拉偏分位数
    print(f"预热 {warmup}s ...")
    warm_end = time.time() + warmup
    await asyncio.gather(*[
        worker(base, headers, conv_id, scenario, warm_end, [])
        for _ in range(min(2, concurrency))
    ])

    print(f"压测 {duration}s | 并发 {concurrency} | 场景 {scenario}")
    results: list = []
    t0 = time.time()
    await asyncio.gather(*[
        worker(base, headers, conv_id, scenario, time.time() + duration, results)
        for _ in range(concurrency)
    ])
    elapsed = time.time() - t0

    oks = [r for r in results if not r["err"]]
    errs = [r for r in results if r["err"]]
    lat = sorted(r["lat"] * 1000 for r in oks)

    def pct(p):
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, int(len(lat) * p))]

    err_kinds = {}
    for r in errs:
        err_kinds[r["err"]] = err_kinds.get(r["err"], 0) + 1

    return {
        "scenario": scenario, "concurrency": concurrency, "duration": round(elapsed, 1),
        "total": len(results), "ok": len(oks), "err": len(errs),
        "qps": round(len(results) / elapsed, 2) if elapsed else 0,
        "qps_ok": round(len(oks) / elapsed, 2) if elapsed else 0,
        "p50": round(pct(0.50), 1), "p95": round(pct(0.95), 1),
        "p99": round(pct(0.99), 1),
        "mean": round(statistics.mean(lat), 1) if lat else 0.0,
        "err_kinds": err_kinds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="后端地址（可指向 nginx）")
    ap.add_argument("--scenario", choices=["light", "health", "search", "chat"],
                    default="light",
                    help="light=只测 Web 层（/openapi.json）；health=依赖探活（很重，每次约2秒）；"
                         "search=检索链路；chat=含 LLM 生成")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--json", default="", help="把结果追加写入该 JSON 文件（供汇总对比）")
    args = ap.parse_args()

    r = asyncio.run(run(args.url, args.scenario, args.concurrency, args.duration))

    print()
    print("=" * 64)
    print(f"结果 | {r['scenario']} | 并发 {r['concurrency']} | 用时 {r['duration']}s")
    print("=" * 64)
    print(f"  总请求 {r['total']}　成功 {r['ok']}　失败 {r['err']}")
    print(f"  QPS（总）  {r['qps']}")
    print(f"  QPS（成功）{r['qps_ok']}")
    print(f"  延迟 ms    p50={r['p50']}  p95={r['p95']}  p99={r['p99']}  均值={r['mean']}")
    if r["err_kinds"]:
        print(f"  失败类型   {r['err_kinds']}")
    print()
    print("⚠️ 不同 scenario 的 QPS 不可直接比较：health 测 Web 层，")
    print("   chat 受限于 LLM 生成，两者相差数量级。")

    if args.json:
        import json
        p = Path(args.json)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        data.append({**r, "url": args.url})
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已追加到 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
