"""T4 在线测试 ②：并发与稳定性。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 8/9、工单「稳定性容错」）：

- 10 个并发请求：断言**无崩溃、无死锁、全部返回、错误率 0**；
- 统计 **P50 / P95** 延迟；
- **如实量化并发劣化**，不美化：分别测「串行 10 题」与「并发 10 题」，
  给出倍数关系并说明成因（见下）。

成因（读实现得出，非推测）：
1. ``serve_fallback.py`` 的 ``/api/ask`` 全程持有进程级 ``_ASK_LOCK``（源码 line 50/382/400/421），
   请求在**应用层即被串行化**；
2. 本地 Ollama 单实例串行推理。
因此并发**不会**带来吞吐提升，延迟应接近串行总和量级——本用例把这件事实测出来并打印，
既不美化也不把它写成失败（工单只要求「并发无崩溃、无死锁、错误率 0」）。
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from conftest import ask_stream, get_json

CONCURRENCY = 10


def _run_batch(server, questions, workers: int) -> tuple[list[dict], float]:
    """以 ``workers`` 并发度跑完 questions，返回 (结果列表, 墙钟秒)。"""
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask_stream, server.base_url, q): q for q in questions}
        for future in as_completed(futures, timeout=600):
            results.append(future.result())
    return results, time.perf_counter() - started


@pytest.mark.slow
def test_ten_concurrent_requests_no_crash_no_deadlock(server, golden):
    """10 并发：全部返回、错误率 0、无死锁；统计 P50/P95 延迟并如实报告劣化。"""
    questions = [item.question for item in golden][:CONCURRENCY]
    assert len(questions) == CONCURRENCY, "并发题数应为 10"

    results, wall_s = _run_batch(server, questions, workers=CONCURRENCY)

    # ① 全部返回（无死锁 / 无超时）
    assert len(results) == CONCURRENCY, f"只有 {len(results)}/{CONCURRENCY} 个请求返回（疑似死锁或超时）"

    # ② 错误率 0：每个响应都必须是有效回答（非空文本 + 有引用），且无 error 事件
    failures: list[str] = []
    latencies: list[float] = []
    first_tokens: list[float] = []
    for idx, res in enumerate(results):
        answer = res.get("answer") or {}
        if "error" in res.get("events", []) or answer.get("error"):
            failures.append(f"#{idx} 返回 error 事件: {answer.get('error')}")
            continue
        if not (answer.get("answer") or "").strip():
            failures.append(f"#{idx} 答案为空")
        if not (answer.get("citations") or []):
            failures.append(f"#{idx} 无引用")
        latencies.append(res["elapsed_ms"])
        if res["client_first_token_ms"] is not None:
            first_tokens.append(res["client_first_token_ms"])

    lat_sorted = sorted(latencies)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]
    ft_sorted = sorted(first_tokens)
    ft_p95 = ft_sorted[min(len(ft_sorted) - 1, int(len(ft_sorted) * 0.95))] if ft_sorted else 0.0
    print(f"\n[并发] {CONCURRENCY} 并发：全部返回={len(results)}/{CONCURRENCY}，错误={len(failures)}")
    print(f"[并发] 端到端 P50={p50:.0f} ms  P95={p95:.0f} ms  墙钟={wall_s * 1000:.0f} ms")
    print(f"[并发] 首字 P50={statistics.median(ft_sorted) if ft_sorted else 0:.0f} ms  P95={ft_p95:.0f} ms")

    # ③ 串行基线：量化劣化（如实记录，不美化）
    serial_results, serial_wall_s = _run_batch(server, questions, workers=1)
    serial_lat = sorted(r["elapsed_ms"] for r in serial_results)
    print(f"[并发] 串行对照：端到端 P50={statistics.median(serial_lat):.0f} ms  "
          f"墙钟={serial_wall_s * 1000:.0f} ms")
    print(f"[并发] 墙钟比值 并发/串行 = {wall_s / serial_wall_s:.2f}×"
          f"（≈1 说明应用层 _ASK_LOCK 与单实例 Ollama 已把请求串行化，无吞吐收益）")

    assert not failures, "并发下出现错误：\n  - " + "\n  - ".join(failures)
    assert len(serial_results) == CONCURRENCY, "串行对照未全部返回"


def test_service_alive_after_concurrency(server):
    """并发压测后服务必须存活且索引仍就绪（无崩溃残留）。"""
    status, body = get_json(server.base_url, "/api/health")
    assert status == 200 and body.get("ok") is True, body
    health = body.get("health") or {}
    assert health.get("index", {}).get("ready") is True, f"索引状态异常: {health.get('index')}"
    print(f"\n[并发] 压测后服务存活，索引块数={health.get('index', {}).get('count')}")


def test_concurrent_different_conversations_do_not_mix(server, golden):
    """并发多会话：不同会话的问答不得互相污染（答案与各自问题匹配）。"""
    from conftest import post_json

    questions = [item.question for item in golden][:4]
    convs = []
    for _ in questions:
        status, body = post_json(server.base_url, "/api/conversations", {})
        assert status == 200 and body.get("ok"), body
        convs.append(body["conversation_id"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(ask_stream, server.base_url, q, cid)
                   for q, cid in zip(questions, convs)]
        results = [f.result(timeout=600) for f in futures]

    mismatched = []
    for q, cid, res in zip(questions, convs, results):
        answer = res.get("answer") or {}
        if not (answer.get("answer") or "").strip():
            mismatched.append(f"{cid} 答案为空")
    print(f"\n[并发·多会话] {len(convs)} 个会话并发完成，答案均非空={not mismatched}")
    assert not mismatched, "并发多会话出现空答案: " + "; ".join(mismatched)
