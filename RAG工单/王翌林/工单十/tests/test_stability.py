# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_stability.py —— Step 9 长时间运行 / 稳定性测试

覆盖（要求 3）：
  1. 模拟连续 100 次问答请求，校验成功率、响应时间分位（p50/p95/p99/max）
  2. 采集进程 RSS 内存，断言无显著内存泄漏
  3. 混合异常请求（空输入 422）穿插压测，验证服务在非法输入下持续可用
  4. 结果落盘 data/test_runs/stability_step9.json，供 docs/04_测试报告.md 引用

策略：FastAPI TestClient（进程内 ASGI 调用，无网络抖动）+ 伪引擎模拟检索/生成耗时，
     不依赖真实 Milvus / MySQL / LLM，保证可重复、秒级完成；
     真实服务长跑表现见 docs/04_测试报告.md 中的 8002 联调记录。
"""
import json
import os
import random
import time
from unittest.mock import MagicMock

import psutil
import pytest
from fastapi.testclient import TestClient

from src.api import app

N_REQUESTS = 100                       # 工单二：模拟 100 次请求
RESULT_PATH = "data/test_runs/stability_step9.json"
P95_LIMIT_MS = 1000.0                  # 伪引擎本地调用，p95 应远低于 1s（留足 CI 余量）
MEM_GROWTH_LIMIT_MB = 100.0            # 100 次请求后 RSS 增长上限（无泄漏应 < 20MB）


class _FakeStableEngine:
    """模拟工单二优化引擎：稳定小延迟 + 5% 缓存命中 + 标准引用结构"""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.calls = 0

    def _payload(self, question):
        self.calls += 1
        time.sleep(self.rng.uniform(0.002, 0.012))  # 模拟检索+生成 2~12ms
        cached = self.calls % 20 == 0
        return {
            "mode": "rag_optimized",
            "answer": f"针对「{question}」的优化链路回答：营业收入详见引用页。",
            "references": [{"page": 12, "chunk_id": f"c_{self.calls:04d}",
                            "score": round(0.8 + self.rng.random() * 0.19, 4),
                            "preview": "报告期内营业收入分别为 7,955.46 万元……"}],
            "latency_ms": round(self.rng.uniform(20, 120), 1),
            "token_usage": {"total_tokens": 150},
            "breakdown": {"retrieve_ms": 30.0, "llm_ms": 60.0},
            "cache_hit": cached,
        }

    def ask_optimized(self, question, lang_override=None, **kw):
        return self._payload(question)

    def ask_rag(self, question):
        self.calls += 1
        time.sleep(0.005)
        return {"mode": "rag", "answer": "基线答案", "references": [],
                "latency_ms": 200.0, "token_usage": {}, "query_understanding": {},
                "breakdown": {"retrieve_ms": 100.0, "llm_ms": 100.0}}

    def ask_llm(self, question):
        self.calls += 1
        return {"mode": "pure_llm", "answer": "纯LLM", "references": [],
                "latency_ms": 50.0, "token_usage": {}}


def _percentile(sorted_vals, pct):
    """线性插值百分位（pct: 0~100）"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


@pytest.fixture
def stability_client(monkeypatch):
    """装配伪引擎的进程内客户端（不触发 lifespan、不写 MySQL）"""
    app.state.rag_engine = _FakeStableEngine()
    monkeypatch.setattr("src.api._log_qa", lambda *a, **k: None)
    yield TestClient(app)
    if hasattr(app.state, "rag_engine"):
        delattr(app.state, "rag_engine")


def test_100_requests_latency_and_memory(stability_client):
    """长稳主测试：100 次请求成功率 100%、p95 达标、内存无显著增长"""
    proc = psutil.Process(os.getpid())
    _ = proc.memory_info().rss  # 预热
    rss_before = proc.memory_info().rss

    latencies, cache_hits, errors = [], 0, []
    t_start = time.perf_counter()
    for i in range(N_REQUESTS):
        body = {"question": f"问题{i:03d}：报告期内公司营业收入分别是多少？", "top_k": 8}
        t0 = time.perf_counter()
        r = stability_client.post("/api/ask", json=body)
        cost_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            errors.append({"i": i, "status": r.status_code, "text": r.text[:200]})
        else:
            latencies.append(cost_ms)
            cache_hits += 1 if r.json().get("cache_hit") else 0
    total_s = time.perf_counter() - t_start
    rss_after = proc.memory_info().rss

    ordered = sorted(latencies)
    stats = {
        "workorder": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
        "total_requests": N_REQUESTS,
        "success": len(latencies),
        "failed": len(errors),
        "success_rate": round(len(latencies) / N_REQUESTS * 100, 2),
        "duration_seconds": round(total_s, 2),
        "qps": round(N_REQUESTS / total_s, 2),
        "cache_hits": cache_hits,
        "latency_ms": {
            "min": round(ordered[0], 2),
            "avg": round(sum(ordered) / len(ordered), 2),
            "p50": round(_percentile(ordered, 50), 2),
            "p95": round(_percentile(ordered, 95), 2),
            "p99": round(_percentile(ordered, 99), 2),
            "max": round(ordered[-1], 2),
        },
        "memory_mb": {
            "rss_before": round(rss_before / 1024 / 1024, 2),
            "rss_after": round(rss_after / 1024 / 1024, 2),
            "growth": round((rss_after - rss_before) / 1024 / 1024, 2),
        },
        "errors": errors,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # —— 稳定性断言（人工智能NLP-RAG-基于PDF文档的问答系统优化）——
    assert not errors, f"存在失败请求: {errors[:3]}"
    assert stats["success_rate"] == 100.0
    assert stats["latency_ms"]["p95"] < P95_LIMIT_MS, stats["latency_ms"]
    assert stats["latency_ms"]["max"] < P95_LIMIT_MS * 3
    assert stats["memory_mb"]["growth"] < MEM_GROWTH_LIMIT_MB, stats["memory_mb"]


def test_mixed_invalid_input_keeps_service_alive(stability_client):
    """异常输入混合压测：每 9 次合法请求插 1 次空输入（422），服务全程可用"""
    ok, rejected = 0, 0
    for i in range(N_REQUESTS):
        if i % 10 == 9:
            r = stability_client.post("/api/ask", json={"question": ""})
            assert r.status_code == 422
            rejected += 1
        else:
            r = stability_client.post("/api/ask", json={"question": f"混合压测问题{i}"})
            assert r.status_code == 200, r.text
            assert r.json()["answer"]
            ok += 1
    assert ok == 90 and rejected == 10
    # 压测后健康检查仍正常
    assert stability_client.get("/api/health").status_code == 200
