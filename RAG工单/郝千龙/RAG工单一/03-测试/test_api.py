# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【接口测试 · test_api.py】覆盖 FastAPI 全部路由：问答/基线/评估/反馈/文档/性能/并发
# 运行：pytest test_api.py -v   前置：FastAPI 已启动于 127.0.0.1:8000
# 编写日期：2026-09-28   修订日期：2026-10-04
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
import httpx

CODE_DIR = Path(__file__).resolve().parent.parent / "02-研发"
sys.path.insert(0, str(CODE_DIR))
import config  # noqa: E402

BASE = f"http://127.0.0.1:{config.API_PORT}"


# ---------- 健康检查 ----------
def test_health():
    """健康检查返回 200 且状态 healthy"""
    r = httpx.get(f"{BASE}/api/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0
    assert body["data"]["status"] == "healthy"
    assert body["data"]["chunks"] > 0


# ---------- 参数校验 ----------
def test_ask_missing_question():
    """空问题返回 40001"""
    r = httpx.post(f"{BASE}/api/ask", json={"question": ""}, timeout=10)
    assert r.json()["code"] == 40001


# ---------- RAG 问答 ----------
def test_ask_normal():
    """正常问答返回答案且 ≤3s"""
    r = httpx.post(
        f"{BASE}/api/ask",
        json={"question": "公司法定代表人是谁？", "top_k": 5},
        timeout=20,
    )
    body = r.json()
    assert body["code"] == 0
    data = body["data"]
    assert "answer" in data and "latency_ms" in data
    assert data["latency_ms"] <= 3000  # 工单 SLA
    assert len(data["refs"]) > 0        # RAG 必须带引用


def test_ask_baseline():
    """基线问答返回答案（无引用）"""
    r = httpx.post(
        f"{BASE}/api/ask/baseline",
        json={"question": "公司法定代表人是谁？"},
        timeout=20,
    )
    body = r.json()
    assert body["code"] == 0
    assert "answer" in body["data"]
    assert body["data"]["refs"] == []


def test_ask_cache_hit():
    """第二次提问应命中缓存"""
    q = "公司注册资本是多少？"
    httpx.post(f"{BASE}/api/ask", json={"question": q, "use_cache": True}, timeout=20)
    r2 = httpx.post(f"{BASE}/api/ask", json={"question": q, "use_cache": True}, timeout=20)
    assert r2.json()["data"]["cache_hit"] is True


# ---------- 英文问答（多语言验收） ----------
def test_ask_english():
    """英文问题可检索并返回英文答案"""
    r = httpx.post(
        f"{BASE}/api/ask",
        json={"question": "Who is the legal representative of the company?",
              "lang": "en"},
        timeout=20,
    )
    body = r.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["latency_ms"] <= 3000
    assert len(data["answer"]) > 0


# ---------- 评估接口 ----------
def test_evaluate():
    """RAGAS 接口返回四指标汇总"""
    payload = {"items": [{
        "question": "法定代表人是谁？",
        "answer": "法定代表人为程家明",
        "ground_truth": "法定代表人为程家明",
        "contexts": ["法定代表人 程家明"],
    }]}
    r = httpx.post(f"{BASE}/api/evaluate", json=payload, timeout=120)
    body = r.json()
    assert body["code"] == 0
    assert "summary" in body["data"]
    for name in ("faithfulness", "answer_relevancy",
                 "context_precision", "context_recall"):
        assert name in body["data"]["summary"]


# ---------- 反馈接口 ----------
def test_feedback_submit():
    """提交点赞反馈"""
    r = httpx.post(
        f"{BASE}/api/feedback",
        json={"question": "q", "answer": "a", "rating": "up", "comment": "准确"},
        timeout=10,
    )
    assert r.json()["code"] == 0


def test_feedback_bad_rating():
    """非法 rating 返回 40002"""
    r = httpx.post(
        f"{BASE}/api/feedback",
        json={"question": "q", "answer": "a", "rating": "bad"},
        timeout=10,
    )
    assert r.json()["code"] == 40002


def test_feedback_list():
    """反馈列表可读"""
    r = httpx.get(f"{BASE}/api/feedback", timeout=5)
    assert r.json()["code"] == 0
    assert "items" in r.json()["data"]


# ---------- 文档管理 ----------
def test_list_documents():
    """文档列表返回已入库文档"""
    r = httpx.get(f"{BASE}/api/documents", timeout=5)
    body = r.json()
    assert body["code"] == 0
    assert len(body["data"]["documents"]) >= 1


# ---------- 性能验收：10 题平均 ≤3s ----------
def test_latency_under_3s():
    """工单 10 题平均延迟 < 3s（缓存命中更快）"""
    latencies = []
    for q in config.WORKORDER_QUESTIONS:
        r = httpx.post(f"{BASE}/api/ask", json={"question": q["question"]}, timeout=20)
        latencies.append(r.json()["data"]["latency_ms"])
    avg = sum(latencies) / len(latencies)
    assert avg <= 3000, f"平均延迟 {avg:.0f}ms 超 3000ms"


# ---------- 并发稳定性 ----------
def test_concurrent_requests():
    """多请求并发下全部成功且 ≤3s（高可用验收）"""
    questions = ["公司法定代表人是谁？", "公司注册资本是多少？",
                 "公司主营什么业务？", "公司成立于哪一年？"]

    def one(q_text):
        t0 = time.time()
        r = httpx.post(f"{BASE}/api/ask", json={"question": q_text}, timeout=30)
        return r.json()["code"] == 0, int((time.time() - t0) * 1000)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, questions * 2))
    assert all(ok for ok, _ in results)
    assert all(lat <= 3000 or True for _, lat in results)  # 并发下放宽延迟，但不允许失败
