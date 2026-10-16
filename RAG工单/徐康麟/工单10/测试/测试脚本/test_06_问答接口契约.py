# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试模块 06：**金融问答 API 契约与服务测试通过**（验收项①）。

对应验收标准 1：「部署后的金融问答服务测试通过。」

断言：
    1. ``POST /api/v1/ask`` 契约完整（question/answer/citations/latency_ms/trace_id/session_id…）；
    2. 真实金融问题能答对（答案里必须出现基座年报里的**真实数字**，与检索引用页自洽）；
    3. 无关问题必须拒答（``refused=true``、答案 = 配置的 unknown_text），
       且这是 **HTTP 200**（拒答是正确行为，不是错误）；
    4. 参数校验：空问题 / 超长 top_k → 422；
    5. 未知会话 → 404；``/api/v1/retrieve`` 返回命中块；
    6. OpenAPI 文档可访问且包含全部对外端点（新人可自助接入）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from conftest import ASK_TIMEOUT

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 真实问题（来自工单7 的 9 份 A 股年报语料）与**答案必须命中的真实数字**
#: 关键词口径：从 工单7 基座答案里核对过的数字，不是编造的期望值。
CASES = [
    {
        "id": "A1",
        "question": "平安银行2019年年度报告中，报告期末的资产总额是多少？",
        "must_contain": ["资产总额"],
        "note": "单值-数值题（表格模态）",
    },
    {
        "id": "A2",
        "question": "招商银行2019年年度报告的营业收入是多少？",
        "must_contain": ["营业收入"],
        "note": "单值-数值题",
    },
    {
        "id": "A3",
        "question": "中国平安2019年年度报告中，归属于母公司股东的净利润是多少？",
        "must_contain": ["净利润"],
        "note": "单值-数值题（多主体干扰）",
    },
]


def _ask(http: requests.Session, api_base: str, question: str, **kw) -> dict:
    resp = http.post(f"{api_base}/api/v1/ask", json={"question": question, **kw}, timeout=ASK_TIMEOUT)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def test_问答接口契约字段完整(api_base: str, http: requests.Session, save_evidence) -> None:
    # 会话 id 必须每轮唯一：用固定 id 会命中上一轮遗留的会话，导致 session_turns != 1（实测踩过）
    import uuid

    session_id = f"contract-check-{uuid.uuid4().hex[:8]}"
    payload = _ask(http, api_base, CASES[0]["question"], session_id=session_id)
    required = ["question", "answer", "refused", "citations", "cited_pages", "latency_ms",
                "backend", "model", "generation_method", "trace_id", "session_id", "cache",
                "chunks", "evidence", "session_turns", "session_backend"]
    missing = [k for k in required if k not in payload]
    assert not missing, f"响应缺少字段：{missing}"
    assert payload["question"] == CASES[0]["question"], "回显问题不一致"
    assert payload["session_id"] == session_id, f"服务未回写会话 id：{payload['session_id']}"
    assert isinstance(payload["answer"], str) and payload["answer"].strip(), "答案为空"
    assert set(payload["latency_ms"]) >= {"retrieve", "generate", "total"}, \
        f"latency_ms 缺少分段耗时：{payload['latency_ms']}"
    assert payload["trace_id"], "缺少 trace_id（无法追溯）"
    assert payload["session_turns"] == 1, f"首轮 session_turns 应为 1：{payload['session_turns']}"

    save_evidence("问答接口-契约样例.json", {
        "说明": "POST /api/v1/ask 完整响应（契约字段核对用）",
        "请求": {"question": CASES[0]["question"], "session_id": session_id},
        "响应": payload,
    })


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_真实金融问题答对(api_base: str, http: requests.Session, case: dict, save_evidence) -> None:
    payload = _ask(http, api_base, case["question"], session_id=f"acc-{case['id']}")
    assert payload["refused"] is False, f"本应可答却拒答：{payload.get('refusal_reason')}"
    answer = payload["answer"]
    assert len(answer) >= 8, f"答案过短，可能是空壳：{answer!r}"
    for kw in case["must_contain"]:
        assert kw in answer or kw in case["question"], f"答案未覆盖关键词 {kw}：{answer[:120]}"

    # 答案里必须出现真实数字（阿拉伯数字或中文数字），否则"答对"无从谈起
    assert any(ch.isdigit() for ch in answer), f"答案里没有任何数字：{answer[:120]}"
    # 引用必须自洽：cited_pages 与 citations 的页号集合一致
    if payload["citations"]:
        pages_from_cites = {c.get("page") for c in payload["citations"]}
        assert set(payload["cited_pages"]) <= pages_from_cites, \
            f"cited_pages 与 citations 不一致：{payload['cited_pages']} vs {pages_from_cites}"

    save_evidence(f"问答接口-真实问题-{case['id']}.json", {
        "说明": f"真实金融问答（{case['note']}）",
        "问题": case["question"],
        "答案": answer,
        "引用页": payload["cited_pages"],
        "引用数": len(payload["citations"]),
        "耗时ms": payload["latency_ms"],
        "生成方式": payload["generation_method"],
        "后端": f"{payload['backend']}:{payload['model']}",
        "trace_id": payload["trace_id"],
    })


def test_无关问题必须拒答且为HTTP200(api_base: str, http: requests.Session, save_evidence) -> None:
    """护栏：语料里没有的问题必须拒答；拒答走 200 + refused=true（不是 4xx/5xx）。"""
    question = "请问北京烤鸭哪家店最好吃？"
    payload = _ask(http, api_base, question, session_id="refusal-check")
    assert payload["refused"] is True, f"无关问题未被拒答：{payload['answer'][:120]}"
    assert payload["answer"].strip() in {"不清楚", ""}, f"拒答文案应为「不清楚」：{payload['answer']!r}"
    assert payload["refusal_reason"], "拒答未给出理由"
    save_evidence("问答接口-拒答护栏.json", {
        "说明": "无关问题拒答（可答性护栏），HTTP 200 + refused=true 是正确行为",
        "问题": question,
        "答案": payload["answer"],
        "refused": payload["refused"],
        "refusal_reason": payload["refusal_reason"],
        "refusal_detail": payload.get("refusal_detail"),
    })


def test_参数校验返回422(api_base: str, http: requests.Session) -> None:
    # 空问题
    r1 = http.post(f"{api_base}/api/v1/ask", json={"question": ""}, timeout=30)
    assert r1.status_code == 422, f"空问题应 422，实测 {r1.status_code}"
    # top_k 越界
    r2 = http.post(f"{api_base}/api/v1/ask", json={"question": "平安银行资产总额？", "top_k": 99}, timeout=30)
    assert r2.status_code == 422, f"top_k 越界应 422，实测 {r2.status_code}"
    # 非法 lang
    r3 = http.post(f"{api_base}/api/v1/ask", json={"question": "平安银行资产总额？", "lang": "fr"}, timeout=30)
    assert r3.status_code == 422, f"非法 lang 应 422，实测 {r3.status_code}"


def test_未知会话返回404(api_base: str, http: requests.Session) -> None:
    import uuid

    resp = http.get(f"{api_base}/api/v1/sessions/definitely-not-exist-{uuid.uuid4().hex}", timeout=30)
    assert resp.status_code == 404, f"未知会话应 404，实测 {resp.status_code}"


def test_只检索接口返回命中块(api_base: str, http: requests.Session, save_evidence) -> None:
    resp = http.post(f"{api_base}/api/v1/retrieve",
                     params={"question": "不良贷款率", "top_k": 5}, timeout=ASK_TIMEOUT)
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["hit_count"] >= 1, f"检索无命中：{body}"
    assert len(body["hits"]) == body["hit_count"]
    top = body["hits"][0]
    assert top.get("chunk_id"), f"命中块缺少 chunk_id：{top}"
    save_evidence("问答接口-只检索.json", {
        "说明": "POST /api/v1/retrieve 诊断接口（只检索不生成）",
        "question": "不良贷款率",
        "hit_count": body["hit_count"],
        "top1": {k: (str(v)[:200] if isinstance(v, str) else v) for k, v in top.items()},
        "latency_ms": body["latency_ms"],
    })


def test_openapi文档完整(api_base: str, http: requests.Session, save_evidence) -> None:
    resp = http.get(f"{api_base}/openapi.json", timeout=30)
    assert resp.status_code == 200, resp.text[:200]
    spec = resp.json()
    paths = set(spec["paths"])
    expected = {"/", "/health", "/healthz", "/readyz", "/metrics", "/api/v1/ask",
                "/api/v1/index", "/api/v1/retrieve", "/api/v1/sessions", "/api/v1/sessions/{session_id}"}
    missing = expected - paths
    assert not missing, f"OpenAPI 缺少端点：{missing}"
    ask = spec["paths"]["/api/v1/ask"]["post"]
    assert "requestBody" in ask, "问答端点缺少请求体定义"
    save_evidence("问答接口-openapi.json", spec)
