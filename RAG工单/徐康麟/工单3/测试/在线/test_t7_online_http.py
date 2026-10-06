# -*- coding: utf-8 -*-
"""在线级：HTTP 服务契约（FastAPI ``app/main.py``，§7）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

覆盖：
    ``/api/health``、``/api/files``、``/api/ask``、``/api/ask/stream``（SSE）、
    ``/api/sessions``、``/api/sessions/{id}/messages``；状态码约定（400 入参非法 / 503 索引未就绪）。
    「不清楚」是**正常答案**（200），不是错误 —— 本文件不做「拒答即 500」的错误断言。
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = [pytest.mark.online]

QUESTION = "武汉兴图新科电子股份有限公司注册资本是多少？"


@pytest.fixture(scope="module")
def client(app_config: Any) -> Any:
    """FastAPI TestClient（触发 startup：setup_logging + build_engine(warmup=True)）。"""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    with TestClient(app) as instance:
        yield instance


def test_health(client: Any) -> None:
    """``/api/health`` 返回索引与 LLM 状态，且索引块数 > 0。"""
    response = client.get("/api/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("ok") is True
    assert int(payload.get("index", {}).get("count", 0)) > 0
    assert int(payload.get("files", 0)) >= 2, f"应发现 2 个 PDF，实测 {payload.get('files')}"


def test_files_listing(client: Any, discovered_pdfs: list[Any]) -> None:
    """``/api/files`` 的文件集合 = 自动发现结果（禁硬编码）。"""
    response = client.get("/api/files")
    assert response.status_code == 200, response.text
    files = response.json().get("files", [])
    assert {row["file_name"] for row in files} == {p.name for p in discovered_pdfs}
    for row in files:
        assert int(row.get("page_count", 0)) > 0


def test_ask_returns_answer_with_citation(client: Any) -> None:
    """``/api/ask`` 返回答案 + 引用 + 首字耗时，且字段齐备。"""
    response = client.post("/api/ask", json={"question": QUESTION, "session_id": "t8-http-1", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in ("text", "citations", "is_unknown", "first_token_ms", "session_id", "trace_id", "chunks"):
        assert key in payload, f"响应缺少字段 {key!r}：{sorted(payload)}"
    assert payload["text"].strip(), "答案正文为空"
    assert payload["citations"], "缺少引用"
    assert float(payload["first_token_ms"]) <= 3000.0, f"首字 {payload['first_token_ms']} ms 超预算"


def test_ask_stream_sse(client: Any) -> None:
    """``/api/ask/stream`` 是 ``text/event-stream``，含 first_token 与 done 事件。"""
    with client.stream("POST", "/api/ask/stream",
                       json={"question": QUESTION, "session_id": "t8-http-stream"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = "".join(chunk for chunk in response.iter_text())
    assert "event: first_token" in body or "first_token_ms" in body, body[:300]
    assert "event: done" in body or "done" in body, body[-300:]
    assert "text" in body


def test_invalid_input_status_codes(client: Any) -> None:
    """入参非法：空问题 → 4xx；不存在的文件名 → 4xx（§7 约定 400）。"""
    empty = client.post("/api/ask", json={"question": "", "session_id": "t8-http-bad"})
    assert empty.status_code in (400, 422), f"空问题应 4xx，实测 {empty.status_code}"
    missing = client.post("/api/ask", json={"question": QUESTION, "file_names": ["不存在的文件.pdf"]})
    assert missing.status_code in (400, 404, 422), f"不存在的文件名应 4xx，实测 {missing.status_code}"


def test_sessions_endpoints(client: Any) -> None:
    """会话列表与消息回放一致（HTTP 层也能读到落库历史）。"""
    client.post("/api/ask", json={"question": QUESTION, "session_id": "t8-http-sess"})
    listing = client.get("/api/sessions")
    assert listing.status_code == 200, listing.text
    sessions = listing.json().get("sessions", [])
    assert any(row.get("session_id") == "t8-http-sess" for row in sessions), sessions[:5]
    messages = client.get("/api/sessions/t8-http-sess/messages")
    assert messages.status_code == 200, messages.text
    rows = messages.json().get("messages", [])
    assert len(rows) >= 2, f"应有问+答两条，实测 {len(rows)}"
    assert rows[0].get("role") == "user"


def test_unknown_is_200_not_error(client: Any) -> None:
    """「不清楚」是正常答案：语料外提问必须 200 且 ``is_unknown=True``、无引用。"""
    response = client.post("/api/ask", json={"question": "银河系外文明的量子计算机专利申请数量是多少？"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("is_unknown") is True, payload.get("text", "")[:60]
    assert not payload.get("citations"), "拒答不得带引用"
