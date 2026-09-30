"""真实端到端测试：手动起 uvicorn 后按序跑 6 个核心场景。

运行前提：先启动服务 `.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000`，
等日志出现 embedder/reranker 预热完成后再运行：`.venv/bin/pytest -m slow -v`。
"""
import json
import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.slow

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000")


def _sid() -> str:
    return f"e2e-{uuid.uuid4().hex[:12]}"


def _post(url: str, payload: dict) -> httpx.Response:
    return httpx.post(url, json=payload, timeout=120.0)


@pytest.fixture(scope="session")
def base_url():
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5.0).raise_for_status()
    except Exception:
        pytest.skip(
            "uvicorn 未启动，请先运行："
            ".venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000"
        )
    return BASE_URL


def test_health(base_url):
    resp = httpx.get(f"{base_url}/health", timeout=10.0)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_single_round(base_url):
    resp = _post(f"{base_url}/chat", {"message": "血压多少算高？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"].strip()
    assert len(data["sources"]) == 4
    for s in data["sources"]:
        # 父子分块后命中带 parent_content 的 source 会多一个 chunk_content 字段
        assert {"content", "page", "similarity"} <= set(s) <= {
            "content", "page", "similarity", "chunk_content",
        }


def test_chat_multi_round_same_session(base_url):
    sid = _sid()
    codeword = f"蓝鲸{uuid.uuid4().hex[:6]}"
    r1 = _post(f"{base_url}/chat", {
        "session_id": sid,
        "message": f"请记住：我给自己起的代号是「{codeword}」，请确认你记住了。",
    })
    assert r1.status_code == 200
    r2 = _post(f"{base_url}/chat", {
        "session_id": sid,
        "message": "我刚才说的代号是什么？",
    })
    assert r2.status_code == 200
    assert codeword in r2.json()["answer"]


def test_chat_session_isolation(base_url):
    codeword = f"蓝鲸{uuid.uuid4().hex[:6]}"
    _post(f"{base_url}/chat", {
        "session_id": _sid(),
        "message": f"请记住：我给自己起的代号是「{codeword}」，请确认你记住了。",
    })
    r2 = _post(f"{base_url}/chat", {
        "session_id": _sid(),
        "message": "我刚才说的代号是什么？",
    })
    assert r2.status_code == 200
    assert codeword not in r2.json()["answer"]


def test_chat_stream(base_url):
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{base_url}/chat/stream", json={"message": "血压多少算高？"}) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            lines = list(resp.iter_lines())

    # 定位 event: sources 后的 sources JSON
    sources = None
    for i, line in enumerate(lines):
        if line == "event: sources":
            for nxt in lines[i + 1:]:
                if nxt.startswith("data: "):
                    sources = json.loads(nxt[len("data: "):])
                    break
            break
    assert sources is not None, "未收到 event: sources"
    assert len(sources) == 4

    data_lines = [l for l in lines if l.startswith("data: ")]
    assert data_lines, "未收到任何 data 行"
    assert data_lines[-1] == "data: [DONE]", "流未以 [DONE] 结束"
    assert len(data_lines) >= 3, "未收到 token 内容"


def test_chat_empty_message_422(base_url):
    resp = _post(f"{base_url}/chat", {"message": ""})
    assert resp.status_code == 422


def test_chat_query_rewrite_multi_round(base_url):
    """多轮指代消解：第二轮「那我该怎么办？」应改写后检索，命中血压相关内容。"""
    sid = "rw_e2e"
    r1 = _post(f"{base_url}/chat", {
        "session_id": sid,
        "message": "我血压150/95",
        "role": "高血压医生",
    })
    assert r1.status_code == 200

    r2 = _post(f"{base_url}/chat", {
        "session_id": sid,
        "message": "那我该怎么办？",
        "role": "高血压医生",
    })
    assert r2.status_code == 200
    data = r2.json()
    assert data["sources"], "第二轮 sources 不应为空"
    contents = " ".join(s["content"] for s in data["sources"])
    assert ("血压" in contents) or ("150" in contents), (
        "改写未生效：第二轮检索结果未命中血压相关内容"
    )
