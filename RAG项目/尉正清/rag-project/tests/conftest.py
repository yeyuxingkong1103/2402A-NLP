# tests/conftest.py
"""pytest 公共夹具。

测试针对**真实运行的服务**发 HTTP 请求，而不是用 TestClient 在进程内调用。
原因：项目里的 BGE-M3、Reranker 都是懒加载单例，进程内测试会额外占一份显存，
而真实服务的表现（含网络、序列化、超时）才是要验的对象。

    python run.py &
    .venv/bin/python -m pytest tests/ -v
"""
import os

import pytest
import requests

BASE_URL = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("RAG_API_TIMEOUT", "180"))


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def require_server():
    """整个测试会话开始前确认服务可用，避免一串无意义的连接错误。"""
    try:
        r = requests.get(BASE_URL + "/api/health", timeout=15)
        r.raise_for_status()
        body = r.json().get("data", {})
    except Exception as e:
        pytest.exit("服务不可用（%s）：请先启动 `python run.py`" % e, returncode=1)

    unhealthy = [k for k, v in body.items() if v is False]
    if unhealthy:
        pytest.exit("依赖未就绪: %s（检查 redis / milvus / mysql）" % unhealthy,
                    returncode=1)


@pytest.fixture(scope="session")
def client():
    """JSON 请求用。

    刻意**不**设置全局 Content-Type：requests 在 json= 时会自动加，
    而全局设置会覆盖 multipart/form-data 的 boundary，导致文件上传解析失败。
    """
    return requests.Session()


@pytest.fixture(scope="session")
def timeout():
    return TIMEOUT


@pytest.fixture(scope="session")
def chat(client, timeout):
    """返回一个「提问」函数，内部处理统一解包。"""
    def _ask(question, role_key="lawyer", user_id=1, session_id=None):
        payload = {"question": question, "role_key": role_key, "user_id": user_id}
        if session_id:
            payload["session_id"] = session_id
        r = client.post(BASE_URL + "/api/chat/ask", json=payload, timeout=timeout)
        return r

    return _ask
