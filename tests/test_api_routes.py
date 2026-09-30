"""路由级冒烟：不需要模型推理的轻量检查（含 favicon 204 空响应回归）。

回归背景：``JSONResponse(status_code=204, content=None)`` 会把 ``None``
渲染成 ``b"null"``。h11 对 204 响应强制按 ``Content-Length: 0`` 分帧，
一旦发送这 4 个字节就会抛
``LocalProtocolError: Too much data for declared Content-Length``
并中断连接（uvicorn 层表现为一次 ERROR 日志）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx", reason="TestClient 依赖 httpx")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from role_rag.api.app import app

    # 不使用 with 语法：避免触发 lifespan（会预热模型，慢且占显存）
    return TestClient(app)


def test_favicon_returns_empty_204(client):
    """204 响应必须没有 body。"""

    response = client.get("/favicon.ico")
    assert response.status_code == 204
    assert response.content == b""


def test_index_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Role RAG_try" in response.text


def test_roles_endpoint_requires_token(client):
    response = client.get("/api/roles")
    assert response.status_code == 401
