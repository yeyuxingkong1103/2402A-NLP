# -*- coding: utf-8 -*-
"""HTTP 层错误契约测试

《接口文档》1.3 承诺「所有错误统一返回如下结构：{"detail": "错误说明（中文）"}」，
且 2.7 明列 `/api/chat/query` 的 `400` 场景为「问题为空」。

但 FastAPI 对请求体校验失败默认返回 **422 + 结构化 errors 数组**，与该承诺冲突
（会把 pydantic 的内部结构泄漏给 curl / Postman 等直接调用方）。
`backend/main.py` 因此注册了 `RequestValidationError` 处理器，把默认的 422
转换为文档承诺的 400 + 中文串。

本文件从 HTTP 层锚定该契约；模型层的校验行为见 `tests/test_input_validation.py`。
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

# 说明：Starlette 的 TestClient 只在 `with` 语句中才触发 lifespan（启动自检会去连
# Milvus / MySQL / Redis）。这里直接实例化，使测试不依赖任何外部服务——
# 参数校验发生在进入路由函数之前，本就不需要这些依赖。
client = TestClient(app)


# ===========================================================================
# 一、校验失败：400 + 中文串 detail
# ===========================================================================

@pytest.mark.parametrize(
    "question, 期望提示片段",
    [
        ("   ", "空白"),        # 纯空白
        ("？？？！！！", "标点"),  # 纯标点
        ("", "空"),              # 空串
        ("问" * 501, "500"),     # 超长
    ],
)
def test_非法问题返回400且detail为中文串(question, 期望提示片段):
    resp = client.post("/api/chat/query", json={"question": question})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert isinstance(body["detail"], str), "detail 必须是字符串，不能是 errors 数组"
    assert 期望提示片段 in body["detail"]
    assert set(body.keys()) == {"detail"}, "错误响应体只应有 detail 一个字段"


def test_缺少问题字段返回400且detail为中文串():
    resp = client.post("/api/chat/query", json={})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert isinstance(body["detail"], str)
    assert body["detail"]


def test_top_k_越界返回400且detail为中文串():
    resp = client.post("/api/chat/query", json={"question": "正常问题", "top_k": 99})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert isinstance(body["detail"], str)
    assert set(body.keys()) == {"detail"}


def test_请求体不是合法_JSON_时也返回400且detail为中文串():
    resp = client.post(
        "/api/chat/query",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert isinstance(body["detail"], str)
    assert set(body.keys()) == {"detail"}


# ===========================================================================
# 二、校验放行：合法问题不应被校验层拦下
# ===========================================================================

def test_合法问题通过校验层(monkeypatch):
    """
    校验层必须放行合法问题（含首尾空白，strip 后合法）。

    链路本身用假实现替换：本用例只关心「校验层是否放行」，
    不应真的去调用模型与向量库。
    """

    class _FakePipeline:
        """只回显参数的假链路，避免真实检索与生成"""

        def answer(self, question, session_id, top_k=None):
            return {
                "session_id": session_id or "fake-session",
                "answer": "假答案：%s" % question,
                "sources": [],
                "pipeline": "fake",
                "latency_ms": 1,
                "cached": False,
            }

    monkeypatch.setattr(
        "backend.api.query.get_pipeline_by_name", lambda name: _FakePipeline()
    )

    resp = client.post(
        "/api/chat/query",
        json={"question": "  数据中心业务连续性分为几个等级？  "},
    )

    assert resp.status_code == 200, resp.text
    # 校验器已 strip，链路拿到的是去掉首尾空白后的问题
    assert resp.json()["answer"] == "假答案：数据中心业务连续性分为几个等级？"
