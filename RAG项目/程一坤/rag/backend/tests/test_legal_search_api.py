"""法律检索接口契约测试。"""

import asyncio
import re

import httpx
import pytest

from app.auth.current_user import get_current_user
from app.auth.session_store import SessionUser
from app.main import app, global_exception_handler
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.query_rewrite import QueryRewriteResult
from app.retrieval.service import RetrievalResult


KNOWN_REQUEST = {
    "query": "经济补偿怎么算",
    "knowledge_base_ids": ["kb_labor_law_001"],
    "jurisdiction": "中国大陆",
    "as_of_date": "2026-09-14",
    "legal_domain": "labor_law",
    "document_types": ["law", "judicial_interpretation", "case"],
    "top_k": 5,
    "enable_hybrid_search": True,
    "enable_rerank": True,
    "user_id": "伪造用户",
}


class RecordingRetrievalService:
    """记录 API 下传参数，并返回完整的固定检索结果。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        self.calls.append((question, kwargs))
        article = RetrievedArticle(
            chunk_key="law-a47",
            content="经济补偿按劳动者在本单位工作的年限计算。",
            document_title="中华人民共和国劳动合同法",
            source_url="https://example.gov.cn/law",
            recall_score=0.74,
            article_number="第四十七条",
            paragraph_number="1",
            document_type="法律",
            jurisdiction="中国大陆",
            effective_date=1577836800,
            expiration_date=0,
            issuing_authority="全国人民代表大会常务委员会",
            is_current=True,
            score_sources=("vector", "keyword"),
            vector_score=0.74,
            keyword_score=7.57,
            fusion_score=0.032,
            rerank_score=0.94,
            document_id="doc-labor-contract-law",
        )
        return RetrievalResult(
            articles=[article],
            context_block="法源上下文",
            stats={
                "vector_recall_count": 11,
                "keyword_recall_count": 20,
                "fused_count": 26,
                "reranked_count": 5,
            },
            query_rewrite=QueryRewriteResult(
                original_query=question,
                rewritten_query="经济补偿 12 期 怎么算",
                changed=True,
                reasons=["命中指代/省略模式", "那", "主题词：经济补偿"],
            ),
        )


class FailingRetrievalService:
    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        raise RuntimeError(
            "SELECT * FROM document_chunks mysql+pymysql://user:secret@127.0.0.1/legal"
        )


async def post_json(payload: dict, *, authorization: str | None = "Bearer test-token"):
    headers = {"Authorization": authorization} if authorization else {}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v1/legal/search", json=payload, headers=headers)


@pytest.fixture
def authenticated_user():
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id="authenticated-user",
        is_admin=False,
    )
    yield
    app.dependency_overrides.pop(get_current_user, None)
    if hasattr(app.state, "legal_search_service"):
        del app.state.legal_search_service


def test_search_returns_envelope_maps_types_dates_and_uses_authenticated_user(
    authenticated_user,
) -> None:
    service = RecordingRetrievalService()
    app.state.legal_search_service = service

    response = asyncio.run(post_json(KNOWN_REQUEST))

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert re.fullmatch(r"req_[0-9a-f]{12}", body["request_id"])
    assert body["data"]["query"] == "经济补偿怎么算"
    assert body["data"]["rewritten_queries"] == ["经济补偿 12 期 怎么算"]
    assert body["data"]["rewrite"]["reasons"] == [
        "命中指代/省略模式",
        "那",
        "主题词：经济补偿",
    ]
    result = body["data"]["results"][0]
    assert result["document_id"] == "doc-labor-contract-law"
    assert result["document_type"] == "law"
    assert result["effective_date"] == "2020-01-01"
    assert result["repeal_date"] is None
    assert result["is_current"] is True
    assert result["score_sources"] == ["vector", "keyword"]
    assert result["vector_score"] == 0.74
    assert result["keyword_score"] == 7.57
    assert result["fusion_score"] == 0.032
    assert result["rerank_score"] == 0.94
    assert body["data"]["stats"]["fused_count"] == 26

    question, kwargs = service.calls[0]
    assert question == "经济补偿怎么算"
    assert kwargs["user_id"] == "authenticated-user"
    assert kwargs["document_types"] == ["法律", "司法解释", "案例材料"]
    assert kwargs["enable_hybrid_search"] is True
    assert kwargs["enable_rerank"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("as_of_date", "2026-02-30"),
        ("document_types", ["unknown"]),
        ("knowledge_base_ids", ["kb_unknown"]),
        ("legal_domain", "criminal_law"),
    ],
)
def test_search_rejects_invalid_contract_values(authenticated_user, field, value) -> None:
    payload = {**KNOWN_REQUEST, field: value}

    response = asyncio.run(post_json(payload))

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 40000
    assert body["data"] is None
    assert re.fullmatch(r"req_[0-9a-f]{12}", body["request_id"])


def test_search_requires_authentication() -> None:
    response = asyncio.run(post_json(KNOWN_REQUEST, authorization=None))

    assert response.status_code == 401
    assert response.json()["code"] == 40100
    assert re.fullmatch(r"req_[0-9a-f]{12}", response.json()["request_id"])


def test_global_exception_handler_hides_internal_details_from_logs(caplog) -> None:
    request = type(
        "RequestStub",
        (),
        {"state": type("StateStub", (), {"request_id": "req_global1234"})()},
    )()
    error = RuntimeError("mysql+pymysql://user:secret@host/db document_chunks")

    response = asyncio.run(global_exception_handler(request, error))

    assert response.status_code == 500
    assert "secret" not in caplog.text.lower()
    record = next(record for record in caplog.records if record.name == "app.error")
    assert record.request_id == "req_global1234"


def test_search_hides_internal_error_details(authenticated_user, caplog) -> None:
    app.state.legal_search_service = FailingRetrievalService()

    response = asyncio.run(post_json(KNOWN_REQUEST))

    assert response.status_code == 500
    body = response.json()
    serialized = response.text.lower()
    assert body["code"] == 50000
    assert body["message"] == "系统内部错误"
    assert body["data"] is None
    assert "document_chunks" not in serialized
    assert "mysql+pymysql" not in serialized
    assert "secret" not in serialized
    assert "traceback" not in serialized
    assert "mysql+pymysql" not in caplog.text.lower()
    assert "secret" not in caplog.text.lower()
    assert re.fullmatch(r"req_[0-9a-f]{12}", body["request_id"])
