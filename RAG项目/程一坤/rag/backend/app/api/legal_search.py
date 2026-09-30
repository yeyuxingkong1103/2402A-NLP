"""法律检索 HTTP 接口：校验契约并把能力层结果转换为公开响应。"""

import logging
from datetime import UTC, date, datetime
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth.current_user import CurrentUser
from app.errors import internal_error
from app.retrieval.assembly import build_default_retrieval_service
from app.retrieval.service import RetrievalResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/legal", tags=["legal-search"])

# API 使用英文稳定枚举，索引使用中文业务值；映射集中在此处，禁止分散转换。
DOCUMENT_TYPE_TO_STORAGE = {
    "law": "法律",
    "administrative_regulation": "行政法规",
    "judicial_interpretation": "司法解释",
    "case": "案例材料",
}
DOCUMENT_TYPE_TO_API = {value: key for key, value in DOCUMENT_TYPE_TO_STORAGE.items()}
KNOWN_KNOWLEDGE_BASE_ID = "kb_labor_law_001"
KNOWN_LEGAL_DOMAIN = "labor_law"


class LegalSearchRequest(BaseModel):
    """首期法律检索请求；固定知识库字段会显式校验，但暂不参与过滤。"""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=2000)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    jurisdiction: str = Field(default="中国大陆", min_length=1)
    as_of_date: date | None = None
    legal_domain: str | None = None
    document_types: list[Literal[
        "law",
        "administrative_regulation",
        "judicial_interpretation",
        "case",
    ]] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    enable_hybrid_search: bool = True
    enable_rerank: bool = True
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query 不能为空")
        return cleaned

    @field_validator("knowledge_base_ids")
    @classmethod
    def validate_knowledge_bases(cls, values: list[str]) -> list[str]:
        if any(value != KNOWN_KNOWLEDGE_BASE_ID for value in values):
            raise ValueError("首期仅支持劳动法知识库")
        return values

    @field_validator("legal_domain")
    @classmethod
    def validate_legal_domain(cls, value: str | None) -> str | None:
        if value not in (None, KNOWN_LEGAL_DOMAIN):
            raise ValueError("首期仅支持 labor_law")
        return value


class LegalSearchResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: dict[str, Any]
    request_id: str


@router.post("/search", response_model=LegalSearchResponse)
def search_legal_documents(
    payload: LegalSearchRequest,
    current_user: CurrentUser,
    request: Request,
) -> LegalSearchResponse:
    """执行检索；身份只取认证上下文，绝不采信请求体中的 user_id。"""
    request_id = request.state.request_id
    try:
        service = _get_retrieval_service(request)
        result = service.retrieve(
            payload.query,
            rerank_top_n=payload.top_k,
            as_of_date=payload.as_of_date.isoformat() if payload.as_of_date else None,
            jurisdiction=payload.jurisdiction,
            document_types=[DOCUMENT_TYPE_TO_STORAGE[item] for item in payload.document_types] or None,
            user_id=current_user.user_id,
            session_id=payload.session_id,
            enable_hybrid_search=payload.enable_hybrid_search,
            enable_rerank=payload.enable_rerank,
            request_id=request_id,
        )
    except Exception as error:
        logger.error(
            "法律检索失败：%s",
            type(error).__name__,
            extra={"request_id": request_id},
        )
        raise internal_error("系统内部错误") from None

    logger.info(
        "法律检索完成：user_id=%s results=%s",
        current_user.user_id,
        len(result.articles),
        extra={"request_id": request_id},
    )
    return LegalSearchResponse(
        data=_serialize_result(payload, result),
        request_id=request_id,
    )


def _get_retrieval_service(request: Request) -> Any:
    service = getattr(request.app.state, "legal_search_service", None)
    if service is None:
        service = build_default_retrieval_service()
        request.app.state.legal_search_service = service
    return service


def _serialize_result(payload: LegalSearchRequest, result: RetrievalResult) -> dict[str, Any]:
    rewrite = result.query_rewrite
    rewritten = rewrite.rewritten_query if rewrite else payload.query
    changed = bool(rewrite and rewrite.changed)
    return {
        "query": payload.query,
        "rewritten_queries": [rewritten] if changed else [],
        "rewrite": {
            "original_query": rewrite.original_query if rewrite else payload.query,
            "rewritten_query": rewritten,
            "changed": changed,
            "reasons": list(rewrite.reasons) if rewrite else [],
        },
        "results": [_serialize_article(article) for article in result.articles],
        "filters": {
            "knowledge_base_ids": payload.knowledge_base_ids,
            "jurisdiction": payload.jurisdiction,
            "as_of_date": payload.as_of_date.isoformat() if payload.as_of_date else None,
            "legal_domain": payload.legal_domain,
            "document_types": payload.document_types,
        },
        "stats": result.stats,
    }


def _serialize_article(article: Any) -> dict[str, Any]:
    document_type = DOCUMENT_TYPE_TO_API.get(article.document_type)
    return {
        "chunk_id": article.chunk_key,
        "document_id": article.document_id,
        "document_type": document_type,
        "law_name": article.document_title,
        "article_number": article.article_number,
        "paragraph_number": article.paragraph_number,
        "item_number": article.item_number,
        "jurisdiction": article.jurisdiction,
        "effective_date": _serialize_date(article.effective_date),
        "repeal_date": _serialize_date(article.expiration_date),
        "issuing_authority": article.issuing_authority,
        "is_current": article.is_current,
        "source": article.source_url,
        "content": article.content,
        "retrieval_score": article.recall_score,
        "score_sources": list(article.score_sources),
        "vector_score": article.vector_score,
        "keyword_score": article.keyword_score,
        "fusion_score": article.fusion_score,
        "rerank_score": article.rerank_score,
    }


def _serialize_date(value: Any) -> str | None:
    if value in (None, 0, "", "0"):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC).date().isoformat()
    if isinstance(value, str):
        return value
    return None
