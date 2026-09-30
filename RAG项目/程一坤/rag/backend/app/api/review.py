"""管理员审核与发布接口（阶段6，契约 6.3 / 6.4）。

权限语义（按裁决）：
- 未登录 → 401（认证依赖拦截，接口"存在"）
- 已登录非管理员 → 403（接口本身存在，不是藏着不让看）
- 管理员 → 正常使用

生产装配：会话工厂复用 chat_runtime 的 MySQL 装配；向量库与 Embedding
客户端按 settings 直连（与 retrieval/assembly.py 同一风格）；
测试通过 app.state 注入替身。
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.auth.current_user import CurrentUser
from app.auth.session_store import SessionUser
from app.core.config import settings
from app.errors import bad_request_error, forbidden_error, not_found_error
from app.review.review_detail import ReviewDetailNotFound, get_review_document_detail
from app.review.review_service import (
    InvalidReviewDecision,
    NoPendingReviewVersion,
    list_review_documents,
    review_document,
)
# 状态词表默认值取自单一来源常量模块（避免接口层再出现裸字面量）
from app.db.version_status import PENDING_REVIEW

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _require_admin(current_user: CurrentUser) -> SessionUser:
    """管理员门卫：非管理员统一 403（不区分接口，管理员接口一律拦）。"""
    if not current_user.is_admin:
        raise forbidden_error("需要管理员权限")
    return current_user


# 依赖注入类型别名：通过门卫的管理员用户
AdminUser = Annotated[SessionUser, Depends(_require_admin)]


class ReviewRequest(BaseModel):
    """审核决定请求体（契约 6.4）。"""

    decision: str = Field(min_length=1, max_length=16)
    review_note: str | None = Field(default=None, max_length=1024)


def _get_review_session_factory(request: Request):
    """审核用的数据库会话工厂（测试可经 app.state 注入 SQLite）。"""
    factory = getattr(request.app.state, "review_session_factory", None)
    if factory is None:
        from app.chat.chat_runtime import build_default_session_factory

        factory = build_default_session_factory()
    return factory


def _get_vector_deps(request: Request) -> tuple[Any, Any]:
    """审核用的（向量库, Embedding 客户端）；测试可经 app.state 注入替身。"""
    store = getattr(request.app.state, "review_vector_store", None)
    client = getattr(request.app.state, "review_embedding_client", None)
    if store is None or client is None:
        from pymilvus import MilvusClient

        from app.db.milvus_store import LegalMilvusStore
        from app.models.embedding import SiliconFlowEmbeddingClient

        store = LegalMilvusStore(
            client=MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
            ),
            collection_name=settings.milvus_collection_name,
            dimension=settings.embedding_dimension,
        )
        store.ensure_collection()
        client = SiliconFlowEmbeddingClient(
            api_url=settings.embedding_api_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout=settings.embedding_timeout_seconds,
        )
    return store, client


@router.get("/documents")
def list_documents_for_review(
    admin: AdminUser,
    request: Request,
    status: str = Query(default=PENDING_REVIEW, max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    """按审核状态分页列出版本（契约 6.3；提交时间倒序）。"""
    try:
        with _get_review_session_factory(request)() as session:
            data = list_review_documents(
                session, status=status, page=page, page_size=page_size
            )
    except ValueError as error:
        # status 非法 → 参数错误
        raise bad_request_error(str(error)) from error
    return {
        "code": 0,
        "message": "success",
        "data": data,
        "request_id": request.headers.get("X-Request-ID", ""),
    }


@router.get("/documents/{document_id}/detail")
def get_document_review_detail(
    document_id: int,
    admin: AdminUser,
    request: Request,
    preview_chunks: int = Query(default=5, ge=1, le=20),
) -> dict:
    """查询审核文档详情（契约 6.5：元数据 + 分块预览）。

    preview_chunks：预览条数，默认 5、上限 20（契约约束）；内容在服务层截断。
    """
    try:
        with _get_review_session_factory(request)() as session:
            data = get_review_document_detail(
                session,
                document_id=document_id,
                preview_chunks=preview_chunks,
            )
    except ReviewDetailNotFound as error:
        raise not_found_error(str(error)) from error
    return {
        "code": 0,
        "message": "success",
        "data": data,
        "request_id": request.headers.get("X-Request-ID", ""),
    }


@router.post("/documents/{document_id}/review")
def review_document_by_id(
    document_id: int,
    payload: ReviewRequest,
    admin: AdminUser,
    request: Request,
) -> dict:
    """审核并发布/驳回文档（契约 6.4；审核人取认证上下文 user_key）。"""
    store, embedding_client = _get_vector_deps(request)
    try:
        with _get_review_session_factory(request)() as session:
            data = review_document(
                session,
                document_id=document_id,
                reviewer_user_key=admin.user_id,
                decision=payload.decision,
                review_note=payload.review_note,
                vector_store=store,
                embedding_client=embedding_client,
            )
    except NoPendingReviewVersion as error:
        raise not_found_error(str(error)) from error
    except InvalidReviewDecision as error:
        raise bad_request_error(str(error)) from error
    return {
        "code": 0,
        "message": "success",
        "data": data,
        "request_id": request.headers.get("X-Request-ID", ""),
    }
