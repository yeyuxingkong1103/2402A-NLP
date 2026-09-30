"""Knowledge-base and document management endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from app.api.v1 import access, deps
from app.api.v1.kb_schemas import (
    DocumentListResponse,
    DocumentResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from app.rag.knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)
router = APIRouter()


def _service() -> KnowledgeBaseService:
    client = deps.get_mysql_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge-base management requires MySQL",
        )
    return KnowledgeBaseService(client)


def _tenant(claimed: int | None) -> int:
    """Resolve tenant scope exclusively from the authenticated caller."""
    return access.tenant_id(claimed)


def _kb(row: dict[str, Any]) -> KnowledgeBaseResponse:
    data = {key: row.get(key) for key in KnowledgeBaseResponse.model_fields}
    if isinstance(data.get("settings"), str):
        try:
            data["settings"] = json.loads(data["settings"])
        except json.JSONDecodeError:
            data["settings"] = {}
    return KnowledgeBaseResponse(**data)


def _doc(row: dict[str, Any]) -> DocumentResponse:
    return DocumentResponse(**{key: row.get(key) for key in DocumentResponse.model_fields})


@router.post("/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=201)
@router.post("/kb", response_model=KnowledgeBaseResponse, status_code=201, include_in_schema=False)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
) -> KnowledgeBaseResponse:
    data = payload.model_dump()
    data["tenant_id"] = _tenant(payload.tenant_id)
    data["owner_id"] = access.user_id(payload.owner_id)
    return _kb(_service().create(data))


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
@router.get("/kb", response_model=KnowledgeBaseListResponse, include_in_schema=False)
async def list_knowledge_bases(
    tenant_id: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> KnowledgeBaseListResponse:
    rows, total = _service().list(_tenant(tenant_id), offset, limit)
    return KnowledgeBaseListResponse(items=[_kb(row) for row in rows], total=total)


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
@router.get("/kb/{kb_id}", response_model=KnowledgeBaseResponse, include_in_schema=False)
async def get_knowledge_base(
    kb_id: int, tenant_id: int | None = Query(default=None, ge=1)
) -> KnowledgeBaseResponse:
    row = _service().get(kb_id, _tenant(tenant_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return _kb(row)


@router.patch("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: int,
    payload: KnowledgeBaseUpdate,
    tenant_id: int | None = Query(default=None, ge=1),
) -> KnowledgeBaseResponse:
    service = _service()
    scope = _tenant(tenant_id)
    if not service.update(kb_id, scope, payload.model_dump(exclude_unset=True)):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return _kb(service.get(kb_id, scope) or {})


@router.delete("/knowledge-bases/{kb_id}", status_code=204)
@router.delete("/kb/{kb_id}", status_code=204, include_in_schema=False)
async def delete_knowledge_base(
    kb_id: int, tenant_id: int | None = Query(default=None, ge=1)
) -> None:
    if not _service().delete(kb_id, _tenant(tenant_id)):
        raise HTTPException(status_code=404, detail="Knowledge base not found")


@router.get("/knowledge-bases/{kb_id}/documents", response_model=DocumentListResponse)
@router.get("/kb/{kb_id}/documents", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents(
    kb_id: int,
    tenant_id: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> DocumentListResponse:
    rows, total = _service().documents(kb_id, _tenant(tenant_id), offset, limit)
    return DocumentListResponse(items=[_doc(row) for row in rows], total=total)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str, tenant_id: int | None = Query(default=None, ge=1)
) -> DocumentResponse:
    row = _service().document(document_id, _tenant(tenant_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc(row)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str, tenant_id: int | None = Query(default=None, ge=1)
) -> None:
    service = _service()
    scope = _tenant(tenant_id)
    row = service.document(document_id, scope)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not service.mark_document_deleted(document_id, scope):
        raise HTTPException(status_code=500, detail="Failed to mark document deleted")
    try:
        from app.api.v1.routes import _get_vector_store

        _get_vector_store().delete_by_document(document_id)
    except Exception as exc:
        logger.exception("Milvus document deletion failed")
        raise HTTPException(status_code=503, detail="Document vector deletion failed; retry required") from exc
