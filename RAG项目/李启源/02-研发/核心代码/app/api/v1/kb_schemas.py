"""Schemas for knowledge-base and document management APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    tenant_id: int | None = Field(default=None, ge=1)
    owner_id: int | None = Field(default=None, ge=1)
    settings: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    settings: dict[str, Any] | None = None
    status: str | None = Field(default=None, pattern="^(active|archived)$")


class KnowledgeBaseResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    tenant_id: int
    owner_id: int | None = None
    status: str
    settings: dict[str, Any] | None = None
    created_at: Any = None
    updated_at: Any = None


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int


class DocumentResponse(BaseModel):
    id: str
    knowledge_base_id: int
    version_no: int
    file_name: str
    file_sha256: str
    status: str
    review_status: str
    chunk_count: int | None = None
    character_count: int | None = None
    created_at: Any = None
    updated_at: Any = None
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
