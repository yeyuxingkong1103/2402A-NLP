"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "1.0.0"


class UploadDocumentRequest(BaseModel):
    """Document upload request."""

    knowledge_base_id: int = Field(default=1, description="Knowledge base ID")
    tenant_id: int = Field(default=1, description="Tenant ID")
    role_id: int = Field(default=0, description="Role ID")
    parser: str = Field(default="auto", description="Parser type: auto, text, pymupdf, pdfplumber")
    chunking_strategy: str = Field(default="paragraph", description="Chunking strategy")
    chunk_size: int = Field(default=800, description="Target chunk size in characters")
    force_reindex: bool = Field(default=False, description="Force reindex even if hash exists")


class UploadDocumentResponse(BaseModel):
    """Document upload response."""

    document_id: str
    file_name: str
    file_sha256: str
    chunks_inserted: int
    status: str


class QueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., min_length=1, description="User query")
    knowledge_base_id: int | None = Field(default=None, ge=1, description="Filter by knowledge base")
    tenant_id: int | None = Field(default=None, ge=1, description="Tenant must match authenticated identity")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    rerank: bool = Field(default=False, description="Enable reranking")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    conversation_id: str | None = Field(default=None, description="Conversation ID for history")


class SourceChunk(BaseModel):
    """Source chunk in response."""

    chunk_id: str
    text: str
    source: str
    score: float
    summary: str = ""


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    query: str
    sources: list[SourceChunk]
    model: str
    retrieval_count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class IndexStatsResponse(BaseModel):
    """Index statistics response."""

    collection_name: str
    total_vectors: int
    dimension: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
