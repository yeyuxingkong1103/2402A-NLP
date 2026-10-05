from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    section_title: str | None = None
    content: str
    score: float | None = None
    document_name: str


class QueryPlan(BaseModel):
    language: Literal["zh", "en"] = "zh"
    intent: str = "fact_lookup"
    entities: list[str] = Field(default_factory=list)
    years: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None
    language: Literal["auto", "zh", "en"] = "auto"
    stream: bool = False


class AskResponse(BaseModel):
    request_id: str
    answer: str
    language: str
    citations: list[Citation] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient"]
    refusal: str | None = None
    latency: dict[str, float] = Field(default_factory=dict)
    model: str
    document_id: str | None = None


class FeedbackRequest(BaseModel):
    helpful: bool
    reason: str | None = None
    note: str | None = None


class DocumentOut(BaseModel):
    id: str
    display_name: str
    original_name: str
    sha256: str
    status: str
    pages: int = 0
    chunks: int = 0
    error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskOut(BaseModel):
    task_id: str
    document_id: str
    status: str
    stage: str
    progress: int
    message: str | None = None
