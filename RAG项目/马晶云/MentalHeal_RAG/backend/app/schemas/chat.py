from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    role_id: str = Field(default="mental-health", max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatSource(BaseModel):
    chunk_id: str
    document_id: str | None = None
    title: str
    page_start: int
    page_end: int
    score: float
    rerank_score: float | None = None
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[ChatSource]
    safety_intervention: bool = False
