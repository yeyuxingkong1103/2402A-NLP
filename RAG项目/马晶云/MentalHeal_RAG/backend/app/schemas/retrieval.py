from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=36)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    title: str
    page_start: int
    page_end: int
    score: float
    text: str


class RetrievalResponse(BaseModel):
    session_id: str
    query: str
    results: list[RetrievedChunk]
