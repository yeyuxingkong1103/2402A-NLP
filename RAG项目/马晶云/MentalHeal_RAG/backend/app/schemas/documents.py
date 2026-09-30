from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: str
    source_file: str
    page_count: int
    total_text_chars: int
    is_enabled: bool
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentJobResponse(BaseModel):
    document_id: str
    job_id: int | None = None
    status: str
    message: str


class IngestJobResponse(BaseModel):
    job_id: int
    job_name: str
    status: str
    document_count: int
    page_count: int
    chunk_count: int
    note: str
    started_at: datetime
    finished_at: datetime


class IngestJobListResponse(BaseModel):
    jobs: list[IngestJobResponse]
