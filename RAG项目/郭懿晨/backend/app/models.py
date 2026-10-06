from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """文档状态。"""

    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class BuildStep(StrEnum):
    """构建步骤。"""

    PENDING = "pending"
    PARSING = "parsing"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """PDF 文档记录。"""

    document_id: str
    file_name: str
    file_path: str
    status: DocumentStatus
    created_at: datetime
    content_hash: str | None = None

    @classmethod
    def new(cls, file_name: str, file_path: str, content_hash: str | None = None) -> "Document":
        """创建已上传文档记录。"""
        return cls(
            document_id=str(uuid4()),
            file_name=file_name,
            file_path=file_path,
            status=DocumentStatus.UPLOADED,
            created_at=datetime.now(timezone.utc),
            content_hash=content_hash,
        )


class BuildTask(BaseModel):
    """构建任务记录。"""

    task_id: str
    document_id: str
    step: BuildStep
    progress: int = Field(ge=0, le=100)
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def new(cls, document_id: str) -> "BuildTask":
        """创建待运行构建任务。"""
        now = datetime.now(timezone.utc)
        return cls(
            task_id=str(uuid4()),
            document_id=document_id,
            step=BuildStep.PENDING,
            progress=0,
            status="pending",
            created_at=now,
            updated_at=now,
        )

    def mark_running(self, step: BuildStep, progress: int) -> "BuildTask":
        """标记任务进入运行步骤。"""
        return self.model_copy(
            update={
                "step": step,
                "progress": progress,
                "status": "running",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def mark_completed(self) -> "BuildTask":
        """标记任务完成。"""
        return self.model_copy(
            update={
                "step": BuildStep.COMPLETED,
                "progress": 100,
                "status": "completed",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def mark_failed(self, message: str) -> "BuildTask":
        """标记任务失败。"""
        return self.model_copy(
            update={
                "step": BuildStep.FAILED,
                "status": "failed",
                "error_message": message,
                "updated_at": datetime.now(timezone.utc),
            }
        )


class Chunk(BaseModel):
    """可溯源文本块。"""

    chunk_id: str
    document_id: str
    page: int = Field(ge=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_span: str = Field(min_length=1)


class Citation(BaseModel):
    """答案引用来源。"""

    document_id: str
    file_name: str
    page: int = Field(ge=1)
    category: str
    text: str
    score: float | None = None


class ChatRequest(BaseModel):
    """问答请求。"""

    question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    """问答响应。"""

    answer: str
    citations: list[Citation]
    fallback: bool
