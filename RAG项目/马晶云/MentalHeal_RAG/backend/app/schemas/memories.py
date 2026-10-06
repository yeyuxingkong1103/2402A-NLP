from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class LongTermMemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    memory_type: str = Field(default="preference", min_length=1, max_length=32)
    confirmed: bool = Field(
        default=False,
        description="必须明确确认后才会保存长期记忆",
    )

    @field_validator("content", "memory_type")
    @classmethod
    def strip_and_require_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("内容不能为空")
        return value


class LongTermMemoryResponse(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    is_confirmed: bool
    created_at: datetime
    updated_at: datetime


class LongTermMemoryListResponse(BaseModel):
    memories: list[LongTermMemoryResponse]
