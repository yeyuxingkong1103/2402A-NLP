"""API 请求 / 响应模型（pydantic v2）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    role_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["dense", "sparse", "bm25", "hybrid"] = "hybrid"
    fusion: Literal["app", "milvus"] = "app"
    stream: bool = True
    use_cache: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_new_tokens: int | None = Field(default=None, ge=16, le=2048)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    role_id: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=6, ge=1, le=50)
    mode: Literal["dense", "sparse", "bm25", "hybrid"] = "hybrid"
    fusion: Literal["app", "milvus"] = "app"
    use_cache: bool = True
    with_text: bool = True


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    role_id: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    roles: list[str] | None = None
    recreate: bool = False


class MemoryClearRequest(BaseModel):
    role_id: str | None = None


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=64)
    roles: list[str] = Field(default_factory=lambda: ["*"])
    is_admin: bool = False


class ApiResponse(BaseModel):
    ok: bool = True
    data: Any = None
    message: str = ""
