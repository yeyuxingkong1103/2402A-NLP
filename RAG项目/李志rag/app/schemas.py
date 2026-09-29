from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    is_admin: bool


class UserResponse(BaseModel):
    id: int
    username: str
    is_admin: bool
    model_config = ConfigDict(from_attributes=True)


class RoleCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    description: str = Field(default="", max_length=500)
    system_prompt: str = Field(min_length=20, max_length=8000)


class RoleResponse(RoleCreate):
    id: int
    is_public: bool
    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: int
    owner_id: int
    role_id: int
    filename: str
    source_url: str
    summary: str
    status: str
    is_public: bool
    chunk_count: int
    error_message: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class CatalogItem(BaseModel):
    document_id: int
    category: str
    title: str
    summary: str
    questions: list[str]
    filename: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    role_id: int
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: int
    text: str
    source: str
    score: float


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    role_id: int
    session_id: str = Field(default="default", min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchHit]
    session_id: str
