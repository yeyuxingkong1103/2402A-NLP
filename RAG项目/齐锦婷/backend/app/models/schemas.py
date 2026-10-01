from typing import Any

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=6, max_length=100)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class KnowledgeBaseOut(BaseModel):
    id: int
    name: str
    description: str

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    error_message: str
    chunk_count: int

    model_config = {"from_attributes": True}


class TaskOut(BaseModel):
    id: int
    document_id: int
    status: str
    current_step: str
    error_message: str

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    knowledge_base_id: int
    question: str = Field(min_length=1)
    session_id: int | None = None


class ReferenceOut(BaseModel):
    index: int
    document_id: int
    filename: str
    chunk_uid: str
    content: str
    score: float
    page_number: int
    title_path: str


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    references: list[ReferenceOut]


class ApiMessage(BaseModel):
    message: str
    data: Any | None = None
