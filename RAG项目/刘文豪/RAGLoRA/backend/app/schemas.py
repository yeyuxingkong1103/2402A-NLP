# -*- coding: utf-8 -*-
"""Pydantic 入参 / 出参模型。"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------- 鉴权
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=64)
    display_name: str | None = Field(default=None, max_length=64)


class LoginIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    display_name: str | None = None
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------------------------------------------------------- 角色
class CharacterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    slug: str
    name: str
    category: str | None = None
    avatar: str | None = None
    description: str | None = None
    kb_collection: str
    is_builtin: bool
    # 人格三层（前端编辑角色时需要）
    identity_block: str | None = None
    style_json: dict | None = None
    domain_constraints: str | None = None
    prompt_template: str | None = None
    recall_top_k: int
    rerank_top_k: int
    temperature: float


class CharacterIn(BaseModel):
    slug: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    category: str | None = None
    avatar: str | None = None
    description: str | None = None
    identity_block: str | None = None
    style_json: dict | None = None
    domain_constraints: str | None = None
    prompt_template: str | None = None
    kb_collection: str
    recall_top_k: int = 20
    rerank_top_k: int = 5
    temperature: float = 0.3


class CharacterUpdateIn(BaseModel):
    name: str | None = None
    category: str | None = None
    avatar: str | None = None
    description: str | None = None
    identity_block: str | None = None
    style_json: dict | None = None
    domain_constraints: str | None = None
    prompt_template: str | None = None
    kb_collection: str | None = None
    recall_top_k: int | None = None
    rerank_top_k: int | None = None
    temperature: float | None = None


# ---------------------------------------------------------------- 会话
class ConversationCreateIn(BaseModel):
    character_id: int
    title: str | None = Field(default=None, max_length=128)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    character_id: int
    title: str | None = None
    message_count: int
    last_message_at: datetime | None = None
    created_at: datetime


class ConversationRenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=128)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    rewritten_query: str | None = None
    sources_json: list | None = None
    latency_ms: int | None = None
    created_at: datetime


# ---------------------------------------------------------------- 对话
class ChatIn(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = None


# ---------------------------------------------------------------- 检索
class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    collection: str | None = None          # 不传则检索所有集合
    top_k: int = Field(default=5, ge=1, le=50)
    recall_k: int = Field(default=20, ge=1, le=100)
    use_rerank: bool = True
    filters: dict | None = None            # 如 {"law_name": "民法典"}


# ---------------------------------------------------------------- 知识库
class IngestIn(BaseModel):
    collection: str
    path: str                              # 文件或目录（绝对路径）
    strategy: str = "auto"                 # auto / article / paragraph
    # 文档解析分流：auto=按文本层自动判定扫描件 / force=强制 OCR / off=强制文本层
    ocr: str = "auto"


class KbDocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    collection: str
    source_path: str
    doc_type: str
    chunk_strategy: str | None = None
    chunk_count: int
    status: str
    error_msg: str | None = None
    created_at: datetime


class CollectionOut(BaseModel):
    name: str
    points: int
    characters: list[str] = []
