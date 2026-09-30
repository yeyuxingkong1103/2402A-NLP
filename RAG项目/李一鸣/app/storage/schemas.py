from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RoleCreate(BaseModel):
    """创建角色时的请求字段和长度校验。"""
    name: str = Field(min_length=1, max_length=120)
    category: str = "friend"
    description: str = ""
    personality: str = "温和、诚实、善于倾听"
    expertise: str = "通用陪伴"
    speaking_style: str = "自然、简洁、有同理心"
    safety_policy: str = "遇到高风险问题时明确说明局限并建议咨询专业人士。"
    knowledge_scope: str = "global"


class RoleRead(RoleCreate):
    """角色响应模型，额外返回数据库 ID、系统提示词和时间。"""
    id: str
    system_prompt: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: str
    filename: str
    source: str
    status: str
    chunk_count: int
    error_message: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    """聊天请求；stream=true 时 API 返回 SSE，而不是完整 JSON。"""
    user_id: str = Field(default="demo-user", min_length=1, max_length=120)
    role_id: str
    conversation_id: str = Field(default="default", min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=10000)
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    retrieval_method: str


class ChatResponse(BaseModel):
    """非流式聊天响应，包含答案和引用来源。"""
    answer: str
    role_id: str
    conversation_id: str
    citations: list[Citation]
    query: str
    trace_id: str


class SearchRequest(BaseModel):
    """独立检索接口的请求参数。"""
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=6, ge=1, le=20)
    role_id: str | None = None


class EvaluationSampleRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""


class EvaluationRequest(BaseModel):
    samples: list[EvaluationSampleRequest] = Field(min_length=1)


class EvaluationResponse(BaseModel):
    provider: str
    sample_count: int
    metrics: dict[str, float]
    samples: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    environment: str
    components: dict[str, Any]
