"""Chat API schemas for Stage 4 RAG question answering."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request schema."""

    query: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    conversation_id: str | None = Field(default=None, description="对话ID，用于多轮对话")
    session_id: str | None = Field(
        default=None,
        description="会话ID。传了就会读写服务端短期记忆（Redis），实现真正的多轮对话；"
        "不传则为无状态单轮问答。会话用 POST /api/v1/sessions 创建。",
    )
    knowledge_base_id: int | None = Field(default=None, ge=1, description="知识库ID过滤")
    tenant_id: int | None = Field(default=None, ge=1, description="租户ID，必须与认证身份一致")

    # Memory options
    enable_memory: bool = Field(
        default=True, description="传了 session_id 时是否读取历史对话作为上下文"
    )
    max_memory_turns: int = Field(
        default=5, ge=1, le=20, description="带入上下文的最大历史轮数"
    )

    # Retrieval options
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")
    enable_rerank: bool = Field(default=True, description="启用重排序")
    enable_query_rewrite: bool = Field(default=True, description="启用查询改写")
    enable_hybrid_search: bool = Field(default=True, description="启用混合检索（向量+BM25）")
    enable_multi_path: bool = Field(default=False, description="启用多路召回")

    # Generation options
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM温度")
    max_length: int = Field(default=500, ge=50, le=2000, description="最大回答长度")
    stream: bool = Field(default=False, description="是否流式输出")

    # Advanced options
    enable_citation: bool = Field(default=True, description="启用来源引用")
    enable_postprocess: bool = Field(default=True, description="启用后处理")


class SourceChunkResponse(BaseModel):
    """来源块响应."""

    chunk_id: str
    text: str
    source: str
    score: float
    summary: str = ""


class ChatMetadata(BaseModel):
    """Chat响应元数据."""

    query_rewritten: str | None = None
    retrieval_count: int = 0
    rerank_enabled: bool = False
    final_chunk_count: int = 0
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    retrieval_sources: dict[str, int] | None = None
    history_turns: int = 0
    memory_enabled: bool = False


class ChatResponse(BaseModel):
    """Chat响应schema."""

    answer: str
    query: str
    conversation_id: str
    session_id: str | None = None
    sources: list[SourceChunkResponse] = []
    metadata: ChatMetadata
    warnings: list[str] = []


class ChatStreamChunk(BaseModel):
    """流式输出的单个块."""

    type: str  # start, content, metadata, done, error
    content: str | None = None
    metadata: dict | None = None
    timestamp: float | None = None
