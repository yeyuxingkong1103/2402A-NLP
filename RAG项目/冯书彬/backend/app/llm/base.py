from dataclasses import dataclass, field
from typing import Any

from backend.app.schemas.retrieval import Citation


@dataclass(frozen=True)
class LlmRequest:
    # messages 是 OpenAI 兼容聊天消息；调用方负责在进入模型前完成业务编排。
    messages: list[dict[str, Any]]
    # citations 只携带引用元数据，DeepSeek 客户端不会主动检索或拼接 RAG 上下文。
    citations: list[Citation] = field(default_factory=list)
    # trace_id 用于串联脱敏日志和模型调用元数据，不包含用户原文。
    trace_id: str = ""
    # prompt_version 记录模板版本，便于后续审计模型行为。
    prompt_version: str = "unknown"
    # knowledge_base_version 记录知识库快照版本，不记录完整检索上下文。
    knowledge_base_version: str = "unknown"
    # model_version 可由上游指定具体模型版本，缺省由客户端使用模型名兜底。
    model_version: str | None = None
    # memory_used 仅表示是否使用长期记忆，不携带记忆正文。
    memory_used: bool = False
    # citation_validation_passed 仅表示引用校验结果，不携带回答正文。
    citation_validation_passed: bool = False
