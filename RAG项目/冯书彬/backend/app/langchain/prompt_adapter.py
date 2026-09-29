from typing import Any

from backend.app.llm.base import LlmRequest
from backend.app.rag.pipeline import RetrievalDecision

_SYSTEM_PROMPT = "你是婚姻家事法律信息助手。只能依据给定资料回答，不能编造依据。"
_MAX_CONTEXT_CHARS_PER_FRAGMENT = 800
_MAX_CONTEXT_CHARS_TOTAL = 2400


def minimize_context(context: str) -> str:
    """沿用旧链路的上下文截断边界，避免迁移改变模型输入规模。"""
    fragments = [part.strip()[:_MAX_CONTEXT_CHARS_PER_FRAGMENT] for part in context.split("\n\n") if part.strip()]
    return "\n\n".join(fragments)[:_MAX_CONTEXT_CHARS_TOTAL]


def build_langchain_llm_request(
    text: str,
    decision: RetrievalDecision,
    trace_id: str,
) -> LlmRequest:
    """使用 ChatPromptTemplate 生成现有 DeepSeekClient 所需的 LlmRequest。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise RuntimeError("langchain-core is not installed") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "用户问题：{question}\n\n可用资料摘要：{context}"),
        ]
    )
    prompt_value = prompt.invoke(
        {
            "question": text,
            "context": minimize_context(decision.context),
        }
    )
    messages: list[dict[str, Any]] = []
    for message in prompt_value.messages:
        messages.append({"role": _message_role(message), "content": str(message.content)})
    return LlmRequest(
        messages=messages,
        citations=decision.citations,
        trace_id=trace_id,
        prompt_version="chat-orchestration-langchain-v1",
        citation_validation_passed=True,
    )


def _message_role(message: Any) -> str:
    role = getattr(message, "type", "")
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return "system"
