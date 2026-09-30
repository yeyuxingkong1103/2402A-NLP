from dataclasses import dataclass, field


@dataclass
class Message:
    # 消息模型保存脱敏后的用户文本摘要和回答状态，避免日志依赖原文。
    id: str
    conversation_id: str
    user_id: str
    user_text: str
    answer: str = ""
    status: str = "created"
    citations: list[dict] = field(default_factory=list)
    regeneration_count: int = 0
    corrected: bool = False
    source_message_id: str | None = None
