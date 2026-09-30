from dataclasses import dataclass, field


@dataclass
class Conversation:
    # 会话只保存编排所需的短期状态，不实现长期记忆。
    id: str
    user_id: str
    follow_up_count: int = 0
    message_ids: list[str] = field(default_factory=list)
    active: bool = True
