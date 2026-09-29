from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScopeResult:
    # 范围判断只给出业务状态，不回显用户原文。
    in_scope: bool
    reason: str


@dataclass(frozen=True)
class ChatRequest:
    # API 请求体限制为用户标识和本轮文本。
    user_id: str
    text: str


@dataclass(frozen=True)
class RegenerateRequest:
    # 重新生成只允许原用户触发。
    user_id: str


@dataclass(frozen=True)
class FactCorrection:
    # confirmed=True 表示用户确认使用更正事实重新生成。
    message_id: str
    correction: str
    confirmed: bool
    user_id: str = ""


@dataclass(frozen=True)
class ChatResult:
    # ChatResult 是服务层和 API 的统一输出结构。
    status: str
    conversation_id: str
    message_id: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    follow_up_count: int = 0
    reason: str = ""
