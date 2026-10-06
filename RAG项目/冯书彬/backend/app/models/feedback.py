from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Feedback:
    # 反馈只关联用户和消息，不保存额外聊天原文。
    id: str
    user_id: str
    message_id: str
    rating: str
    risk_level: str
    category: str | None = None
    reason: str | None = None
    alert_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeedbackAlert:
    # 高风险反馈生成后台告警，元数据必须脱敏后保存。
    id: str
    feedback_id: str
    risk_level: str
    category: str
    metadata: dict
    notification_sent: bool
    created_at: datetime = field(default_factory=datetime.utcnow)
