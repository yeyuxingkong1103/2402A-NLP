from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    # 用户只提交消息 ID、评价方向和可选原因，不允许指定风险等级。
    message_id: str = Field(min_length=1, max_length=128)
    rating: Literal["up", "down", "report"]
    category: Literal[
        "wrong_legal_basis",
        "not_helpful",
        "citation_unavailable",
        "fact_misunderstood",
        "concluded_with_insufficient_info",
        "unsafe_guidance",
        "privacy_leak",
        "high_risk_handling_error",
        "other",
    ] | None = None
    reason: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    rating: str
    category: str | None
    reason: str | None
    risk_level: str
    alert_id: str | None
