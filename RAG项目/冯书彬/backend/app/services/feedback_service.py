import logging
from typing import Protocol
from uuid import uuid4

from backend.app.api.v1.chat import chat_service
from backend.app.core.security import utc_now
from backend.app.models.feedback import Feedback, FeedbackAlert
from backend.app.services.notification_service import AdminAlert, send_admin_alert
from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)

_VALID_RATINGS = {"up", "down", "report"}
_VALID_CATEGORIES = {
    "wrong_legal_basis",
    "not_helpful",
    "citation_unavailable",
    "fact_misunderstood",
    "concluded_with_insufficient_info",
    "unsafe_guidance",
    "privacy_leak",
    "high_risk_handling_error",
    "other",
}
_HIGH_RISK_CATEGORIES = {"wrong_legal_basis", "unsafe_guidance", "privacy_leak", "high_risk_handling_error"}
_MAX_REASON_CHARS = 500
_feedback_store: dict[str, Feedback] = {}
_alert_store: dict[str, FeedbackAlert] = {}
_repository: "FeedbackRepository | None" = None


class FeedbackRepository(Protocol):
    # 反馈服务只依赖最小持久化操作，默认仍可使用内存 store。
    def save_feedback(self, feedback: Feedback) -> None: ...
    def save_alert(self, alert: FeedbackAlert) -> None: ...
    def get_feedback(self, feedback_id: str) -> Feedback | None: ...
    def get_alert(self, alert_id: str) -> FeedbackAlert | None: ...
    def delete_user_data(self, user_id: str) -> dict[str, int]: ...


class FeedbackServiceError(ValueError):
    # 服务层统一抛业务错误，API 层按 status_code 映射。
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def set_feedback_repository(repository: FeedbackRepository | None) -> None:
    # 生产启动时显式注入 SQL repository；测试默认保留内存行为。
    global _repository
    _repository = repository


def reset_feedback_store_for_tests() -> None:
    # 测试专用清理入口，避免跨用例共享反馈和告警。
    _feedback_store.clear()
    _alert_store.clear()
    set_feedback_repository(None)


def delete_feedback_data_for_user(user_id: str) -> dict[str, int]:
    # 注销时先删除用户反馈，再删除其关联告警，避免留下可回溯的个人数据。
    if _repository is not None:
        deleted = _repository.delete_user_data(user_id)
        logger.info(
            "user feedback data deleted",
            extra={"user_id": user_id, "deleted_feedback": deleted["feedback"], "deleted_alerts": deleted["alerts"]},
        )
        return deleted
    feedback_ids = {feedback.id for feedback in _feedback_store.values() if feedback.user_id == user_id}
    deleted_feedback = sum(1 for feedback_id in feedback_ids if _feedback_store.pop(feedback_id, None) is not None)
    deleted_alerts = sum(
        1
        for alert_id, alert in list(_alert_store.items())
        if alert.feedback_id in feedback_ids and _alert_store.pop(alert_id, None) is not None
    )
    logger.info(
        "user feedback data deleted",
        extra={"user_id": user_id, "deleted_feedback": deleted_feedback, "deleted_alerts": deleted_alerts},
    )
    return {"feedback": deleted_feedback, "alerts": deleted_alerts}


def classify_feedback_risk(category: str | None, reason: str | None) -> str:
    # 高风险分类由枚举优先决定，原因关键词兜底识别安全处置错误。
    normalized = (category or "other").strip()
    if normalized in _HIGH_RISK_CATEGORIES:
        return "high"
    reason_text = reason or ""
    if any(keyword in reason_text for keyword in ["危险", "自杀", "家暴", "报警", "隐私泄露", "手机号泄露"]):
        return "high"
    return "normal"


def _validate_feedback_input(rating: str, category: str | None) -> str | None:
    # 评价方向必须在受控集合内，避免脏数据进入后台统计。
    if rating not in _VALID_RATINGS:
        raise FeedbackServiceError("不支持的反馈类型", status_code=400)
    if category is None:
        return None
    normalized = category.strip()
    if normalized not in _VALID_CATEGORIES:
        raise FeedbackServiceError("不支持的反馈分类", status_code=400)
    return normalized


def _get_owned_message(user_id: str, message_id: str):
    # 反馈只能针对本人已完成的 AI 回答，且不存在与越权统一返回，避免枚举消息。
    message = chat_service.messages.get(message_id)
    if message is None or message.user_id != user_id:
        raise FeedbackServiceError("消息不存在或无权反馈", status_code=404)
    if message.status not in {"answered", "corrected"} or not message.answer:
        raise FeedbackServiceError("该消息暂不可反馈", status_code=409)
    return message


def _redact_reason(reason: str | None) -> str | None:
    # 原因可能包含 PII 或聊天原文，只保留有限长度的脱敏内容。
    if reason is None:
        return None
    trimmed = reason.strip()
    if len(trimmed) > _MAX_REASON_CHARS:
        raise FeedbackServiceError("反馈原因不能超过500个字符", status_code=400)
    return redact_pii(trimmed).text


def _save_feedback(feedback: Feedback) -> None:
    if _repository is not None:
        _repository.save_feedback(feedback)
    else:
        _feedback_store[feedback.id] = feedback


def _save_alert(alert: FeedbackAlert) -> None:
    if _repository is not None:
        _repository.save_alert(alert)
    else:
        _alert_store[alert.id] = alert


def _create_high_risk_alert(feedback: Feedback) -> FeedbackAlert:
    # 高风险反馈生成后台告警，邮件通知只携带脱敏元数据。
    alert_id = f"alert-{uuid4().hex}"
    metadata = {"message_id": feedback.message_id, "category": feedback.category, "reason": feedback.reason}
    result = send_admin_alert(AdminAlert(feedback.id, feedback.risk_level, feedback.category or "other", metadata))
    alert = FeedbackAlert(
        id=alert_id,
        feedback_id=feedback.id,
        risk_level=feedback.risk_level,
        category=feedback.category or "other",
        metadata=result.metadata,
        notification_sent=result.sent,
        created_at=utc_now(),
    )
    _save_alert(alert)
    feedback.alert_id = alert.id
    _save_feedback(feedback)
    return alert


def submit_feedback(user_id: str, message_id: str, rating: str, reason: str | None, category: str | None) -> Feedback:
    # 提交反馈时先校验消息归属，再分类风险和触发必要告警。
    _get_owned_message(user_id, message_id)
    normalized_category = _validate_feedback_input(rating, category)
    safe_reason = _redact_reason(reason)
    risk_level = classify_feedback_risk(normalized_category, safe_reason)
    feedback = Feedback(
        id=f"fb-{uuid4().hex}",
        user_id=user_id,
        message_id=message_id,
        rating=rating,
        category=normalized_category,
        reason=safe_reason,
        risk_level=risk_level,
        created_at=utc_now(),
    )
    _save_feedback(feedback)
    if risk_level == "high":
        _create_high_risk_alert(feedback)
    logger.info("feedback submitted", extra={"user_id": user_id, "message_id": message_id, "feedback_id": feedback.id, "risk_level": risk_level})
    return feedback


def get_feedback(feedback_id: str) -> Feedback | None:
    return _repository.get_feedback(feedback_id) if _repository is not None else _feedback_store.get(feedback_id)


def get_alert(alert_id: str) -> FeedbackAlert | None:
    return _repository.get_alert(alert_id) if _repository is not None else _alert_store.get(alert_id)
