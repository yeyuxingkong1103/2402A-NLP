import logging
from dataclasses import dataclass
from typing import Any

from backend.app.services.audit_service import _REDACTED_VALUE
from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdminAlert:
    # 告警只携带反馈定位信息和脱敏元数据，不携带聊天原文。
    feedback_id: str
    risk_level: str
    category: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NotificationResult:
    # MVP 阶段模拟邮件发送，保留发送结果供测试和后台追踪。
    sent: bool
    channel: str
    metadata: dict[str, Any]


def _sanitize_alert_value(key: str, value: Any) -> Any:
    # 敏感字段按 key 直接替换，避免完整手机号或聊天内容进入邮件。
    if key.lower() in {"phone", "token", "code", "answer", "prompt", "chat_text"}:
        return _REDACTED_VALUE
    if isinstance(value, dict):
        return {str(child_key): _sanitize_alert_value(str(child_key), child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_alert_value(key, item) for item in value]
    if isinstance(value, str):
        return redact_pii(value).text
    return value


def _sanitize_alert_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # 逐字段脱敏，返回新的 dict，避免修改调用方对象。
    return {str(key): _sanitize_alert_value(str(key), value) for key, value in metadata.items()}


def send_admin_alert(alert: AdminAlert) -> NotificationResult:
    # 内部测试版不真正发邮件，只记录模拟发送成功和脱敏后的元数据。
    safe_metadata = _sanitize_alert_metadata(alert.metadata)
    logger.warning(
        "high risk feedback alert prepared",
        extra={"feedback_id": alert.feedback_id, "risk_level": alert.risk_level, "category": alert.category},
    )
    return NotificationResult(sent=True, channel="email", metadata=safe_metadata)
