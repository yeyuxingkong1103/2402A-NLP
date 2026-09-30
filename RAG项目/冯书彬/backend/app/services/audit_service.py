import logging
from copy import deepcopy
from datetime import timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from backend.app.core.security import utc_now
from backend.app.models.audit_log import AuditLog
from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {
    "answer",
    "chat",
    "chat_text",
    "ciphertext",
    "code",
    "encrypted",
    "encrypted_data_key",
    "encrypted_payload",
    "encrypted_value",
    "key",
    "phone",
    "prompt",
    "raw_chat",
    "secret",
    "token",
}
_REDACTED_VALUE = "[已脱敏]"
_audit_logs: list[AuditLog] = []


class AuditAction(str, Enum):
    # 管理员登录后台。
    ADMIN_LOGIN = "ADMIN_LOGIN"
    # 知识库内容审核通过前的审核动作。
    KB_REVIEW = "KB_REVIEW"
    # 知识库内容发布上线。
    KB_PUBLISH = "KB_PUBLISH"
    # 知识库内容审核拒绝。
    KB_REJECT = "KB_REJECT"
    # 知识库内容废弃下线。
    KB_DEPRECATE = "KB_DEPRECATE"
    # 来源白名单变更。
    WHITELIST_CHANGE = "WHITELIST_CHANGE"
    # 知识库抓取快照创建。
    KB_SNAPSHOT_CREATE = "KB_SNAPSHOT_CREATE"
    # 知识库材料提交审核。
    KB_SUBMIT_REVIEW = "KB_SUBMIT_REVIEW"
    # 用户或系统数据导出。
    DATA_EXPORT = "DATA_EXPORT"
    # 完整会话访问，属于高敏操作。
    FULL_CONVERSATION_ACCESS = "FULL_CONVERSATION_ACCESS"
    # 用户数据删除。
    DATA_DELETE = "DATA_DELETE"


def _is_sensitive_key(key: str) -> bool:
    # 统一小写匹配，覆盖 phone_number、api_token 等组合字段。
    normalized = key.lower()
    return any(marker in normalized for marker in _SENSITIVE_KEYS)


def _redact_string(value: str) -> str:
    # 文本先走 PII 脱敏，命中手机号等格式时保留低敏上下文。
    result = redact_pii(value)
    return result.text


def _sanitize_metadata_value(key: str, value: Any) -> Any:
    # 敏感 key 的值直接替换，不保留密钥、验证码、聊天原文等内容。
    if _is_sensitive_key(key):
        return _REDACTED_VALUE
    # 字典递归处理，确保嵌套字段同样脱敏。
    if isinstance(value, dict):
        return {str(child_key): _sanitize_metadata_value(str(child_key), child_value) for child_key, child_value in value.items()}
    # 列表逐项处理，列表项没有字段名时使用父字段名判断敏感度。
    if isinstance(value, list):
        return [_sanitize_metadata_value(key, item) for item in value]
    # 普通字符串只替换手机号、身份证、银行卡等可识别 PII。
    if isinstance(value, str):
        return _redact_string(value)
    # 其他 JSON 标量不含文本敏感模式，直接返回深拷贝后的值。
    return value


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # 深拷贝输入，避免调用方后续修改影响审计快照。
    snapshot = deepcopy(metadata)
    return {str(key): _sanitize_metadata_value(str(key), value) for key, value in snapshot.items()}


def record_audit(action: AuditAction, actor_id: str, target_type: str, target_id: str, metadata: dict[str, Any]) -> AuditLog:
    # 审计记录只追加，不提供更新或删除入口。
    safe_metadata = _sanitize_metadata(metadata)
    log = AuditLog(
        id=str(uuid4()),
        action=action.value,
        actor_id=actor_id,
        target_type=target_type,
        target_id=target_id,
        metadata=safe_metadata,
        created_at=utc_now().astimezone(timezone.utc),
    )
    _audit_logs.append(log)
    logger.info(
        "audit log appended",
        extra={
            "action": log.action,
            "actor": actor_id,
            "target_type": target_type,
            "target_id": target_id,
            "metadata_keys": sorted(safe_metadata.keys()),
        },
    )
    return log


def list_audit_logs() -> tuple[AuditLog, ...]:
    # 返回不可变元组，避免外部直接修改内部追加列表。
    return tuple(_audit_logs)
