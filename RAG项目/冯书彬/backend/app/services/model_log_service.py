import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)

_BLOCKED_FIELDS = {
    "api_key",
    "answer",
    "chat",
    "chat_text",
    "context",
    "full_context",
    "messages",
    "prompt",
    "raw_chat",
    "retrieval_context",
}
# MVP 限制：当前模型调用日志只保存在进程内存，重启会丢失，不提供生产删除接口。
# 后续接入数据库时仍必须沿用 sanitize_model_metadata 的字段白名单和脱敏边界。
_model_call_logs: list[dict[str, Any]] = []
LogReasonCode = Literal["controlled_unavailable", "transport_or_stream_error", "unknown"]


@dataclass(frozen=True)
class ModelCallMetadata:
    # provider 和 model 用于统计模型供应商与模型名称，不包含请求正文。
    provider: str
    model: str
    # prompt_version 只记录模板版本，不能记录完整 Prompt。
    prompt_version: str
    # knowledge_base_version 只记录知识库版本，不能记录完整检索上下文。
    knowledge_base_version: str
    # token_count 由调用方或客户端估算，仅用于成本与异常分析。
    token_count: int
    # duration_ms 记录端到端模型调用耗时。
    duration_ms: int
    # status 记录 success 或 failure 等状态。
    status: str
    # error_type 只记录异常类别，不记录异常中的敏感正文。
    error_type: str | None
    # trace_id 用于关联请求链路，必须由上游保证不含用户原文。
    trace_id: str
    # model_version 可独立于 model 记录具体版本。
    model_version: str | None = None
    # redaction_applied 仅记录是否做过脱敏。
    redaction_applied: bool = False
    # memory_used 仅记录是否使用长期记忆。
    memory_used: bool = False
    # case_citation_used 仅记录是否出现案例引用。
    case_citation_used: bool = False
    # citation_validation_passed 仅记录引用校验是否通过。
    citation_validation_passed: bool = False
    # reason_code 只能是短代码，禁止使用自由文本记录 Prompt、回答或上下文。
    reason_code: LogReasonCode = "unknown"
    # 以下兼容字段用于防御性过滤，sanitize 时绝不输出。
    prompt: str | None = None
    answer: str | None = None
    api_key: str | None = None
    context: str | None = None
    messages: list[dict[str, Any]] | None = None


def _is_blocked_field(key: str) -> bool:
    # 采用包含匹配，覆盖 deepseek_api_key、full_prompt 等派生字段。
    normalized_key = key.lower()
    return any(blocked in normalized_key for blocked in _BLOCKED_FIELDS)


def _sanitize_value(value: Any) -> Any:
    # 字符串只保留 PII 脱敏后的低敏内容。
    if isinstance(value, str):
        return redact_pii(value).text
    # 列表递归处理，避免安全备注数组中夹带手机号等 PII。
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    # 字典递归处理，并同步过滤敏感键。
    if isinstance(value, dict):
        return {str(key): _sanitize_value(child) for key, child in value.items() if not _is_blocked_field(str(key))}
    # 其他标量直接返回，日志层不做业务转换。
    return value


def sanitize_model_metadata(metadata: ModelCallMetadata) -> dict[str, Any]:
    # dataclass 转字典后统一过滤，保证新增字段默认走同一套安全规则。
    raw_metadata = asdict(metadata)
    sanitized: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        # None 字段不写入日志，减少无意义数据面。
        if value is None:
            continue
        # 完整 Prompt、回答、上下文、API Key 和聊天正文一律丢弃。
        if _is_blocked_field(key):
            continue
        sanitized[key] = _sanitize_value(value)
    return sanitized


def record_model_call(metadata: ModelCallMetadata) -> dict[str, Any]:
    # 仅追加脱敏后的元数据，内存列表用于当前 MVP 的单元测试和本地审计。
    sanitized = sanitize_model_metadata(metadata)
    _model_call_logs.append(sanitized)
    logger.info(
        "模型调用元数据已记录",
        extra={
            "provider": sanitized.get("provider"),
            "model": sanitized.get("model"),
            "status": sanitized.get("status"),
            "error_type": sanitized.get("error_type"),
            "trace_id": sanitized.get("trace_id"),
            "token_count": sanitized.get("token_count"),
            "duration_ms": sanitized.get("duration_ms"),
        },
    )
    return sanitized


def list_model_call_logs() -> tuple[dict[str, Any], ...]:
    # 返回浅拷贝元组，避免调用方直接修改内部追加列表。
    return tuple(dict(item) for item in _model_call_logs)
