import json
import logging
import uuid
from typing import Any, Protocol

from backend.app.core.crypto import EncryptedValue, decrypt_text, encrypt_text
from backend.app.core.security import utc_now
from backend.app.models.memory import MemoryCandidate, MemoryRecord
from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)
_memories: dict[str, MemoryRecord] = {}
_candidates: dict[str, MemoryCandidate] = {}
_auto_extract_enabled: dict[str, bool] = {}
_repository: "MemoryRepository | None" = None


class MemoryRepository(Protocol):
    # 长期记忆服务只依赖这组最小持久化操作，默认仍可退回内存实现。
    def save_memory(self, memory: MemoryRecord) -> None: ...
    def get_memory(self, memory_id: str, user_id: str) -> MemoryRecord | None: ...
    def get_memory_by_id(self, memory_id: str) -> MemoryRecord | None: ...
    def list_memories(self, user_id: str) -> list[MemoryRecord]: ...
    def delete_memory(self, memory_id: str, user_id: str) -> bool: ...
    def set_auto_extract_enabled(self, user_id: str, enabled: bool) -> None: ...
    def is_auto_extract_enabled(self, user_id: str) -> bool: ...
    def save_candidate(self, candidate: MemoryCandidate) -> None: ...
    def get_candidate(self, candidate_id: str) -> MemoryCandidate | None: ...
    def delete_memory_data(self, user_id: str) -> dict[str, int]: ...


def set_memory_repository(repository: MemoryRepository | None) -> None:
    # 生产启动时显式注入 SQL repository；测试默认不注入以保持内存路径稳定。
    global _repository
    _repository = repository


def reset_memory_store_for_tests() -> None:
    # 测试专用清理入口，避免跨用例共享长期记忆状态。
    _memories.clear()
    _candidates.clear()
    _auto_extract_enabled.clear()
    set_memory_repository(None)


def is_memory_auto_extraction_enabled(user_id: str) -> bool:
    # 长期记忆默认启用，用户关闭后只停止新增抽取。
    if _repository is not None:
        return _repository.is_auto_extract_enabled(user_id)
    return _auto_extract_enabled.get(user_id, True)


def set_memory_auto_extraction(user_id: str, enabled: bool) -> bool:
    # 只修改用户偏好，不删除既有记忆。
    if _repository is not None:
        _repository.set_auto_extract_enabled(user_id, enabled)
    else:
        _auto_extract_enabled[user_id] = enabled
    logger.info("memory auto extraction changed", extra={"user_id": user_id, "enabled": enabled})
    return enabled


def _encrypted_value_to_dict(value: EncryptedValue) -> dict[str, str]:
    # 导出时只暴露加密元数据，不把密钥或原始明文写入日志。
    return {
        "ciphertext": value.ciphertext,
        "nonce": value.nonce,
        "encrypted_data_key": value.encrypted_data_key,
        "key_version": value.key_version,
    }


def _redact_facts(value: Any) -> Any:
    # 结构化事实写入前递归脱敏，避免保存完整手机号等 PII。
    if isinstance(value, str):
        return redact_pii(value).text
    if isinstance(value, dict):
        return {str(key): _redact_facts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_facts(item) for item in value]
    return value


def _serialize_facts(facts: dict[str, Any]) -> str:
    # 结构化事实稳定序列化，便于加密前后测试和审计。
    return json.dumps(_redact_facts(facts), ensure_ascii=False, sort_keys=True)


def create_memory_from_conversation(user_id: str, conversation_id: str, facts: dict[str, Any], summary: str) -> MemoryRecord | None:
    # 用户关闭自动抽取后不创建新记忆，但保留既有记忆。
    if not is_memory_auto_extraction_enabled(user_id):
        logger.info("memory creation skipped by user preference", extra={"user_id": user_id, "conversation_id": conversation_id})
        return None
    # 摘要再次脱敏，防止调用方误传入手机号等个人信息。
    redacted_summary = redact_pii(summary).text
    # 仅加密结构化事实，不存完整聊天原文。
    encrypted_facts = encrypt_text(_serialize_facts(facts), purpose="memory-facts")
    encrypted_summary = encrypt_text(redacted_summary, purpose="memory-summary")
    now = utc_now()
    first_for_user = len(list_user_memories(user_id)) == 0
    memory = MemoryRecord(
        id=f"mem-{uuid.uuid4().hex}",
        user_id=user_id,
        encrypted_facts=encrypted_facts,
        encrypted_summary=encrypted_summary,
        source_conversation_ids=[conversation_id],
        enabled=True,
        version=1,
        created_at=now,
        updated_at=now,
        prompt_required=first_for_user,
    )
    if _repository is not None:
        _repository.save_memory(memory)
    else:
        _memories[memory.id] = memory
    logger.info("memory created", extra={"user_id": user_id, "conversation_id": conversation_id, "memory_id": memory.id, "fact_keys": sorted(facts.keys())})
    return memory


def list_user_memories(user_id: str) -> list[MemoryRecord]:
    # 返回用户自己的未删除记忆，调用方不能跨用户读取。
    if _repository is not None:
        return _repository.list_memories(user_id)
    return [memory for memory in _memories.values() if memory.user_id == user_id]


def get_memory(memory_id: str, user_id: str) -> MemoryRecord | None:
    # 读取时校验归属，避免 ID 枚举泄露。
    if _repository is not None:
        return _repository.get_memory(memory_id, user_id)
    memory = _memories.get(memory_id)
    if memory is None or memory.user_id != user_id:
        return None
    return memory


def update_memory(memory_id: str, user_id: str, facts: dict[str, Any], summary: str) -> MemoryRecord | None:
    # 用户编辑后直接替换加密内容，并递增版本。
    memory = get_memory(memory_id, user_id)
    if memory is None:
        return None
    memory.encrypted_facts = encrypt_text(_serialize_facts(facts), purpose="memory-facts")
    memory.encrypted_summary = encrypt_text(redact_pii(summary).text, purpose="memory-summary")
    memory.version += 1
    memory.updated_at = utc_now()
    if _repository is not None:
        _repository.save_memory(memory)
    logger.info("memory updated", extra={"user_id": user_id, "memory_id": memory_id, "fact_keys": sorted(facts.keys())})
    return memory


def delete_memory(memory_id: str, user_id: str) -> bool:
    # 手动删除只允许删除本人记忆。
    if _repository is not None:
        deleted = _repository.delete_memory(memory_id, user_id)
        if deleted:
            logger.info("memory deleted", extra={"user_id": user_id, "memory_id": memory_id})
        return deleted
    memory = get_memory(memory_id, user_id)
    if memory is None:
        return False
    _memories.pop(memory_id, None)
    logger.info("memory deleted", extra={"user_id": user_id, "memory_id": memory_id})
    return True


def delete_exclusive_memories_for_conversation(user_id: str, conversation_id: str) -> list[str]:
    # 只删除唯一来源为该会话的记忆；多会话共享记忆必须保留。
    deleted: list[str] = []
    for memory in list_user_memories(user_id):
        if memory.source_conversation_ids == [conversation_id] and delete_memory(memory.id, user_id):
            deleted.append(memory.id)
    logger.info("exclusive memories deleted", extra={"user_id": user_id, "conversation_id": conversation_id, "deleted_count": len(deleted)})
    return deleted


def delete_all_memories_for_user(user_id: str) -> int:
    # 账号注销时删除所有可关联长期记忆。
    if _repository is not None:
        deleted = _repository.delete_memory_data(user_id).get("memories", 0)
        logger.info("user memories deleted", extra={"user_id": user_id, "deleted_count": deleted})
        return deleted
    deleted = 0
    for memory_id, memory in list(_memories.items()):
        if memory.user_id == user_id:
            _memories.pop(memory_id, None)
            deleted += 1
    logger.info("user memories deleted", extra={"user_id": user_id, "deleted_count": deleted})
    return deleted


def mark_candidate_update(memory_id: str, new_value: dict[str, Any]) -> MemoryCandidate:
    # 候选更新只记录脱敏后的结构化新值，等待用户确认。
    memory = _repository.get_memory_by_id(memory_id) if _repository is not None else _memories.get(memory_id)
    if memory is None:
        raise ValueError("记忆不存在")
    redacted_value = _redact_facts(new_value)
    candidate = MemoryCandidate(id=f"cand-{uuid.uuid4().hex}", memory_id=memory_id, user_id=memory.user_id, new_value=redacted_value, status="pending", created_at=utc_now())
    if _repository is not None:
        _repository.save_candidate(candidate)
    else:
        _candidates[candidate.id] = candidate
    logger.info("memory candidate created", extra={"user_id": memory.user_id, "memory_id": memory_id, "candidate_id": candidate.id, "keys": sorted(redacted_value.keys())})
    return candidate


def confirm_memory_update(candidate_id: str, user_id: str) -> MemoryRecord:
    # 确认时才把候选事实写回长期记忆。
    candidate = _repository.get_candidate(candidate_id) if _repository is not None else _candidates.get(candidate_id)
    if candidate is None:
        raise ValueError("候选记忆不存在")
    if candidate.user_id != user_id:
        raise ValueError("候选记忆不属于该用户")
    memory = get_memory(candidate.memory_id, user_id)
    if memory is None:
        raise ValueError("记忆不存在")
    summary = str(candidate.new_value.get("summary", "已更新的案件事实摘要"))
    facts = dict(candidate.new_value.get("facts", candidate.new_value))
    updated = update_memory(memory.id, user_id, facts, summary)
    if updated is None:
        raise ValueError("记忆不存在")
    candidate.status = "confirmed"
    candidate.confirmed_at = utc_now()
    if _repository is not None:
        _repository.save_candidate(candidate)
    logger.info("memory candidate confirmed", extra={"user_id": user_id, "memory_id": memory.id, "candidate_id": candidate_id})
    return updated


def memory_to_export_dict(memory: MemoryRecord) -> dict[str, Any]:
    # 用户导出包含可读的结构化事实和脱敏摘要，不包含密钥。
    return {
        "id": memory.id,
        "facts": json.loads(decrypt_text(memory.encrypted_facts, purpose="memory-facts")),
        "summary": decrypt_text(memory.encrypted_summary, purpose="memory-summary"),
        "source_conversation_ids": list(memory.source_conversation_ids),
        "enabled": memory.enabled,
        "version": memory.version,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
    }
