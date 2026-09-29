from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.app.core.crypto import EncryptedValue


@dataclass
class MemoryRecord:
    # 长期记忆只保存加密后的结构化事实和脱敏摘要，不保存跨会话原文。
    id: str
    user_id: str
    encrypted_facts: EncryptedValue
    encrypted_summary: EncryptedValue
    source_conversation_ids: list[str]
    enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None
    prompt_required: bool = False


@dataclass
class MemoryCandidate:
    # 候选更新先等待用户确认，避免模型自动覆盖重要事实。
    id: str
    memory_id: str
    user_id: str
    new_value: dict[str, Any]
    status: str
    created_at: datetime
    confirmed_at: datetime | None = None


@dataclass
class ExportJob:
    # 导出文件为二次加密 ZIP，下载控制由作业元数据表达。
    id: str
    user_id: str
    encrypted_zip: EncryptedValue
    password_hmac: str
    expires_at: datetime
    max_downloads: int = 3
    download_count: int = 0
    revoked: bool = False
    excluded_from_backup: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
