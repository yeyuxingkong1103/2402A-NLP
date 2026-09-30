from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AuditLog:
    # 审计日志为不可变数据对象，避免业务层修改历史记录。
    id: str
    # action 按字符串保存，便于后续数据库索引和跨语言查询。
    action: str
    # actor_id 仅保存内部用户 ID，不保存手机号等敏感身份信息。
    actor_id: str
    # target_type 描述被操作资源类型，例如 user、knowledge_base。
    target_type: str
    # target_id 保存资源内部 ID，不保存聊天原文或外部密钥。
    target_id: str
    # metadata 必须由服务层先脱敏再写入。
    metadata: dict[str, Any]
    # created_at 记录追加时间，审计日志不提供更新入口。
    created_at: datetime
