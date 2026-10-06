"""ORM 模型统一导出。

把分散在各 model 文件中的模型集中在此导出，外部只需
`from src.models import User, Message` 即可，无需关心具体模块路径。
另外，只有这些模型类被导入后，SQLAlchemy 才会把它们注册到 Base.metadata，
从而支持建表/迁移。
"""
# 声明式基类：所有模型共享，也用于 metadata.create_all 建表
from src.models.base import Base
# 会话与消息
from src.models.conversation import Conversation, Message
# 知识库文档与分块
from src.models.knowledge import KnowledgeChunk, KnowledgeDoc
# 心理医生人设
from src.models.persona import CounselorPersona
# 用户/角色/偏好/日志
from src.models.user import (AuditLog, LoginLog, SysRole, User, UserPersonaPreference,
                             UserSysRole)

# __all__ 声明 `from src.models import *` 时导出的名字，明确对外公开的接口
__all__ = [
    "Base",
    "User",
    "SysRole",
    "UserSysRole",
    "UserPersonaPreference",
    "AuditLog",
    "LoginLog",
    "CounselorPersona",
    "Conversation",
    "Message",
    "KnowledgeDoc",
    "KnowledgeChunk",
]