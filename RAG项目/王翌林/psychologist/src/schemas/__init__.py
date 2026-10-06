"""Schema 统一导出（包入口模块）。

本模块把散落在 7 个子模块里的 Pydantic 模型集中“搬”到 src.schemas 这一层，
这样其他代码只需写 `from src.schemas import ChatRequest`，
而不必关心它究竟定义在哪个具体文件里（降低调用方与文件结构的耦合）。

约定：
- 只做导入与再导出，不在此处定义任何模型或业务逻辑；
- `__all__` 显式声明本包对外暴露的名字，作用有两个：
  1) 让 `from src.schemas import *` 只导入这里的名字；
  2) 给 IDE / 静态检查工具一份权威的公开 API 清单。
"""
# 鉴权相关：注册/登录/刷新令牌/令牌响应/当前用户信息
from src.schemas.auth import (LoginRequest, RefreshRequest, RegisterRequest,
                              TokenResponse, UserInfo)
# 通用响应包装：统一响应体、分页结果、健康检查结果
from src.schemas.common import ApiResponse, HealthResult, PageResult
# 会话与消息：会话创建/会话条目/消息条目/聊天请求与响应/引用来源
from src.schemas.conversation import (ChatReference, ChatRequest, ChatResponse,
                                      ConversationCreateRequest, ConversationItem,
                                      MessageItem, MessageListResult)
# 知识库：文档条目/上传结果/重建索引/检索请求与命中/RAG 评测
from src.schemas.knowledge import (KnowledgeDocItem, KnowledgeRebuildRequest,
                                   KnowledgeSearchHit, KnowledgeSearchRequest,
                                   KnowledgeSearchResult, KnowledgeUploadResult,
                                   RagasEvalRequest, RagasEvalResult)
# 心理医生角色（persona）：角色基础信息/详情/增改请求/列表结果
from src.schemas.persona import (PersonaBase, PersonaCreateRequest, PersonaDetail,
                                 PersonaListResult, PersonaUpdateRequest)
# 用户管理：资料更新/改密码/启停状态/用户列表/偏好设置
from src.schemas.user import (PasswordChangeRequest, PreferenceItem, PreferenceRequest,
                              UserListItem, UserStatusRequest, UserUpdateRequest)

# 对外公开的名字清单，按“通用 → 鉴权 → 用户 → 角色 → 会话 → 知识库”的逻辑分组排列
__all__ = [
    "ApiResponse", "PageResult", "HealthResult",
    "RegisterRequest", "LoginRequest", "RefreshRequest", "TokenResponse", "UserInfo",
    "UserUpdateRequest", "PasswordChangeRequest", "UserStatusRequest", "UserListItem",
    "PreferenceRequest", "PreferenceItem",
    "PersonaBase", "PersonaDetail", "PersonaCreateRequest", "PersonaUpdateRequest",
    "PersonaListResult",
    "ConversationCreateRequest", "ConversationItem", "MessageItem", "MessageListResult",
    "ChatRequest", "ChatResponse", "ChatReference",
    "KnowledgeDocItem", "KnowledgeUploadResult", "KnowledgeRebuildRequest",
    "KnowledgeSearchRequest", "KnowledgeSearchHit", "KnowledgeSearchResult",
    "RagasEvalRequest", "RagasEvalResult",
]