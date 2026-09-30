from dataclasses import dataclass
from enum import Enum


class RoleName(str, Enum):
    # 超级管理员可执行后台高危配置和数据操作。
    SUPER_ADMIN = "super_admin"
    # 内容审核员只负责知识内容审核，不可修改安全白名单。
    CONTENT_REVIEWER = "content_reviewer"
    # 客服运营只能处理受限工单和用户支持场景。
    SUPPORT_OPERATOR = "support_operator"


SEEDABLE_ROLE_NAMES = tuple(role.value for role in RoleName)


@dataclass(frozen=True)
class Role:
    # 角色模型保持简单，后续可由迁移脚本按名称种子化。
    name: str
    # 描述不参与权限判断，仅用于后台展示。
    description: str = ""


def is_known_role(role: str) -> bool:
    # 权限依赖只接受预定义后台角色，避免拼写错误静默放权。
    return role in SEEDABLE_ROLE_NAMES
