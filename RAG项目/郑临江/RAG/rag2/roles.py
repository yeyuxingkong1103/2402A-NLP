# -*- coding: utf-8 -*-
"""角色定义与加载。

角色是「同一套检索 + 生成」之上的差异化人设：每个角色有自己的
``system_prompt``（人设 + 红线）与推荐问题，检索仍然共用同一个 Milvus 集合。

    from rag2 import load_roles, get_role

    roles = load_roles("roles.yaml")
    role = get_role("agriculture_expert", roles)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import PROJECT_ROOT
from .logging_config import get_logger

logger = get_logger("roles")


@dataclass
class Role:
    """单个角色定义。"""

    role_id: str
    name: str
    system_prompt: str
    description: str = ""
    avatar: str = "🙂"
    followups: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "description": self.description,
            "avatar": self.avatar,
            "followups": list(self.followups),
        }

    def to_dict(self) -> dict[str, Any]:
        """完整字段（含 system_prompt），供管理员后台读写 roles.yaml。"""
        return {
            "role_id": self.role_id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "avatar": self.avatar,
            "followups": list(self.followups),
        }


# 内置兜底角色：roles.yaml 缺失/损坏时仍可启动
DEFAULT_ROLES: list[Role] = [
    Role(
        role_id="agriculture_expert",
        name="农业专家",
        avatar="🌾",
        description="解答农作物种植、病虫害防治、施肥管理等农业问题",
        system_prompt=(
            "你是一名经验丰富的农业技术专家，熟悉作物种植、病虫害防治、土壤肥料、"
            "畜牧养殖等农业知识。你的回答要专业、准确、接地气，优先给出可操作的建议。"
        ),
        followups=["小麦常见病虫害有哪些？", "如何科学施肥提高产量？", "水稻种植的关键环节是什么？"],
    ),
    Role(
        role_id="tech_support",
        name="技术支持",
        avatar="🔧",
        description="解答软件、系统、设备使用与故障排查问题",
        system_prompt=(
            "你是一名耐心的技术支持工程师，擅长排查系统、软件与设备使用问题，"
            "能够把复杂的操作步骤讲清楚。回答要条理清晰、步骤明确、可执行。"
        ),
        followups=["如何排查服务无法启动的问题？", "Redis 连接失败怎么办？", "如何查看系统日志？"],
    ),
    Role(
        role_id="general",
        name="通用助手",
        avatar="🤖",
        description="通用知识问答助手",
        system_prompt="你是一名乐于助人的智能助手，能回答各类知识性问题。回答要准确、简洁、友好。",
        followups=["什么是 RAG？", "如何理解向量检索？"],
    ),
    Role(
        role_id="customer_service",
        name="客服",
        avatar="💬",
        description="面向用户的咨询与售后支持",
        system_prompt=(
            "你是一名专业的客服人员，态度亲切、有耐心，善于理解用户诉求并提供解决方案。"
            "回答要礼貌、周到、以用户为中心。"
        ),
        followups=["我想了解产品的使用方法", "遇到问题该如何反馈？"],
    ),
]


def _from_item(item: dict[str, Any]) -> Role:
    return Role(
        role_id=str(item.get("role_id", "")).strip(),
        name=str(item.get("name", item.get("role_id", ""))),
        system_prompt=str(item.get("system_prompt", "")).strip(),
        description=str(item.get("description", "")),
        avatar=str(item.get("avatar", "🙂")),
        followups=[str(x) for x in (item.get("followups") or [])],
    )


def load_roles(path: str | Path | None = None) -> list[Role]:
    """从 yaml 加载角色表；缺失/解析失败时回退内置默认角色。"""
    yaml_path = Path(path) if path else (PROJECT_ROOT / "roles.yaml")
    if not yaml_path.is_file():
        logger.warning("角色文件不存在，使用内置默认角色：%s", yaml_path)
        return list(DEFAULT_ROLES)
    try:
        import yaml  # 懒加载

        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        raw_roles = payload.get("roles") or []
        if not raw_roles:
            raise ValueError("roles.yaml 未定义任何角色")
        roles = [_from_item(item) for item in raw_roles if item.get("role_id")]
        if not roles:
            raise ValueError("roles.yaml 中缺少有效角色")
        return roles
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都回退默认
        logger.warning("角色文件加载失败，使用内置默认角色：%s", exc)
        return list(DEFAULT_ROLES)


def get_role(role_id: str, roles: Sequence[Role] | None = None) -> Role:
    """按 role_id 查询角色；不存在则抛 ``KeyError``。"""
    for role in (roles if roles is not None else DEFAULT_ROLES):
        if role.role_id == role_id:
            return role
    raise KeyError(f"角色不存在：{role_id}")


def list_roles(roles: Sequence[Role] | None = None) -> list[Role]:
    return list(roles if roles is not None else DEFAULT_ROLES)


def role_from_dict(item: dict[str, Any]) -> Role:
    """从字典构建角色（供管理员后台新增/编辑）。"""
    return _from_item(item)


def save_roles(roles: Sequence[Role], path: str | Path | None = None) -> Path:
    """把角色表落盘到 roles.yaml（覆盖，原子写入）。"""
    yaml_path = Path(path) if path else (PROJECT_ROOT / "roles.yaml")
    import yaml  # 懒加载

    header = (
        "# ============================================================\n"
        "# 内置角色表（由管理员后台在线保存，也可手改）\n"
        "# 每项：role_id（唯一，小写字母/数字/下划线）、name、system_prompt、\n"
        "#       description、avatar（前端表情）、followups（推荐问题）\n"
        "# ============================================================\n"
    )
    body = yaml.safe_dump(
        {"roles": [r.to_dict() for r in roles]},
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    tmp = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    tmp.write_text(header + body, encoding="utf-8")
    os.replace(tmp, yaml_path)
    return yaml_path
