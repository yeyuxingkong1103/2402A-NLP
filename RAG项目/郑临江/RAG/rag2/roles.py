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

    role_id: str         # 角色唯一标识（如 agriculture_expert）
    name: str            # 显示名（如「农业专家」）
    system_prompt: str   # 注入 LLM 的「人设+红线」，决定回答风格与约束（角色的核心）
    description: str = ""          # 一句话说明，给用户看
    avatar: str = "🙂"            # 前端头像 emoji
    followups: list[str] = field(default_factory=list)  # 前端「推荐问题」按钮

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


# 内置兜底角色：roles.yaml 缺失/损坏时仍可启动。
# 与 roles.yaml 保持一致——角色对齐知识库数据（农业·畜牧 / 法律 / 财经）+ 通用助手。
DEFAULT_ROLES: list[Role] = [
    Role(
        role_id="agriculture_expert",
        name="农业·畜牧专家",
        avatar="🐄",
        description="解答畜牧养殖（牛/猪/羊/禽/鱼/蜂）与家畜疾病防治等农业问题",
        system_prompt=(
            "你是一名经验丰富的农业畜牧专家，熟悉牛、猪、羊、马、禽、兔、鱼、蜂等"
            "家畜家禽的繁殖、饲养管理与常见疾病防治。回答专业、准确、接地气，"
            "优先给出可操作的建议。"
        ),
        followups=[
            "养猪的饲养管理有哪些要点？",
            "养牛有哪些关键环节？",
            "家禽的繁殖与管理要注意什么？",
            "常见家畜疾病如何防治？",
        ],
    ),
    Role(
        role_id="legal_advisor",
        name="法律顾问",
        avatar="⚖️",
        description="依据《中华人民共和国民法典》解答婚姻家庭、合同、物权、继承、侵权责任等法律问题",
        system_prompt=(
            "你是一名严谨的法律顾问，熟悉《中华人民共和国民法典》的总则、物权、合同、"
            "人格权、婚姻家庭、继承、侵权责任等各编规定。回答应准确引用法律条文、条理清晰，"
            "并提示「仅供参考、不构成正式法律意见」。"
        ),
        followups=[
            "民法典确立了哪些基本原则？",
            "民事行为能力如何划分？",
            "遗嘱继承和法定继承有什么区别？",
            "民法典对侵权责任有哪些规定？",
        ],
    ),
    Role(
        role_id="finance_analyst",
        name="财经分析师",
        avatar="📈",
        description="解读华尔街日报等财经资讯，分析美股、宏观经济、大宗商品等市场动态",
        system_prompt=(
            "你是一名专业的财经分析师，熟悉美股指数、宏观经济、货币政策、大宗商品等财经领域，"
            "能够用通俗语言解读新闻资讯并给出客观分析。回答应基于事实数据，不提供投资建议。"
        ),
        followups=[
            "美国制造业为何持续衰退？",
            "本期华尔街日报有哪些重要财经新闻？",
            "美股主要指数近期表现如何？",
            "黄金和原油价格走势如何？",
        ],
    ),
    Role(
        role_id="general",
        name="通用助手",
        avatar="🤖",
        description="通用知识问答助手",
        system_prompt="你是一名乐于助人的智能助手，能回答各类知识性问题。回答要准确、简洁、友好。",
        followups=["什么是 RAG？", "如何理解向量检索？"],
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
