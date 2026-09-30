"""角色注册表与角色护栏。

角色定义放在 configs/roles.yaml（11 个内置角色），本模块负责：

* 加载 / 校验角色表，提供按 id 查询与「启用角色」列表；
* 组装角色人设提示词（persona + guardrails）；
* 角色护栏兜底：答案命中风险词时，强制追加免责声明（不依赖模型自觉）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import Config
from .errors import ConfigError, NotFoundError

# ---------------------------------------------------------------------------
# 风险词表：命中即在答案末尾强制补免责声明。
# key 为角色 id，"*" 表示对所有角色生效。
# ---------------------------------------------------------------------------
RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "*": ("包治百病", "绝对安全", "零风险", "百分之百有效"),
    "financial_planner": (
        "稳赚", "必赚", "一定涨", "保证收益", "保本保息", "无风险高收益",
        "推荐买", "满仓", "全部投入", "抄底", "梭哈",
    ),
    "stock_analyst": ("必涨", "涨停", "抄底", "满仓", "一定涨", "目标价", "买入评级"),
    "lawyer": ("包赢", "保证胜诉", "肯定能赢", "百分之百胜诉", "一定赢"),
    "doctor": ("可以确诊", "不用去医院", "自己吃药就行", "按这个剂量吃"),
    "psychologist": ("你就是抑郁症", "确诊为", "停药", "加药"),
    "scientist": ("绝对证明", "已经彻底证实", "100% 确定", "百分百确定"),
    "english_tutor": ("绝对正确", "唯一正确说法"),
}

DISCLAIMER_MARK = "⚠️"

# 用户问题本身命中风险词时的「前置纠正」指令（优先于知识片段）
SAFETY_TEMPLATE = """# 安全纠正（优先级最高）
用户的问题中出现了这些表述：{hits}
它们预设了「{role_name}」明确不允许的结论。你必须：
1. 先用一句话指出该预设不成立（例如：不存在保证收益的产品 / 无法承诺结果）；
2. 再依据下面的知识片段给出客观、中立的说明；
3. 不得顺着用户的假设继续作答，不得给出保证性结论或具体操作建议。"""


@dataclass(slots=True)
class Role:
    """单个角色定义。"""

    id: str
    name: str
    persona: str = ""
    avatar: str = "🙂"
    category: str = "通用"
    tagline: str = ""
    enabled: bool = False
    kb_dirs: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    disclaimer: str = ""
    followups: list[str] = field(default_factory=list)
    temperature: float = 0.35
    max_new_tokens: int = 700

    @property
    def kb_scopes(self) -> list[str]:
        """该角色在 Milvus 中的过滤作用域（shared 表示公共知识库）。"""

        scopes = list(self.kb_dirs)
        if "shared" not in scopes:
            scopes.append("shared")
        return scopes

    def persona_block(self) -> str:
        """拼装进系统提示词的人设段落。"""

        lines = [f"# 你的角色：{self.name}", self.persona.strip()]
        if self.guardrails:
            lines.append("\n# 你的红线（必须遵守，违反即为回答失败）")
            lines.extend(f"- {item}" for item in self.guardrails)
        return "\n".join(line for line in lines if line)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "avatar": self.avatar,
            "category": self.category,
            "tagline": self.tagline,
            "enabled": self.enabled,
            "kb_dirs": list(self.kb_dirs),
            "followups": list(self.followups),
            "temperature": self.temperature,
        }


class RoleRegistry:
    """角色表：加载、查询、护栏判定。"""

    def __init__(self, roles: Iterable[Role], defaults: dict[str, Any] | None = None) -> None:
        self._roles: dict[str, Role] = {role.id: role for role in roles}
        self.defaults = defaults or {}

    # ---------------------------------------------------------------- 加载
    @classmethod
    def from_config(cls, config: Config) -> "RoleRegistry":
        payload = config.roles_config()
        defaults = payload.get("defaults") or {}
        raw_roles = payload.get("roles") or []
        if not raw_roles:
            raise ConfigError("configs/roles.yaml 未定义任何角色")

        roles: list[Role] = []
        seen: set[str] = set()
        for item in raw_roles:
            role_id = str(item.get("id", "")).strip()
            if not role_id:
                raise ConfigError("roles.yaml 中存在缺少 id 的角色")
            if role_id in seen:
                raise ConfigError(f"roles.yaml 中角色 id 重复：{role_id}")
            seen.add(role_id)
            if not re.fullmatch(r"[a-z0-9_]+", role_id):
                raise ConfigError(f"角色 id 只允许小写字母/数字/下划线：{role_id}")
            roles.append(
                Role(
                    id=role_id,
                    name=str(item.get("name", role_id)),
                    persona=str(item.get("persona", "")).strip(),
                    avatar=str(item.get("avatar", "🙂")),
                    category=str(item.get("category", "通用")),
                    tagline=str(item.get("tagline", "")),
                    enabled=bool(item.get("enabled", False)),
                    kb_dirs=[str(x) for x in (item.get("kb_dirs") or [])],
                    guardrails=[str(x) for x in (item.get("guardrails") or [])],
                    disclaimer=str(item.get("disclaimer", "")).strip(),
                    followups=[str(x) for x in (item.get("followups") or [])],
                    temperature=float(item.get("temperature", defaults.get("temperature", 0.35))),
                    max_new_tokens=int(item.get("max_new_tokens", defaults.get("max_new_tokens", 700))),
                )
            )
        return cls(roles, defaults)

    # ---------------------------------------------------------------- 查询
    def get(self, role_id: str) -> Role:
        role = self._roles.get(role_id)
        if role is None:
            raise NotFoundError(f"角色不存在：{role_id}", role_id=role_id)
        return role

    def require_enabled(self, role_id: str) -> Role:
        role = self.get(role_id)
        if not role.enabled:
            raise NotFoundError(f"角色未启用：{role_id}", role_id=role_id)
        return role

    def all(self) -> list[Role]:
        return list(self._roles.values())

    def enabled(self) -> list[Role]:
        return [role for role in self._roles.values() if role.enabled]

    def enabled_ids(self) -> list[str]:
        return [role.id for role in self.enabled()]

    def __contains__(self, role_id: object) -> bool:
        return role_id in self._roles

    def __len__(self) -> int:
        return len(self._roles)

    # ---------------------------------------------------------------- 护栏
    def risk_hits(self, role: Role, text: str) -> list[str]:
        patterns = list(RISK_KEYWORDS.get("*", ())) + list(RISK_KEYWORDS.get(role.id, ()))
        return [word for word in patterns if word in text]

    def apply_guardrails(self, role: Role, answer: str) -> tuple[str, list[str]]:
        """命中风险词时补免责声明；返回 (最终答案, 命中的风险词)。"""

        hits = self.risk_hits(role, answer)
        if not hits or not role.disclaimer:
            return answer, hits
        if DISCLAIMER_MARK in answer or role.disclaimer.strip() in answer:
            return answer, hits
        return f"{answer.rstrip()}\n\n{role.disclaimer.strip()}", hits

    def safety_notice(self, role: Role, question: str) -> str:
        """问题本身踩到红线时，生成前置纠正指令（空字符串表示无需纠正）。"""

        hits = self.risk_hits(role, question)
        if not hits:
            return ""
        return SAFETY_TEMPLATE.format(hits="、".join(hits), role_name=role.name)
