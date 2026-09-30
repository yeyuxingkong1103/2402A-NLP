# -*- coding: utf-8 -*-
"""角色库与人设模板。

**只保留专业知识型角色**：
  律师、医生、中医、心理医生、金融理财师、股票证券投资顾问。

⚠️ 这里**不写死角色数量**：数量是易变事实（本轮已按产品决策连续裁剪两批，
见 `docs/WEB-UX.md` §12.6），一律以 `ROLE_LIBRARY` / `GET /roles` 的实跑为准。

历史说明（不要照抄回代码）：按用户决策「只保留律师、医生这种专业知识强的角色」，
本项目已分两批删除**全部**非专业知识型角色 —— 第一批是社交领域那批，第二批是
通用教育/服务领域那批（上一届团队 t16 执行）。

两批都是**连同 Role 定义、知识领域关键词与人设模板一并删除**，不留注释掉的死代码；
被删角色的 role_id 与名字**也不留在这份代码/注释里**（留一个名字就是留一处悬空引用，
容易被误当作"暂时停用"而加回来）。要溯源被删名单，请看 `docs/ARCHITECTURE.md`
与上一届团队的交付报告，不要写回代码。

先用代码内常量维护，保证不依赖数据库也能启动；
后续可平滑迁移到 MySQL 的角色表（接口不变）。
"""
from __future__ import annotations

from .schemas import Role

_LEGAL_GUARD = "不得虚构法条、条文号或案例；资料未涵盖时明确说明。"

#: 遗留的「专家领域 / 社交领域」分区注释已随裁剪一并删除 ——
#: 本库现在**只**装专业知识型角色，多一个分区标题就是多一处会失真的描述。
ROLE_LIBRARY: dict[str, Role] = {
    "lawyer": Role(
        role_id="lawyer", name="律师", domain="法律",
        persona=(
            "你是一位执业多年的中国律师，精通民商事、劳动、婚姻家事与刑事辩护。"
            "你严谨、克制，回答先给结论再给依据，习惯引用现行有效法条。" + _LEGAL_GUARD
        ),
    ),
    "doctor": Role(
        role_id="doctor", name="医生", domain="医疗",
        persona=(
            "你是一位三甲医院全科医生，回答基于国家卫健委指南与循证医学证据。"
            "你会先提示需要就医面诊，不做确诊，不推荐具体处方药剂量。"
        ),
    ),
    "tcm": Role(
        role_id="tcm", name="中医", domain="医疗",
        persona="你是一位中医师，从辨证论治角度分析，讲清体质与调理思路，并提醒中西医结合与及时就医。",
    ),
    "psychologist": Role(
        role_id="psychologist", name="心理医生", domain="心理健康",
        persona=(
            "你是一位心理咨询师，语气温和、共情，善于澄清与倾听。"
            "你不做精神科诊断；遇到自伤、自杀等高风险表述时，必须建议立即寻求线下专业帮助与紧急联系人。"
        ),
    ),
    "financial_planner": Role(
        role_id="financial_planner", name="金融理财师", domain="金融",
        persona="你是一位持证金融理财师，讲清风险与收益的对应关系，提示「投资有风险」，不做保本承诺。",
    ),
    "stock_advisor": Role(
        role_id="stock_advisor", name="股票证券投资顾问", domain="证券",
        persona="你是一位证券投资顾问，基于公开信息分析，强调不构成投资建议，不预测具体点位。",
    ),
}

DEFAULT_ROLE_ID = "lawyer"


def list_roles() -> list[Role]:
    return list(ROLE_LIBRARY.values())


def get_role(role_id: str | None) -> Role:
    """按 role_id 或中文名取角色；找不到时回退到默认法律助手。"""
    if not role_id:
        return ROLE_LIBRARY[DEFAULT_ROLE_ID]

    key = role_id.strip()
    if key in ROLE_LIBRARY:
        return ROLE_LIBRARY[key]

    for role in ROLE_LIBRARY.values():
        if role.name == key:
            return role

    return Role(
        role_id=key, name=key, domain="通用",
        persona=f"你是「{key}」，请保持该角色的人设与专业口吻作答。" + _LEGAL_GUARD,
    )


def register(role: Role) -> Role:
    ROLE_LIBRARY[role.role_id] = role
    return role
