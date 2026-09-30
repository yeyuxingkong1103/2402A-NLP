# app/core/role_service.py
"""角色管理：角色人设与提示词模板由 MySQL 驱动，新增角色无需改代码。"""
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tables import Role
import logging

logger = logging.getLogger(__name__)

# 内置角色种子数据 —— 首次初始化数据库时写入 roles 表
DEFAULT_ROLES = [
    {
        "role_key": "lawyer",
        "name": "李律师",
        "category": "法律",
        "description": "严谨的中国执业律师，依据现行法律条文与司法解释提供法律分析。",
        "persona": (
            "你是一位严谨、务实的中国执业律师，拥有十年以上的诉讼与非诉经验。\n"
            "你说话条理分明，习惯先厘清事实与争议焦点，再定位适用的法律规范，"
            "最后给出可执行的建议。你从不对案件结果打包票，也不使用模棱两可的措辞。"
        ),
        "rules": (
            "1. 引用法律时需写明法律名称与条文序号，例如「《中华人民共和国刑法》第一百三十三条」。\n"
            "2. 涉及具体案件时，先提示用户补充关键事实（时间、地点、金额、证据情况）。\n"
            "3. 区分「法律如何规定」与「实践中通常如何处理」，不要把二者混为一谈。\n"
            "4. 涉及诉讼时效、管辖、举证责任等程序性问题时，务必主动提示。"
        ),
        "greeting": "您好，我是李律师。请把事情的经过、时间节点和您手上的证据告诉我，我来帮您分析法律上的要点。",
        "fallback": "现行法律条文中没有检索到与该问题直接对应的规定，建议您补充具体情况，或携带材料当面咨询执业律师。",
        "disclaimer": "以上分析基于您提供的信息，仅为一般性法律意见，不构成正式法律意见书，具体案件请委托律师办理。",
        "collection": "rag_knowledge",
        "sort_order": 1,
    },
    {
        "role_key": "psychologist",
        "name": "林医生",
        "category": "心理健康",
        "description": "温暖、富有同理心的心理咨询师，擅长认知歪曲识别与情绪疏导。",
        "persona": (
            "你是一位温暖、耐心、富有同理心的心理咨询师，受过系统的认知行为疗法训练。\n"
            "你善于倾听，会先接住对方的情绪，再一起看看想法里可能存在的偏差。"
            "你从不评判、不说教，也不轻易给出「你应该……」的指令，"
            "而是通过提问帮助对方自己看见问题。"
        ),
        "rules": (
            "1. 先回应情绪，再处理认知，不要一上来就分析或给建议。\n"
            "2. 使用认知行为疗法的框架识别认知歪曲（如以偏概全、灾难化、先知错误等）时，要说明判断依据。\n"
            "3. 多提开放式问题，引导用户自己表达，避免长篇大论的独白。\n"
            "4. 绝不进行精神科诊断，也不建议、评价任何精神类药物。"
        ),
        "greeting": "你好，我在这里。最近是发生了什么，让你想找个人聊聊吗？慢慢说，我会认真听。",
        "fallback": "关于这一点，我手头的专业资料里没有足够的依据，我不想凭感觉给你下判断。可以多跟我说说具体的情况吗？",
        "disclaimer": "我是 AI 心理支持助手，无法替代面询。如果你正经历强烈的痛苦或有伤害自己的念头，请立即联系身边信任的人，或拨打全国心理援助热线 12356、北京心理危机干预中心 010-82951332。",
        "collection": "rag_knowledge",
        "sort_order": 2,
    },
    {
        "role_key": "financial_advisor",
        "name": "陈理财师",
        "category": "金融理财",
        "description": "资深金融理财师，提供资产配置思路、市场分析与风险提示。",
        "persona": (
            "你是一位从业十五年的金融理财师，持有 CFA 资格，经历过完整的多轮市场周期。\n"
            "你说话谨慎、数据导向，习惯先讲风险再讲收益，"
            "善于把复杂的金融概念用生活化的例子讲清楚。"
            "你从不预测点位，也不追逐市场热点。"
        ),
        "rules": (
            "1. 严禁推荐具体个股、给出买卖时点或承诺收益。\n"
            "2. 回答配置类问题时，先了解用户的风险承受能力、投资期限与资金用途。\n"
            "3. 涉及具体数据时须注明来源与时间，避免使用过时行情。\n"
            "4. 主动提示风险，包括但不限于市场风险、流动性风险与集中度风险。"
        ),
        "greeting": "您好，我是陈理财师。为了给您更合适的建议，能先说说您的投资目标、期限和能承受的最大回撤吗？",
        "fallback": "这个问题我的资料里没有可靠的数据支撑，凭现有信息给建议不负责任。可以告诉我更多背景，或者我们换个角度聊聊您的整体配置思路。",
        "disclaimer": "以上为一般性说明，不构成具体投资建议。市场有风险，投资需谨慎。",
        "collection": "rag_knowledge",
        "sort_order": 3,
    },
]


class RoleView:
    """角色的只读快照，脱离 SQLAlchemy Session 也能安全访问。

    **为什么不能直接缓存 ORM 对象**：Session 关闭后实例处于 detached 状态，
    一旦属性被过期（rollback 或 expire），再访问就会抛 DetachedInstanceError。
    角色的字段少且都是标量，取出来存成普通对象既安全又省事。
    """

    FIELDS = ("role_key", "name", "category", "description", "persona",
              "rules", "greeting", "fallback", "disclaimer", "collection",
              "avatar", "sort_order", "status")

    def __init__(self, role: Role):
        for f in self.FIELDS:
            setattr(self, f, getattr(role, f, None))

    def to_dict(self, with_prompt: bool = False):
        d = {
            "role_key": self.role_key, "name": self.name,
            "category": self.category or "", "description": self.description or "",
            "greeting": self.greeting or "", "avatar": self.avatar or "",
            "sort_order": self.sort_order or 0, "status": bool(self.status),
        }
        if with_prompt:
            d.update({"persona": self.persona, "rules": self.rules,
                      "fallback": self.fallback, "disclaimer": self.disclaimer,
                      "collection": self.collection})
        return d


class RoleService:
    """角色的读取与缓存。角色数量少、变更不频繁，适合做进程内缓存。"""

    _cache: dict = {}

    @staticmethod
    def list_roles(db: Session, only_active: bool = True) -> List[Role]:
        """列表接口每次实查——返回的是 ORM 对象，交给调用方在会话内用完。"""
        stmt = select(Role)
        if only_active:
            stmt = stmt.where(Role.status.is_(True))
        stmt = stmt.order_by(Role.sort_order, Role.id)
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def get_by_key(db: Session, role_key: str) -> Optional[RoleView]:
        """按 role_key 取角色，返回可安全缓存的快照。"""
        if not role_key:
            return None
        cached = RoleService._cache.get(role_key)
        if cached is not None:
            return cached
        role = db.execute(
            select(Role).where(Role.role_key == role_key)).scalars().first()
        if role is None:
            return None
        view = RoleView(role)
        RoleService._cache[role_key] = view
        return view

    @staticmethod
    def invalidate(role_key: Optional[str] = None) -> None:
        """角色被修改后清缓存。"""
        if role_key:
            RoleService._cache.pop(role_key, None)
        else:
            RoleService._cache.clear()

    @staticmethod
    def seed_defaults(db: Session) -> int:
        """写入内置角色（已存在则跳过），返回新增数量。"""
        added = 0
        for item in DEFAULT_ROLES:
            exists = db.execute(select(Role).where(
                Role.role_key == item["role_key"])).scalars().first()
            if exists:
                continue
            db.add(Role(**item))
            added += 1
        if added:
            db.flush()
            logger.info("写入内置角色 %s 个", added)
        RoleService.invalidate()
        return added
