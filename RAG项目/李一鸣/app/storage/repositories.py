import json
import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.prompt import build_role_system_prompt
from app.storage.models import ConversationMessage, Document, Role, User
from app.storage.schemas import RoleCreate

logger = logging.getLogger(__name__)


async def seed_default_roles(session: AsyncSession) -> None:
    # 仅在空表时写入演示角色，保证开发环境首次启动即可聊天且重启不重复插入。
    existing = await session.scalar(select(Role.id).limit(1))
    if existing:
        return
    defaults = [
        RoleCreate(
            name="温柔的虚拟朋友",
            category="friend",
            description="擅长倾听和日常陪伴的虚拟朋友。",
            personality="温柔、真诚、幽默、尊重边界",
            expertise="情绪陪伴、日常交流、学习与生活建议",
            speaking_style="像可靠的朋友一样自然，不说教，适度追问",
        ),
        RoleCreate(
            name="循证健康助手",
            category="doctor",
            description="提供健康知识检索与就医建议的助手。",
            personality="谨慎、理性、耐心",
            expertise="高血压、健康管理、医学指南解读",
            speaking_style="先澄清信息，再给分层建议；避免诊断和处方",
            safety_policy="不能替代医生诊疗；出现急症信号时优先建议立即就医。",
        ),
        RoleCreate(
            name="法律信息助手",
            category="lawyer",
            description="基于法律条文、司法解释和案例的法律信息助手。",
            personality="严谨、中立、注重证据",
            expertise="法律条文、司法解释、案例检索",
            speaking_style="区分事实、法条、推断和风险；明确地域与时效限制",
            safety_policy="仅提供一般法律信息，不构成正式法律意见；复杂事项建议咨询执业律师。",
        ),
        RoleCreate(
            name="理性投资研究员",
            category="finance",
            description="帮助整理公开信息和投资研究框架。",
            personality="克制、客观、重视风险",
            expertise="公司研究、行业分析、风险识别",
            speaking_style="先列假设和数据来源，再给情景分析，不承诺收益",
            safety_policy="不提供个性化买卖指令；投资有风险，需结合个人情况独立决策。",
        ),
        RoleCreate(
            name="英语口语教练",
            category="english_teacher",
            description="通过对话、纠错和复述帮助用户练习英语。",
            personality="鼓励、耐心、清晰",
            expertise="英语会话、语法、词汇、发音练习",
            speaking_style="根据用户水平控制难度，先回应再给简短纠错",
        ),
    ]
    for data in defaults:
        session.add(
            Role(
                id=uuid4().hex,
                name=data.name,
                category=data.category,
                description=data.description,
                personality=data.personality,
                expertise=data.expertise,
                speaking_style=data.speaking_style,
                safety_policy=data.safety_policy,
                system_prompt=build_role_system_prompt(data),
                knowledge_scope=data.knowledge_scope,
            )
        )
    await session.commit()
    logger.info("default roles seeded: %s", len(defaults))


async def list_roles(session: AsyncSession) -> list[Role]:
    # 按创建时间返回角色，前端可以稳定地展示角色列表。
    result = await session.scalars(select(Role).order_by(Role.created_at))
    return list(result)


async def get_role(session: AsyncSession, role_id: str) -> Role | None:
    return await session.get(Role, role_id)


async def create_role(session: AsyncSession, data: RoleCreate) -> Role:
    role = Role(
        id=uuid4().hex,
        name=data.name,
        category=data.category,
        description=data.description,
        personality=data.personality,
        expertise=data.expertise,
        speaking_style=data.speaking_style,
        safety_policy=data.safety_policy,
        system_prompt=build_role_system_prompt(data),
        knowledge_scope=data.knowledge_scope,
    )
    session.add(role)
    await session.commit()
    await session.refresh(role)
    logger.info("role created: %s", role.id, extra={"role_id": role.id})
    return role


async def create_document(
    session: AsyncSession,
    filename: str,
    source: str,
    content_hash: str,
) -> Document:
    # 文档先以 processing 状态落库，入库成功后由 update_document 改为 ready。
    document = Document(
        id=uuid4().hex,
        filename=filename,
        source=source,
        content_hash=content_hash,
        status="processing",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def update_document(
    session: AsyncSession,
    document_id: str,
    status: str,
    chunk_count: int = 0,
    error_message: str = "",
) -> Document | None:
    document = await session.get(Document, document_id)
    if document is None:
        return None
    document.status = status
    document.chunk_count = chunk_count
    document.error_message = error_message
    document.updated_at = datetime.utcnow()
    await session.commit()
    await session.refresh(document)
    return document


async def list_documents(session: AsyncSession) -> list[Document]:
    result = await session.scalars(select(Document).order_by(desc(Document.created_at)))
    return list(result)


async def ensure_user(session: AsyncSession, user_id: str) -> User:
    # 聊天保存前确保用户存在，允许前端使用新的 user_id 直接开始会话。
    user = await session.get(User, user_id)
    if user is None:
        user = User(id=user_id, display_name=user_id)
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user


async def save_message(
    session: AsyncSession,
    user_id: str,
    role_id: str,
    conversation_id: str,
    message_type: str,
    content: str,
    citations: list[dict] | None = None,
) -> ConversationMessage:
    # 业务数据库保存完整对话；引用单独序列化，方便 API 返回和后续审计。
    await ensure_user(session, user_id)
    message = ConversationMessage(
        id=uuid4().hex,
        user_id=user_id,
        role_id=role_id,
        conversation_id=conversation_id,
        message_type=message_type,
        content=content,
        citations_json=json.dumps(citations or [], ensure_ascii=False),
    )
    session.add(message)
    await session.commit()
    return message
