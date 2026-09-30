"""FastAPI 应用入口：初始化数据库、默认账号/角色并注册四组业务路由。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select

from app.auth import hash_password
from app.config import get_settings
from app.database import SessionLocal, initialize_database
from app.models import Role, User
from app.routers import auth_routes, chat_routes, knowledge_routes, role_routes

settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """你是严谨、耐心的医疗健康教育助手。
你的职责是根据知识库解释一般健康知识，帮助用户理解指南和就医准备。
你不能冒充医生、诊断疾病、替代面诊、擅自推荐处方药或调整剂量。
遇到急症信号时，优先建议立即拨打120或前往急诊。
资料不足时直接说明，并建议咨询有资质的医疗专业人员。"""


def seed_defaults() -> None:
    """首次启动时创建 .env 指定的管理员和默认医疗角色。"""
    with SessionLocal() as database:
        admin = database.scalar(select(User).where(User.username == settings.admin_username))
        if not admin:
            database.add(
                User(
                    username=settings.admin_username,
                    password_hash=hash_password(settings.admin_password),
                    is_admin=True,
                )
            )
        role = database.scalar(select(Role).where(Role.name == "医疗健康教育助手"))
        if not role:
            database.add(
                Role(
                    name="医疗健康教育助手",
                    description="依据公开指南回答一般健康教育问题，不提供诊断。",
                    system_prompt=DEFAULT_PROMPT,
                )
            )
        database.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """创建基础数据，并恢复上次关机时中断的 PDF 入库任务。"""
    initialize_database()
    seed_defaults()
    knowledge_routes.recover_processing_documents()
    logger.info("Application initialized in %s mode", settings.app_env)
    yield


app = FastAPI(
    title="Medical Role RAG API",
    version="1.0.0",
    description="基于 DeepSeek、轻量 BGE、Milvus 和 Redis 的医疗健康教育 RAG。",
    lifespan=lifespan,
)
app.include_router(auth_routes.router, prefix="/api/v1")
app.include_router(role_routes.router, prefix="/api/v1")
app.include_router(knowledge_routes.router, prefix="/api/v1")
app.include_router(chat_routes.router, prefix="/api/v1")


@app.get("/health", tags=["系统"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}
