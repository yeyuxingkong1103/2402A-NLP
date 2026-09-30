from collections.abc import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """所有 SQLAlchemy ORM 模型的基类。"""

    pass


settings = get_settings()
# 使用异步引擎，避免数据库 I/O 阻塞 FastAPI 的事件循环。
engine = create_async_engine(settings.database_url, echo=False, future=True)
# expire_on_commit=False 让提交后对象仍可直接读取属性，适合 API 返回 ORM 数据。
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # 每个请求获取一个独立会话；请求结束后由上下文管理器自动释放连接。
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    # create_all 只创建缺少的表，不会自动删除已有数据。
    from app.storage.models import Document, Role  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_run_lightweight_migrations)


def _run_lightweight_migrations(sync_connection) -> None:
    """Add newly introduced nullable text fields for existing development databases."""
    inspector = inspect(sync_connection)
    if "roles" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("roles")}
    # 项目使用轻量级迁移兼容早期 SQLite 数据库；生产环境建议使用 Alembic。
    for name in ("personality", "expertise", "speaking_style", "safety_policy"):
        if name not in existing:
            sync_connection.execute(text(f"ALTER TABLE roles ADD COLUMN {name} TEXT DEFAULT ''"))
