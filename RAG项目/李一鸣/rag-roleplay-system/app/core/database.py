from collections.abc import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
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
    for name in ("personality", "expertise", "speaking_style", "safety_policy"):
        if name not in existing:
            sync_connection.execute(text(f"ALTER TABLE roles ADD COLUMN {name} TEXT DEFAULT ''"))
