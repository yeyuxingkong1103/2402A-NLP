from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.models.base import Base

settings = get_settings()

_engine_kwargs: dict = {}
if settings.database_url.startswith("sqlite"):
    _engine_kwargs = {
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False},
    }

engine = create_async_engine(settings.database_url, echo=False, **_engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


@asynccontextmanager
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
