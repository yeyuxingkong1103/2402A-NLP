import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base, get_db
from app.main import create_app


class _NoopMilvus:
    """角色创建/编辑/删除会触发设定索引，测试环境用桩替代真实 Milvus。"""
    def __init__(self):
        self.calls = []

    async def delete_by_filter(self, collection, filter_expr):
        self.calls.append(("delete", collection, filter_expr))

    async def upsert_settings(self, character_id, chunks):
        self.calls.append(("upsert", character_id, len(chunks)))


@pytest.fixture
async def client(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with factory() as s:
            yield s

    # 角色 CRUD 现在会触发 Milvus 设定索引，全局桩掉，避免测试打真实 Milvus
    import app.api.routes.characters as characters_mod
    monkeypatch.setattr(characters_mod, "get_milvus", lambda: _NoopMilvus())

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()
