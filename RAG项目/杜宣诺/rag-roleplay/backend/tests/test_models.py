import pytest
from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.base import Base
from app.db.models import Character, Document, Message, Session, User, MemoryTask


def test_pk_type_is_bigint_on_mysql_and_integer_on_sqlite():
    # 锁定 with_variant 的双端行为：MySQL 保持 BIGINT，SQLite 退化为 INTEGER。
    from sqlalchemy.schema import CreateTable
    mysql_ddl = str(CreateTable(User.__table__).compile(dialect=mysql.dialect()))
    sqlite_ddl = str(CreateTable(User.__table__).compile(dialect=sqlite.dialect()))

    assert "BIGINT" in mysql_ddl
    assert "INTEGER" in sqlite_ddl


async def test_models_create_tables():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    assert "documents" in Base.metadata.tables
    assert "users" in Base.metadata.tables
    await engine.dispose()


async def test_pk_autoincrements_on_sqlite():
    # SQLite 只对 INTEGER PRIMARY KEY 自动自增；验证 BigIntPK 的 with_variant 生效，
    # 否则后续任务的 SQLite 测试在插入 User 时会出现 NOT NULL 主键错误。
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with factory() as s:
        s.add(User(username="a", password_hash="x"))
        s.add(User(username="b", password_hash="y"))
        await s.commit()
    async with factory() as s:
        u = await s.get(User, 1)
        assert u is not None
        assert u.username == "a"
    await engine.dispose()
