# -*- coding: utf-8 -*-
"""数据库连接与会话管理（同步 SQLAlchemy + pymysql）。"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from . import config


class Base(DeclarativeBase):
    pass


engine = create_engine(
    config.DB_URL,
    echo=config.SQL_ECHO,
    pool_pre_ping=True,     # 长连接失效自动重连
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每请求一个会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """建表（幂等）。需先 import models 让元数据注册。"""
    from .. import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
