"""MySQL 元数据存储：users / roles / sessions 的基础 CRUD。

只建表 + 增删改查，不接业务逻辑；import 不建立连接（Engine 惰性）。
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import ForeignKey, String, Text, create_engine, select
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()

_DEFAULT_URL = "mysql+pymysql://root:@127.0.0.1:3306/role_rag?charset=utf8mb4"
MYSQL_URL = os.getenv("MYSQL_URL", _DEFAULT_URL)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=datetime.now)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    collection_name: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=datetime.now)


class ChatSession(Base):
    """会话元数据；类名用 ChatSession 以免遮蔽 sqlalchemy.orm.Session。"""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # 对应 rag.py 里的 session_id 字符串；id 才是自增主键。
    session_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    # 挂用户/角色属于 B3.2 业务改造，这里先允许为空。
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    role_id: Mapped[int | None] = mapped_column(ForeignKey("roles.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=datetime.now)
    last_active_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=6), default=datetime.now, onupdate=datetime.now)


engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_all() -> None:
    """建表（幂等）。需保证 MYSQL_URL 指向的库已存在（见 init_db.py）。"""
    Base.metadata.create_all(engine)


@contextmanager
def get_db() -> Iterator[Session]:
    """提供一个事务 session：正常提交、异常回滚、最后关闭。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------- users ----------
def create_user(username: str) -> User:
    with get_db() as s:
        user = User(username=username)
        s.add(user)
        s.flush()  # 触发 INSERT，填充自增 id
        return user


def get_user(user_id: int) -> User | None:
    with get_db() as s:
        return s.get(User, user_id)


def get_user_by_username(username: str) -> User | None:
    with get_db() as s:
        return s.scalars(select(User).where(User.username == username)).first()


def update_user(user_id: int, username: str) -> User | None:
    with get_db() as s:
        user = s.get(User, user_id)
        if user is None:
            return None
        user.username = username
        return user


def delete_user(user_id: int) -> bool:
    with get_db() as s:
        user = s.get(User, user_id)
        if user is None:
            return False
        s.delete(user)
        return True


# ---------- roles ----------
def create_role(name: str, persona: str, collection_name: str) -> Role:
    with get_db() as s:
        role = Role(name=name, persona=persona, collection_name=collection_name)
        s.add(role)
        s.flush()
        return role


def get_role(role_id: int) -> Role | None:
    with get_db() as s:
        return s.get(Role, role_id)


def get_role_by_name(name: str) -> Role | None:
    with get_db() as s:
        return s.scalars(select(Role).where(Role.name == name)).first()


def update_role(role_id: int, **fields) -> Role | None:
    """按字段更新角色，只接受 name/persona/collection_name，其余忽略。"""
    allowed = {"name", "persona", "collection_name"}
    with get_db() as s:
        role = s.get(Role, role_id)
        if role is None:
            return None
        for key, value in fields.items():
            if key in allowed:
                setattr(role, key, value)
        return role


def delete_role(role_id: int) -> bool:
    with get_db() as s:
        role = s.get(Role, role_id)
        if role is None:
            return False
        s.delete(role)
        return True


# ---------- sessions ----------
def _find_session(s: Session, session_id: str) -> ChatSession | None:
    return s.scalars(select(ChatSession).where(ChatSession.session_id == session_id)).first()


def create_session(
    session_id: str, user_id: int | None = None, role_id: int | None = None
) -> ChatSession:
    with get_db() as s:
        sess = ChatSession(session_id=session_id, user_id=user_id, role_id=role_id)
        s.add(sess)
        s.flush()
        return sess


def get_session(session_id: str) -> ChatSession | None:
    with get_db() as s:
        return _find_session(s, session_id)


def touch_session(session_id: str) -> ChatSession | None:
    """刷新 last_active_at 为当前时间；不存在返回 None。"""
    with get_db() as s:
        sess = _find_session(s, session_id)
        if sess is None:
            return None
        sess.last_active_at = datetime.now()
        return sess


def delete_session(session_id: str) -> bool:
    with get_db() as s:
        sess = _find_session(s, session_id)
        if sess is None:
            return False
        s.delete(sess)
        return True
